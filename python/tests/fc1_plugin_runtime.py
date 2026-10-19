"""Run fc1 checks without importing Triton, torch, or the cubin producer."""

import argparse
import gc
import json
import os
import sys
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from safetensors.numpy import load_file
from tensoracle import (
    BuilderSettings,
    EngineBuild,
    EngineIO,
    InputShape,
    TensorSpec,
    build_serialized_engine,
    open_engine,
)
from tensoracle.bundle import inspect_bundle

from cubin_plugin_runtime import _cuda

WIDTH = 1280
OUTPUT = 5120
NAME = "LffwFp16Fc1"
NAMESPACE = "listenflow.fastest_whisper"


def _creator(version="1"):
    creator = trt.get_plugin_registry().get_creator(NAME, version, NAMESPACE)
    assert creator is not None, "the single-token fc1 plugin was not registered"
    return creator


def graph_for(
    digest, weights, *, shape=(1, 1, WIDTH), dtype=trt.float16, constants=None, production=True
):
    owners = []

    def graph(network):
        arrays = load_file(weights)
        for name, (constant_shape, constant_dtype) in (constants or {}).items():
            arrays[name] = np.zeros(constant_shape, dtype=constant_dtype)
        # TensorRT may borrow constant weight arrays until the build completes.
        owners.append(arrays)
        if production:
            from listenflow.fastest_whisper._network import _Graph

            checkpoint = SimpleNamespace(
                config=None, tensor=lambda key: arrays[key.rsplit(".", 1)[-1]]
            )
            product = _Graph(network, checkpoint, "fp16")
            owners.append(product)
            features = network.add_input("features", dtype, shape)
            output = product.decoder_fc1(features, "fc1")
            output.name = "fc1"
            network.mark_output(output)
            return
        digest_bytes = np.frombuffer(digest.encode("ascii"), dtype=np.int8)
        plugin = _creator().create_plugin(
            "fc1",
            trt.PluginFieldCollection(
                [trt.PluginField("bundle_sha256", digest_bytes, trt.PluginFieldType.CHAR)]
            ),
            trt.TensorRTPhase.BUILD,
        )
        assert plugin is not None, "valid fc1 bundle field was rejected"
        features = network.add_input("features", dtype, shape)
        inputs = [features]
        for name in ("weight", "bias"):
            values = arrays[name]
            constant = network.add_constant(values.shape, values)
            constant.name = name
            inputs.append(constant.get_output(0))
        layer = network.add_plugin_v3(inputs, [], plugin)
        assert layer is not None, "valid fc1 input/constant shapes were rejected"
        output = layer.get_output(0)
        output.name = "fc1"
        network.mark_output(output)

    return graph


def declaration(shape=(1, 1, WIDTH)):
    return EngineBuild({"features": InputShape.fixed(shape)}, BuilderSettings(optimization_level=5))


def _reject(digest, weights, library):
    from tensoracle._plugins import register_libraries

    register_libraries((library,))
    good = np.frombuffer(digest.encode("ascii"), dtype=np.int8)
    invalid = [
        [],
        [trt.PluginField("bundle_sha256", good[:-1], trt.PluginFieldType.CHAR)],
        [
            trt.PluginField(
                "bundle_sha256", np.frombuffer(b"0" * 64, dtype=np.int8), trt.PluginFieldType.CHAR
            )
        ],
        [trt.PluginField("bundle_sha256", np.zeros(64, np.float32), trt.PluginFieldType.FLOAT32)],
        [
            trt.PluginField("bundle_sha256", good, trt.PluginFieldType.CHAR),
            trt.PluginField("bias", np.zeros(OUTPUT, np.float16), trt.PluginFieldType.FLOAT16),
        ],
    ]
    for fields in invalid:
        assert (
            _creator().create_plugin(
                "bad", trt.PluginFieldCollection(fields), trt.TensorRTPhase.BUILD
            )
            is None
        )
    invalid_inputs = [
        {"shape": shape, "dtype": dtype}
        for shape, dtype in (
            ((WIDTH,), trt.float16),
            ((1, 2, WIDTH), trt.float16),
            ((1, 1, WIDTH - 1), trt.float16),
            ((1, 1, WIDTH), trt.float32),
        )
    ]
    invalid_inputs += [
        {"constants": {name: (shape, dtype)}}
        for name, shape, dtype in (
            ("weight", (WIDTH, OUTPUT), np.float16),
            ("weight", (1, OUTPUT, WIDTH), np.float16),
            ("weight", (OUTPUT, WIDTH), np.float32),
            ("bias", (OUTPUT - 1,), np.float16),
            ("bias", (OUTPUT,), np.float32),
        )
    ]
    for arguments in invalid_inputs:
        try:
            build_serialized_engine(
                graph_for(digest, weights, production=False, **arguments),
                declaration(arguments.get("shape", (1, 1, WIDTH))),
                plugins=(library,),
            )
        except (RuntimeError, AssertionError):
            pass
        else:
            raise AssertionError(f"Accepted incompatible fc1 input: {arguments}")
    return {"fields_rejected": len(invalid), "inputs_rejected": len(invalid_inputs)}


def infer_cases(handle, cases, *, execution=None):
    outputs = []
    errors = []
    with np.load(cases) as reference, ExitStack() as resources:
        if execution is None:
            execution = resources.enter_context(handle.create_execution())
        assert execution.get_tensor_shape("fc1") == (1, 1, OUTPUT)
        assert execution.get_tensor_dtype("fc1") == trt.float16
        source = _cuda(cudart.cudaMalloc(WIDTH * 2))
        resources.callback(lambda: _cuda(cudart.cudaFree(source)))
        target = _cuda(cudart.cudaMalloc(OUTPUT * 2))
        resources.callback(lambda: _cuda(cudart.cudaFree(target)))
        stream = _cuda(cudart.cudaStreamCreate())
        resources.callback(lambda: _cuda(cudart.cudaStreamDestroy(stream)))
        resources.callback(lambda: _cuda(cudart.cudaStreamSynchronize(stream)))
        for values, expected in zip(reference["inputs"], reference["expected"], strict=True):
            actual = np.full((OUTPUT,), np.nan, dtype=np.float16)
            _cuda(
                cudart.cudaMemcpy(
                    source,
                    values.ctypes.data,
                    values.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )
            execution.enqueue({"features": int(source), "fc1": int(target)}, stream=int(stream))
            _cuda(cudart.cudaStreamSynchronize(stream))
            _cuda(
                cudart.cudaMemcpy(
                    actual.ctypes.data,
                    target,
                    actual.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
            )
            # Each network edge rounds to half, including the dot before bias.
            # Float32 tree-order error may cross one half quantization boundary;
            # erf approximation and the 1+erf cancellation can add one half ULP.
            # 2^-10 relative (<=2 ULP) and 2^-11 absolute cover those boundaries
            # around GELU's negative lobe; no tanh approximation is accepted.
            np.testing.assert_allclose(actual, expected, rtol=2**-10, atol=2**-11)
            if "exact" in reference and reference["exact"].item():
                np.testing.assert_array_equal(actual, expected)
            errors.append(
                float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
            )
            outputs.append(actual.tobytes())
    return {
        "cases": len(outputs),
        "shape": [1, 1, OUTPUT],
        "max_abs_errors": errors,
        "output_sha256": sha256(b"".join(outputs)).hexdigest(),
    }


def _driver_events():
    path = Path(os.environ["LFFW_QKV_DRIVER_LOG"])
    return (
        [
            (event, int(context), int(pointer), int(size), int(status))
            for event, context, pointer, size, status in (
                line.split("|") for line in path.read_text().splitlines()
            )
        ]
        if path.exists()
        else []
    )


def _resource_snapshot():
    counts = dict.fromkeys(("alloc", "upload", "free", "load", "unload", "fc1_launch"), 0)
    live = {}
    uploaded = 0
    constants = {name: set() for name in ("fc1_weight", "fc1_bias")}
    for event, context, pointer, size, status in _driver_events():
        assert status == 0, (event, status)
        if event in constants:
            constants[event].add((context, pointer))
            continue
        counts[event] += 1
        if event == "alloc":
            live[context, pointer] = size
        elif event == "free":
            del live[context, pointer]
        elif event == "upload":
            uploaded += size
    rss = next(
        line.split()[1]
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("VmRSS:")
    )
    status, free, total = cudart.cudaMemGetInfo()
    _cuda((status,))
    return {
        **counts,
        "upload_bytes": uploaded,
        "live_bytes": sum(live.values()),
        "host_rss_bytes": int(rss) * 1024,
        "device_used_bytes": int(total - free),
        "constant_pointers": {name: sorted(pointers) for name, pointers in constants.items()},
    }


def context_lifecycle(handle, cases):
    """Use real TRT contexts; observer counts only calls made by our plugin library."""
    metadata = {"path": str(handle.path), "sha256": sha256(handle.path.read_bytes()).hexdigest()}
    resources = {}
    executions = []
    hashes = []
    try:
        for index in range(4):
            execution = handle.create_execution()
            executions.append(execution)
            hashes.append(infer_cases(None, cases, execution=execution)["output_sha256"])
            if index == 0:
                resources["one_context"] = _resource_snapshot()
        resources["four_contexts"] = _resource_snapshot()
        executions.pop(0).close()
        survivor_hashes = [
            infer_cases(None, cases, execution=item)["output_sha256"] for item in executions
        ]
        resources["one_closed"] = _resource_snapshot()
        replacement = handle.create_execution()
        executions.append(replacement)
        replacement_hash = infer_cases(None, cases, execution=replacement)["output_sha256"]
        resources["replacement"] = _resource_snapshot()
        # The remaining executions retain the engine. Its last plugin owner must
        # release the resources even after the caller has discarded its handle.
        del handle
        while len(executions) > 1:
            executions.pop(0).close()
        resources["last_owner"] = _resource_snapshot()
    finally:
        for item in executions:
            item.close()
    gc.collect()
    resources["closed"] = _resource_snapshot()
    events = _driver_events()
    balanced = all(
        sorted((context, pointer) for event, context, pointer, _, _ in events if event == acquire)
        == sorted(
            (context, pointer) for event, context, pointer, _, _ in events if event == release
        )
        for acquire, release in (("alloc", "free"), ("load", "unload"))
    )
    return {
        **metadata,
        "resources": resources,
        "output_hashes": hashes,
        "survivor_hashes": survivor_hashes,
        "replacement_hash": replacement_hash,
        "balanced_pointers": balanced,
        "cuda_contexts": len({context for _, context, _, _, _ in events}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "reuse", "reject", "contexts"))
    parser.add_argument("bundle", type=Path)
    parser.add_argument("library", type=Path)
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()
    from listenflow.fastest_whisper import _plugins

    _plugins._fc1_library = lambda: args.library
    digest = inspect_bundle(args.bundle, plugin=args.library)
    _cuda(cudart.cudaSetDevice(0))
    if args.mode == "reject":
        print(json.dumps(_reject(digest, args.weights, args.library)))
        return
    graph = graph_for(digest, args.weights)
    if args.mode in ("reuse", "contexts"):

        def graph(_):
            raise AssertionError("saved fc1 engine invoked graph or loaded its weight tensors")

    def open_handle():
        return open_engine(
            graph,
            declaration(),
            name="lffw-fc1",
            graph_version="1",
            weights=(args.weights,),
            plugins=(args.library,),
            io=EngineIO(
                inputs={"features": TensorSpec(trt.float16, (1, 1, WIDTH))},
                outputs={"fc1": TensorSpec(trt.float16, (1, 1, OUTPUT))},
            ),
        )

    if args.mode == "contexts":
        result = context_lifecycle(open_handle(), args.weights.with_suffix(".cases.npz"))
        result["producer_loaded"] = any(
            name in sys.modules for name in ("tensoracle.producer", "triton", "torch")
        )
        print(json.dumps(result))
        return
    handle = open_handle()
    result = infer_cases(handle, args.weights.with_suffix(".cases.npz"))
    result.update(
        path=str(handle.path),
        sha256=sha256(handle.path.read_bytes()).hexdigest(),
        producer_loaded=any(
            name in sys.modules for name in ("tensoracle.producer", "triton", "torch")
        ),
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
