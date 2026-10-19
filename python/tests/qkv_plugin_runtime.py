"""Run QKV checks without importing Triton, torch, or the cubin producer."""

import argparse
import gc
import json
import os
import sys
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from safetensors.numpy import load_file
from tensoracle import (
    BuilderSettings,
    CorruptEngineError,
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
NAME = "LffwFp16Qkv"
NAMESPACE = "listenflow.fastest_whisper"


def _creator(version="2"):
    creator = trt.get_plugin_registry().get_creator(NAME, version, NAMESPACE)
    assert creator is not None, "the single-token QKV plugin was not registered"
    return creator


def _creator_registered(version):
    return trt.get_plugin_registry().get_creator(NAME, version, NAMESPACE) is not None


def _fields(digest, weights):
    arrays = {
        "bundle_sha256": np.frombuffer(digest.encode("ascii"), dtype=np.int8),
        **load_file(weights),
    }
    return arrays, [
        trt.PluginField(
            name,
            value.ravel(),
            trt.PluginFieldType.CHAR if name == "bundle_sha256" else trt.PluginFieldType.FLOAT16,
        )
        for name, value in arrays.items()
    ]


def legacy_graph_for(digest, weights, *, shape=(1, 1, WIDTH), dtype=trt.float16):
    def graph(network):
        arrays, fields = _fields(digest, weights)
        plugin = _creator("1").create_plugin(
            "qkv", trt.PluginFieldCollection(fields), trt.TensorRTPhase.BUILD
        )
        assert plugin is not None, "valid QKV fields were rejected"
        # Plugin fields must own their data once create_plugin returns.
        for name, value in arrays.items():
            if name != "bundle_sha256":
                value.fill(np.nan)
        features = network.add_input("features", dtype, shape)
        layer = network.add_plugin_v3([features], [], plugin)
        assert layer is not None, "valid single-token QKV shape was rejected"
        output = layer.get_output(0)
        output.name = "qkv"
        network.mark_output(output)

    return graph


def graph_for(digest, weights, *, shape=(1, 1, WIDTH), dtype=trt.float16, constants=None):
    owners = []

    def graph(network):
        arrays = load_file(weights)
        arrays["qkv_weight"] = arrays["qkv_weight"].reshape(3 * WIDTH, WIDTH)
        for name, (constant_shape, constant_dtype) in (constants or {}).items():
            arrays[name] = np.zeros(constant_shape, dtype=constant_dtype)
        # TensorRT may borrow constant weight arrays until the build completes.
        owners.append(arrays)
        digest_bytes = np.frombuffer(digest.encode("ascii"), dtype=np.int8)
        plugin = _creator().create_plugin(
            "qkv",
            trt.PluginFieldCollection(
                [trt.PluginField("bundle_sha256", digest_bytes, trt.PluginFieldType.CHAR)]
            ),
            trt.TensorRTPhase.BUILD,
        )
        assert plugin is not None, "valid QKV bundle field was rejected"
        features = network.add_input("features", dtype, shape)
        inputs = [features]
        for name in ("qkv_weight", "q_bias", "v_bias"):
            values = arrays[name]
            constant = network.add_constant(values.shape, values)
            constant.name = name
            inputs.append(constant.get_output(0))
        layer = network.add_plugin_v3(inputs, [], plugin)
        assert layer is not None, "valid QKV input/constant shapes were rejected"
        output = layer.get_output(0)
        output.name = "qkv"
        network.mark_output(output)

    return graph


def declaration(shape=(1, 1, WIDTH)):
    return EngineBuild({"features": InputShape.fixed(shape)}, BuilderSettings(optimization_level=5))


def _reject(digest, weights, library):
    class FieldsChecked(Exception):
        pass

    rejected = 0

    def graph(_):
        nonlocal rejected
        arrays, fields = _fields(digest, weights)
        by_name = {field.name: field for field in fields}
        invalid = [
            [],
            fields,  # Version 1's weight-bearing schema is not version 2.
            [
                trt.PluginField(
                    "bundle_sha256", arrays["bundle_sha256"][:-1], trt.PluginFieldType.CHAR
                )
            ],
            [
                trt.PluginField(
                    "bundle_sha256",
                    np.frombuffer(b"0" * 64, dtype=np.int8),
                    trt.PluginFieldType.CHAR,
                )
            ],
            [
                trt.PluginField(
                    "bundle_sha256", np.zeros(64, dtype=np.float32), trt.PluginFieldType.FLOAT32
                )
            ],
            [
                by_name["bundle_sha256"],
                trt.PluginField("q_bias", arrays["q_bias"], trt.PluginFieldType.FLOAT16),
            ],
        ]
        for values in invalid:
            assert (
                _creator().create_plugin(
                    "bad-qkv", trt.PluginFieldCollection(values), trt.TensorRTPhase.BUILD
                )
                is None
            )
            rejected += 1
        raise FieldsChecked

    try:
        build_serialized_engine(graph, declaration(), plugins=(library,))
    except FieldsChecked:
        pass
    else:
        raise AssertionError("field checks did not run")
    inputs_rejected = 0
    invalid_inputs = [
        {"shape": shape, "dtype": dtype}
        for shape, dtype in (
            ((WIDTH,), trt.float16),
            ((1, 1, WIDTH - 1), trt.float16),
            ((1, 2, WIDTH), trt.float16),
            ((2, 1, WIDTH), trt.float16),
            ((1, 1, WIDTH), trt.float32),
        )
    ]
    invalid_inputs += [
        {"constants": {name: (shape, dtype)}}
        for name, shape, dtype in (
            ("qkv_weight", (WIDTH, 3 * WIDTH), np.float16),
            ("qkv_weight", (3, WIDTH, WIDTH), np.float16),
            ("q_bias", (1, WIDTH), np.float16),
            ("v_bias", (WIDTH - 1,), np.float16),
            ("qkv_weight", (3 * WIDTH, WIDTH), np.float32),
            ("q_bias", (WIDTH,), np.float32),
            ("v_bias", (WIDTH,), np.float32),
        )
    ]
    for arguments in invalid_inputs:
        try:
            build_serialized_engine(
                graph_for(digest, weights, **arguments),
                declaration(arguments.get("shape", (1, 1, WIDTH))),
                plugins=(library,),
            )
        except (RuntimeError, AssertionError):
            inputs_rejected += 1
        else:
            raise AssertionError(f"QKV accepted incompatible input: {arguments}")
    return {"fields_rejected": rejected, "inputs_rejected": inputs_rejected}


def infer_cases(handle, cases, *, execution=None):
    outputs = []
    errors = []
    with np.load(cases) as reference, ExitStack() as resources:
        if execution is None:
            execution = resources.enter_context(handle.create_execution())
        assert execution.get_tensor_shape("qkv") == (1, 1, 3 * WIDTH)
        assert execution.get_tensor_dtype("qkv") == trt.float16
        source = _cuda(cudart.cudaMalloc(WIDTH * 2))
        resources.callback(lambda: _cuda(cudart.cudaFree(source)))
        target = _cuda(cudart.cudaMalloc(3 * WIDTH * 2))
        resources.callback(lambda: _cuda(cudart.cudaFree(target)))
        stream = _cuda(cudart.cudaStreamCreate())
        resources.callback(lambda: _cuda(cudart.cudaStreamDestroy(stream)))
        resources.callback(lambda: _cuda(cudart.cudaStreamSynchronize(stream)))
        for values, expected in zip(reference["inputs"], reference["expected"], strict=True):
            actual = np.full((3, WIDTH), np.nan, dtype=np.float16)
            _cuda(
                cudart.cudaMemcpy(
                    source,
                    values.ctypes.data,
                    values.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )
            execution.enqueue({"features": int(source), "qkv": int(target)}, stream=int(stream))
            _cuda(cudart.cudaStreamSynchronize(stream))
            _cuda(
                cudart.cudaMemcpy(
                    actual.ctypes.data,
                    target,
                    actual.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
            )
            # The fp16 relative spacing bound (2^-10), plus 2^-12 near cancellation.
            # Products of these fp16 inputs are exact in fp32. For the fixed
            # cases, gamma_12 * sum(abs(x*w), abs(bias)) < 1.86e-5 bounds the
            # padded 2048-wide tree reduction and bias addition. NumPy's fp32
            # reference differs from a float64 calculation by at most 1.64e-6.
            # Their sum is below the 2^-12 absolute allowance near cancellation.
            np.testing.assert_allclose(actual, expected, rtol=2**-10, atol=2**-12)
            if np.count_nonzero(values) <= 1:
                np.testing.assert_array_equal(actual, expected)
            errors.append(
                float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
            )
            outputs.append(actual.tobytes())
    return {
        "cases": len(outputs),
        "shape": [1, 1, 3 * WIDTH],
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
    counts = dict.fromkeys(("alloc", "upload", "free", "load", "unload"), 0)
    live = {}
    uploaded = 0
    constants = {name: set() for name in ("weight", "q_bias", "v_bias")}
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
    parser.add_argument(
        "mode", choices=("build", "reuse", "reject", "contexts", "legacy-build", "legacy-reject")
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("library", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    digest = inspect_bundle(args.bundle, plugin=args.library)
    _cuda(cudart.cudaSetDevice(0))
    if args.mode == "legacy-build":
        plan = build_serialized_engine(
            legacy_graph_for(digest, args.weights), declaration(), plugins=(args.library,)
        )
        args.plan.write_bytes(plan)
        print(json.dumps({"path": str(args.plan)}))
        return
    if args.mode == "legacy-reject":
        # Exercise the same deserialization/error path used by open_engine,
        # using a raw legacy plan so cache-library identity cannot mask the
        # missing plugin version.
        from tensoracle._cache import _open_plan, register_libraries

        register_libraries((args.library,))
        try:
            handle = _open_plan(
                args.plan.read_bytes(),
                args.plan,
                declaration(),
                EngineIO(
                    inputs={"features": TensorSpec(trt.float16, (1, 1, WIDTH))},
                    outputs={"qkv": TensorSpec(trt.float16, (1, 1, 3 * WIDTH))},
                ),
                trt.Logger(trt.Logger.ERROR),
            )
            # TensorRT 11.4 defers plugin creation until the first execution
            # context, so opening a plan alone can still succeed.
            with handle.create_execution():
                pass
        except (CorruptEngineError, RuntimeError) as error:
            print(
                json.dumps(
                    {
                        "rejected": True,
                        "error": str(error),
                        "legacy_creator_present": _creator_registered("1"),
                    }
                )
            )
            return
        raise AssertionError("version-1 QKV engine was accepted by the upgraded plugin")
    if args.mode == "reject":
        print(json.dumps(_reject(digest, args.weights, args.library)))
        return
    graph = graph_for(digest, args.weights)
    if args.mode in ("reuse", "contexts"):

        def graph(_):
            raise AssertionError("saved QKV engine invoked graph or loaded its weight tensors")

    def open_handle():
        return open_engine(
            graph,
            declaration(),
            name="lffw-qkv",
            graph_version="1",
            weights=(args.weights,),
            plugins=(args.library,),
            io=EngineIO(
                inputs={"features": TensorSpec(trt.float16, (1, 1, WIDTH))},
                outputs={"qkv": TensorSpec(trt.float16, (1, 1, 3 * WIDTH))},
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
