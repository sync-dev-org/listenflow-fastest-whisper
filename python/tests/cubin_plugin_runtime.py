"""Exercise the development plugin in a process that never imports its producer."""

import argparse
import json
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
    EngineBuild,
    EngineHandle,
    EngineIO,
    InputShape,
    TensorSpec,
    build_serialized_engine,
    open_engine,
)
from tensoracle.bundle import inspect_bundle

BIAS = np.float16(-0.375)
COUNTS = (1, 127, 128, 129, 4097)


def _cuda(result):
    status, *values = result
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA returned {status}")
    return values[0] if values else None


def _graph(digest, weights):
    def graph(network):
        creator = trt.get_plugin_registry().get_creator(
            "LffwFp16Bias", "1", "listenflow.fastest_whisper"
        )
        assert creator is not None, "plugin was not registered before graph construction"
        digest_bytes = np.frombuffer(digest.encode("ascii"), dtype=np.int8)
        bias = load_file(weights)["bias"].copy()
        fields = trt.PluginFieldCollection(
            [
                trt.PluginField("bundle_sha256", digest_bytes, trt.PluginFieldType.CHAR),
                trt.PluginField("bias", bias, trt.PluginFieldType.FLOAT16),
            ]
        )
        plugin = creator.create_plugin("bias", fields, trt.TensorRTPhase.BUILD)
        assert plugin is not None, "valid fp16 bias fields were rejected"
        bias[:] = 99  # The caller may release or reuse field storage after create_plugin.
        features = network.add_input("features", trt.float16, (-1,))
        layer = network.add_plugin_v3([features], [], plugin)
        assert layer is not None
        output = layer.get_output(0)
        output.name = "result"
        network.mark_output(output)

    return graph


def _invalid_fields(digest):
    rejected = 0

    class FieldsChecked(Exception):
        pass

    def graph(_):
        nonlocal rejected
        creator = trt.get_plugin_registry().get_creator(
            "LffwFp16Bias", "1", "listenflow.fastest_whisper"
        )
        assert creator is not None
        digest_bytes = np.frombuffer(digest.encode("ascii"), dtype=np.int8)
        wrong_digest = np.frombuffer(b"0" * 64, dtype=np.int8)
        bias = np.array([BIAS], dtype=np.float16)
        valid_digest = trt.PluginField("bundle_sha256", digest_bytes, trt.PluginFieldType.CHAR)
        valid_bias = trt.PluginField("bias", bias, trt.PluginFieldType.FLOAT16)
        invalid = [
            [valid_digest],
            [valid_bias],
            [trt.PluginField("bundle_sha256", wrong_digest, trt.PluginFieldType.CHAR), valid_bias],
            [
                valid_digest,
                trt.PluginField("bias", bias.astype(np.float32), trt.PluginFieldType.FLOAT32),
            ],
            [
                valid_digest,
                trt.PluginField("bias", np.array([BIAS, BIAS]), trt.PluginFieldType.FLOAT16),
            ],
            [valid_digest, valid_bias, trt.PluginField("extra", bias, trt.PluginFieldType.FLOAT16)],
        ]
        for fields in invalid:
            assert (
                creator.create_plugin(
                    "bad-bias", trt.PluginFieldCollection(fields), trt.TensorRTPhase.BUILD
                )
                is None
            )
            rejected += 1
        raise FieldsChecked

    return graph, FieldsChecked, lambda: rejected


def _infer(handle, count):
    values = np.random.default_rng(count).uniform(-5, 5, count).astype(np.float16)
    expected = (values.astype(np.float32) + np.float32(BIAS)).astype(np.float16)
    with ExitStack() as resources:
        execution = resources.enter_context(handle.create_execution())
        execution.set_input_shapes({"features": values.shape})
        assert execution.get_tensor_shape("result") == values.shape
        assert execution.get_tensor_dtype("result") == trt.float16
        actual = np.full(values.shape, np.nan, dtype=np.float16)
        source = _cuda(cudart.cudaMalloc(values.nbytes))
        resources.callback(lambda: _cuda(cudart.cudaFree(source)))
        target = _cuda(cudart.cudaMalloc(actual.nbytes))
        resources.callback(lambda: _cuda(cudart.cudaFree(target)))
        stream = _cuda(cudart.cudaStreamCreate())
        resources.callback(lambda: _cuda(cudart.cudaStreamDestroy(stream)))
        # This callback runs first even if enqueue or a copy fails.
        resources.callback(lambda: _cuda(cudart.cudaStreamSynchronize(stream)))
        _cuda(
            cudart.cudaMemcpy(
                source,
                values.ctypes.data,
                values.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        )
        execution.enqueue({"features": int(source), "result": int(target)}, stream=int(stream))
        _cuda(cudart.cudaStreamSynchronize(stream))
        _cuda(
            cudart.cudaMemcpy(
                actual.ctypes.data,
                target,
                actual.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
        )
        np.testing.assert_array_equal(actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "reuse", "serialized", "fields"))
    parser.add_argument("bundle", type=Path)
    parser.add_argument("library", type=Path)
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()
    digest = inspect_bundle(args.bundle, plugin=args.library)
    _cuda(cudart.cudaSetDevice(0))
    declaration = EngineBuild(
        {"features": InputShape.ranged((1,), (128,), (4097,))},
        BuilderSettings(optimization_level=0),
    )
    graph = _graph(digest, args.weights)
    if args.mode == "fields":
        graph, checked, count = _invalid_fields(digest)
        try:
            build_serialized_engine(graph, declaration, plugins=(args.library,))
        except checked:
            print(json.dumps({"rejected": count()}))
            return
        raise AssertionError("field checks did not run")
    if args.mode == "serialized":
        plan = build_serialized_engine(graph, declaration, plugins=(args.library,))
        assert len(plan) > 0
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(plan)
        assert engine is not None
        handle = EngineHandle(engine, runtime, Path("unsaved.engine"))
    else:
        if args.mode == "reuse":

            def graph(_):
                raise AssertionError("the saved engine called its graph while reopening")

        handle = open_engine(
            graph,
            declaration,
            name="lffw-cubin-smoke",
            graph_version="1",
            weights=(args.weights,),
            plugins=(args.library,),
            io=EngineIO(
                inputs={"features": TensorSpec(trt.float16, (-1,))},
                outputs={"result": TensorSpec(trt.float16, (-1,))},
            ),
        )
    for count in COUNTS:
        _infer(handle, count)
    result = {
        "counts": list(COUNTS),
        "producer_loaded": any(
            name in sys.modules for name in ("tensoracle.producer", "triton", "torch")
        ),
    }
    if args.mode != "serialized":
        result.update(path=str(handle.path), sha256=sha256(handle.path.read_bytes()).hexdigest())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
