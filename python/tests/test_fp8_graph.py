"""Reference quantization rules and native FP8 calibration inputs."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import tensorrt as trt
from tokenizers import Tokenizer, models

from listenflow.fastest_whisper import _inference
from listenflow.fastest_whisper._checkpoint import ModelConfig
from listenflow.fastest_whisper._network import _Graph
from test_native_resources import native_boundary as native_boundary  # noqa: PLC0414


class Tensor:
    def __init__(self, shape, dtype, producer=None, value=None):
        self.shape, self.dtype, self.producer, self.value = tuple(shape), dtype, producer, value
        self.name = ""


class Layer:
    def __init__(self, kind, inputs, shape, dtype, value=None):
        self.kind, self.inputs = kind, inputs
        self.axis, self.name = 0, ""
        self.tensor = Tensor(shape, dtype, self, value)

    def get_output(self, index):
        return self.tensor

    def get_input(self, index):
        return self.inputs[index]


class Shuffle(Layer):
    def __init__(self, tensor):
        super().__init__("shuffle", [tensor], tensor.shape, tensor.dtype)
        self.first_transpose = self.second_transpose = self.reshape_dims = None

    def get_output(self, index):
        shape = self.inputs[0].shape
        if self.first_transpose is not None:
            shape = tuple(shape[i] for i in self.first_transpose)
        if self.reshape_dims is not None:
            shape = tuple(shape[i] if n == 0 else n for i, n in enumerate(self.reshape_dims))
        if self.second_transpose is not None:
            shape = tuple(shape[i] for i in self.second_transpose)
        self.tensor.shape = shape
        return self.tensor


class Network:
    """Record only the external TensorRT declarations; execute the real graph builder."""

    def __init__(self):
        self.layers, self.outputs = [], {}

    def layer(self, kind, inputs, shape=None, dtype=None, value=None):
        result = Layer(kind, inputs, shape or inputs[0].shape, dtype or inputs[0].dtype, value)
        self.layers.append(result)
        return result

    def add_input(self, name, dtype, shape):
        return Tensor(shape, dtype)

    def add_constant(self, shape, value):
        # Scalar scales must retain their rank, including shape ().
        dtype = {
            np.dtype("float16"): trt.float16,
            np.dtype("float32"): trt.float32,
            np.dtype("int32"): trt.int32,
        }[value.dtype]
        result = Layer("constant", [], shape, dtype, value)
        self.layers.append(result)
        return result

    def add_cast(self, x, dtype):
        return self.layer("cast", [x], dtype=dtype)

    def add_quantize(self, x, scale, dtype):
        return self.layer("quantize", [x, scale], dtype=dtype)

    def add_dequantize(self, x, scale, dtype):
        return self.layer("dequantize", [x, scale], dtype=dtype)

    def add_matrix_multiply(self, x, x_op, y, y_op):
        last = y.shape[-2] if y_op == trt.MatrixOperation.TRANSPOSE else y.shape[-1]
        return self.layer("matmul", [x, y], (*x.shape[:-1], last))

    def add_elementwise(self, x, y, operation):
        dtype = trt.bool if operation == trt.ElementWiseOperation.EQUAL else x.dtype
        return self.layer("elementwise", [x, y], dtype=dtype)

    def add_normalization_v2(self, x, scale, bias, axes):
        return self.layer("normalization", [x, scale, bias])

    def add_unary(self, x, operation):
        return self.layer("unary", [x])

    def add_softmax(self, x):
        return self.layer("softmax", [x])

    def add_shuffle(self, x):
        layer = Shuffle(x)
        self.layers.append(layer)
        return layer

    def add_convolution_nd(self, x, channels, kernel, weight, bias):
        return self.layer("convolution", [x], (1, channels, x.shape[2], 1))

    def add_concatenation(self, tensors):
        return self.layer("concat", tensors)

    def add_gather(self, x, indices, axis):
        return self.layer(
            "gather", [x, indices], (*x.shape[:axis], *indices.shape, *x.shape[axis + 1 :])
        )

    def add_select(self, condition, x, y):
        return self.layer("select", [condition, x, y], dtype=x.dtype)

    def add_reduce(self, x, operation, axes, keep_dims):
        shape = tuple(1 if axes & (1 << i) else n for i, n in enumerate(x.shape))
        return self.layer("reduce", [x], shape)

    def add_plugin_v3(self, tensors, shape_inputs, plugin):
        return self.layer("plugin", tensors, (1, 1, tensors[1].shape[0]))

    def mark_output(self, tensor):
        self.outputs[tensor.name] = tensor


def checkpoint(tmp_path):
    config = ModelConfig(2, 4, 4, 2, 8, 2, 2, 8, 2, 448, 12)
    Tokenizer(models.WordLevel({"<|nospeech|>": 0})).save(str(tmp_path / "tokenizer.json"))

    def tensor(key):
        if key.endswith("embed_tokens.weight"):
            shape = (12, 4)
        elif key.endswith("embed_positions.weight"):
            shape = (4, 4) if ".encoder." in key else (448, 4)
        elif ".conv" in key:
            shape = (4, 2 if ".conv1." in key else 4, 3) if key.endswith("weight") else (4,)
        elif "layer_norm" in key:
            shape = (4,)
        elif key.endswith("bias"):
            shape = (8 if ".fc1." in key else 4,)
        else:
            shape = (8, 4) if ".fc1." in key else (4, 8) if ".fc2." in key else (4, 4)
        return (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1) / 100

    return SimpleNamespace(directory=tmp_path, config=config, tensor=tensor, weight_map=None)


def linear_keys():
    keys = set()
    for part in ("encoder", "decoder"):
        for index in range(2):
            root = f"model.{part}.layers.{index}"
            for attention in ("self_attn", "encoder_attn") if part == "decoder" else ("self_attn",):
                keys.update(f"{root}.{attention}.{name}_proj" for name in ("q", "k", "v", "out"))
            keys.update(f"{root}.{name}" for name in ("fc1", "fc2"))
    return keys


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.req("SP-LFFASTESTWHISPER-004")
@pytest.mark.parametrize("part", ["encoder", "decoder"])
def test_fp8_quantizes_each_linear_input_and_output_column_but_keeps_other_ops_half(tmp_path, part):
    """Each selected projection has E4M3 Q/DQ; attention, conv and vocabulary stay half."""
    from listenflow.fastest_whisper._fp8 import _Fp8Graph

    source, net = checkpoint(tmp_path), Network()
    scales = {key: 0.125 for key in linear_keys()}
    graph = _Fp8Graph(net, source, scales)
    getattr(graph, part)()
    quantized = [layer for layer in net.layers if layer.kind == "quantize"]
    expected = (
        {
            key
            for key in linear_keys()
            if (
                ".encoder." in key or ".encoder_attn.k_proj" in key or ".encoder_attn.v_proj" in key
            )
        }
        if part == "encoder"
        else {
            key
            for key in linear_keys()
            if ".decoder." in key
            and not key.endswith(("encoder_attn.k_proj", "encoder_attn.v_proj"))
        }
    )
    assert {layer.name for layer in quantized} == {
        f"{key}.{which}.Q" for key in expected for which in ("input", "weight")
    }
    assert len(quantized) == 2 * len(expected)
    assert not any(layer.kind == "plugin" for layer in net.layers)
    for layer in quantized:
        assert layer.tensor.dtype == trt.fp8
        key, which, _ = layer.name.rsplit(".", 2)
        scale = layer.inputs[1]
        if which == "input":
            assert scale.shape == ()
            assert scale.value.item() == 0.125
        else:
            weights = source.tensor(f"{key}.weight")
            assert scale.shape == (weights.shape[0],)
            assert layer.axis == 2
            np.testing.assert_allclose(scale.value, np.max(abs(weights), axis=1) / 448)
    for layer in net.layers:
        if layer.kind == "matmul":
            assert all(t.dtype == trt.float16 for t in layer.inputs)
            dq = [t.producer for t in layer.inputs if t.producer.kind == "dequantize"]
            assert len(dq) in (0, 2)
        if layer.kind == "convolution":
            assert layer.inputs[0].dtype == trt.float16
    for output in net.outputs.values():
        if output.name not in ("token", "greedy"):
            assert output.dtype == trt.float16


@pytest.mark.req("SP-LFFASTESTWHISPER-004")
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("part", ["encoder", "decoder"])
def test_reference_graphs_keep_original_dtypes_and_fc1_without_any_quantization(
    tmp_path, monkeypatch, precision, part
):
    """The two existing graph precisions never acquire Q/DQ or lose their selected fc1."""
    monkeypatch.setattr("listenflow.fastest_whisper._network._fc1_plugin", lambda: object())
    net = Network()
    getattr(_Graph(net, checkpoint(tmp_path), precision), part)()
    assert not any(layer.kind in ("quantize", "dequantize") for layer in net.layers)
    assert sum(layer.kind == "plugin" for layer in net.layers) == (
        2 if part == "decoder" and precision == "fp16" else 0
    )
    expected = trt.float16 if precision == "fp16" else trt.float32
    assert net.outputs["cross_keys" if part == "encoder" else "logits"].dtype == expected


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_sampled_fp16_batch_uses_existing_single_row_fc1_with_shared_weights(tmp_path, monkeypatch):
    """Each candidate uses the same existing fc1 kernel and shares one set of layer weights."""
    monkeypatch.setattr("listenflow.fastest_whisper._network._fc1_plugin", lambda: object())

    class NativeNetwork(Network):
        def add_slice(self, x, start, shape, stride):
            return self.layer("slice", [x], shape)

        def add_plugin_v3(self, tensors, shape_inputs, plugin):
            assert tensors[0].shape == (1, 1, 4), "existing fc1 accepts one candidate per call"
            return super().add_plugin_v3(tensors, shape_inputs, plugin)

    net = NativeNetwork()
    graph = _Graph(net, checkpoint(tmp_path), "fp16")
    graph.decoder_fc1(Tensor((5, 1, 4), trt.float16), "model.decoder.layers.0.fc1")
    calls = [layer for layer in net.layers if layer.kind == "plugin"]
    assert len(calls) == 5
    assert all(layer.inputs[1] is calls[0].inputs[1] for layer in calls)
    assert all(layer.inputs[2] is calls[0].inputs[2] for layer in calls)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_fp8_scales_reach_native_identity_without_loading_fc1(
    source_checkpoint, native_boundary, monkeypatch, tmp_path
):
    """Calibration is passed separately from safetensors and FP8 never selects the FP16 plugin."""
    from listenflow.fastest_whisper._cached_model import CachedWhisper

    calls, _ = native_boundary
    scales = tmp_path / "scales.json"
    values = {key: 0.125 for key in linear_keys()}
    scales.write_text(json.dumps(values))
    monkeypatch.setattr(_inference, "FP8_SCALES", scales)
    monkeypatch.setattr(
        _inference, "_fc1_library", lambda: pytest.fail("FP8 must not load FP16 fc1")
    )
    with CachedWhisper(source_checkpoint, precision="fp8"):
        pass
    source, precision = calls[0][:2]
    assert precision == "fp8"
    assert source["scale_file"] == str(scales)
    assert source["scales"] == values
    assert source["fc1_library"] == ""
    assert all(path.endswith(".safetensors") for path in source["weights"])


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("scales", [None, []])
def test_invalid_calibration_file_cannot_select_the_product_fp16_graph(
    source_checkpoint, native_boundary, tmp_path, monkeypatch, scales
):
    """An invalid scale collection fails before opening native inference or selecting another graph."""
    from listenflow.fastest_whisper._cached_model import CachedWhisper

    path = tmp_path / "scales.json"
    path.write_text(json.dumps(scales))
    monkeypatch.setattr(_inference, "FP8_SCALES", path)
    calls, _ = native_boundary
    with pytest.raises(ValueError, match="scales"):
        CachedWhisper(source_checkpoint, precision="fp8")
    assert calls == []


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("invalid", [0, -0.1, float("nan"), float("inf")])
def test_fp8_rejects_invalid_or_missing_calibration_before_quantizing(tmp_path, invalid):
    """A layer without a positive finite scale cannot silently use a different precision."""
    from listenflow.fastest_whisper._fp8 import _Fp8Graph

    source, net = checkpoint(tmp_path), Network()
    key = "model.decoder.layers.0.fc1"
    for scales in ({}, {key: invalid}):
        graph = _Fp8Graph(net, source, scales)
        with pytest.raises(ValueError, match=key):
            graph.linear(Tensor((1, 1, 4), trt.float16), key)
