"""Evaluate the real sampling graph with a NumPy TensorRT boundary, without CUDA."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorrt as trt

from listenflow.fastest_whisper._longform import apply_timestamp_rules
from listenflow.fastest_whisper._network import _Graph


class ArrayLayer:
    def __init__(self, value):
        self.value = value
        self.axis = self.axes = 0

    def get_output(self, _index):
        value = np.asarray(self.value(self) if callable(self.value) else self.value)
        return SimpleNamespace(value=value, shape=value.shape)


class ArrayNetwork:
    """Execute only the external TensorRT operators, retaining graph construction."""

    def add_constant(self, shape, value):
        return ArrayLayer(np.asarray(value).reshape(shape))

    def add_cast(self, tensor, dtype):
        return ArrayLayer(tensor.value.astype(trt.nptype(dtype)))

    def add_elementwise(self, x, y, operation):
        functions = {
            trt.ElementWiseOperation.SUM: np.add,
            trt.ElementWiseOperation.SUB: np.subtract,
            trt.ElementWiseOperation.PROD: np.multiply,
            trt.ElementWiseOperation.DIV: np.divide,
            trt.ElementWiseOperation.GREATER: np.greater,
            trt.ElementWiseOperation.LESS: np.less,
            trt.ElementWiseOperation.AND: np.logical_and,
        }
        return ArrayLayer(functions[operation](x.value, y.value))

    def add_reduce(self, x, operation, axes, keep_dims):
        axes = tuple(i for i in range(len(x.shape)) if axes & (1 << i))
        function = {
            trt.ReduceOperation.MAX: np.max,
            trt.ReduceOperation.MIN: np.min,
            trt.ReduceOperation.SUM: np.sum,
        }[operation]
        return ArrayLayer(function(x.value, axis=axes, keepdims=keep_dims))

    def add_unary(self, x, operation):
        return ArrayLayer(
            {trt.UnaryOperation.EXP: np.exp, trt.UnaryOperation.LOG: np.log}[operation](x.value)
        )

    def add_softmax(self, x):
        def evaluate(layer):
            axes = tuple(i for i in range(len(x.shape)) if layer.axes & (1 << i))
            weights = np.exp(x.value - x.value.max(axis=axes, keepdims=True))
            return weights / weights.sum(axis=axes, keepdims=True)

        return ArrayLayer(evaluate)

    def add_cumulative(self, x, axis, operation, exclusive, reverse):
        assert operation == trt.CumulativeOperation.SUM and not exclusive and not reverse
        return ArrayLayer(np.cumsum(x.value, axis=int(axis.value)))

    def add_select(self, condition, x, y):
        return ArrayLayer(np.where(condition.value, x.value, y.value))

    def add_slice(self, x, start, shape, stride):
        return ArrayLayer(
            x.value[
                tuple(slice(a, a + n * d, d) for a, n, d in zip(start, shape, stride, strict=True))
            ]
        )

    def add_gather_v2(self, x, indices, mode):
        assert mode == trt.GatherMode.ELEMENT
        return ArrayLayer(lambda layer: np.take_along_axis(x.value, indices.value, axis=layer.axis))

    def add_gather(self, x, indices, axis):
        return ArrayLayer(np.take(x.value, indices.value, axis=axis))

    def add_concatenation(self, tensors):
        return ArrayLayer(lambda layer: np.concatenate([x.value for x in tensors], axis=layer.axis))


def tensor(value):
    return ArrayLayer(value).get_output(0)


def sample_graph(logits, uniforms, *, temperature=0.4, suppression=None, timestamps=False):
    logits = np.asarray(logits, np.float32)
    graph = _Graph(ArrayNetwork(), SimpleNamespace(config=SimpleNamespace(vocab_size=6)), "fp32")
    return graph._sampling_result(
        tensor(logits),
        tensor(np.zeros_like(logits) if suppression is None else suppression),
        tensor(np.full((1, 1, 1), temperature, np.float32)),
        tensor(np.asarray(uniforms, np.float32).reshape(logits.shape[0], 1, 1)),
        tensor(np.full((1, 1, 1), timestamps, bool)),
        timestamp_begin=4,
        no_speech=5,
    ).value


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("temperature", [0.2, 0.8, 2.0])
def test_sampling_graph_matches_softmax_frequencies_and_unscaled_logprobability(temperature):
    """Temperature controls token frequencies within six binomial standard deviations, not scoring."""
    count = 40_000
    logits = np.array([1_000_000, 999_999.9375, 999_999.875, -np.inf, -np.inf, -np.inf])
    uniforms = np.random.default_rng(421).random(count, dtype=np.float32)
    actual = sample_graph(np.tile(logits, (count, 1, 1)), uniforms, temperature=temperature)
    weights = np.exp((logits[:3] - logits[0]) / temperature)
    expected = weights / weights.sum()
    tokens = actual[:, 0, 0].astype(int)
    counts = np.bincount(tokens, minlength=6)
    assert not counts[3:].any()
    assert np.all(
        abs(counts[:3] - count * expected) < 6 * np.sqrt(count * expected * (1 - expected))
    )
    unscaled = logits[:3] - logits[0] - np.log(np.exp(logits[:3] - logits[0]).sum())
    np.testing.assert_allclose(actual[:, 0, 1], unscaled[tokens], atol=2e-6)


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_sampling_graph_obeys_timestamp_mass_rule_and_skips_zero_probability_tokens():
    """Combined timestamp probability wins over a text maximum, and zero-mass tokens cannot win."""
    logits = np.array([[[-np.inf, 1, -np.inf, -np.inf, 0.9, 0.9]]], np.float32)
    for timestamps in (False, True):
        expected_logits = logits[0, 0].copy()
        if timestamps:
            apply_timestamp_rules(expected_logits, [1], timestamp_begin=4, eot=3)
        weights = np.exp((expected_logits - expected_logits.max()) / 0.4)
        cdf = np.cumsum(weights / weights.sum())
        for uniform in (0, 0.4, np.nextafter(np.float32(1), np.float32(0))):
            actual = sample_graph(logits, [uniform], timestamps=timestamps)
            assert actual[0, 0, 0] == np.searchsorted(cdf, uniform, side="right")
