"""Greedy selection on the device preserves suppression and deterministic ties."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorrt as trt

from test_gpu_mel import Layer, NativeNetwork, Tensor


class SelectionNetwork(NativeNetwork):
    def add_concatenation(self, tensors):
        class Concatenation:
            axis = 0

            def get_output(self, index):
                return Tensor(np.concatenate([tensor.array for tensor in tensors], axis=self.axis))

        return Concatenation()

    def add_softmax(self, tensor):
        values = tensor.array - tensor.array.max(axis=-1, keepdims=True)
        weights = np.exp(values)
        return Layer(weights / weights.sum(axis=-1, keepdims=True))

    def add_cast(self, tensor, dtype):
        return Layer(tensor.array.astype(trt.nptype(dtype)))

    def add_elementwise(self, left, right, operation):
        if operation == trt.ElementWiseOperation.EQUAL:
            return Layer(left.array == right.array)
        return super().add_elementwise(left, right, operation)

    def add_select(self, condition, left, right):
        return Layer(np.where(condition.array, left.array, right.array))

    def add_reduce(self, tensor, operation, axes, keep_dims):
        if operation == trt.ReduceOperation.MIN:
            reduced = tuple(index for index in range(tensor.array.ndim) if axes & (1 << index))
            return Layer(np.min(tensor.array, axis=reduced, keepdims=keep_dims))
        return super().add_reduce(tensor, operation, axes, keep_dims)


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("begin", [False, True])
def test_device_selection_excludes_suppressed_tokens_and_chooses_first_maximum(precision, begin):
    """Suppression wins over larger scores, and equal scores select the lowest eligible token ID."""
    from listenflow.fastest_whisper._network import _Graph

    graph = _Graph.__new__(_Graph)
    graph.precision = precision
    graph.config = SimpleNamespace(vocab_size=5)
    graph._weights = []
    graph.net = SelectionNetwork(None)
    dtype = np.float16 if precision == "fp16" else np.float32
    logits = Tensor(np.array([[[5, 99, 5, 99, 4]]], dtype=dtype))
    mask = np.zeros((1, 1, 5), dtype=np.float32)
    mask[..., [1, 3]] = -np.inf
    if begin:
        mask[..., 0] = -np.inf
    selected = graph._select_token(logits, Tensor(mask))
    assert int(selected.array.item()) == (2 if begin else 0)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_device_greedy_result_carries_quality_scores_without_vocabulary_transfer(precision):
    """The device returns the selected token, its filtered score, and unfiltered silence probability."""
    from listenflow.fastest_whisper._network import _Graph

    graph = _Graph.__new__(_Graph)
    graph.precision = precision
    graph.config = SimpleNamespace(vocab_size=5)
    graph._weights = []
    graph.net = SelectionNetwork(None)
    dtype = np.float16 if precision == "fp16" else np.float32
    raw = np.array([[[5, 99, 5, 99, 4]]], dtype=dtype)
    mask = np.zeros((1, 1, 5), dtype=np.float32)
    mask[..., [1, 3]] = -np.inf

    logits, suppression = Tensor(raw), Tensor(mask)
    token = graph._select_token(logits, suppression)
    result = graph._greedy_result(logits, suppression, token=token, no_speech=3)

    filtered = raw.astype(np.float32) + mask
    expected_score = 5 - np.logaddexp.reduce(filtered, axis=-1).item()
    expected_silence = np.exp(99 - np.logaddexp.reduce(raw.astype(np.float32), axis=-1).item())
    assert result.array.shape == (1, 1, 3)
    assert result.array.dtype == np.float32
    np.testing.assert_allclose(result.array[0, 0], [0, expected_score, expected_silence], atol=3e-6)
