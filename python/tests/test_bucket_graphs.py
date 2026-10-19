"""Completed decoder work keeps compact history while sharing bounded CUDA graphs."""

import ctypes
from pathlib import Path

import numpy as np
import pytest
from tensoracle import EngineHandle

from listenflow.fastest_whisper._inference import _HostIO
from test_latency_profile import cpu_native as cpu_native  # noqa: PLC0414


@pytest.fixture
def bucket_native(cpu_native, monkeypatch):
    model, cudart, _, _, _ = cpu_native
    engine = model._decoder.execution._handle.engine
    engine_type = type(engine)
    shape = engine_type.get_tensor_shape
    profile = engine_type.get_tensor_profile_shape

    def tensor_shape(self, name):
        if name in ("self_keys", "self_values", "present_keys", "present_values"):
            return (2, 1, 2, -1, 2)
        if name in ("cross_keys", "cross_values"):
            return (2, 1, 2, 1, 2)
        return shape(self, name)

    def tensor_profile(self, name, index):
        if name == "mask":
            return (1, 1, 1, 2), (1, 1, 1, 225), (1, 1, 1, 449)
        if name in ("self_keys", "self_values"):
            return (2, 1, 2, 1, 2), (2, 1, 2, 224, 2), (2, 1, 2, 448, 2)
        return profile(self, name, index)

    monkeypatch.setattr(engine_type, "get_tensor_shape", tensor_shape)
    monkeypatch.setattr(engine_type, "get_tensor_profile_shape", tensor_profile)
    model._decoder.close()
    model._encoder.close()
    model._decoder = _HostIO(EngineHandle(engine, object(), Path("bucket.engine")))
    model._encoder = _HostIO(EngineHandle(engine_type(False), object(), Path("encoder.engine")))
    model.config.text_positions = 448
    model.config.text_layers = model.config.text_heads = 2
    model.config.audio_state = 4
    success = cudart.cudaError_t.cudaSuccess

    def copy2d(dst, dpitch, src, spitch, width, height, kind, stream):
        for row in range(height):
            ctypes.memmove(dst + row * dpitch, src + row * spitch, width)
        return (success,)

    monkeypatch.setattr(cudart, "cudaMemcpy2DAsync", copy2d)
    monkeypatch.setattr(
        cudart,
        "cudaMemcpy",
        lambda dst, src, size, kind: (ctypes.memmove(dst, src, size), (success,))[1],
    )
    monkeypatch.setattr(
        cudart,
        "cudaMemsetAsync",
        lambda ptr, value, size, stream: (ctypes.memset(ptr, value, size), (success,))[1],
    )
    observed = []
    context_type = type(model._decoder.execution._context)
    original = context_type.perform

    def perform(context, stream):
        if not context.engine.decoder:
            return original(context, stream)

        def array(name, dtype):
            dimensions = context.get_tensor_shape(name)
            size = int(np.prod(dimensions))
            raw = (ctypes.c_ubyte * (size * np.dtype(dtype).itemsize)).from_address(
                context.addresses[name]
            )
            return np.ctypeslib.as_array(raw).view(dtype).reshape(dimensions)

        keys = array("self_keys", np.float16)
        token = int(array("tokens", np.int32).item())
        position = int(array("positions", np.int32).item())
        mask = array("mask", np.float32).flatten()
        observed.append((position, mask.copy(), keys.copy()))
        for target, source in (("present_keys", "self_keys"), ("present_values", "self_values")):
            previous = array(source, np.float16)
            value = np.full((2, 1, 2, 1, 2), token, dtype=np.float16)
            value[..., 0] += np.arange(4, dtype=np.float16).reshape(2, 1, 2, 1) * 100
            value[..., 1] = position + 1
            array(target, np.float16)[...] = np.concatenate((previous, value), axis=3)
        valid = keys[0, 0, 0, :, 0][mask[:-1] == 0]
        # The visible history selects a token, so stale, padded, or wrong-row data changes it.
        next_token = 8 + (int(valid.sum()) + token) % 3
        logits = array("logits", np.float16)
        logits[...] = 0
        logits[..., next_token] = 10
        array("token", np.int32)[...] = next_token
        return True

    monkeypatch.setattr(context_type, "perform", perform)
    return model, observed


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_boundary_lengths_keep_tokens_and_each_rows_compact_history(bucket_native):
    """Every boundary keeps the same next token and contiguous history across all cache rows."""
    model, observed = bucket_native
    encoder = model.encode(np.zeros((128, 3000), np.float32))
    prefix = [1] * 448
    for length in (1, 7, 8, 9, 15, 16, 17, 447, 448):
        values = model.logits(prefix[:length], encoder)
        assert int(np.argmax(values[-1])) == 8 + length % 3
        cache = model._prefix_states[tuple(prefix[:length])].tensors["self_keys"].numpy()
        assert cache.shape == (2, 1, 2, length + 1, 2)
        np.testing.assert_array_equal(
            cache[..., 1:, 0],
            np.broadcast_to(1 + np.arange(4).reshape(2, 1, 2, 1) * 100, (2, 1, 2, length)),
        )
        np.testing.assert_array_equal(
            cache[..., 1:, 1], np.broadcast_to(np.arange(1, length + 1), (2, 1, 2, length))
        )
        assert len(model._decoder._graphs.entries) <= 16
    position, mask, keys = observed[-2]
    assert position == 446 and keys.shape[3] == 448
    assert np.isneginf(mask[0]) and np.isneginf(mask[-2]) and mask[-1] == 0


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_released_outputs_share_one_capture_within_each_length_interval(bucket_native):
    """Lengths in one interval reuse captured work while retained results keep their own data."""
    model, _ = bucket_native
    encoder = model.encode(np.zeros((128, 3000), np.float32))
    for length in (2, 3, 4, 5, 6, 7, 8):
        model.logits([1] * length, encoder)
    assert len(model._decoder._graphs.entries) == 2  # Dummy-only length and the first interval.
