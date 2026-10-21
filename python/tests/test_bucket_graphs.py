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
        next_token = 8 + (int(valid.sum(dtype=np.float32)) + token) % 3
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


def _greedy_history(model, encoder, count):
    prefix = [1]
    model._clear_greedy_cache()
    for _ in range(count):
        token, _, _ = model._greedy_step(prefix, encoder, first=False)
        prefix.append(token)
    return prefix


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_greedy_preserves_every_rows_history_across_all_bucket_boundaries(bucket_native):
    """Greedy tokens and each layer/head's history survive every bucket change and restart."""
    model, observed = bucket_native
    encoder = model.encode(np.zeros((128, 3000), np.float32))
    prefix = _greedy_history(model, encoder, 448)
    for position, mask, keys in observed:
        expected_token = 8 + sum(prefix[: position + 1]) % 3
        assert prefix[position + 1] == expected_token
        np.testing.assert_array_equal(
            keys[..., 1 : position + 1, 0],
            np.asarray(prefix[:position])[None, None, None, :]
            + np.arange(4).reshape(2, 1, 2, 1) * 100,
        )
        np.testing.assert_array_equal(
            keys[..., 1 : position + 1, 1],
            np.broadcast_to(np.arange(1, position + 1), (2, 1, 2, position)),
        )
        assert np.isneginf(mask[0]) and mask[-1] == 0
        assert np.all(mask[1 : position + 1] == 0)
        assert np.all(np.isneginf(mask[position + 1 : -1]))
    assert _greedy_history(model, encoder, 20) == prefix[:21]


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_greedy_writes_only_one_new_cache_position_between_bucket_changes(
    bucket_native, monkeypatch
):
    """A continued greedy step copies one position per row and never clears existing history."""
    from cuda.bindings import runtime as cudart

    model, _ = bucket_native
    encoder = model.encode(np.zeros((128, 3000), np.float32))
    operations = []
    copy, clear = cudart.cudaMemcpy2DAsync, cudart.cudaMemsetAsync

    def copy2d(dst, dpitch, src, spitch, width, height, kind, stream):
        operations.append(("copy", width, height))
        return copy(dst, dpitch, src, spitch, width, height, kind, stream)

    def memset(ptr, value, size, stream):
        operations.append(("clear", size))
        return clear(ptr, value, size, stream)

    monkeypatch.setattr(cudart, "cudaMemcpy2DAsync", copy2d)
    monkeypatch.setattr(cudart, "cudaMemsetAsync", memset)
    prefix = [1]
    model._clear_greedy_cache()
    for position in range(20):
        operations.clear()
        token, _, _ = model._greedy_step(prefix, encoder, first=False)
        prefix.append(token)
        # Steps adjacent to a bucket change may move the history in full.
        if position in (2, 3, 4, 5, 9, 10, 11, 12, 17, 18):
            assert operations == [("copy", 4, 4), ("copy", 4, 4)]


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
@pytest.mark.parametrize("invalid", ["dtype", "shape", "device", "tensor-closed", "io-closed"])
def test_greedy_rejects_invalid_continuations(bucket_native, invalid):
    """A continued greedy chain rejects invalid masks, resident devices and closed owners."""
    model, _ = bucket_native
    io = model._decoder
    inputs = _inputs(3)
    outputs = io.run(inputs, device_outputs=("present_keys", "present_values"), reuse_cache=True)
    keys, values = outputs["present_keys"], outputs["present_values"]
    inputs.update(self_keys=keys, self_values=values, mask=np.zeros((1, 1, 1, 5), np.float32))
    if invalid == "dtype":
        inputs["mask"] = inputs["mask"].astype(np.float16)
    elif invalid == "shape":
        inputs["mask"] = np.zeros((1, 1, 1, 6), np.float32)
    elif invalid == "device":
        keys.device = io.device + 1
    elif invalid == "tensor-closed":
        keys.close()
    else:
        io.close()
    try:
        with pytest.raises(
            RuntimeError if "closed" in invalid else ValueError,
            match="dtype|cache length|CUDA device|closed",
        ):
            io.run(inputs, device_outputs=("present_keys", "present_values"), reuse_cache=True)
    finally:
        keys.close()
        values.close()


def _inputs(length):
    keys = np.arange(8 * length, dtype=np.float16).reshape(2, 1, 2, length, 2)
    return {
        "tokens": np.array([[2]], np.int32),
        "positions": np.array([[length]], np.int32),
        "mask": np.zeros((1, 1, 1, length + 1), np.float32),
        "self_keys": keys,
        "self_values": keys + 10,
        "cross_keys": np.zeros((2, 1, 2, 1, 2), np.float16),
        "cross_values": np.zeros((2, 1, 2, 1, 2), np.float16),
        "suppression": np.zeros((1, 1, 12), np.float32),
    }


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_warmed_bucket_layouts_keep_shapes_and_retained_results_without_repeated_output_queries(
    bucket_native, monkeypatch
):
    """Reused layouts keep each output's data and avoid resolving unchanged output shapes."""
    model, _ = bucket_native
    io = model._decoder
    context = io.execution._context
    context_type = type(context)
    reads = []
    original = context_type.get_tensor_shape

    def shape(current, name):
        if current is context:
            reads.append(name)
        return original(current, name)

    monkeypatch.setattr(context_type, "get_tensor_shape", shape)
    retained = None
    try:
        for index, length in enumerate((3, 15, 5, 3)):
            previous_reads = len(reads)
            inputs = _inputs(length)
            keys = inputs["self_keys"]
            output = io.run(inputs, device_outputs=("present_keys",) if index == 0 else ())
            actual = output["present_keys"]
            if index == 0:
                retained = actual
                actual = actual.numpy()
                saved = actual.copy()
            assert actual.shape == (2, 1, 2, length + 1, 2)
            np.testing.assert_array_equal(actual[..., :length, :], keys)
            np.testing.assert_array_equal(output["present_values"][..., :length, :], keys + 10)
            assert int(output["token"].item()) == 8 + (int(keys[0, 0, 0, :, 0].sum()) + 2) % 3
            if index == 3:
                assert not set(reads[previous_reads:]).intersection(io.outputs)
        np.testing.assert_array_equal(retained.numpy(), saved)
    finally:
        if retained is not None:
            retained.close()


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_failed_bucket_switch_can_return_to_the_previous_layout(bucket_native, monkeypatch):
    """A rejected shape change does not prevent a subsequent call using the previous bucket."""
    model, _ = bucket_native
    io = model._decoder
    io.run(_inputs(3))
    io.run(_inputs(15))
    context = io.execution._context
    context_type = type(context)
    original = context_type.set_input_shape

    def reject(current, name, shape):
        if current is context:
            return False
        return original(current, name, shape)

    with monkeypatch.context() as patch:
        patch.setattr(context_type, "set_input_shape", reject)
        with pytest.raises(RuntimeError, match="rejected"):
            io.run(_inputs(3))
    output = io.run(_inputs(15))
    np.testing.assert_array_equal(output["present_keys"][..., :15, :], _inputs(15)["self_keys"])


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_host_outputs_survive_bucket_changes_with_one_pinned_allocation_per_output(
    bucket_native, monkeypatch
):
    """Different bucket shapes reuse bounded pinned storage while every returned array keeps its data."""
    from cuda.bindings import runtime as cu

    model, _ = bucket_native
    allocations = []
    allocate = cu.cudaMallocHost

    def host(size):
        allocations.append(size)
        return allocate(size)

    monkeypatch.setattr(cu, "cudaMallocHost", host)
    saved = []
    for length in (3, 15, 5, 3):
        output = model._decoder.run(_inputs(length))
        assert len(allocations) == len(model._decoder.outputs)
        assert output["present_keys"].shape == (2, 1, 2, length + 1, 2)
        np.testing.assert_array_equal(
            output["present_keys"][..., :length, :], _inputs(length)["self_keys"]
        )
        saved.append((output, {name: array.copy() for name, array in output.items()}))
    model._decoder.close()
    for output, expected in saved:
        for name in output:
            np.testing.assert_array_equal(output[name], expected[name])
