"""One-token decoding preserves each candidate's history without replaying its prefix."""

import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorrt as trt
from tensoracle import EngineHandle

from listenflow.fastest_whisper._inference import TensorRTWhisper, _HostIO


@pytest.fixture
def incremental_model():
    calls = []
    windows = []

    class Encoder:
        def run(self, inputs, *, device_outputs):
            assert set(device_outputs) == {"encoder", "cross_keys", "cross_values"}
            window = len(windows) + 1
            windows.append(window)
            return {
                "encoder": np.full((1, 2, 1), window, dtype=np.float32),
                "cross_keys": np.full((1, 1, 1, 2, 1), window, dtype=np.float32),
                "cross_values": np.full((1, 1, 1, 2, 1), window + 10, dtype=np.float32),
            }

        def close(self):
            pass

    class Decoder:
        def run(self, inputs, *, device_outputs):
            assert set(device_outputs) == {"present_keys", "present_values", "greedy"}
            assert set(inputs) == {
                "tokens",
                "positions",
                "mask",
                "self_keys",
                "self_values",
                "cross_keys",
                "cross_values",
                "suppression",
            }
            assert inputs["tokens"].shape == inputs["positions"].shape == (1, 1)
            history = inputs["self_keys"].flatten()[1:].astype(int).tolist()
            token = int(inputs["tokens"][0, 0])
            position = int(inputs["positions"][0, 0])
            assert position == len(history)
            assert inputs["mask"].shape == (1, 1, 1, position + 2)
            assert np.isneginf(inputs["mask"].flat[0])
            assert np.all(inputs["mask"].flatten()[1:] == 0)
            window = int(inputs["cross_keys"].flat[0])
            assert int(inputs["cross_values"].flat[0]) == window + 10
            calls.append((window, history, token))
            keys = np.array([0, *history, token], dtype=np.float32).reshape(1, 1, 1, -1, 1)
            # Different branch histories must produce different next-token scores.
            score = sum((index + 1) * value for index, value in enumerate(history + [token]))
            logits = np.arange(8, dtype=np.float32) + score + window * 100
            return {
                "logits": logits[None, None],
                "present_keys": keys,
                "present_values": keys.copy(),
            }

        def close(self):
            pass

    model = TensorRTWhisper.__new__(TensorRTWhisper)
    model.config = SimpleNamespace(
        n_mels=1,
        audio_positions=2,
        audio_state=1,
        text_layers=1,
        text_heads=1,
        text_positions=16,
        vocab_size=8,
    )
    model.precision = "fp32"
    model.device = 0
    model._encoder = Encoder()
    model._decoder = Decoder()
    yield model, calls, windows
    model.close()


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_prefix_extension_executes_only_the_new_token(incremental_model):
    """A continued prefix uses past keys and values and returns the same position logits."""
    model, calls, _ = incremental_model
    encoder = model.encode(np.zeros((1, 4), dtype=np.float32))

    first = model.logits([1, 2], encoder)
    extended = model.logits([1, 2, 3], encoder)
    repeated = model.logits([1, 2, 3], encoder)

    assert calls == [(1, [], 1), (1, [1], 2), (1, [1, 2], 3)]
    np.testing.assert_array_equal(first[:, 0], [101, 105])
    np.testing.assert_array_equal(extended[:, 0], [101, 105, 114])
    np.testing.assert_array_equal(repeated, extended)
    np.testing.assert_array_equal(np.asarray(encoder), [[1], [1]])


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_several_new_tokens_continue_from_the_longest_cached_prefix(incremental_model):
    """A caller adding several tokens reuses every already evaluated position."""
    model, calls, _ = incremental_model
    encoder = model.encode(np.zeros((1, 4), dtype=np.float32))
    model.logits([1, 2], encoder)
    model.logits([1, 2, 3], encoder)
    actual = model.logits([1, 2, 3, 4, 5], encoder)

    assert [token for _, _, token in calls] == [1, 2, 3, 4, 5]
    assert actual[-1, 0] == 155


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_beam_reordering_uses_the_selected_parent_cache(incremental_model):
    """Two branches share their initial cache and continue with their own selected histories."""
    model, calls, _ = incremental_model
    encoder = model.encode(np.zeros((1, 4), dtype=np.float32))
    model.logits([1, 2], encoder)
    model.logits([1, 2, 3], encoder)
    model.logits([1, 2, 4], encoder)
    selected = model.logits([1, 2, 4, 5], encoder)
    other = model.logits([1, 2, 3, 6], encoder)

    assert calls[-4:] == [
        (1, [1, 2], 3),
        (1, [1, 2], 4),
        (1, [1, 2, 4], 5),
        (1, [1, 2, 3], 6),
    ]
    assert selected[-1, 0] == 137
    assert other[-1, 0] == 138


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_independent_samples_restart_from_the_initial_cache(incremental_model):
    """A new sample or temperature attempt uses the original prefix without replaying it."""
    model, calls, _ = incremental_model
    encoder = model.encode(np.zeros((1, 4), dtype=np.float32))
    model.logits([1, 2], encoder)
    model.logits([1, 2, 3], encoder)
    model.logits([1, 2, 3, 4], encoder)
    model.logits([1, 2], encoder)
    second = model.logits([1, 2, 5], encoder)

    assert calls[-1] == (1, [1, 2], 5)
    assert len(calls) == 5
    assert second[-1, 0] == 120


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_new_audio_window_replaces_cross_attention_and_prefix_cache(incremental_model):
    """A second window recomputes its own cross-attention and starts with empty token history."""
    model, calls, windows = incremental_model
    first = model.encode(np.zeros((1, 4), dtype=np.float32))
    model.logits([1, 2], first)
    second = model.encode(np.ones((1, 4), dtype=np.float32))
    actual = model.logits([1, 2], second)

    assert windows == [1, 2]
    assert calls == [(1, [], 1), (1, [1], 2), (2, [], 1), (2, [1], 2)]
    np.testing.assert_array_equal(actual[:, 0], [201, 205])


@pytest.fixture
def native_io(monkeypatch):
    """Replace CUDA and TensorRT only; use real tensoracle execution and buffer ownership."""
    from cuda.bindings import runtime as cudart

    allocations, copies = {}, []
    fail = SimpleNamespace(enqueue=False, malloc_calls=0, free_calls=0, sync_calls=0)
    success = cudart.cudaError_t.cudaSuccess

    def allocate(size):
        fail.malloc_calls += 1
        buffer = ctypes.create_string_buffer(size)
        pointer = ctypes.addressof(buffer)
        allocations[pointer] = buffer
        return success, pointer

    def free(pointer):
        fail.free_calls += 1
        assert pointer in allocations, "an allocation was returned twice"
        del allocations[pointer]
        return (success,)

    def copy(destination, source, size, kind, *stream):
        copies.append(kind)
        ctypes.memmove(destination, source, size)
        return (success,)

    def synchronize(stream):
        fail.sync_calls += 1
        return (success,)

    class Context:
        def __init__(self):
            self.addresses = {}

        def set_input_shape(self, name, shape):
            return True

        def infer_shapes(self):
            return []

        def get_tensor_shape(self, name):
            return (1, 4)

        def set_tensor_address(self, name, pointer):
            self.addresses[name] = pointer
            return True

        def execute_async_v3(self, stream):
            if fail.enqueue:
                return False
            source = np.ctypeslib.as_array(
                (ctypes.c_float * 4).from_address(self.addresses["input"])
            )
            for name in ("kept", "host"):
                target = np.ctypeslib.as_array(
                    (ctypes.c_float * 4).from_address(self.addresses[name])
                )
                target[:] = source * 2
            return True

    class Engine:
        num_io_tensors = 3

        def get_tensor_name(self, index):
            return ("input", "kept", "host")[index]

        def get_tensor_mode(self, name):
            return trt.TensorIOMode.INPUT if name == "input" else trt.TensorIOMode.OUTPUT

        def get_tensor_shape(self, name):
            return (1, 4)

        def get_tensor_dtype(self, name):
            return trt.float32

        def get_tensor_location(self, name):
            return trt.TensorLocation.DEVICE

        def is_shape_inference_io(self, name):
            return False

        def create_execution_context(self):
            return Context()

    monkeypatch.setattr(cudart, "cudaSetDevice", lambda device: (success,))
    monkeypatch.setattr(cudart, "cudaStreamCreate", lambda: (success, 10))
    monkeypatch.setattr(cudart, "cudaStreamSynchronize", synchronize)
    monkeypatch.setattr(cudart, "cudaStreamDestroy", lambda stream: (success,))
    monkeypatch.setattr(cudart, "cudaMalloc", allocate)
    monkeypatch.setattr(cudart, "cudaFree", free)
    monkeypatch.setattr(cudart, "cudaMemcpyAsync", copy)
    monkeypatch.setattr(cudart, "cudaMemcpy", copy)
    io = _HostIO(EngineHandle(Engine(), object(), Path("kv-test.engine")))
    yield io, allocations, copies, fail, cudart.cudaMemcpyKind
    io.close()


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_resident_outputs_pass_to_the_next_execution_without_a_host_copy(native_io):
    """Completed device outputs are borrowed by the next execution and remain caller-owned."""
    io, allocations, copies, _, kind = native_io
    first = io.run({"input": np.arange(4, dtype=np.float32)[None]}, device_outputs=("kept",))
    second = io.run({"input": first["kept"]}, device_outputs=("kept",))

    assert copies == [
        kind.cudaMemcpyHostToDevice,
        kind.cudaMemcpyDeviceToHost,
        kind.cudaMemcpyDeviceToHost,
    ]
    assert len(allocations) == 4  # Retained results and reusable host I/O storage.
    np.testing.assert_array_equal(second["host"], [[0, 4, 8, 12]])
    np.testing.assert_array_equal(first["kept"].numpy(), [[0, 2, 4, 6]])
    first["kept"].close()
    second["kept"].close()
    second["kept"].close()
    io.close()
    assert not allocations


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_failed_execution_returns_its_outputs_and_preserves_borrowed_cache(native_io):
    """A retained enqueue error releases new buffers and leaves the supplied cache usable."""
    io, allocations, _, fail, _ = native_io
    first = io.run({"input": np.ones((1, 4), dtype=np.float32)}, device_outputs=("kept",))
    fail.enqueue = True
    with pytest.raises(RuntimeError, match="enqueue") as error:
        io.run({"input": first["kept"]}, device_outputs=("kept",))
    assert error.value.__traceback__ is not None
    np.testing.assert_array_equal(first["kept"].numpy(), [[2, 2, 2, 2]])
    fail.enqueue = False
    recovered = io.run({"input": first["kept"]}, device_outputs=("kept",))
    np.testing.assert_array_equal(recovered["host"], [[4, 4, 4, 4]])
    recovered["kept"].close()
    first["kept"].close()
    io.close()
    assert not allocations


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_failed_final_synchronization_returns_unclaimed_resident_outputs(native_io, monkeypatch):
    """A retained CUDA completion error returns every output that the caller never received."""
    from cuda.bindings import runtime as cudart

    io, allocations, _, _, _ = native_io
    calls = 0

    def synchronize(stream):
        nonlocal calls
        calls += 1
        return (
            cudart.cudaError_t.cudaErrorUnknown if calls == 1 else cudart.cudaError_t.cudaSuccess,
        )

    monkeypatch.setattr(cudart, "cudaStreamSynchronize", synchronize)
    with pytest.raises(RuntimeError, match="cudaStreamSynchronize") as error:
        io.run({"input": np.ones((1, 4), dtype=np.float32)}, device_outputs=("kept",))
    assert error.value.__traceback__ is not None
    io.close()
    assert not allocations


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_completed_inference_reuses_buffers_without_overwriting_retained_outputs(native_io):
    """Repeated inference reuses released device storage while retained results keep their values."""
    io, allocations, _, calls, _ = native_io
    retained = io.run({"input": np.ones((1, 4), dtype=np.float32)}, device_outputs=("kept",))
    warm = io.run({"input": np.full((1, 4), 3, dtype=np.float32)}, device_outputs=("kept",))
    warm["kept"].close()
    warmed_addresses = set(allocations)
    warmed_calls = calls.malloc_calls, calls.free_calls
    for value in (4, 5, 6):
        result = io.run(
            {"input": np.full((1, 4), value, dtype=np.float32)}, device_outputs=("kept",)
        )
        np.testing.assert_array_equal(result["host"], np.full((1, 4), value * 2))
        np.testing.assert_array_equal(retained["kept"].numpy(), [[2, 2, 2, 2]])
        result["kept"].close()
        assert set(allocations) == warmed_addresses
        assert (calls.malloc_calls, calls.free_calls) == warmed_calls
    retained["kept"].close()
    io.close()
    assert not allocations


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_completed_exchange_waits_for_cuda_only_once(native_io):
    """A successful exchange returns completed outputs without waiting twice for the same work."""
    io, _, _, calls, _ = native_io
    result = io.run({"input": np.ones((1, 4), dtype=np.float32)})
    np.testing.assert_array_equal(result["host"], [[2, 2, 2, 2]])
    assert calls.sync_calls == 1
