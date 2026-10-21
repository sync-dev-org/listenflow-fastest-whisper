"""Host exchanges retain validation and ownership when tensor layouts are reused."""

import numpy as np
import pytest

from listenflow.fastest_whisper._inference import _DeviceTensor
from test_kv_cache import native_io as native_io  # noqa: PLC0414


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
@pytest.mark.parametrize(
    "array,reason",
    [(np.ones((1, 4), np.float16), "dtype"), (np.ones((4, 1), np.float32), "shape")],
)
def test_exchange_rejects_incompatible_input_after_warmup(native_io, array, reason):
    """A warmed exchange still refuses an input with the wrong dtype or dimensions."""
    io, _, _, _, _ = native_io
    io.run({"input": np.ones((1, 4), np.float32)})
    with pytest.raises(ValueError, match=reason):
        io.run({"input": array})


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_exchange_rejects_closed_inference(native_io):
    """Closing host inference prevents another exchange, including a warmed layout."""
    io, _, _, _, _ = native_io
    io.run({"input": np.ones((1, 4), np.float32)})
    io.close()
    with pytest.raises(RuntimeError, match="closed"):
        io.run({"input": np.ones((1, 4), np.float32)})


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
@pytest.mark.parametrize("closed", [False, True])
def test_exchange_rejects_unusable_resident_input(native_io, closed):
    """A warmed exchange rejects a closed resident tensor or one on another CUDA device."""
    io, _, _, _, _ = native_io
    output = io.run({"input": np.ones((1, 4), np.float32)}, device_outputs=("kept",))
    resident = output["kept"]
    assert isinstance(resident, _DeviceTensor)
    try:
        if closed:
            resident.close()
        else:
            resident.device = io.device + 1
        with pytest.raises(RuntimeError if closed else ValueError, match="closed|CUDA device"):
            io.run({"input": resident})
    finally:
        resident.close()


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_exchange_copies_strided_host_inputs(native_io):
    """Strided host arrays transfer their logical values in contiguous tensor order."""
    io, _, _, _, _ = native_io
    source = np.arange(8, dtype=np.float32)[None, ::2]
    assert not source.flags.c_contiguous
    output = io.run({"input": source})
    np.testing.assert_array_equal(output["host"], [[0, 4, 8, 12]])


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_host_results_keep_their_values_after_another_exchange_and_close(native_io):
    """Each returned host array remains valid after later exchanges and owner closure."""
    io, _, _, _, _ = native_io
    first = io.run({"input": np.arange(4, dtype=np.float32)[None]})
    second = io.run({"input": np.full((1, 4), 7, dtype=np.float32)})
    io.close()
    for name in ("kept", "host"):
        assert first[name].shape == second[name].shape == (1, 4)
        assert first[name].dtype == second[name].dtype == np.float32
        np.testing.assert_array_equal(first[name], [[0, 2, 4, 6]])
        np.testing.assert_array_equal(second[name], [[14, 14, 14, 14]])


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_output_copies_reuse_pinned_storage_and_release_it_once(native_io, monkeypatch):
    """Host outputs reuse one pinned destination per tensor and closing frees each allocation once."""
    from cuda.bindings import runtime as cu

    io, _, _, calls, kind = native_io
    destinations = []
    original = cu.cudaMemcpyAsync

    def copy(dst, src, size, direction, stream):
        if direction == kind.cudaMemcpyDeviceToHost:
            assert dst in calls.host_allocations
            destinations.append(dst)
        return original(dst, src, size, direction, stream)

    monkeypatch.setattr(cu, "cudaMemcpyAsync", copy)
    retained = []
    for value in range(5):
        result = io.run({"input": np.full((1, 4), value, np.float32)})
        retained.append(result)
        assert calls.host_malloc_calls == 2
    assert len(set(destinations)) == 2
    for value, result in enumerate(retained):
        for array in result.values():
            np.testing.assert_array_equal(array, np.full((1, 4), value * 2, np.float32))
    io.close()
    io.close()
    assert calls.host_free_calls == 2
    assert not calls.host_allocations


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_resident_outputs_allocate_pinned_storage_only_when_requested_on_host(native_io):
    """Changing which output stays on the device never aliases the host outputs' staging storage."""
    io, _, _, calls, _ = native_io
    source = {"input": np.ones((1, 4), np.float32)}
    resident = io.run(source, device_outputs=("kept", "host"))
    assert calls.host_malloc_calls == 0
    for tensor in resident.values():
        tensor.close()
    first = io.run(source, device_outputs=("kept",))
    assert calls.host_malloc_calls == 1
    second = io.run({"input": np.full((1, 4), 3, np.float32)}, device_outputs=("host",))
    assert calls.host_malloc_calls == 2
    np.testing.assert_array_equal(first["host"], [[2, 2, 2, 2]])
    np.testing.assert_array_equal(second["kept"], [[6, 6, 6, 6]])
    first["kept"].close()
    second["host"].close()


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_failed_pinned_allocation_can_retry_without_leaking_output_storage(native_io, monkeypatch):
    """A rejected pinned allocation reports CUDA failure and leaves the exchange usable."""
    from cuda.bindings import runtime as cu

    io, _, _, calls, _ = native_io
    with monkeypatch.context() as patch:
        patch.setattr(
            cu, "cudaMallocHost", lambda size: (cu.cudaError_t.cudaErrorMemoryAllocation,)
        )
        with pytest.raises(RuntimeError, match="cudaMallocHost"):
            io.run({"input": np.ones((1, 4), np.float32)})
    assert not calls.host_allocations
    actual = io.run({"input": np.ones((1, 4), np.float32)})
    np.testing.assert_array_equal(actual["host"], [[2, 2, 2, 2]])


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_close_releases_pinned_storage_even_when_synchronization_fails(native_io, monkeypatch):
    """A close-time CUDA error still releases host storage and retains returned arrays."""
    from cuda.bindings import runtime as cu

    io, _, _, calls, _ = native_io
    result = io.run({"input": np.ones((1, 4), np.float32)})
    with monkeypatch.context() as patch:
        patch.setattr(
            cu, "cudaStreamSynchronize", lambda stream: (cu.cudaError_t.cudaErrorUnknown,)
        )
        with pytest.raises(RuntimeError, match="cudaStreamSynchronize"):
            io.close()
    assert calls.host_free_calls == 2
    assert not calls.host_allocations
    np.testing.assert_array_equal(result["host"], [[2, 2, 2, 2]])
