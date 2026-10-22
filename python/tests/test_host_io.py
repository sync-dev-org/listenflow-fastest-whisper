"""Host exchanges retain validation and ownership when tensor layouts are reused."""

from pathlib import Path

import numpy as np
import pytest

from listenflow.fastest_whisper._inference import _DeviceTensor
from test_kv_cache import native_io as native_io  # noqa: PLC0414


@pytest.fixture
def timer_slack(monkeypatch):
    import ctypes
    import errno
    from types import SimpleNamespace

    from listenflow.fastest_whisper import _inference

    state = SimpleNamespace(value=50_000, calls=[], fail=None)

    def prctl(option, value, *_unused):
        state.calls.append((option, value))
        if state.fail == (option, value):
            ctypes.set_errno(errno.EPERM)
            return -1
        if option == 30:  # PR_GET_TIMERSLACK
            return state.value
        assert option == 29  # PR_SET_TIMERSLACK
        state.value = value
        return 0

    monkeypatch.setattr(_inference, "_prctl", prctl, raising=False)
    return state


@pytest.mark.req("SP-LFFASTESTWHISPER-007")
@pytest.mark.parametrize("machine", ["aarch64", "riscv64"])
@pytest.mark.parametrize("value", [None, 1])
def test_timer_slack_rejects_unsupported_architecture_without_syscalls(
    timer_slack, monkeypatch, machine, value
):
    """Unsupported machines reject timer-slack reads and writes without calling a system call."""
    import platform

    from listenflow.fastest_whisper import _inference

    monkeypatch.setattr(platform, "machine", lambda: machine)
    with pytest.raises(RuntimeError) as caught:
        _inference._timer_slack(value)
    assert machine in str(caught.value)
    assert "unsupported" in str(caught.value).lower()
    assert timer_slack.calls == []


@pytest.mark.req("SP-LFFASTESTWHISPER-017")
@pytest.mark.parametrize("previous_slack", [3_000_000_000, 2**32 - 1, 2**32, 2**63 + 1])
def test_wait_restores_wide_linux_timer_slack(native_io, previous_slack):
    """Completion restores the calling Linux thread's timer slack beyond signed 32 bit values."""
    from listenflow.fastest_whisper import _inference

    io, *_ = native_io
    slack = Path("/proc/self/timerslack_ns")
    initial = int(slack.read_text())
    try:
        _inference._timer_slack(previous_slack)
        assert int(slack.read_text()) == previous_slack
        io._wait()
        assert int(slack.read_text()) == previous_slack
    finally:
        _inference._timer_slack(initial)
        assert int(slack.read_text()) == initial


@pytest.mark.req("SP-LFFASTESTWHISPER-017")
def test_wait_restores_linux_timer_slack_if_interrupted_after_setting_it(native_io, monkeypatch):
    """An interrupt after the kernel sets 1 ns still restores the caller's original timer slack."""
    from listenflow.fastest_whisper import _inference

    io, *_ = native_io
    slack = Path("/proc/self/timerslack_ns")
    initial = int(slack.read_text())
    original_prctl = _inference._prctl
    error = KeyboardInterrupt("interrupted after setting timer slack")

    def prctl(option, value, *unused):
        result = original_prctl(option, value, *unused)
        if option == 29 and value == 1 and result == 0:
            assert int(slack.read_text()) == 1
            raise error
        return result

    try:
        _inference._timer_slack(123_456)
        monkeypatch.setattr(_inference, "_prctl", prctl)
        with pytest.raises(KeyboardInterrupt) as caught:
            io._wait()
        assert caught.value is error
        assert int(slack.read_text()) == 123_456
    finally:
        assert original_prctl(29, initial, 0, 0, 0) == 0
        assert int(slack.read_text()) == initial


@pytest.mark.req("SP-LFFASTESTWHISPER-017")
def test_wait_limits_timer_slack_to_polling_and_restores_previous_value(
    native_io, timer_slack, monkeypatch
):
    """Completion queries and 50 microsecond sleeps use 1 ns slack, then restore the caller's value."""
    from cuda.bindings import runtime as cu

    from listenflow.fastest_whisper import _inference

    io, *_ = native_io
    timer_slack.value = 123_456
    statuses = iter([cu.cudaError_t.cudaErrorNotReady, cu.cudaError_t.cudaSuccess])
    sleeps = []

    def query(event):
        assert timer_slack.value == 1
        return (next(statuses),)

    def sleep(interval):
        assert timer_slack.value == 1
        sleeps.append(interval)

    monkeypatch.setattr(cu, "cudaEventQuery", query)
    monkeypatch.setattr(_inference, "sleep", sleep)
    io._wait()
    assert sleeps == [50e-6]
    assert timer_slack.value == 123_456
    assert timer_slack.calls == [(30, 0), (29, 1), (29, 123_456)]


@pytest.mark.req("SP-LFFASTESTWHISPER-017")
@pytest.mark.parametrize("failure", ["query", "sleep", "interrupt"])
def test_wait_restores_timer_slack_when_completion_raises(
    native_io, timer_slack, monkeypatch, failure
):
    """CUDA errors, failed sleeps and interruptions restore the thread's original timer slack."""
    from cuda.bindings import runtime as cu

    from listenflow.fastest_whisper import _inference

    io, *_ = native_io
    status = (
        cu.cudaError_t.cudaErrorUnknown if failure == "query" else cu.cudaError_t.cudaErrorNotReady
    )
    monkeypatch.setattr(cu, "cudaEventQuery", lambda event: (status,))
    error = (
        KeyboardInterrupt("interrupted wait")
        if failure == "interrupt"
        else RuntimeError("sleep failed")
    )

    def sleep(interval):
        raise error

    monkeypatch.setattr(_inference, "sleep", sleep)
    with pytest.raises(RuntimeError if failure == "query" else type(error)) as caught:
        io._wait()
    if failure != "query":
        assert caught.value is error
    assert timer_slack.value == 50_000
    assert timer_slack.calls == [(30, 0), (29, 1), (29, 50_000)]


@pytest.mark.req("SP-LFFASTESTWHISPER-017")
@pytest.mark.parametrize("operation,value", [(30, 0), (29, 1), (29, 50_000)])
def test_wait_reports_timer_slack_failure_with_errno(
    native_io, timer_slack, monkeypatch, operation, value
):
    """A rejected timer-slack read, change or restoration reaches the caller with errno."""
    import errno

    from cuda.bindings import runtime as cu

    io, *_ = native_io
    timer_slack.fail = (operation, value)
    queries = []
    monkeypatch.setattr(
        cu,
        "cudaEventQuery",
        lambda event: (queries.append(event), (cu.cudaError_t.cudaSuccess,))[1],
    )
    with pytest.raises(OSError, match="prctl") as caught:
        io._wait()
    assert caught.value.errno == errno.EPERM
    assert len(queries) == (1 if value == 50_000 else 0)


@pytest.mark.req("SP-LFFASTESTWHISPER-017")
def test_wait_retains_original_error_if_timer_slack_restoration_also_fails(
    native_io, timer_slack, monkeypatch
):
    """A restoration failure annotates the original CUDA exception without replacing it."""
    from cuda.bindings import runtime as cu

    io, *_ = native_io
    timer_slack.fail = (29, 50_000)
    monkeypatch.setattr(cu, "cudaEventQuery", lambda event: (cu.cudaError_t.cudaErrorUnknown,))
    with pytest.raises(RuntimeError, match="cudaEventQuery") as caught:
        io._wait()
    assert timer_slack.calls == [(30, 0), (29, 1), (29, 50_000)]
    assert any("Errno 1" in note and "restor" in note for note in caught.value.__notes__)


@pytest.mark.req("SP-LFFASTESTWHISPER-017")
def test_exchange_polls_completion_before_returning_copied_values(native_io, monkeypatch):
    """Pending output copies finish before return, with one 50 microsecond sleep per retry."""
    from cuda.bindings import runtime as cu

    from listenflow.fastest_whisper import _inference

    io, _, _, calls, kind = native_io
    pending, sleeps, queries = [], [], []
    original_copy = cu.cudaMemcpyAsync
    statuses = iter([cu.cudaError_t.cudaErrorNotReady] * 2 + [cu.cudaError_t.cudaSuccess])

    def copy(*args):
        if args[3] == kind.cudaMemcpyDeviceToHost:
            pending.append(args)
            return (cu.cudaError_t.cudaSuccess,)
        return original_copy(*args)

    def query(event):
        assert calls.event_records[-1] == (event, io.stream)
        queries.append(event)
        status = next(statuses)
        if status == cu.cudaError_t.cudaSuccess:
            for args in pending:
                original_copy(*args)
            pending.clear()
        return (status,)

    monkeypatch.setattr(cu, "cudaMemcpyAsync", copy)
    monkeypatch.setattr(cu, "cudaEventQuery", query)
    monkeypatch.setattr(_inference, "sleep", sleeps.append, raising=False)
    result = io.run({"input": np.arange(4, dtype=np.float32)[None]})
    assert len(queries) == 3
    assert sleeps == [50e-6, 50e-6]
    assert calls.sync_calls == 0
    assert not pending
    for array in result.values():
        assert array.shape == (1, 4)
        np.testing.assert_array_equal(array, [[0, 2, 4, 6]])


@pytest.mark.req("SP-LFFASTESTWHISPER-017")
@pytest.mark.parametrize("operation", ["cudaEventRecord", "cudaEventQuery"])
def test_completion_failure_releases_outputs_and_allows_retry(native_io, monkeypatch, operation):
    """A completion error reaches the caller, synchronizes cleanup, and releases unclaimed leases."""
    from cuda.bindings import runtime as cu

    io, _, _, calls, _ = native_io
    with monkeypatch.context() as patch:
        patch.setattr(cu, operation, lambda *_args: (cu.cudaError_t.cudaErrorUnknown,))
        with pytest.raises(RuntimeError, match=operation):
            io.run({"input": np.ones((1, 4), np.float32)}, device_outputs=("kept",))
    assert calls.sync_calls == 1
    assert all(buffer.lease is None for entries in io._buffers.values() for buffer in entries)
    result = io.run({"input": np.full((1, 4), 3, np.float32)})
    np.testing.assert_array_equal(result["host"], [[6, 6, 6, 6]])


@pytest.mark.req("SP-LFFASTESTWHISPER-017")
def test_completion_event_is_reused_and_released_once(native_io):
    """Repeated exchanges share one event without timing, and close destroys it exactly once."""
    from cuda.bindings import runtime as cu

    io, _, _, calls, _ = native_io
    for _ in range(3):
        io.run({"input": np.ones((1, 4), np.float32)})
    assert calls.event_flags == [cu.cudaEventDisableTiming]
    assert len(calls.events) == 1
    event = next(iter(calls.events))
    assert calls.event_records == [(event, io.stream)] * 3
    io.close()
    io.close()
    assert calls.destroyed_events == [event]
    assert not calls.events


@pytest.mark.req("SP-LFFASTESTWHISPER-017")
def test_event_creation_failure_returns_the_owned_stream(native_io, monkeypatch):
    """A rejected completion-event allocation closes the already created CUDA stream."""
    from cuda.bindings import runtime as cu

    from listenflow.fastest_whisper._inference import _HostIO

    io, _, _, _, _ = native_io
    destroyed = []
    monkeypatch.setattr(
        cu, "cudaEventCreateWithFlags", lambda flags: (cu.cudaError_t.cudaErrorMemoryAllocation,)
    )
    monkeypatch.setattr(
        cu,
        "cudaStreamDestroy",
        lambda stream: (destroyed.append(stream), (cu.cudaError_t.cudaSuccess,))[1],
    )
    with pytest.raises(RuntimeError, match="cudaEventCreateWithFlags"):
        _HostIO(io.execution._handle)
    assert destroyed == [io.stream]


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
    assert len(calls.destroyed_events) == 1
    assert not calls.events
    np.testing.assert_array_equal(result["host"], [[2, 2, 2, 2]])
