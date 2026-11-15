"""Real C ABI cleanup survives failures at its external CUDA/backend boundaries."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from diagnostic_runtime_fixture import diagnostic_build_directory


@pytest.fixture(scope="module")
def fault_observer():
    output = diagnostic_build_directory() / "cpp-build/liblffw_runtime_diagnostic_observer.so"
    assert output.is_file(), (
        "Build python/ with LFFW_RUNTIME_DIAGNOSTICS=ON in build/python-diagnostics "
        "before running the native failure tests"
    )
    return output


def run(fault_observer, source_checkpoint, body):
    prefix = f"""
import ctypes
from pathlib import Path
import numpy as np
from listenflow.fastest_whisper import _native
from listenflow.fastest_whisper._cached_model import CachedWhisper
observer = ctypes.CDLL({str(fault_observer)!r})
model = CachedWhisper(Path({str(source_checkpoint)!r}))
features = np.zeros((80, 3000), np.float32)
"""
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(fault_observer)
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root / "tests")))
    result = subprocess.run(
        [sys.executable, "-c", prefix + body], env=env, check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("fault,status", [(1, 6), (2, 3), (3, 4)])
def test_failed_exchange_releases_unclaimed_outputs_and_retries_with_bounded_storage(
    fault_observer, source_checkpoint, fault, status
):
    """Retaining completion, pinned-allocation or enqueue errors cannot retain output leases."""
    run(
        fault_observer,
        source_checkpoint,
        f"""
observer.lffw_test_fail_next({fault})
retained = []
try:
    model.encode(features)
except _native.NativeError as error:
    retained.append(error)
    assert error.status == {status}, (error.status, str(error))
else:
    raise AssertionError('injected failure was ignored')
first = np.asarray(model.encode(features))
expected = first.copy()
allocated = observer.lffw_test_total_allocations()
second = np.asarray(model.encode(features))
assert observer.lffw_test_total_allocations() == allocated
assert not np.array_equal(first, second)
model.close()
np.testing.assert_array_equal(first, expected)
assert observer.lffw_test_live_resources() == 0
assert retained
""",
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("fault", [1, 3, 5])
def test_failed_prefix_extension_keeps_borrowed_history_and_releases_capture(
    fault_observer, source_checkpoint, fault
):
    """Completion, enqueue and graph-creation failures leave the retained parent usable."""
    run(
        fault_observer,
        source_checkpoint,
        f"""
audio = model.encode(features)
chain = audio.decoder(_native.PREFIX)
parent, previous = chain.prefix_step(None, 4)
observer.lffw_test_fail_next({fault})
retained = []
try:
    chain.prefix_step(parent, 5)
except _native.NativeError as error:
    retained.append(error)
else:
    raise AssertionError('injected failure was ignored')
child, actual = chain.prefix_step(parent, 5)
assert np.isfinite(actual).all() and np.isfinite(previous).all()
captures = observer.lffw_test_graph_captures()
grandchild, _ = chain.prefix_step(child, 1)
assert observer.lffw_test_graph_captures() == captures
model.close()
assert observer.lffw_test_live_resources() == 0
grandchild.close(); child.close(); parent.close(); chain.close(); audio.close()
assert observer.lffw_test_live_resources() == 0
assert retained
""",
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_native_input_validation_and_host_copy_failure_preserve_owned_audio(
    fault_observer, source_checkpoint
):
    """Typed inputs, closed/foreign mel handles and failed copies keep the model usable."""
    run(
        fault_observer,
        source_checkpoint,
        f"""
for bad in (features.astype(np.float64), features.astype(np.float16), features[:, :-1]):
    try:
        model._native_model.encode_features(bad)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError('invalid native features were accepted')
mel = model._native_model.mel(np.zeros(160, np.float32))
other = CachedWhisper(Path({str(source_checkpoint)!r}))
try:
    other._native_model.encode(mel)
except ValueError:
    pass
else:
    raise AssertionError('foreign model mel was accepted')
other.close()
mel.close()
try:
    model._native_model.encode(mel)
except _native.NativeError as error:
    assert error.status == 5
else:
    raise AssertionError('closed mel was accepted')
audio = model.encode(features)
allocated = observer.lffw_test_total_allocations()
observer.lffw_test_fail_next(4)
retained = []
try:
    np.asarray(audio)
except _native.NativeError as error:
    retained.append(error)
    assert error.status == 6
else:
    raise AssertionError('copy failure was ignored')
assert observer.lffw_test_total_allocations() == allocated + 1
first = np.asarray(audio)
assert np.isfinite(first).all()
assert observer.lffw_test_total_allocations() == allocated + 1
another = model._native_model.encode_features(features)
np.testing.assert_array_equal(np.asarray(audio), first)
assert not np.array_equal(np.asarray(another), first)
initial = model.initial_tokens
submitted = observer.lffw_test_graph_submissions()
model.logits(initial, audio)
assert observer.lffw_test_graph_submissions() == submitted + len(initial)
prefix = initial + [1, 2, 1]
model.logits(prefix, audio)
assert observer.lffw_test_graph_submissions() == submitted + len(prefix)
model.logits(prefix, audio)
assert observer.lffw_test_graph_submissions() == submitted + len(prefix)
model.logits(initial + [2], audio)
assert observer.lffw_test_graph_submissions() == submitted + len(prefix) + 1
model.close()
assert observer.lffw_test_live_resources() == 0
another.close()
assert retained
""",
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_mutable_chain_rejects_invalid_input_and_replaces_failed_execution(
    fault_observer, source_checkpoint
):
    """Invalid tokens keep a chain usable; an enqueue failure requires a fresh chain."""
    run(
        fault_observer,
        source_checkpoint,
        """
audio = model.encode(features)
chain = audio.decoder(_native.GREEDY)
try:
    chain.greedy_step(-1, True, False)
except ValueError:
    pass
else:
    raise AssertionError('invalid token was accepted')
chain.greedy_step(4, True, False)
observer.lffw_test_fail_next(3)
try:
    chain.greedy_step(5, False, False)
except _native.NativeError as error:
    retained = error
    assert error.status == 4
else:
    raise AssertionError('enqueue failure was ignored')
try:
    chain.greedy_step(5, False, False)
except _native.NativeError as error:
    assert error.status == 8
else:
    raise AssertionError('failed mutable state was reused')
fresh = audio.decoder(_native.GREEDY)
fresh.greedy_step(4, True, False)
model.close()
chain.close(); fresh.close(); audio.close()
assert observer.lffw_test_live_resources() == 0
assert retained
""",
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_busy_model_preserves_handles_until_the_current_operation_finishes(
    fault_observer, source_checkpoint
):
    """Concurrent native info and close report BUSY, then the same model closes normally."""
    run(
        fault_observer,
        source_checkpoint,
        """
import threading, time
native = model._native_model
result = []
observer.lffw_test_hold_completion(1)
thread = threading.Thread(target=lambda: result.append(native.encode_features(features)))
thread.start()
try:
    deadline = time.monotonic() + 5
    while not observer.lffw_test_completion_entered():
        assert time.monotonic() < deadline, 'native completion did not start'
        time.sleep(0.001)
    for operation in (lambda: native.info, native.close):
        try:
            operation()
        except _native.NativeError as error:
            assert error.status == 10
        else:
            raise AssertionError('concurrent operation did not report BUSY')
finally:
    observer.lffw_test_hold_completion(0)
    thread.join(5)
assert not thread.is_alive() and len(result) == 1
native.close()
model.close()
result[0].close()
assert observer.lffw_test_live_resources() == 0
""",
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_all_context_positions_use_only_sixteen_native_graphs(fault_observer, tmp_path_factory):
    """All 448 positions share the sixteen native buckets and close releases every graph."""
    from native_runtime_fixture import prepare_source_checkpoint

    checkpoint = prepare_source_checkpoint(tmp_path_factory, text_positions=448)
    run(
        fault_observer,
        checkpoint,
        """
audio = model.encode(features)
tokens = [4] + [1] * 447
for length in range(1, 449):
    model.logits(tokens[:length], audio)
assert observer.lffw_test_graph_captures() == 16
assert observer.lffw_test_graph_submissions() == 448
model.close()
assert observer.lffw_test_live_resources() == 0
""",
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_product_cancellation_probe_counts_only_completed_native_steps(
    fault_observer, source_checkpoint
):
    """The product probe cancels after four real C ABI steps and restores normal decoding."""
    run(
        fault_observer,
        source_checkpoint,
        """
from fc1_product_runtime import cancel_prefix_after_steps
audio = model.encode(features)
steps = cancel_prefix_after_steps(model, audio, model.initial_tokens + [1] * 32)
assert steps == observer.lffw_test_graph_submissions() == 4
assert model._cancel_requested is None
assert np.isfinite(np.asarray(model.encode(features))).all()
model.close()
assert observer.lffw_test_live_resources() == 0
""",
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_long_prompt_keeps_released_prefix_capacity_linear(fault_observer, tmp_path_factory):
    """Growing a prompt through 448 positions retains bounded reusable KV storage."""
    from native_runtime_fixture import prepare_source_checkpoint

    checkpoint = prepare_source_checkpoint(tmp_path_factory, text_positions=448)
    print(
        run(
            fault_observer,
            checkpoint,
            """
observer.lffw_test_device_bytes.restype = ctypes.c_size_t
audio = model.encode(features)
tokens = [4] + [1] * 447
model.logits(tokens[:1], audio)
baseline = observer.lffw_test_device_bytes()
# Reserve eight maximum-sized KV pairs above the warmed fixed buffers. The
# bound allows retained root/parent/output and reusable capacities, not one
# allocation for each of the 448 historical lengths.
maximum_pair = 2 * (model.config.text_positions + 1) * model.config.text_layers * model.config.audio_state * 4
limit = baseline + 8 * maximum_pair
for length in range(2, 449):
    model.logits(tokens[:length], audio)
peak = observer.lffw_test_device_bytes()
model._clear_prefix_states()
released = observer.lffw_test_device_bytes()
print('prefix device bytes:', baseline, peak, released, 'limit:', limit)
assert peak <= limit, ('growing prefixes retain released lengths', baseline, peak, limit)
assert released <= limit, ('cleared prefixes retain quadratic storage', released, limit)
saved = model.logits(tokens[:64], audio).copy()
model.logits(tokens[:64] + [2] * 16, audio)
np.testing.assert_array_equal(model.logits(tokens[:64], audio), saved)
model.close()
assert observer.lffw_test_live_resources() == 0
""",
        )
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize(
    "fault,operation",
    [
        (6, "cuMemFreeHost"),
        (7, "cuGraphExecDestroy"),
        (8, "cuMemFree"),
        (9, "cuEventDestroy"),
        (10, "cuStreamDestroy"),
        (11, "cuDevicePrimaryCtxRelease"),
    ],
)
def test_explicit_close_reports_release_failure_and_retains_retry_ownership(
    fault_observer, source_checkpoint, fault, operation
):
    """A failed native release is reported and a second close returns its retained resource."""
    run(
        fault_observer,
        source_checkpoint,
        f"""
audio = model.encode(features)
model.logits(model.initial_tokens, audio)
native = model._native_model
for attempt in range(2):
    observer.lffw_test_fail_next({fault})
    try:
        native.close()
    except _native.NativeError as error:
        retained = error
        assert error.status == 6, (error.status, str(error))
        assert {operation!r} in str(error), str(error)
    else:
        raise AssertionError('explicit close discarded {operation} failure')
try:
    native.encode_features(features)
except _native.NativeError as error:
    assert error.status == 5
else:
    raise AssertionError('failed close left inference usable')
native.close()
assert observer.lffw_test_live_resources() == 0, 'failed release lost retry ownership'
model.close()
assert retained
""",
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize("fault,operation", [(6, "cuMemFreeHost"), (7, "cuGraphExecDestroy")])
@pytest.mark.parametrize("owner", ["adapter", "engine"])
def test_python_close_retries_failed_cleanup_while_error_tracebacks_remain_owned(
    fault_observer, source_checkpoint, fault, operation, owner
):
    """Adapter close and engine release finish native cleanup with the first error retained."""
    run(
        fault_observer,
        source_checkpoint,
        f"""
import asyncio
from listenflow import OfflineModelDirectorySource
from listenflow.fastest_whisper import FastestWhisperEngine
audio = model.encode(features)
model.logits(model.initial_tokens, audio)
if {owner!r} == 'engine':
    engine = FastestWhisperEngine(
        model_source=OfflineModelDirectorySource(str(model.checkpoint_dir)),
        precision='fp32', device=0, language='en',
    )
    engine._model = model
    def close():
        asyncio.run(engine._release())
else:
    close = model.close
retained = []
observer.lffw_test_fail_next({fault})
try:
    close()
except _native.NativeError as error:
    retained.append(error)
    assert error.status == 6, (error.status, str(error))
    assert {operation!r} in str(error), str(error)
else:
    raise AssertionError('explicit {owner} close discarded {operation} failure')
remaining = observer.lffw_test_live_resources()
assert remaining > 0, 'injected release failure did not retain resources'
for infer in (lambda: model.encode(features), lambda: model.logits(model.initial_tokens, audio)):
    try:
        infer()
    except RuntimeError as error:
        assert 'closed' in str(error).lower(), str(error)
    else:
        raise AssertionError('failed close left inference usable')
close()
after_retry = observer.lffw_test_live_resources()
print('close retry:', {owner!r}, {operation!r}, remaining, after_retry)
assert after_retry == 0, ('{owner} lost cleanup ownership', remaining, after_retry)
assert retained[0].__traceback__ is not None
close()
assert observer.lffw_test_live_resources() == 0
""",
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_prepare_releases_failed_model_before_loading_its_replacement(
    fault_observer, source_checkpoint
):
    """Preparation keeps failed cleanup owned and opens a replacement only after release succeeds."""
    run(
        fault_observer,
        source_checkpoint,
        """
import asyncio
from listenflow import EngineState, EnvironmentUnavailableError, OfflineModelDirectorySource
from listenflow.fastest_whisper import FastestWhisperEngine
audio = model.encode(features)
model.logits(model.initial_tokens, audio)
engine = FastestWhisperEngine(
    model_source=OfflineModelDirectorySource(str(model.checkpoint_dir)),
    precision='fp32', device=0, language='en',
)
engine._model = model
retained = []
async def scenario():
    observer.lffw_test_fail_next(6)
    try:
        await engine._release()
    except _native.NativeError as error:
        retained.append(error)
        assert error.status == 6 and 'cuMemFreeHost' in str(error), str(error)
    else:
        raise AssertionError('cleanup failure was ignored')
    previous = engine._model
    assert previous is not None
    remaining = observer.lffw_test_live_resources()
    allocated = observer.lffw_test_total_allocations()
    assert remaining > 0
    observer.lffw_test_fail_next(6)
    try:
        await engine.prepare()
    except EnvironmentUnavailableError as error:
        retained.append(error)
        assert 'cuMemFreeHost' in str(error), str(error)
    else:
        raise AssertionError('prepare replaced the model before cleanup succeeded')
    assert engine._model is previous, 'prepare lost the failed model'
    assert observer.lffw_test_live_resources() == remaining
    assert observer.lffw_test_total_allocations() == allocated, 'prepare allocated a replacement'
    assert engine.health().state is EngineState.UNPREPARED
    await engine.prepare()
    assert engine._model is not previous
    assert engine.health().state is EngineState.READY
    await engine.aclose()
    assert observer.lffw_test_live_resources() == 0, 'prepare abandoned the failed cleanup'
    assert all(error.__traceback__ is not None for error in retained)
asyncio.run(scenario())
""",
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize(
    "faults,primary,additional",
    [
        ((1 << 6) | (1 << 7), "cuGraphExecDestroy", "cuMemFreeHost"),
        ((1 << 6) | (1 << 8), "cuMemFreeHost", "cuMemFree failed"),
        ((1 << 9) | (1 << 10), "cuEventDestroy", "cuStreamDestroy"),
    ],
)
def test_explicit_close_preserves_primary_and_additional_release_failures(
    fault_observer, source_checkpoint, faults, primary, additional
):
    """Graph and pinned failures both reach the caller while independent cleanup continues."""
    run(
        fault_observer,
        source_checkpoint,
        f"""
audio = model.encode(features)
model.logits(model.initial_tokens, audio)
native = model._native_model
before = observer.lffw_test_live_resources()
observer.lffw_test_fail_cleanup({faults})
try:
    native.close()
except _native.NativeError as error:
    retained = error
    assert error.status == 6
    message = str(error)
    assert {primary!r} in message and {additional!r} in message, message
    assert message.index({primary!r}) < message.index({additional!r}), message
else:
    raise AssertionError('explicit close discarded simultaneous release failures')
assert observer.lffw_test_live_resources() < before, 'independent cleanup did not continue'
native.close()
assert observer.lffw_test_live_resources() == 0
model.close()
assert retained
""",
    )
