"""CPU behavior checks for the issue 40 decoder diagnostic."""

import csv
import io

import pytest
from scripts import decoder_host_profile as profile

pytest_plugins = ("test_latency_profile",)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_partition_accounts_for_python_between_calls_and_rejects_gaps():
    """Seven exclusive sections account for a whole step and expose missing time."""
    ledger = profile.Phases(100)
    ledger.switch("pad_cache", 120)
    ledger.switch("python_before", 150)
    ledger.switch("h2d", 160)
    ledger.switch("python_before", 170)
    ledger.switch("graph_launch", 180)
    ledger.switch("python_after", 190)
    ledger.switch("compact_cache", 200)
    ledger.switch("python_after", 220)
    ledger.switch("d2h_sync", 230)
    ledger.switch("python_after", 290)
    result = ledger.finish(300)
    assert result["total_ns"] == 200
    assert result["python_before_ns"] == 40
    assert result["python_after_ns"] == 30
    assert result["pad_cache_ns"] == 30
    assert result["h2d_ns"] == result["graph_launch_ns"] == 10
    assert result["compact_cache_ns"] == 20
    assert result["d2h_sync_ns"] == 60
    assert profile.partition_error(result) == 0
    result["h2d_ns"] -= 3
    assert profile.partition_error(result) == pytest.approx(0.015)
    with pytest.raises(ValueError, match="partition"):
        profile.summarize([dict(result, condition="L0", workload="controlled", bucket=8)])
    with pytest.raises(ValueError, match="monotonic"):
        ledger.switch("h2d", 299)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_wait_split_intersects_only_wait_spans_with_calibrated_graph_end():
    """A graph endpoint splits copy and synchronization time with calibration bounds."""
    result = profile.split_wait([(100, 120), (140, 200)], 160, 10)
    assert result == {
        "gpu_pending_ns": 40,
        "after_graph_ns": 40,
        "gpu_pending_lower_ns": 30,
        "gpu_pending_upper_ns": 50,
    }
    assert profile.split_wait([(100, 200)], 50, 0)["after_graph_ns"] == 100
    assert profile.split_wait([(100, 200)], 250, 0)["gpu_pending_ns"] == 100


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_summary_pools_samples_by_load_workload_and_bucket_and_interpolates_p95():
    """Each condition and cache bucket retains its median and interpolated tail."""
    rows = []
    for condition, bucket, values in [("L1", 8, [10, 20, 40]), ("L0", 64, [100])]:
        for value in values:
            row = {f"{name}_ns": value for name in profile.SECTIONS}
            rows.append(
                dict(
                    row,
                    total_ns=7 * value,
                    condition=condition,
                    workload="controlled",
                    bucket=bucket,
                )
            )
    summary = profile.summarize(rows)
    assert len(summary) == 2
    group = next(item for item in summary if item["condition"] == "L1")
    assert group["count"] == 3
    assert group["sections"]["pad_cache"]["median_ns"] == 20
    assert group["sections"]["pad_cache"]["p95_ns"] == pytest.approx(38)
    assert group["total"]["median_ns"] == 140
    rows[0]["pad_cache_ns"] = -1
    with pytest.raises(ValueError, match="negative"):
        profile.summarize(rows)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_proc_cpu_usage_includes_user_and_system_ticks_and_handles_spaces():
    """Actual worker CPU use includes both clocks and is normalized by wall time."""
    fields = ["S"] + ["0"] * 10 + ["130", "20"] + ["0"] * 10
    assert profile.proc_cpu_ticks("123 (worker (audio) load) " + " ".join(fields)) == 150
    before = {123: 20, 456: 10}
    after = {123: 150, 456: 140}
    assert profile.cpu_cores(before, after, elapsed_ns=2_000_000_000, hz=100) == 1.3
    with pytest.raises(ValueError, match="workers"):
        profile.cpu_cores(before, {123: 150}, elapsed_ns=2_000_000_000, hz=100)
    with pytest.raises(ValueError, match="elapsed"):
        profile.cpu_cores(before, after, elapsed_ns=0, hz=100)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_host_cpu_busy_excludes_idle_iowait_and_duplicate_guest_ticks():
    """Host CPU use includes all busy cores while retaining idle and I/O wait separately."""
    assert profile.host_busy_ticks("cpu 100 20 30 400 50 6 7 8 90 10") == 171


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_load_schedule_uses_both_audio_periods_and_requested_core_totals():
    """Synthetic audio workers retain both periods and the specified CPU totals."""
    assert profile.load_schedule("L0") == []
    for condition, total in [("L1", 1.3), ("L2", 3.0)]:
        schedule = profile.load_schedule(condition)
        assert {period for period, _ in schedule} == {10_700_000, 16_000_000}
        assert sum(busy / period for period, busy in schedule) == pytest.approx(total)
        assert all(0 < busy < period for period, busy in schedule)
    with pytest.raises(ValueError, match="condition"):
        profile.load_schedule("unknown")


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_csv_has_timestamps_partition_fields_and_round_trips_quoted_ids(tmp_path):
    """Raw records preserve nanosecond boundaries and every section in a fixed schema."""
    row = {name: 1 for name in profile.STEP_FIELDS}
    row.update(condition="L1", workload="utterance", sample_id='a,"b', run_id="01-L1")
    target = tmp_path / "steps.csv"
    profile.write_steps(target, [row])
    with target.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(profile.STEP_FIELDS)
        loaded = list(reader)
    assert loaded[0]["sample_id"] == 'a,"b'
    assert all(loaded[0][f"{name}_ns"] == "1" for name in profile.SECTIONS)
    assert "graph_end_lower_ns" in loaded[0]
    with pytest.raises(FileExistsError):
        profile.write_steps(target, [row])
    assert len(list(csv.reader(io.StringIO(target.read_text())))) == 2


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_observer_preserves_real_product_tokens_and_covers_each_run(cpu_native, monkeypatch):
    """Wrapping the actual product retains its output and counts each decoder invocation."""
    import numpy as np

    model, cu, clock, _, _ = cpu_native
    monkeypatch.setattr(cu, "cudaEventSynchronize", lambda event: (cu.cudaError_t.cudaSuccess,))
    encoder = model.encode(np.zeros((128, 3000), dtype=np.float32))
    expected = model.greedy_decode(encoder)
    with profile.StepProfiler(model, cu, clock=lambda: round(clock() * 1e9)) as observer:
        actual, rows = observer.capture(lambda: model.greedy_decode(encoder))
        drift = observer.check_clock()
    assert actual.tokens == expected.tokens == [8, 5]
    assert len(rows) == 5
    assert [row["bucket"] for row in rows] == [1, 8, 8, 8, 8]
    assert [row["output_token"] for row in rows] == [8, 8, 8, 8, 5]
    assert all(profile.partition_error(row) == 0 for row in rows)
    assert all(row["graph_gpu_ns"] == 2_000_000 for row in rows)
    assert all(row["h2d_ns"] > 0 and row["d2h_sync_ns"] > 0 for row in rows)
    assert rows[-1]["end_ns"] - rows[0]["start_ns"] == sum(row["total_ns"] for row in rows)
    assert abs(drift["drift_ns"]) <= drift["uncertainty_ns"]


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_observer_restores_product_methods_and_rejects_cold_graphs(cpu_native, monkeypatch):
    """Measurement never silently includes a graph build or leaves hooks after a failure."""
    import numpy as np

    from listenflow.fastest_whisper._inference import _HostIO

    model, cu, _, _, _ = cpu_native
    original_run, original_copy = _HostIO.run, cu.cudaMemcpyAsync
    monkeypatch.setattr(cu, "cudaEventSynchronize", lambda event: (cu.cudaError_t.cudaSuccess,))
    encoder = model.encode(np.zeros((128, 3000), dtype=np.float32))
    with pytest.raises(RuntimeError, match="warm"), profile.StepProfiler(model, cu) as observer:
        observer.capture(lambda: model.greedy_decode(encoder))
    assert _HostIO.run is original_run
    assert cu.cudaMemcpyAsync is original_copy


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_load_workers_report_real_proc_usage_and_stop_after_failure():
    """Periodic standard library workers expose real CPU clocks and leave no child running."""
    import time

    load = profile.CpuLoad("L1")
    with pytest.raises(RuntimeError, match="measurement failed"), load:
        before = load.snapshot()
        time.sleep(0.15)
        after = load.snapshot()
        assert after["ticks"] > before["ticks"]
        assert after["elapsed_ns"] > before["elapsed_ns"]
        raise RuntimeError("measurement failed")
    assert all(not child.is_alive() for child in load.workers)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_wall_control_times_product_without_adding_timing_events(cpu_native, monkeypatch):
    """Whole-step observation preserves tokens and records only product completion events."""
    import numpy as np

    model, cu, clock, _, _ = cpu_native
    encoder = model.encode(np.zeros((128, 3000), dtype=np.float32))
    expected = model.greedy_decode(encoder)

    def forbidden(*_args):
        raise AssertionError("event-free control used a CUDA event")

    original_record = cu.cudaEventRecord

    def record(event, stream):
        assert event == model._decoder._completion
        return original_record(event, stream)

    monkeypatch.setattr(cu, "cudaEventRecord", record)
    monkeypatch.setattr(cu, "cudaEventCreate", forbidden)
    with profile.WallProfiler(model, clock=lambda: round(clock() * 1e9)) as observer:
        actual, rows = observer.capture(lambda: model.greedy_decode(encoder))
    assert actual.tokens == expected.tokens
    assert len(rows) == 5
    assert all(row["total_ns"] > 0 for row in rows)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_tail_partition_retains_stream_gap_copy_and_host_return_without_overlap():
    """The interval after the graph includes stream preparation, output copy and host return."""
    result = profile.split_tail(100, 130, 160, 220)
    assert result == {
        "tail_ns": 120,
        "before_copy_ns": 30,
        "copy_window_ns": 30,
        "host_return_ns": 60,
    }
    assert (
        sum(result[name] for name in ("before_copy_ns", "copy_window_ns", "host_return_ns"))
        == result["tail_ns"]
    )
    # Calibration can put the endpoint just beyond the host's return. Keep raw
    # endpoints separately, but never report a negative elapsed interval.
    assert profile.split_tail(230, 240, 250, 220)["tail_ns"] == 0
    with pytest.raises(ValueError, match="copy"):
        profile.split_tail(100, 160, 130, 220)


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_tail_observer_times_copy_and_wait_cpu_without_changing_product(cpu_native, monkeypatch):
    """Detailed observation preserves tokens and partitions every post-graph interval exactly."""
    import numpy as np

    model, cu, clock, _, _ = cpu_native
    monkeypatch.setattr(cu, "cudaEventSynchronize", lambda event: (cu.cudaError_t.cudaSuccess,))
    encoder = model.encode(np.zeros((128, 3000), dtype=np.float32))
    expected = model.greedy_decode(encoder)
    with profile.StepProfiler(
        model, cu, clock=lambda: round(clock() * 1e9), detailed=True
    ) as observer:
        actual, rows = observer.capture(lambda: model.greedy_decode(encoder))
    assert actual.tokens == expected.tokens
    for row in rows:
        assert row["copy_event_window_ns"] == 1_000_000
        assert row["copy_bytes"] == 4
        assert row["wait_cpu_ns"] >= 0
        assert row["wait_wall_ns"] > 0
        assert row["run_return_ns"] <= row["end_ns"]
        assert row["tail_ns"] == sum(
            row[name] for name in ("before_copy_ns", "copy_window_ns", "host_return_ns")
        )
        assert profile.partition_error(row) == 0


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_wait_observer_retains_tokens_and_records_cpu_without_cuda_events(cpu_native, monkeypatch):
    """Wait-policy comparisons record CPU spent in copy/sync calls without recording events."""
    import numpy as np

    model, cu, _, _, _ = cpu_native
    encoder = model.encode(np.zeros((128, 3000), dtype=np.float32))
    expected = model.greedy_decode(encoder)

    def forbidden(*_args):
        raise AssertionError("wait-policy measurement recorded an event")

    monkeypatch.setattr(cu, "cudaEventCreate", forbidden)
    cpu_ticks = iter(range(0, 10_000, 10))
    monkeypatch.setattr(profile.time, "thread_time_ns", lambda: next(cpu_ticks))
    with profile.WaitProfiler(model, cu) as observer:
        actual, rows = observer.capture(lambda: model.greedy_decode(encoder))
    assert actual.tokens == expected.tokens
    assert len(rows) == 5
    assert all(row["wait_cpu_ns"] >= 0 and row["wait_wall_ns"] > 0 for row in rows)
    assert all(row["wait_cpu_ns"] == 20 for row in rows)
    assert all(row["copy_cpu_ns"] == row["sync_cpu_ns"] == 10 for row in rows)
    assert all(row["wait_wall_ns"] == row["copy_wall_ns"] + row["sync_wall_ns"] for row in rows)


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_schedule_is_set_only_when_requested_and_refuses_a_rejected_policy():
    """Only a diagnostic request changes device scheduling, and CUDA errors stop the run."""
    from types import SimpleNamespace

    from scripts.decoder_host_capture import configure_schedule

    calls = []
    cu = SimpleNamespace(cudaSetDeviceFlags=lambda flag: (calls.append(flag), (0,))[1])
    configure_schedule(cu, None)
    assert calls == []
    for name in ("auto", "spin", "yield", "blocking"):
        configure_schedule(cu, name)
    assert calls == [0, 1, 2, 4]
    cu.cudaSetDeviceFlags = lambda flag: (17,)
    with pytest.raises(RuntimeError, match="scheduling"):
        configure_schedule(cu, "spin")


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_wait_observer_counts_queries_and_retains_each_actual_sleep(cpu_native, monkeypatch):
    """Each measured decoder step retains its own query count and individual sleep durations."""
    import itertools
    import json

    import numpy as np

    from listenflow.fastest_whisper import _inference

    model, cu, _, _, _ = cpu_native
    encoder = model.encode(np.zeros((128, 3000), dtype=np.float32))
    expected = model.greedy_decode(encoder)
    statuses = itertools.cycle(
        [cu.cudaError_t.cudaErrorNotReady] * 2 + [cu.cudaError_t.cudaSuccess]
    )
    original_query = cu.cudaEventQuery
    sleep_clock = [0]
    intervals = []

    def sleep(interval):
        intervals.append(interval)
        sleep_clock[0] += len(intervals) * 100

    monkeypatch.setattr(cu, "cudaEventQuery", lambda event: (next(statuses),))
    monkeypatch.setattr(_inference, "sleep", sleep)
    monkeypatch.setattr(profile.time, "perf_counter_ns", lambda: sleep_clock[0])
    query_before = cu.cudaEventQuery
    with profile.WaitProfiler(model, cu) as observer:
        actual, rows = observer.capture(lambda: model.greedy_decode(encoder))
    assert actual.tokens == expected.tokens
    assert len(rows) == 5
    assert [row["query_count"] for row in rows] == [3] * 5
    assert [json.loads(row["sleep_ns"]) for row in rows] == [
        [100, 200],
        [300, 400],
        [500, 600],
        [700, 800],
        [900, 1000],
    ]
    assert intervals == [50e-6] * 10
    assert cu.cudaEventQuery is query_before
    assert _inference.sleep is sleep
    monkeypatch.setattr(cu, "cudaEventQuery", original_query)
