"""Measure product decoder host sections under periodic synthetic audio CPU loads.

Run from python/ as ``python -m scripts.decoder_host_profile --help``. No product
function body or builder setting is replaced. CPU-only aggregation is importable
without importing CUDA, TensorRT, NumPy, or the inference package.
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing
import statistics
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

SECTIONS = (
    "python_before",
    "pad_cache",
    "h2d",
    "graph_launch",
    "compact_cache",
    "d2h_sync",
    "python_after",
)
STEP_FIELDS = (
    "run_id",
    "condition",
    "workload",
    "sample_id",
    "step",
    "bucket",
    "input_token",
    "output_token",
    "start_ns",
    "end_ns",
    "total_ns",
    *(f"{name}_ns" for name in SECTIONS),
    "graph_gpu_ns",
    "graph_end_ns",
    "graph_end_lower_ns",
    "graph_end_upper_ns",
    "gpu_pending_ns",
    "after_graph_ns",
    "gpu_pending_lower_ns",
    "gpu_pending_upper_ns",
    "event_record_ns",
    "partition_error",
    "tail_ns",
    "before_copy_ns",
    "copy_window_ns",
    "host_return_ns",
    "copy_event_window_ns",
    "copy_bytes",
    "wait_cpu_ns",
    "wait_wall_ns",
    "run_return_ns",
    "copy_cpu_ns",
    "copy_wall_ns",
    "sync_cpu_ns",
    "sync_wall_ns",
    "query_count",
    "sleep_ns",
    "sleeps",
    "endpoint_method",
    "input_kind",
    "history",
    "precision",
)


class Phases:
    """Assign consecutive host intervals to exactly one of seven sections."""

    def __init__(self, start_ns):
        self.start = self.last = start_ns
        self.section = "python_before"
        self.totals = dict.fromkeys(SECTIONS, 0)
        self.wait_spans = []

    def switch(self, section, now):
        if now < self.last:
            raise ValueError("timestamps must be monotonic")
        if section not in self.totals:
            raise ValueError(f"unknown section: {section}")
        self.totals[self.section] += now - self.last
        if self.section == "d2h_sync":
            self.wait_spans.append((self.last, now))
        self.last, self.section = now, section

    def finish(self, now):
        self.switch(self.section, now)
        return {
            "start_ns": self.start,
            "end_ns": now,
            "total_ns": now - self.start,
            **{f"{name}_ns": value for name, value in self.totals.items()},
        }


def partition_error(row):
    total = row["total_ns"]
    if total <= 0:
        raise ValueError("step duration must be positive")
    return abs(sum(row[f"{name}_ns"] for name in SECTIONS) - total) / total


def split_wait(spans, graph_end_ns, uncertainty_ns):
    """Intersect the graph-pending interval with D2H and synchronization spans."""

    def pending(end):
        return sum(max(0, min(stop, end) - start) for start, stop in spans)

    value = pending(graph_end_ns)
    return {
        "gpu_pending_ns": value,
        "after_graph_ns": sum(stop - start for start, stop in spans) - value,
        "gpu_pending_lower_ns": pending(graph_end_ns - uncertainty_ns),
        "gpu_pending_upper_ns": pending(graph_end_ns + uncertainty_ns),
    }


def split_tail(graph_end, copy_start, copy_end, returned):
    """Partition elapsed time after graph completion, clipping calibration overshoot.

    CUDA event brackets include any GPU idle time while the host enqueues the
    second marker. Only CUPTI memcpy activity timestamps isolate actual DMA.
    """
    if copy_end < copy_start:
        raise ValueError("copy endpoints must be ordered")
    graph_end = min(graph_end, returned)
    copy_start = min(max(copy_start, graph_end), returned)
    copy_end = min(max(copy_end, copy_start), returned)
    return {
        "tail_ns": returned - graph_end,
        "before_copy_ns": copy_start - graph_end,
        "copy_window_ns": copy_end - copy_start,
        "host_return_ns": returned - copy_end,
    }


def distribution(values):
    values = sorted(values)
    position = (len(values) - 1) * 0.95
    low, high = math.floor(position), math.ceil(position)
    return {
        "median_ns": statistics.median(values),
        "p95_ns": values[low] + (values[high] - values[low]) * (position - low),
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if any(row[f"{name}_ns"] < 0 for name in SECTIONS):
            raise ValueError("negative section duration")
        if partition_error(row) > 0.01:
            raise ValueError("partition differs from whole step by more than 1%")
        groups[(row["condition"], row["workload"], int(row["bucket"]))].append(row)
    return [
        {
            "condition": condition,
            "workload": workload,
            "bucket": bucket,
            "count": len(group),
            "total": distribution([row["total_ns"] for row in group]),
            "sections": {
                name: distribution([row[f"{name}_ns"] for row in group])
                for name in (*SECTIONS, "gpu_pending", "after_graph", "graph_gpu")
                if all(f"{name}_ns" in row for row in group)
            },
        }
        for (condition, workload, bucket), group in sorted(groups.items())
    ]


def proc_cpu_ticks(text):
    # comm may contain spaces and parentheses; state starts after the last ')'.
    fields = text[text.rfind(")") + 2 :].split()
    return int(fields[11]) + int(fields[12])


def host_busy_ticks(text):
    fields = [int(value) for value in text.split()[1:]]
    return sum(fields[index] for index in (0, 1, 2, 5, 6, 7))


def cpu_cores(before, after, *, elapsed_ns, hz):
    if before.keys() != after.keys():
        raise ValueError("CPU snapshots must contain the same workers")
    if elapsed_ns <= 0 or hz <= 0:
        raise ValueError("elapsed time and clock frequency must be positive")
    ticks = sum(after[pid] - before[pid] for pid in before)
    if ticks < 0:
        raise ValueError("worker CPU clocks moved backwards")
    return ticks / hz / (elapsed_ns / 1e9)


def load_schedule(condition):
    counts = {"L0": 0, "L1": 4, "L2": 6}
    if condition not in counts:
        raise ValueError(f"unknown condition: {condition}")
    duty = 0.325 if condition == "L1" else 0.5
    return [
        (period, round(period * duty))
        for period in ([10_700_000, 16_000_000] * (counts[condition] // 2))
    ]


def write_steps(path: Path, rows):
    with path.open("x", newline="") as stream:
        fields = [name for name in STEP_FIELDS if any(name in row for row in rows)]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _cpu_worker(period_ns, busy_ns, stop, ready):
    ready.wait(timeout=30)
    deadline = time.perf_counter_ns()
    while not stop.is_set():
        cpu_end = time.thread_time_ns() + busy_ns
        while time.thread_time_ns() < cpu_end:
            pass
        deadline += period_ns
        remaining = deadline - time.perf_counter_ns()
        if remaining > 0:
            stop.wait(remaining / 1e9)
        else:
            # Missed releases are not accumulated into a sustained catch-up burst.
            deadline = time.perf_counter_ns()


class CpuLoad:
    """Own periodic workers and sample their actual Linux user/system CPU clocks."""

    def __init__(self, condition):
        self.schedule = load_schedule(condition)
        self.context = multiprocessing.get_context("spawn")
        self.stop = self.context.Event()
        self.ready = self.context.Barrier(len(self.schedule) + 1)
        self.workers = []

    def __enter__(self):
        try:
            for period, busy in self.schedule:
                worker = self.context.Process(
                    target=_cpu_worker, args=(period, busy, self.stop, self.ready)
                )
                worker.start()
                self.workers.append(worker)
            self.ready.wait(timeout=30)
            self.started = time.perf_counter_ns()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def snapshot(self):
        clocks = {
            worker.pid: proc_cpu_ticks(Path(f"/proc/{worker.pid}/stat").read_text())
            for worker in self.workers
        }
        if any(not worker.is_alive() for worker in self.workers):
            raise RuntimeError("CPU load worker exited during measurement")
        return {
            "at_ns": time.perf_counter_ns(),
            "elapsed_ns": time.perf_counter_ns() - self.started,
            "ticks": sum(clocks.values()),
            "workers": clocks,
        }

    def __exit__(self, *_exc):
        self.stop.set()
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)


def _native_steps(records, started, ended, *, events, detailed):
    """Partition host timestamps using the innermost recorded native operation."""
    operations = [
        row
        for row in records
        if row["name"] in ("lffw_greedy_step", "lffw_prefix_step", "lffw_sampled_step")
    ]
    if not operations:
        raise RuntimeError("measurement executed no native decoder steps")
    result = []
    previous = started
    categories = {
        "cache_prepare": "pad_cache",
        "zero": "pad_cache",
        "h2d": "h2d",
        "enqueue": "graph_launch",
        "cache_append": "compact_cache",
        "d2d": "compact_cache",
        "d2h": "d2h_sync",
        "wait": "d2h_sync",
    }
    for index, operation in enumerate(operations):
        stop = ended if index == len(operations) - 1 else operation["end_ns"]
        spans = operation["spans"]
        enqueues = [span for span in spans if span["phase"] == "enqueue"]
        bucket = next((span["bucket"] for span in enqueues), 0)
        row = {
            "start_ns": previous,
            "end_ns": stop,
            "total_ns": stop - previous,
            "step": index,
            "bucket": bucket,
            "input_token": operation["input_token"],
            "output_token": operation["output_token"],
        }
        points = sorted(
            {
                previous,
                stop,
                *(
                    max(previous, min(stop, span[key]))
                    for span in spans
                    for key in ("start_ns", "end_ns")
                ),
            }
        )
        totals = dict.fromkeys(SECTIONS, 0)
        first_enqueue = min((span["start_ns"] for span in enqueues), default=stop)
        for lower, upper in pairwise(points):
            covering = [
                span for span in spans if span["start_ns"] <= lower and span["end_ns"] >= upper
            ]
            if covering:
                inner = min(covering, key=lambda span: span["end_ns"] - span["start_ns"])
                category = categories.get(
                    inner["phase"], "python_after" if lower >= first_enqueue else "python_before"
                )
            else:
                category = "python_after" if lower >= first_enqueue else "python_before"
            totals[category] += upper - lower
        row.update({f"{name}_ns": value for name, value in totals.items()})
        copies = [span for span in spans if span["phase"] == "d2h"]
        waits = [span for span in spans if span["phase"] == "wait"]
        row.update(
            copy_cpu_ns=sum(span["cpu_ns"] for span in copies),
            copy_wall_ns=sum(span["end_ns"] - span["start_ns"] for span in copies),
            sync_cpu_ns=sum(span["cpu_ns"] for span in waits),
            sync_wall_ns=sum(span["end_ns"] - span["start_ns"] for span in waits),
            query_count=sum(span["query_count"] for span in waits),
            sleeps=operation["sleeps"],
            sleep_ns=json.dumps(
                [sleep["end_ns"] - sleep["start_ns"] for sleep in operation["sleeps"]]
            ),
        )
        row["wait_cpu_ns"] = row["copy_cpu_ns"] + row["sync_cpu_ns"]
        row["wait_wall_ns"] = row["copy_wall_ns"] + row["sync_wall_ns"]
        if events:
            if any(span["graph_capture"] for span in spans):
                raise RuntimeError("decoder graph must be warm before measurement")
            if not enqueues or any(span["gpu_ms"] is None for span in enqueues):
                raise RuntimeError("native GPU timing events are unavailable")
            row["graph_gpu_ns"] = round(sum(span["gpu_ms"] for span in enqueues) * 1e6)
            lower = min(span["start_ns"] for span in enqueues)
            upper = max((span["end_ns"] for span in waits), default=operation["end_ns"])
            midpoint, uncertainty = (lower + upper) // 2, (upper - lower + 1) // 2
            row.update(
                graph_end_ns=midpoint,
                graph_end_lower_ns=lower,
                graph_end_upper_ns=upper,
                endpoint_method="host bounds; no event synchronization calibration",
            )
            row.update(
                split_wait(
                    [(span["start_ns"], span["end_ns"]) for span in waits], midpoint, uncertainty
                )
            )
            if detailed:
                row.update(
                    copy_bytes=sum(span["bytes"] for span in copies),
                    copy_event_window_ns=round(sum(span["gpu_ms"] or 0 for span in copies) * 1e6),
                    run_return_ns=operation["end_ns"],
                    **split_tail(
                        midpoint,
                        min((span["start_ns"] for span in copies), default=stop),
                        max((span["end_ns"] for span in copies), default=stop),
                        operation["end_ns"],
                    ),
                )
        row["partition_error"] = partition_error(row)
        result.append(row)
        previous = stop
    return result


class WallProfiler:
    """Read real native token calls without adding timing events."""

    flags = 1

    def __init__(self, model, *, clock=time.perf_counter_ns):
        self.model, self.clock = model, clock
        self.session = None
        self.detailed = False

    def __enter__(self):
        from scripts.profile_latency import diagnostic_session

        self.session = diagnostic_session(self.flags)
        return self

    def capture(self, function, **_kwargs):
        started = self.clock()
        with self.session:
            result = function()
            ended = self.clock()
        rows = _native_steps(
            self.session.records,
            started,
            ended,
            events=bool(self.flags & 2),
            detailed=self.detailed,
        )
        return result, rows

    def __exit__(self, *_exc):
        if self.session is not None:
            self.session.close()
            self.session = None


class StepProfiler(WallProfiler):
    """Read native copy/replay events and disjoint host processing times."""

    flags = 7

    def __init__(self, model, cudart=None, *, clock=time.perf_counter_ns, detailed=False):
        super().__init__(model, clock=clock)
        self.detailed = detailed


class WaitProfiler(WallProfiler):
    """Read copy/wait CPU times and real polling without timing events."""

    flags = 5

    def __init__(self, model, cudart=None):
        super().__init__(model)
