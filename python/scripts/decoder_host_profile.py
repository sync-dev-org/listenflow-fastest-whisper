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
from contextlib import ExitStack
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


class StepProfiler:
    """Wrap product functions, retaining exclusive host intervals and CUDA events.

    Between decoder calls, Python token handling belongs to the following step's
    section 1. Final output assembly belongs to the last step's section 7. Sections
    2/3/4/5/6 time the corresponding function/API calls; remaining buffer and shape
    handling belongs to section 1 before enqueue and section 7 after enqueue.
    Event record overhead belongs to those Python sections and is also reported.
    """

    def __init__(self, model, cudart, *, clock=time.perf_counter_ns, detailed=False):
        self.model, self.cu, self.clock = model, cudart, clock
        self.detailed = detailed
        self.stack = ExitStack()
        self.events = []
        self.active = False
        self.current = None

    def cuda(self, result):
        if result[0] != self.cu.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA operation failed: {result[0]}")
        return result[1] if len(result) > 1 else None

    def event(self):
        event = self.cuda(self.cu.cudaEventCreate())
        self.events.append(event)
        return event

    def __enter__(self):
        from unittest.mock import patch

        from listenflow.fastest_whisper._inference import _HostIO

        self.io = self.model._decoder
        try:
            self.origin = self.event()
            self.check_event = self.event()
            self.pairs = [
                (self.event(), self.event()) for _ in range(self.model.config.text_positions + 8)
            ]
            if self.detailed:
                self.copy_pairs = [(self.event(), self.event()) for _ in self.pairs]
            original_run = _HostIO.run
            original_pad, original_compact = _HostIO._pad_cache, _HostIO._compact_cache
            original_enqueue = _HostIO._enqueue
            original_copy, original_sync = self.cu.cudaMemcpyAsync, self.cu.cudaStreamSynchronize

            def region(section, function, *args, **kwargs):
                ledger = self.current
                if ledger is None:
                    return function(*args, **kwargs)
                previous = ledger.section
                ledger.switch(section, self.clock())
                cpu_before = (
                    time.thread_time_ns() if self.detailed and section == "d2h_sync" else None
                )
                try:
                    return function(*args, **kwargs)
                finally:
                    if cpu_before is not None:
                        self.wait_cpu += time.thread_time_ns() - cpu_before
                    ledger.switch(previous, self.clock())

            def run(io, inputs, **kwargs):
                if not self.active or io is not self.io:
                    return original_run(io, inputs, **kwargs)
                ledger = self.current = Phases(self.next_start)
                index = len(self.records)
                if index >= len(self.pairs):
                    raise RuntimeError("decoder event pool exhausted")
                self.event_overhead = 0
                self.wait_cpu = self.copy_bytes = 0
                self.copy_started = False
                self.bucket = None
                try:
                    outputs = original_run(io, inputs, **kwargs)
                    ended = self.clock()
                finally:
                    self.current = None
                row = ledger.finish(ended)
                token_array = outputs.get("token")
                if not hasattr(token_array, "reshape"):
                    token_array = outputs.get("greedy")
                row.update(
                    bucket=self.bucket,
                    input_token=int(inputs["tokens"].reshape(-1)[0]),
                    output_token=int(token_array.reshape(-1)[0]),
                    step=index,
                    event_record_ns=self.event_overhead,
                )
                if self.detailed:
                    row.update(
                        copy_bytes=self.copy_bytes,
                        wait_cpu_ns=self.wait_cpu,
                        wait_wall_ns=sum(stop - start for start, stop in ledger.wait_spans),
                        run_return_ns=ended,
                    )
                self.records.append((row, ledger))
                self.next_start = ended
                return outputs

            def pad(io, *args, **kwargs):
                return region("pad_cache", original_pad, io, *args, **kwargs)

            def compact(io, *args, **kwargs):
                return region("compact_cache", original_compact, io, *args, **kwargs)

            def copy(*args):
                kind = args[3]
                section = (
                    "h2d" if kind == self.cu.cudaMemcpyKind.cudaMemcpyHostToDevice else "d2h_sync"
                )
                if self.detailed and self.current is not None and section == "d2h_sync":
                    self.copy_bytes += args[2]
                    if not self.copy_started:
                        self.record_copy_event(0, args[4])
                        self.copy_started = True
                return region(section, original_copy, *args)

            def synchronize(stream):
                if self.detailed and self.current is not None and self.copy_started:
                    self.record_copy_event(1, stream)
                return region("d2h_sync", original_sync, stream)

            if hasattr(_HostIO, "_wait"):
                original_wait = _HostIO._wait

                def wait(io):
                    if self.detailed and self.current is not None and self.copy_started:
                        self.record_copy_event(1, io.stream)
                    return region("d2h_sync", original_wait, io)

                self.stack.enter_context(patch.object(_HostIO, "_wait", wait))

            def enqueue(io, addresses, *, stream):
                if self.current is None or io is not self.io:
                    return original_enqueue(io, addresses, stream=stream)
                shapes = tuple(
                    (name, io.execution.get_tensor_shape(name)) for name in io.execution._ranges
                )
                entry = io._graphs.entries.get(shapes)
                if entry is None or entry[2] != tuple(addresses.items()):
                    raise RuntimeError("decoder graph must be warm before measurement")
                self.bucket = io.execution.get_tensor_shape("self_keys")[-2]
                start, end = self.pairs[len(self.records)]
                before = self.clock()
                self.cuda(self.cu.cudaEventRecord(start, stream))
                after = self.clock()
                self.event_overhead += after - before
                self.current.switch("graph_launch", after)
                result = original_enqueue(io, addresses, stream=stream)
                self.current.switch("python_after", self.clock())
                before = self.clock()
                self.cuda(self.cu.cudaEventRecord(end, stream))
                self.event_overhead += self.clock() - before
                return result

            for owner, name, replacement in (
                (_HostIO, "run", run),
                (_HostIO, "_pad_cache", pad),
                (_HostIO, "_compact_cache", compact),
                (_HostIO, "_enqueue", enqueue),
                (self.cu, "cudaMemcpyAsync", copy),
                (self.cu, "cudaStreamSynchronize", synchronize),
            ):
                self.stack.enter_context(patch.object(owner, name, replacement))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def record_copy_event(self, index, stream):
        before = self.clock()
        self.cuda(self.cu.cudaEventRecord(self.copy_pairs[len(self.records)][index], stream))
        self.event_overhead += self.clock() - before

    def calibrate(self):
        self.cuda(self.cu.cudaStreamSynchronize(self.io.stream))
        lower = self.clock()
        self.cuda(self.cu.cudaEventRecord(self.origin, self.io.stream))
        self.cuda(self.cu.cudaEventSynchronize(self.origin))
        upper = self.clock()
        self.anchor = {"lower_ns": lower, "upper_ns": upper}

    def check_clock(self):
        lower = self.clock()
        self.cuda(self.cu.cudaEventRecord(self.check_event, self.io.stream))
        self.cuda(self.cu.cudaEventSynchronize(self.check_event))
        upper = self.clock()
        offset = round(
            float(self.cuda(self.cu.cudaEventElapsedTime(self.origin, self.check_event))) * 1e6
        )
        first = self.anchor
        uncertainty = (first["upper_ns"] - first["lower_ns"] + upper - lower + 1) // 2
        uncertainty += math.ceil(abs(offset) * 2**-23) + 1_000
        drift = (lower + upper - first["lower_ns"] - first["upper_ns"]) // 2 - offset
        return {
            "first": first,
            "last": {"lower_ns": lower, "upper_ns": upper},
            "drift_ns": drift,
            "uncertainty_ns": uncertainty,
            "within_uncertainty": abs(drift) <= uncertainty,
        }

    def capture(self, function, *, recalibrate=True):
        self.records = []
        if recalibrate:
            self.calibrate()
        lower, upper = self.anchor["lower_ns"], self.anchor["upper_ns"]
        self.next_start = self.clock()
        self.active = True
        try:
            result = function()
            ended = self.clock()
        finally:
            self.active = False
        if not self.records:
            raise RuntimeError("measurement executed no decoder steps")
        row, ledger = self.records[-1]
        ledger.switch("python_after", row["end_ns"])
        row.update(ledger.finish(ended))
        midpoint, uncertainty = (lower + upper) // 2, (upper - lower + 1) // 2
        for index, (row, ledger) in enumerate(self.records):
            start, end = self.pairs[index]
            graph_gpu = round(float(self.cuda(self.cu.cudaEventElapsedTime(start, end))) * 1e6)
            offset = round(float(self.cuda(self.cu.cudaEventElapsedTime(self.origin, end))) * 1e6)
            # CUDA event elapsed time is a float in milliseconds. Include its
            # floating-point representation error for a long utterance anchor.
            error = uncertainty + math.ceil(abs(offset) * 2**-23) + 1_000
            graph_end = midpoint + offset
            row.update(
                graph_gpu_ns=graph_gpu,
                graph_end_ns=graph_end,
                graph_end_lower_ns=graph_end - error,
                graph_end_upper_ns=graph_end + error,
                **split_wait(ledger.wait_spans, graph_end, error),
                partition_error=partition_error(row),
            )
            if self.detailed:
                copy_start, copy_end = self.copy_pairs[index]
                offsets = [
                    round(float(self.cuda(self.cu.cudaEventElapsedTime(self.origin, event))) * 1e6)
                    for event in (copy_start, copy_end)
                ]
                row.update(
                    copy_event_window_ns=round(
                        float(self.cuda(self.cu.cudaEventElapsedTime(copy_start, copy_end))) * 1e6
                    ),
                    **split_tail(
                        graph_end,
                        midpoint + offsets[0],
                        midpoint + offsets[1],
                        row["run_return_ns"],
                    ),
                )
        return result, [row for row, _ in self.records]

    def __exit__(self, *_exc):
        self.active = False
        self.current = None
        self.stack.close()
        try:
            if self.events:
                self.cuda(self.cu.cudaDeviceSynchronize())
        finally:
            for event in reversed(self.events):
                self.cuda(self.cu.cudaEventDestroy(event))
            self.events.clear()


class WallProfiler:
    """Measure the same decoder step intervals without sections or CUDA events."""

    def __init__(self, model, *, clock=time.perf_counter_ns):
        self.model, self.clock = model, clock
        self.active = False

    def __enter__(self):
        from unittest.mock import patch

        from listenflow.fastest_whisper._inference import _HostIO

        original = _HostIO.run

        def run(io, inputs, **kwargs):
            if not self.active or io is not self.model._decoder:
                return original(io, inputs, **kwargs)
            outputs = original(io, inputs, **kwargs)
            ended = self.clock()
            token = outputs.get("token")
            if not hasattr(token, "reshape"):
                token = outputs.get("greedy")
            self.rows.append(
                {
                    "start_ns": self.next_start,
                    "end_ns": ended,
                    "total_ns": ended - self.next_start,
                    "bucket": io.execution.get_tensor_shape("self_keys")[-2],
                    "step": len(self.rows),
                    "input_token": int(inputs["tokens"].reshape(-1)[0]),
                    "output_token": int(token.reshape(-1)[0]),
                }
            )
            self.next_start = ended
            return outputs

        self.patch = patch.object(_HostIO, "run", run)
        self.patch.__enter__()
        return self

    def capture(self, function, **_kwargs):
        self.rows = []
        self.next_start = self.clock()
        self.active = True
        try:
            result = function()
            ended = self.clock()
        finally:
            self.active = False
        if not self.rows:
            raise RuntimeError("measurement executed no decoder steps")
        row = self.rows[-1]
        row["total_ns"] += ended - row["end_ns"]
        row["end_ns"] = ended
        return result, self.rows

    def __exit__(self, *_exc):
        self.active = False
        self.patch.__exit__(None, None, None)


class WaitProfiler(WallProfiler):
    """Measure D2H and completion-wait CPU/wall clocks without adding timing events."""

    def __init__(self, model, cudart):
        super().__init__(model)
        self.cu = cudart
        self.stack = ExitStack()
        self.in_wait = False

    def __enter__(self):
        from unittest.mock import patch

        from listenflow.fastest_whisper import _inference
        from listenflow.fastest_whisper._inference import _HostIO

        super().__enter__()
        self.stack.callback(super().__exit__, None, None, None)
        original_run = _HostIO.run

        def run(io, inputs, **kwargs):
            measured = self.active and io is self.model._decoder
            if measured:
                self.wait_cpu = self.wait_wall = 0
                self.query_count = 0
                self.sleep_durations = []
                self.wait_details = dict.fromkeys(
                    ("copy_cpu_ns", "copy_wall_ns", "sync_cpu_ns", "sync_wall_ns"), 0
                )
            outputs = original_run(io, inputs, **kwargs)
            if measured:
                self.rows[-1].update(
                    wait_cpu_ns=self.wait_cpu,
                    wait_wall_ns=self.wait_wall,
                    query_count=self.query_count,
                    sleep_ns=json.dumps(self.sleep_durations),
                    **self.wait_details,
                )
            return outputs

        def wrap(function, is_copy):
            def call(*args):
                measured = self.active and (
                    not is_copy or args[3] == self.cu.cudaMemcpyKind.cudaMemcpyDeviceToHost
                )
                if not measured:
                    return function(*args)
                cpu, wall = time.thread_time_ns(), time.perf_counter_ns()
                if not is_copy:
                    self.in_wait = True
                try:
                    return function(*args)
                finally:
                    self.in_wait = False
                    cpu_elapsed = time.thread_time_ns() - cpu
                    wall_elapsed = time.perf_counter_ns() - wall
                    self.wait_cpu += cpu_elapsed
                    self.wait_wall += wall_elapsed
                    section = "copy" if is_copy else "sync"
                    self.wait_details[f"{section}_cpu_ns"] += cpu_elapsed
                    self.wait_details[f"{section}_wall_ns"] += wall_elapsed

            return call

        original_query = self.cu.cudaEventQuery

        def query(*args):
            if self.active and self.in_wait:
                self.query_count += 1
            return original_query(*args)

        original_sleep = getattr(_inference, "sleep", None)

        def sleep(interval):
            if not self.active or not self.in_wait:
                return original_sleep(interval)
            started = time.perf_counter_ns()
            try:
                return original_sleep(interval)
            finally:
                self.sleep_durations.append(time.perf_counter_ns() - started)

        self.stack.enter_context(patch.object(_HostIO, "run", run))
        self.stack.enter_context(patch.object(self.cu, "cudaEventQuery", query))
        if original_sleep is not None:
            self.stack.enter_context(patch.object(_inference, "sleep", sleep))
        for name, is_copy in (("cudaMemcpyAsync", True), ("cudaStreamSynchronize", False)):
            self.stack.enter_context(
                patch.object(self.cu, name, wrap(getattr(self.cu, name), is_copy))
            )
        if hasattr(_HostIO, "_wait"):
            self.stack.enter_context(patch.object(_HostIO, "_wait", wrap(_HostIO._wait, False)))
        return self

    def __exit__(self, *_exc):
        self.active = False
        self.stack.close()
