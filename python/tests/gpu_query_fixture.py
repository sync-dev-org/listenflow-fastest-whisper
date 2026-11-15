"""Replace only the external GPU command and OS clock for observation tests."""

import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch


@contextmanager
def gpu_queries():
    from scripts import benchmark_latency

    original = benchmark_latency.subprocess.check_output
    state = SimpleNamespace(seconds=1000.0, calls=[], sleeps=[], completed=0, competing_at=None)

    def sleep(seconds):
        state.sleeps.append((state.completed, seconds))
        state.seconds += seconds

    def query(args, **kwargs):
        if args[0] != "nvidia-smi":
            return original(args, **kwargs)
        state.calls.append((args, state.completed, state.seconds))
        pids = [os.getpid()]
        if state.completed == state.competing_at:
            pids.append(os.getpid() + 1)
        if "--query-compute-apps=pid" in args:
            return "\n".join(map(str, pids)) + "\n"
        if "--query-gpu=clocks.sm,utilization.gpu,memory.used" in args:
            return "2500, 50, 4000\n"
        if "--query-gpu=name" in args:
            return "CPU test GPU\n"
        if "--query-gpu=driver_version" in args:
            return "600.00\n"
        assert args == ["nvidia-smi", "-q", "-x"]
        processes = "".join(f"<process_info><pid>{pid}</pid></process_info>" for pid in pids)
        return f"""<nvidia_smi_log>
          <driver_version>600.00</driver_version>
          <gpu><product_name>CPU test GPU</product_name>
            <clocks><sm_clock>2500 MHz</sm_clock></clocks>
            <utilization><gpu_util>50 %</gpu_util></utilization>
            <fb_memory_usage><used>4000 MiB</used></fb_memory_usage>
            <processes>{processes}</processes>
          </gpu>
        </nvidia_smi_log>"""

    with (
        patch.object(benchmark_latency.subprocess, "check_output", query),
        patch.object(benchmark_latency.time, "monotonic", lambda: state.seconds),
        patch.object(benchmark_latency.time, "sleep", sleep),
    ):
        yield state
