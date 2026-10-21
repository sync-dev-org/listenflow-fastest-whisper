"""Resource wrapper for one externally authorized issue 40 GPU invocation.

Read-only snapshots retain disk, CPU, and GPU compute-process state. Engine cache
and process temporary files stay in the worktree. An external lead must grant the
GPU/CPU reservation before this entry point is invoked.
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def stamp():
    return datetime.now().astimezone().isoformat()


def snapshot():
    cpu_stat = Path("/proc/stat").read_text().splitlines()[0]
    cpu_at = time.perf_counter_ns()
    commands = (
        ["df", "-h", "/"],
        ["nvidia-smi", "--query-gpu=name,uuid,memory.used,utilization.gpu", "--format=csv"],
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
        ["ps", "-eo", "pid,comm,pcpu,pmem", "--sort=-pcpu"],
    )
    values = {
        "at": stamp(),
        "load_average": os.getloadavg(),
        "cpu_stat": cpu_stat,
        "cpu_at_ns": cpu_at,
    }
    for command in commands:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        values[" ".join(command)] = {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("build", "sections", "wall", "trace"), required=True)
    parser.add_argument("--condition", choices=("L0", "L1", "L2"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    scratch = repo / ".nf/tmp/issue40-stage0"
    args.record_dir = args.record_dir.resolve()
    args.record_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.record_dir / f"{args.run_id}-resource.json"
    if metadata_path.exists():
        raise FileExistsError(metadata_path)
    output = args.record_dir / args.run_id
    disk = (
        subprocess.check_output(["df", "-B1", "--output=size,used,avail,pcent", "/"], text=True)
        .splitlines()[1]
        .split()
    )
    used, available = int(disk[1]), int(disk[2])
    # Existing private engines are already included in 'used'. Reserve room for
    # another complete model and the largest trace/export during each invocation.
    extra_bytes = 6 * 1024**3
    projected = (used + extra_bytes) / (used + available)
    if projected > 0.85:
        raise RuntimeError(f"projected root usage {projected:.2%} exceeds 85%")
    environment = os.environ.copy()
    for name, leaf in (("XDG_CACHE_HOME", "cache"), ("TMPDIR", "temp")):
        target = scratch / leaf
        target.mkdir(parents=True, exist_ok=True)
        environment[name] = str(target)
    python = repo / "python/.venv/bin/python"
    command = [
        str(python),
        "-m",
        "scripts.decoder_host_capture",
        "--mode",
        args.mode,
        "--condition",
        args.condition,
        "--run-id",
        args.run_id,
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(output),
    ]
    if args.mode == "trace":
        command += ["--nvtx-library", str(scratch / "libnvtx.so")]
        command = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--export=sqlite",
            "--output",
            str(args.record_dir / f"{args.run_id}-trace"),
            *command,
        ]
    command = [
        "systemd-run",
        "--user",
        "--scope",
        "-p",
        "MemoryMax=16G",
        "-p",
        "MemorySwapMax=0",
        "timeout",
        "--signal=TERM",
        "--kill-after=10s",
        "25m",
        *command,
    ]
    metadata = {
        "command": command,
        "cwd": str(repo / "python"),
        "started_at": stamp(),
        "before": snapshot(),
        "extra_disk_forecast_bytes": extra_bytes,
        "projected_root_fraction": projected,
        "environment_overrides": {name: environment[name] for name in ("TMPDIR", "XDG_CACHE_HOME")},
    }
    started = time.perf_counter()
    try:
        with (args.record_dir / f"{args.run_id}.log").open("x") as log:
            result = subprocess.run(
                command,
                cwd=repo / "python",
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        metadata["exit_code"] = result.returncode
    finally:
        metadata.update(
            finished_at=stamp(), elapsed_seconds=time.perf_counter() - started, after=snapshot()
        )
        from scripts.decoder_host_profile import host_busy_ticks

        metadata["host_cpu_cores"] = (
            (
                host_busy_ticks(metadata["after"]["cpu_stat"])
                - host_busy_ticks(metadata["before"]["cpu_stat"])
            )
            / os.sysconf("SC_CLK_TCK")
            / ((metadata["after"]["cpu_at_ns"] - metadata["before"]["cpu_at_ns"]) / 1e9)
        )
        with metadata_path.open("x") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "exit_code": result.returncode,
                "elapsed_seconds": metadata["elapsed_seconds"],
            }
        )
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
