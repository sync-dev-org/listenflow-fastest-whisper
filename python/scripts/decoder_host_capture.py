"""Run one issue 40 load condition in a separately resource-limited process.

All GPU runs need an external exclusive:gpu/exclusive:cpu reservation. This script
does not acquire one. Set XDG_CACHE_HOME and TMPDIR inside the diagnostic worktree
before starting; launch with systemd-run MemoryMax=16G, MemorySwapMax=0 and timeout
25m. The trace mode uses whole-step host observation only, with no CUDA events.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

from scripts.decoder_host_profile import (
    CpuLoad,
    StepProfiler,
    WaitProfiler,
    WallProfiler,
    summarize,
    write_steps,
)

TOKENS = (50258, 50259, 50359, 50363, 440, 1234, 567, 890)


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def configure_schedule(cu, schedule):
    """Change scheduling only in an explicitly requested diagnostic process."""
    if schedule is not None:
        flag = {"auto": 0, "spin": 1, "yield": 2, "blocking": 4}[schedule]
        result = cu.cudaSetDeviceFlags(flag)
        if result[0] != 0:
            raise RuntimeError(f"CUDA scheduling configuration failed: {result[0]}")


def run(args):
    import numpy as np
    from cuda.bindings import runtime as cu
    from scripts.benchmark_latency import _cache_engine_hashes, _load_waveform
    from scripts.profile_latency import select_utterances

    from listenflow.fastest_whisper._cached_model import CachedWhisper
    from listenflow.fastest_whisper._decoding import decoding_settings
    from listenflow.fastest_whisper._inference import _cuda, _DeviceTensor, _release_tensors

    configure_schedule(cu, args.schedule)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "started_at": datetime.now().astimezone().isoformat(),
        "pid": os.getpid(),
        "mode": args.mode,
        "condition": args.condition,
        "run_id": args.run_id,
        "input_tensors": {},
        "clock_checks": [],
        "anchors": [],
        "tokens": {},
        "load_samples": [],
        "load_average": os.getloadavg(),
        "requested_schedule": args.schedule,
    }
    samples = [(row, _load_waveform(row)) for row in select_utterances(args.manifest, per_band=10)]
    record["selected_ids"] = [row["id"] for row, _ in samples]
    record["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    owned = []

    def tensor(name, shape, seed):
        array = (np.random.default_rng(seed).standard_normal(shape, dtype=np.float32) * 0.1).astype(
            np.float16
        )
        record["input_tensors"][name] = {
            "shape": shape,
            "seed": seed,
            "sha256": hashlib.sha256(array).hexdigest(),
        }
        pointer = int(_cuda(cu.cudaMalloc(array.nbytes), "allocate diagnostic tensor"))
        result = _DeviceTensor(pointer, shape, array.dtype, 0)
        owned.append(result)
        _cuda(
            cu.cudaMemcpy(
                pointer, array.ctypes.data, array.nbytes, cu.cudaMemcpyKind.cudaMemcpyHostToDevice
            ),
            "upload diagnostic tensor",
        )
        return result

    try:
        with CachedWhisper(args.checkpoint_dir, precision="fp16", device=0) as model:
            record["device_flags"] = int(_cuda(cu.cudaGetDeviceFlags(), "read scheduling flags"))
            if args.mode == "build":
                model.warmup()
                record["engines"] = _cache_engine_hashes()
                return
            cross = {
                name: tensor(name, (32, 1, 20, 1500, 64), seed)
                for name, seed in (("cross_keys", 3301), ("cross_values", 3302))
            }
            controls = {}
            resident = tuple(name for name in model._decoder.outputs if name != "token")

            def controlled(inputs):
                output = model._decoder.run(inputs, device_outputs=resident)
                token = int(output["token"].reshape(-1)[0])
                _release_tensors(output)
                return token

            for bucket in (8, 64, 128):
                shape = (32, 1, 20, bucket - 1, 64)
                inputs = {
                    **cross,
                    "self_keys": tensor(f"keys_{bucket}", shape, 3401 + bucket),
                    "self_values": tensor(f"values_{bucket}", shape, 3402 + bucket),
                    "tokens": np.array([[TOKENS[0]]], np.int32),
                    "positions": np.array([[bucket - 1]], np.int32),
                    "mask": np.zeros((1, 1, 1, bucket), np.float32),
                    "suppression": np.zeros((1, 1, model.config.vocab_size), np.float32),
                }
                for index in range(24):
                    inputs["tokens"][0, 0] = TOKENS[index % len(TOKENS)]
                    controlled(inputs)
                controls[bucket] = inputs
            # Warm every natural decoder shape and retain an uninstrumented
            # product reference in each fresh process, before introducing load.
            settings = decoding_settings(beam_size=1, temperature=0, best_of=1)
            baseline = {}
            for row, audio in samples:
                mel = model._mel_spectrogram(audio)
                result = model.decode_short(model.encode(mel), [], settings)
                baseline[row["id"]] = result.tokens
            record["baseline_tokens"] = baseline
            record["engines"] = _cache_engine_hashes()

            # Natural decoding changes the live buffer leases. Rewarm controlled
            # signatures before every bucket outside its measurement interval.
            bridge = None
            if args.mode == "trace":
                if args.nvtx_library is None:
                    raise ValueError("trace requires --nvtx-library")
                bridge = ctypes.CDLL(str(args.nvtx_library.resolve()))
                bridge.issue40_push.argtypes = [ctypes.c_char_p]
                bridge.issue40_pop.argtypes = []
            sections = args.mode in ("sections", "tail")
            observer = (
                StepProfiler(model, cu, detailed=args.mode == "tail")
                if sections
                else WaitProfiler(model, cu)
                if args.mode == "wait"
                else WallProfiler(model)
            )
            all_rows = []
            with observer, CpuLoad(args.condition) as load:
                time.sleep(0.25)
                load_before = load.snapshot()

                def collect(function, sample_id, workload, *, recalibrate=True):
                    if bridge:
                        bridge.issue40_push(f"issue40/{workload}/{sample_id}".encode())
                    try:
                        value, rows = observer.capture(function, recalibrate=recalibrate)
                    finally:
                        if bridge:
                            bridge.issue40_pop()
                    for row in rows:
                        row.update(
                            run_id=args.run_id,
                            condition=args.condition,
                            workload=workload,
                            sample_id=sample_id,
                        )
                    all_rows.extend(rows)
                    return value

                if bridge:
                    _cuda(cu.cudaProfilerStart(), "start trace")
                try:
                    for bucket, inputs in controls.items():
                        for index in range(24):
                            inputs["tokens"][0, 0] = TOKENS[index % len(TOKENS)]
                            controlled(inputs)
                        if sections:
                            observer.calibrate()
                            record["anchors"].append(
                                {"sample_id": f"bucket-{bucket}", **observer.anchor}
                            )
                        for index in range(24):
                            inputs["tokens"][0, 0] = TOKENS[index % len(TOKENS)]
                            name = f"bucket-{bucket}-pair-{index:02}"
                            token = collect(
                                lambda inputs=inputs: controlled(inputs),
                                name,
                                "controlled",
                                recalibrate=False,
                            )
                            record["tokens"][name] = [token]
                        if sections:
                            record["clock_checks"].append(
                                {"sample_id": f"bucket-{bucket}", **observer.check_clock()}
                            )
                        record["load_samples"].append(load.snapshot())
                    for row, audio in samples:
                        mel = model._mel_spectrogram(audio)
                        encoder = model.encode(mel)
                        # Restore natural-input graph signatures after the
                        # controlled tensors, outside all measured NVTX ranges.
                        model.decode_short(encoder, [], settings)
                        result = collect(
                            lambda encoder=encoder: model.decode_short(encoder, [], settings),
                            row["id"],
                            "utterance",
                        )
                        record["tokens"][row["id"]] = result.tokens
                        if result.tokens != baseline[row["id"]]:
                            raise RuntimeError(f"observer changed product tokens: {row['id']}")
                        if sections:
                            record["anchors"].append({"sample_id": row["id"], **observer.anchor})
                            record["clock_checks"].append(
                                {"sample_id": row["id"], **observer.check_clock()}
                            )
                        record["load_samples"].append(load.snapshot())
                finally:
                    if bridge:
                        _cuda(cu.cudaProfilerStop(), "stop trace")
                load_after = load.snapshot()
                hz = os.sysconf("SC_CLK_TCK")
                from scripts.decoder_host_profile import cpu_cores

                record["load"] = {
                    "before": load_before,
                    "after": load_after,
                    "hz": hz,
                    "schedule": load.schedule,
                    "cores": cpu_cores(
                        load_before["workers"],
                        load_after["workers"],
                        elapsed_ns=load_after["at_ns"] - load_before["at_ns"],
                        hz=hz,
                    ),
                }
            if sections:
                write_steps(args.output_dir / "steps.csv", all_rows)
                record["summary"] = summarize(all_rows)
            else:
                fields = list(all_rows[0])
                import csv

                with (args.output_dir / "steps.csv").open("x", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(all_rows)
            record["step_count"] = len(all_rows)
            with (args.output_dir / "cpu.csv").open("x") as stream:
                import csv

                writer = csv.DictWriter(stream, fieldnames=("at_ns", "elapsed_ns", "ticks"))
                writer.writeheader()
                writer.writerows(
                    {name: item[name] for name in writer.fieldnames}
                    for item in [load_before, *record["load_samples"], load_after]
                )
    finally:
        for item in owned:
            item.close()
        record["finished_at"] = datetime.now().astimezone().isoformat()
        save(args.output_dir / "run.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("build", "sections", "tail", "wall", "wait", "trace"), required=True
    )
    parser.add_argument("--schedule", choices=("auto", "spin", "yield", "blocking"))
    parser.add_argument("--condition", choices=("L0", "L1", "L2"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nvtx-library", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
