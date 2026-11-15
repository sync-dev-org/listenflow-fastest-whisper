"""Run one issue 40 load condition in a separately resource-limited process.

All GPU runs need an external resource reservation. This script does not acquire
one or choose a cache directory. Use the existing GPU test wrapper for disposable
storage and apply the reserved process limits. Trace mode retains native host
observations without adding CUDA timing events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def capture_transcriptions(model, samples):
    """Retain product text and close the native trace on every exit."""
    from scripts.profile_latency import capture_native

    rows = []
    for row, waveform in samples:
        segments, records = capture_native(
            lambda waveform=waveform: model.transcribe(
                waveform, beam_size=1, temperature=0, best_of=1
            )
        )
        rows.append(
            {
                "id": row["id"],
                "frames": int(row["frames"]),
                "output": " ".join(segment.text for segment in segments),
                "native_operations": records,
            }
        )
    return rows


def run(args, *, per_band=10, iterations=24):
    """Run the requested observation using product audio and prefix-computed KV."""
    import csv
    import hashlib
    import os
    from datetime import datetime

    from scripts.benchmark_latency import _cache_engine_hashes, _load_waveform
    from scripts.decoder_host_profile import (
        CpuLoad,
        StepProfiler,
        WaitProfiler,
        WallProfiler,
        cpu_cores,
        summarize,
        write_steps,
    )
    from scripts.fp8_trial import _prefix_probe
    from scripts.profile_latency import select_utterances

    from listenflow.fastest_whisper._cached_model import CachedWhisper
    from listenflow.fastest_whisper._decoding import decoding_settings

    if per_band < 1 or iterations < 1:
        raise ValueError("per-band and iterations must be positive")
    if args.schedule is not None:
        from cuda.bindings import runtime as cu

        configure_schedule(cu, args.schedule)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "started_at": datetime.now().astimezone().isoformat(),
        "pid": os.getpid(),
        "mode": args.mode,
        "condition": args.condition,
        "run_id": args.run_id,
        "input_kind": "product_prefix",
        "precision": "fp16",
        "iterations": iterations,
        "requested_schedule": args.schedule,
        "load_samples": [],
        "tokens": {},
        "native_trace": [],
        "timing": "host wall/CPU wait; CUDA events for stream copies/replay; host bounds for GPU completion",
    }
    try:
        with CachedWhisper(args.checkpoint_dir, precision="fp16", device=0) as model:
            if args.mode == "build":
                model.warmup()
                record["engines"] = _cache_engine_hashes()
                return
            samples = [
                (row, _load_waveform(row))
                for row in select_utterances(args.manifest, per_band=per_band)
            ]
            record["selected_ids"] = [row["id"] for row, _ in samples]
            record["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
            settings = decoding_settings(beam_size=1, temperature=0, best_of=1)
            baseline = {}
            for row, wave in samples:
                audio = model.encode(model._mel_spectrogram(wave))
                baseline[row["id"]] = model.decode_short(audio, [], settings).tokens
            record["baseline_tokens"] = baseline
            record["engines"] = _cache_engine_hashes()
            sections = args.mode in ("sections", "tail")
            profiler = (
                StepProfiler(model, detailed=args.mode == "tail")
                if sections
                else WaitProfiler(model)
                if args.mode == "wait"
                else WallProfiler(model)
            )
            rows = []
            with profiler, CpuLoad(args.condition) as load:
                before = load.snapshot()
                encoded = model.encode(model._mel_spectrogram(samples[0][1]))
                for bucket in (8, 64, 128):
                    if bucket > model.config.text_positions:
                        raise ValueError(f"checkpoint cannot reach controlled bucket {bucket}")
                    for index in range(iterations):
                        with _prefix_probe(model, encoded, bucket) as (call, _prefix):
                            call()
                        with _prefix_probe(model, encoded, bucket) as (call, prefix):
                            token, observed = profiler.capture(call)
                        name = f"bucket-{bucket}-pair-{index:02}"
                        for row in observed:
                            row.update(
                                run_id=args.run_id,
                                condition=args.condition,
                                workload="controlled",
                                sample_id=name,
                                input_kind="product_prefix",
                                history=bucket,
                                precision=model.precision,
                            )
                        record["tokens"][name] = [int(token[0])]
                        record.setdefault("input_tokens", {})[name] = prefix
                        rows.extend(observed)
                for sample, wave in samples:
                    audio = model.encode(model._mel_spectrogram(wave))
                    model.decode_short(audio, [], settings)
                    result, observed = profiler.capture(
                        lambda audio=audio: model.decode_short(audio, [], settings)
                    )
                    if result.tokens != baseline[sample["id"]]:
                        raise RuntimeError(f"observer changed product tokens: {sample['id']}")
                    for row in observed:
                        row.update(
                            run_id=args.run_id,
                            condition=args.condition,
                            workload="utterance",
                            sample_id=sample["id"],
                            input_kind="product_prefix",
                            precision=model.precision,
                        )
                    rows.extend(observed)
                    record["tokens"][sample["id"]] = result.tokens
                    if args.mode == "trace":
                        record["native_trace"].append(
                            {"id": sample["id"], "operations": profiler.session.records}
                        )
                    record["load_samples"].append(load.snapshot())
                after = load.snapshot()
                hz = os.sysconf("SC_CLK_TCK")
                record["load"] = {
                    "before": before,
                    "after": after,
                    "hz": hz,
                    "schedule": load.schedule,
                    "cores": cpu_cores(
                        before["workers"],
                        after["workers"],
                        elapsed_ns=after["at_ns"] - before["at_ns"],
                        hz=hz,
                    ),
                }
            write_steps(args.output_dir / "steps.csv", rows)
            record["step_count"] = len(rows)
            if sections:
                record["summary"] = summarize(rows)
            with (args.output_dir / "cpu.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("at_ns", "elapsed_ns", "ticks"))
                writer.writeheader()
                writer.writerows(
                    {name: item[name] for name in writer.fieldnames}
                    for item in [before, *record["load_samples"], after]
                )
    finally:
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
