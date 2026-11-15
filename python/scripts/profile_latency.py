"""Profile warmed batch-one fp16 greedy latency from a LibriSpeech manifest.

Run from a standalone checkout with ``uv run --group evaluation python -m
scripts.profile_latency --checkpoint-dir PATH --manifest PATH --output-dir PATH``.
The default selects the first 50 test-clean utterances in each duration band,
preserving manifest order. Model loading, warmup, audio reading and report writing
are outside utterance timings. CUDA events bracket native enqueue or replay on its own
stream; host time is wall time minus enqueue GPU time, including copies, buffer
management, Python token selection and profiling overhead. It is not CPU busy time.
This diagnostic does not acquire exclusive GPU access (排他なし).
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

BANDS = ("0-5", "5-10", "10-20", "20-30")
COMPONENTS = ("mel", "encoder", "decoder_gpu", "decoder_host", "other")
SAMPLE_RATE = 16_000
MAX_FRAMES = 30 * SAMPLE_RATE


def diagnostic_session(flags=7):
    """Load the development extension only when a caller requests observation."""
    try:
        from _lffw_diagnostics import Session
    except ImportError as error:
        raise RuntimeError(
            "Build python/ with -DLFFW_RUNTIME_DIAGNOSTICS=ON and add its build directory "
            "to PYTHONPATH before running native diagnostics"
        ) from error
    return Session(flags)


def capture_native(function, *, flags=7):
    """Observe a synchronous product call and always detach the session."""
    session = diagnostic_session(flags)
    try:
        with session:
            result = function()
        return result, session.records
    finally:
        session.close()


class DecoderProfiler:
    """Read native timing spans while the real product chooses its tokens."""

    def __init__(self, model, cudart=None, *, clock=time.perf_counter):
        self.model, self.clock = model, clock
        self.session = None

    def __enter__(self):
        self.session = diagnostic_session()
        return self

    def __exit__(self, *_exc):
        if self.session is not None:
            self.session.close()
            self.session = None

    def measure(self, row, samples):
        from contextlib import ExitStack
        from unittest.mock import patch

        starts, boundaries = [], {}
        original_step = self.model._greedy_step
        original_mel, original_encode = self.model._mel_spectrogram, self.model.encode

        def step(*args, **kwargs):
            starts.append(round(self.clock() * 1e9))
            return original_step(*args, **kwargs)

        def mel(*args, **kwargs):
            result = original_mel(*args, **kwargs)
            boundaries["mel"] = round(self.clock() * 1e9)
            return result

        def encode(*args, **kwargs):
            result = original_encode(*args, **kwargs)
            boundaries["encoder"] = round(self.clock() * 1e9)
            return result

        def decode(*args, **kwargs):
            result = original_decode(*args, **kwargs)
            boundaries["transcription"] = result
            return result

        with ExitStack() as stack:
            stack.enter_context(patch.object(self.model, "_greedy_step", step))
            stack.enter_context(patch.object(self.model, "_mel_spectrogram", mel))
            stack.enter_context(patch.object(self.model, "encode", encode))
            if hasattr(self.model, "transcribe"):
                original_decode = self.model.decode_short
                stack.enter_context(patch.object(self.model, "decode_short", decode))
            with self.session:
                started = round(self.clock() * 1e9)
                if hasattr(self.model, "transcribe"):
                    segments = self.model.transcribe(samples, beam_size=1, temperature=0, best_of=1)
                    transcription = boundaries["transcription"]
                    if " ".join(segment.text for segment in segments) != transcription.text:
                        raise RuntimeError("product segments differ from decoder text")
                    entrypoint = "CachedWhisper.transcribe"
                else:
                    transcription = self.model.greedy_decode(
                        self.model.encode(self.model._mel_spectrogram(samples))
                    )
                    entrypoint = "TensorRTWhisper.greedy_decode"
                finished = round(self.clock() * 1e9)
        records = self.session.records
        operations = [item for item in records if item["name"] == "lffw_greedy_step"]
        if not operations or not starts:
            raise RuntimeError("decoder executed no timed native token steps")
        times = [boundaries["encoder"], *starts[1:], finished]
        steps, buckets = [], []
        for lower, upper in pairwise(times):
            spans = [
                span
                for operation in operations
                if lower <= operation["start_ns"] < upper
                for span in operation["spans"]
                if span["phase"] == "enqueue"
            ]
            if any(span["gpu_ms"] is None for span in spans):
                raise RuntimeError("native decoder timing events are unavailable")
            gpu = sum(span["gpu_ms"] / 1000 for span in spans)
            wall = (upper - lower) / 1e9
            if gpu <= 0 or gpu > wall + 1e-6:
                raise RuntimeError("native GPU timing is inconsistent with host wall time")
            steps.append({"gpu_seconds": min(gpu, wall), "host_seconds": max(0.0, wall - gpu)})
            buckets.extend(
                {"bucket": span["bucket"], "gpu_seconds": span["gpu_ms"] / 1000} for span in spans
            )
        return {
            "id": row["id"],
            "frames": int(row["frames"]),
            "total_seconds": (finished - started) / 1e9,
            "mel_seconds": (boundaries["mel"] - started) / 1e9,
            "encoder_seconds": (boundaries["encoder"] - boundaries["mel"]) / 1e9,
            "decoder_gpu_seconds": sum(item["gpu_seconds"] for item in steps),
            "decoder_host_seconds": sum(item["host_seconds"] for item in steps),
            "other_seconds": 0.0,
            "decoder_steps": steps,
            "decoder_enqueues": len(buckets),
            "entrypoint": entrypoint,
            "decoder_buckets": buckets,
            "generated_tokens": len(transcription.tokens),
            "tokens": transcription.tokens,
            "output": transcription.text,
            "native_operations": records,
        }


def _band(frames: int) -> str:
    if not 0 <= frames <= MAX_FRAMES:
        raise ValueError(f"invalid audio frames: {frames}")
    for index, upper in enumerate((5, 10, 20)):
        if frames < upper * SAMPLE_RATE:
            return BANDS[index]
    return BANDS[-1]


def select_utterances(manifest: Path, *, per_band: int = 50) -> list[dict[str, str]]:
    """Take the first requested number per band, then retain their manifest order."""
    if per_band < 1:
        raise ValueError("per-band must be positive")
    counts = dict.fromkeys(BANDS, 0)
    selected = []
    seen = set()
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"subset", "id", "audio", "reference", "frames"} <= set(reader.fieldnames or []):
            raise ValueError("manifest requires subset, id, audio, reference and frames")
        for row in reader:
            if row["subset"] != "test-clean":
                continue
            frames = int(row["frames"])
            if frames > MAX_FRAMES:
                continue
            band = _band(frames)
            if row["id"] in seen:
                raise ValueError(f"duplicate test-clean ID: {row['id']}")
            seen.add(row["id"])
            if counts[band] < per_band:
                selected.append(row)
                counts[band] += 1
    for band, count in counts.items():
        if count != per_band:
            raise ValueError(f"band {band} has {count} inputs; requires {per_band}")
    return selected


def summarize(rows: list[dict]) -> dict[str, dict]:
    """Report medians and fractions of summed measured wall time per duration band."""
    grouped: dict[str, list[dict]] = {band: [] for band in BANDS}
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"duplicate measurement: {row['id']}")
        seen.add(row["id"])
        if not row["decoder_steps"]:
            raise ValueError("measurement has no timed decoder steps")
        values = [row["total_seconds"], *(row[f"{part}_seconds"] for part in COMPONENTS)]
        values += [value for step in row["decoder_steps"] for value in step.values()]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("measurement contains invalid timing")
        if not math.isclose(sum(values[1:6]), row["total_seconds"], abs_tol=1e-8):
            raise ValueError("components do not sum to measured wall time")
        for part in ("gpu", "host"):
            if not math.isclose(
                sum(step[f"{part}_seconds"] for step in row["decoder_steps"]),
                row[f"decoder_{part}_seconds"],
                abs_tol=1e-8,
            ):
                raise ValueError("decoder steps do not sum to decoder timing")
        grouped[_band(row["frames"])].append(row)

    def median(values):
        return statistics.median(values) if values else None

    def p95(values):
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * 0.95
        lower = math.floor(position)
        return ordered[lower] + (ordered[math.ceil(position)] - ordered[lower]) * (position - lower)

    result = {}
    for band, observations in grouped.items():
        total = sum(row["total_seconds"] for row in observations)
        steps = [step for row in observations for step in row["decoder_steps"]]
        result[band] = {
            "count": len(observations),
            "total_median_seconds": median([row["total_seconds"] for row in observations]),
            "total_p95_seconds": p95([row["total_seconds"] for row in observations]),
            "components": {
                part: {
                    "median_seconds": median([row[f"{part}_seconds"] for row in observations]),
                    "p95_seconds": p95([row[f"{part}_seconds"] for row in observations]),
                    "share_of_total": (
                        sum(row[f"{part}_seconds"] for row in observations) / total
                        if total
                        else None
                    ),
                }
                for part in COMPONENTS
            },
            "decoder_steps_median": median([len(row["decoder_steps"]) for row in observations]),
            "decoder_step_gpu_median_seconds": median([step["gpu_seconds"] for step in steps]),
            "decoder_step_host_median_seconds": median([step["host_seconds"] for step in steps]),
            "decoder_step_gpu_p95_seconds": p95([step["gpu_seconds"] for step in steps]),
            "decoder_step_host_p95_seconds": p95([step["host_seconds"] for step in steps]),
        }
    return result


def markdown_table(bands: dict[str, dict]) -> str:
    """Show component seconds with their share of total and per-step milliseconds."""
    lines = [
        (
            "| Duration (s) | N | Total (s) | Mel (s; %) | Encoder (s; %) | Decoder GPU (s; %) | "
            "Decoder host (s; %) | Other (s; %) | Steps | GPU/step (ms) | Host/step (ms) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for band, summary in bands.items():
        if not summary["count"]:
            continue
        parts = [
            f"{part['median_seconds']:.4f}; {part['share_of_total']:.1%}"
            for part in summary["components"].values()
        ]
        cells = [band, str(summary["count"]), f"{summary['total_median_seconds']:.4f}", *parts]
        cells += [
            f"{summary['decoder_steps_median']:g}",
            f"{summary['decoder_step_gpu_median_seconds'] * 1e3:.3f}",
            f"{summary['decoder_step_host_median_seconds'] * 1e3:.3f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(output_dir: Path, selected: list[dict], rows: list[dict], metadata: dict) -> None:
    """Reject partial or mismatched runs before writing their JSON and Markdown reports."""
    expected = [(row["id"], int(row["frames"])) for row in selected]
    measured = [(row["id"], row["frames"]) for row in rows]
    if measured != expected:
        raise ValueError("measurements do not match selected inputs in manifest order")
    bands = summarize(rows)
    paths = [output_dir / name for name in ("report.json", "report.md")]
    if any(path.exists() for path in paths):
        raise FileExistsError("a profile report already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "metadata": metadata,
        "selected_utterance_ids": [row["id"] for row in selected],
        "bands": bands,
    }
    paths[0].write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    paths[1].write_text(
        "GPU access: 排他なし. Components show median seconds; percentages use summed "
        "component time / summed total time. Step medians pool all steps in each band. "
        "Host time is wall time minus enqueue GPU time, not CPU busy time. The final "
        "step includes text assembly. Profiling overhead is included.\n\n" + markdown_table(bands),
        encoding="utf-8",
    )


def run(checkpoint_dir: Path, manifest: Path, output_dir: Path, *, per_band: int = 50) -> None:
    """Build once, warm once, and record each of the selected utterances once."""
    from scripts.benchmark_latency import (
        _cache_engine_hashes,
        _gpu_sample,
        _load_waveform,
    )
    from scripts.evaluate_wer import _sha256, checkpoint_hashes

    selected = select_utterances(manifest, per_band=per_band)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("report.json", "report.md", "utterances.jsonl", "utterances.jsonl.partial")
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("profile output already exists; use a fresh output directory")

    from listenflow.fastest_whisper._cached_model import CachedWhisper

    metadata = {
        "started_at": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_sha256": checkpoint_hashes(checkpoint_dir),
        "per_band": per_band,
        "selection": "first per-band test-clean inputs <=30 s, preserving manifest order",
        "precision": "fp16",
        "batch_size": 1,
        "decode": {
            "beam_size": 1,
            "temperature": 0,
            "language": "en",
            "task": "transcribe",
            "without_timestamps": True,
            "initial_prompt": None,
            "condition_on_previous_text": False,
        },
        "gpu_exclusive": False,
        "gpu_access": "排他なし",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("tensoracle", "tensorrt-cu13", "numpy", "cuda-python")
        },
        "timing": {
            "total": "prepared waveform to full text after native completion polling",
            "mel": "log-mel wall time, including its transfers",
            "encoder": "encode wall time, including its I/O and any decoder preparation",
            "decoder_gpu": "CUDA events before/after native enqueue/replay on its stream; "
            "includes any device idle gaps during host submission",
            "decoder_host": "GPU の計算以外の残余: decode wall time minus enqueue event spans; "
            "includes allocations, frees, transfers, Python/NumPy selection and text assembly; "
            "not CPU busy time",
            "other": "host work outside the recorded stages; no extra device synchronization",
            "step": "one _greedy_step call and following token handling until next _greedy_step call; "
            "first step includes prefix setup and last includes text assembly",
            "step_count": "number of _greedy_step calls, including the call that selects EOT",
            "decoder_enqueues": "one-token executions, including all initial prefix positions",
            "generated_tokens": "returned tokens including EOT; may include a synthetic EOT "
            "on context exhaustion",
            "shares": "sum of component seconds / sum of total seconds per band",
            "step_medians": "pooled steps across all utterances in each band",
            "overhead": "profiling overhead included; event queries and report I/O excluded",
        },
    }
    rows = []
    metadata["gpu_before_load"] = _gpu_sample(0)
    metadata["gpu"] = metadata["gpu_before_load"]["name"]
    metadata["driver"] = metadata["gpu_before_load"]["driver_version"]
    metadata["gpu_observations"] = [metadata["gpu_before_load"]]
    metadata["engine_sha256_before_load"] = _cache_engine_hashes()
    load_started = time.perf_counter()
    with CachedWhisper(checkpoint_dir, precision="fp16", device=0) as model:
        metadata["load_seconds"] = time.perf_counter() - load_started
        metadata["engine_build_seconds"] = model.build_seconds
        metadata["engine_sha256_loaded"] = _cache_engine_hashes()
        metadata["decode"]["suppressed_tokens"] = sorted(model.suppressed)
        metadata["decode"]["begin_suppressed_tokens"] = sorted(model.begin_suppressed)
        first_samples = _load_waveform(selected[0])
        with DecoderProfiler(model) as profiler:
            warmup = profiler.measure(selected[0], first_samples)
            metadata["warmup_utterance_id"] = selected[0]["id"]
            metadata["warmup_seconds"] = warmup["total_seconds"]
            partial = output_dir / "utterances.jsonl.partial"
            with partial.open("x", encoding="utf-8") as stream:
                for index, row in enumerate(selected, 1):
                    samples = first_samples if index == 1 else _load_waveform(row)
                    observation = profiler.measure(row, samples)
                    rows.append(observation)
                    stream.write(json.dumps(observation, allow_nan=False) + "\n")
                    stream.flush()
                    if index % 25 == 0 or index == len(selected):
                        print(f"profile: {index}/{len(selected)}", flush=True)
            metadata["gpu_observations"].append(_gpu_sample(len(selected)))
            partial.rename(output_dir / "utterances.jsonl")
        metadata["engine_sha256_after_measurement"] = _cache_engine_hashes()
    metadata["finished_at"] = datetime.now(UTC).isoformat()
    write_report(output_dir, selected, rows, metadata)
    print((output_dir / "report.md").read_text(encoding="utf-8"), flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-band", type=int, default=50)
    args = parser.parse_args(argv)
    if args.per_band < 1:
        parser.error("--per-band must be positive")
    run(args.checkpoint_dir, args.manifest, args.output_dir, per_band=args.per_band)


if __name__ == "__main__":
    main()
