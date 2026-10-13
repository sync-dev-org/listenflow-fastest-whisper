"""Profile warmed batch-one fp16 greedy latency from a LibriSpeech manifest.

Run from a standalone checkout with ``uv run --group evaluation python -m
scripts.profile_latency --checkpoint-dir PATH --manifest PATH --output-dir PATH``.
The default selects the first 50 test-clean utterances in each duration band,
preserving manifest order. Model loading, warmup, audio reading and report writing
are outside utterance timings. CUDA events bracket tensoracle enqueue on its own
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
from pathlib import Path

BANDS = ("0-5", "5-10", "10-20", "20-30")
COMPONENTS = ("mel", "encoder", "decoder_gpu", "decoder_host", "other")
SAMPLE_RATE = 16_000
MAX_FRAMES = 30 * SAMPLE_RATE
_MISSING = object()


class DecoderProfiler:
    """Temporarily instrument one sequential model without changing its decoding.

    Enqueue interception is process-wide while this context is open; run one model
    on one thread. Events are pooled outside utterance timings and read after the
    normal inference synchronization, without adding a synchronization per enqueue.
    This also observes executions created later by a cached decoder session.
    """

    def __init__(self, model, cudart, *, clock=time.perf_counter):
        self.model, self.cudart, self.clock = model, cudart, clock
        self.events = []
        self.active = False
        self.used = []
        self.step_walls = []
        self.step_started = None

    def _cuda(self, result, operation):
        if result[0] != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{operation} failed: {result[0]}")
        return result[1] if len(result) > 1 else None

    def _create_pair(self):
        start = self._cuda(self.cudart.cudaEventCreate(), "cudaEventCreate")
        try:
            stop = self._cuda(self.cudart.cudaEventCreate(), "cudaEventCreate")
        except BaseException:
            self._cuda(self.cudart.cudaEventDestroy(start), "cudaEventDestroy")
            raise
        self.events.append((start, stop))

    def __enter__(self):
        from tensoracle import EngineExecution

        from listenflow.fastest_whisper._inference import _HostIO

        self.execution_type = _HostIO if hasattr(_HostIO, "_enqueue") else EngineExecution
        self.enqueue_name = "_enqueue" if self.execution_type is _HostIO else "enqueue"
        self.original_enqueue = getattr(self.execution_type, self.enqueue_name)
        self.original_methods = {
            name: getattr(self.model, name)
            for name in ("logits", "_greedy_step")
            if hasattr(self.model, name)
        }
        self.instance_methods = {
            name: self.model.__dict__.get(name, _MISSING) for name in self.original_methods
        }
        try:
            # Enough for two enqueues per generated token; grows if a later graph needs more.
            for _ in range(self.model.config.text_positions):
                self._create_pair()

            def enqueue(execution, addresses, *, stream):
                if not self.active:
                    return self.original_enqueue(execution, addresses, stream=stream)
                index = len(self.used)
                if index == len(self.events):
                    self._create_pair()
                start, stop = self.events[index]
                self._cuda(self.cudart.cudaEventRecord(start, stream), "cudaEventRecord start")
                result = self.original_enqueue(execution, addresses, stream=stream)
                self._cuda(self.cudart.cudaEventRecord(stop, stream), "cudaEventRecord stop")
                self.used.append((len(self.step_walls), start, stop))
                return result

            def instrument(original):
                def step(*args, **kwargs):
                    if self.active:
                        now = self.clock()
                        if self.used:
                            self.step_walls.append(now - self.step_started)
                            self.step_started = now
                    return original(*args, **kwargs)

                return step

            setattr(self.execution_type, self.enqueue_name, enqueue)
            for name, original in self.original_methods.items():
                setattr(self.model, name, instrument(original))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_exc):
        self.active = False
        setattr(self.execution_type, self.enqueue_name, self.original_enqueue)
        for name, original in self.instance_methods.items():
            if original is _MISSING:
                self.model.__dict__.pop(name, None)
            else:
                setattr(self.model, name, original)
        try:
            self._cuda(self.cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
        finally:
            pairs, self.events = self.events, []
            for pair in pairs:
                for event in pair:
                    self._cuda(self.cudart.cudaEventDestroy(event), "cudaEventDestroy")

    def measure(self, row: dict[str, str], samples) -> dict:
        """Time from prepared waveform to full text and completed GPU work."""
        from listenflow.fastest_whisper._audio import log_mel_spectrogram

        self._cuda(self.cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
        self.used, self.step_walls = [], []
        started = self.clock()
        mel = (
            self.model._mel_spectrogram(samples)
            if getattr(self.model, "_mel", None) is not None
            else log_mel_spectrogram(samples, n_mels=self.model.config.n_mels)
        )
        mel_finished = self.clock()
        encoder = self.model.encode(mel)
        encoder_finished = self.clock()
        self.active = True
        self.step_started = encoder_finished
        try:
            transcription = self.model.greedy_decode(encoder)
        finally:
            self.active = False
        decode_finished = self.clock()
        self.step_walls.append(decode_finished - self.step_started)
        self._cuda(self.cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
        finished = self.clock()
        if not self.used:
            raise RuntimeError("decoder did not use tensoracle enqueue; GPU timing unavailable")
        gpu_times = [0.0] * len(self.step_walls)
        for step, start, stop in self.used:
            gpu_times[step] += (
                self._cuda(self.cudart.cudaEventElapsedTime(start, stop), "cudaEventElapsedTime")
                / 1000
            )
        steps = []
        for wall, gpu in zip(self.step_walls, gpu_times, strict=True):
            if gpu <= 0 or gpu > wall + 1e-6:
                raise RuntimeError("decoder CUDA timing is inconsistent with wall time")
            steps.append({"gpu_seconds": min(wall, gpu), "host_seconds": max(0.0, wall - gpu)})
        return {
            "id": row["id"],
            "frames": int(row["frames"]),
            "total_seconds": finished - started,
            "mel_seconds": mel_finished - started,
            "encoder_seconds": encoder_finished - mel_finished,
            "decoder_gpu_seconds": sum(step["gpu_seconds"] for step in steps),
            "decoder_host_seconds": sum(step["host_seconds"] for step in steps),
            "other_seconds": finished - decode_finished,
            "decoder_steps": steps,
            "decoder_enqueues": len(self.used),
            "generated_tokens": len(transcription.tokens),
            "tokens": transcription.tokens,
            "output": transcription.text,
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

    result = {}
    for band, observations in grouped.items():
        total = sum(row["total_seconds"] for row in observations)
        steps = [step for row in observations for step in row["decoder_steps"]]
        result[band] = {
            "count": len(observations),
            "total_median_seconds": median([row["total_seconds"] for row in observations]),
            "components": {
                part: {
                    "median_seconds": median([row[f"{part}_seconds"] for row in observations]),
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
    from scripts.benchmark_latency import _gpu_field, _gpu_sample, _load_waveform
    from scripts.evaluate_wer import _sha256, checkpoint_hashes

    selected = select_utterances(manifest, per_band=per_band)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("report.json", "report.md", "utterances.jsonl", "utterances.jsonl.partial")
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("profile output already exists; use a fresh output directory")

    from cuda.bindings import runtime as cudart

    from listenflow.fastest_whisper._inference import TensorRTWhisper

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
        "gpu": _gpu_field("name"),
        "driver": _gpu_field("driver_version"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("tensoracle", "tensorrt-cu13", "numpy", "cuda-python")
        },
        "timing": {
            "total": "prepared waveform to full text and cudaDeviceSynchronize completion",
            "mel": "log-mel wall time, including its transfers",
            "encoder": "encode wall time, including its I/O and any decoder preparation",
            "decoder_gpu": "CUDA events before/after each tensoracle enqueue on its stream; "
            "includes any device idle gaps during host submission",
            "decoder_host": "GPU の計算以外の残余: decode wall time minus enqueue event spans; "
            "includes allocations, frees, transfers, Python/NumPy selection and text assembly; "
            "not CPU busy time",
            "other": "final device synchronization and surrounding Python overhead",
            "step": "one logits call and following token selection until next logits call; "
            "first step includes prefix setup and last includes text assembly",
            "step_count": "number of logits calls, including the call that selects EOT",
            "decoder_enqueues": "one-token executions, including all initial prefix positions",
            "generated_tokens": "returned tokens including EOT; may include a synthetic EOT "
            "on context exhaustion",
            "shares": "sum of component seconds / sum of total seconds per band",
            "step_medians": "pooled steps across all utterances in each band",
            "overhead": "profiling overhead included; event queries and report I/O excluded",
        },
    }
    rows = []
    load_started = time.perf_counter()
    with TensorRTWhisper(checkpoint_dir, precision="fp16") as model:
        metadata["load_seconds"] = time.perf_counter() - load_started
        metadata["engine_build_seconds"] = model.build_seconds
        metadata["decode"]["suppressed_tokens"] = sorted(model.suppressed)
        metadata["decode"]["begin_suppressed_tokens"] = sorted(model.begin_suppressed)
        first_samples = _load_waveform(selected[0])
        with DecoderProfiler(model, cudart) as profiler:
            warmup = profiler.measure(selected[0], first_samples)
            metadata["warmup_utterance_id"] = selected[0]["id"]
            metadata["warmup_seconds"] = warmup["total_seconds"]
            metadata["gpu_observations"] = [_gpu_sample(0)]
            partial = output_dir / "utterances.jsonl.partial"
            with partial.open("x", encoding="utf-8") as stream:
                for index, row in enumerate(selected, 1):
                    samples = first_samples if index == 1 else _load_waveform(row)
                    observation = profiler.measure(row, samples)
                    rows.append(observation)
                    stream.write(json.dumps(observation, allow_nan=False) + "\n")
                    stream.flush()
                    if index % 25 == 0 or index == len(selected):
                        metadata["gpu_observations"].append(_gpu_sample(index))
                        print(f"profile: {index}/{len(selected)}", flush=True)
            partial.rename(output_dir / "utterances.jsonl")
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
