"""Measure product fp8, fp16 and fp32 recognition and latency.

Use the existing GPU test wrapper for disposable storage. The latency mode requires an
externally coordinated quiet interval and records its identifier; the script does
not obtain exclusive resources. FP8 uses the calibration data shipped with the package.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

from scripts.benchmark_latency import _cache_engine_hashes, _gpu_sample, _load_waveform
from scripts.evaluate_wer import EXPECTED_INCLUDED, checkpoint_hashes, read_csv, word_errors

from listenflow.fastest_whisper._audio import log_mel_spectrogram
from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._decoding import decoding_settings


class MeasuredWhisper(CachedWhisper):
    """Expose the native preparation values without changing product selection."""


def resource_sample(completed):
    sample = _gpu_sample(completed)
    sample["load_average"] = list(os.getloadavg())
    return sample


def completed_run(operation):
    """Complete a product call and release any temporary native output it owns."""
    result = operation()
    if hasattr(result, "close"):
        result.close()
    return result


def gpu_times(operation, *, iterations, warmup=1):
    """Read only enqueue/replay CUDA times from synchronous native calls."""
    import statistics

    from scripts.profile_latency import capture_native

    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup nonnegative")
    for _ in range(warmup):
        completed_run(operation)
    values = []
    for _ in range(iterations):
        _, records = capture_native(lambda: completed_run(operation))
        spans = [span for row in records for span in row["spans"] if span["phase"] == "enqueue"]
        if not spans or any(span["gpu_ms"] is None for span in spans):
            raise RuntimeError("native enqueue timing is unavailable")
        values.append(sum(span["gpu_ms"] for span in spans))
    ordered = sorted(values)
    offset = (len(ordered) - 1) * 0.95
    low = int(offset)
    return {
        "samples_ms": values,
        "median_ms": statistics.median(values),
        "p95_ms": ordered[low]
        + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (offset - low),
    }


def _prefix_probe(model, encoded, bucket):
    """Own one native chain prepared immediately before the requested history."""
    from contextlib import contextmanager

    from listenflow.fastest_whisper import _native

    @contextmanager
    def prepare():
        if not 1 <= bucket <= model.config.text_positions:
            raise ValueError("probe history exceeds checkpoint context")
        chain = encoded.decoder(_native.GREEDY)
        try:
            token = model.initial_tokens[0]
            prefix = []
            for position in range(bucket - 1):
                prefix.append(token)
                selected, _, _ = chain.greedy_step(
                    token, position < len(model.initial_tokens), False
                )
                token = (
                    model.initial_tokens[position + 1]
                    if position + 1 < len(model.initial_tokens)
                    else selected
                )
            prefix.append(token)
            yield (
                lambda: chain.greedy_step(token, bucket <= len(model.initial_tokens), False),
                prefix,
            )
        finally:
            chain.close()

    return prepare()


def isolated_timings(model, samples, *, iterations):
    """Visit every native bucket with real audio and prefix-computed KV."""
    import statistics

    from scripts.profile_latency import capture_native
    from scripts.profile_layers import decoder_buckets

    if iterations < 1:
        raise ValueError("iterations must be positive")
    buckets = decoder_buckets()
    if any(bucket > model.config.text_positions for bucket in buckets):
        raise ValueError("checkpoint context cannot reach every product cache bucket")
    mel = model._mel_spectrogram(samples)
    encoded = model.encode(mel)
    encoder = gpu_times(lambda: model._native_model.encode(mel), iterations=iterations)
    decoder = {}
    for bucket in buckets:
        times = []
        for _ in range(iterations):
            with _prefix_probe(model, encoded, bucket) as (call, prefix):
                _, records = capture_native(call)
                spans = [
                    span for row in records for span in row["spans"] if span["phase"] == "enqueue"
                ]
                if len(spans) != 1 or spans[0]["bucket"] != bucket or spans[0]["gpu_ms"] is None:
                    raise RuntimeError(f"native probe did not execute cache bucket {bucket}")
                times.append(spans[0]["gpu_ms"])
        ordered = sorted(times)
        offset = (len(ordered) - 1) * 0.95
        low = int(offset)
        decoder[str(bucket)] = {
            "samples_ms": times,
            "median_ms": statistics.median(times),
            "p95_ms": ordered[low]
            + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (offset - low),
            "input_kind": "product_prefix",
            "input_tokens": prefix,
            "input_token": prefix[-1],
            "history": bucket,
            "bucket": bucket,
            "precision": model.precision,
        }
    return {"encoder": encoder, "decoder": decoder}


def latency(model, manifest, output, *, per_band, iterations):
    """Retain manifest selection and real product tokens while reading native timings."""
    from scripts.profile_latency import DecoderProfiler, select_utterances, summarize

    selected = select_utterances(manifest, per_band=per_band)
    waves = [_load_waveform(row) for row in selected]
    for wave in waves:
        model.transcribe(wave, beam_size=1, temperature=0, best_of=1)
    records = []
    with DecoderProfiler(model) as profiler:
        for row, wave in zip(selected, waves, strict=True):
            records.append(profiler.measure(row, wave))
    with (output / "utterances.jsonl").open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    return {
        "utterances": summarize(records),
        "ids": [row["id"] for row in selected],
        "isolated": isolated_timings(model, waves[0], iterations=iterations),
    }


def evaluate(model, manifest, output):
    from whisper.normalizers import EnglishTextNormalizer

    rows = [row for row in read_csv(manifest) if int(row["frames"]) <= 480_000]
    for split, expected in EXPECTED_INCLUDED.items():
        if sum(row["subset"] == split for row in rows) != expected:
            raise ValueError(f"unexpected evaluation count in {split}")
    if len(rows) != sum(EXPECTED_INCLUDED.values()):
        raise ValueError("unexpected evaluation split")
    totals = {
        split: {"utterances": 0, "reference_words": 0, "errors": 0} for split in EXPECTED_INCLUDED
    }
    normalizer = EnglishTextNormalizer()
    settings = decoding_settings(beam_size=1, temperature=0.0, best_of=1)
    partial = output / "wer.csv.partial"
    start = time.perf_counter()
    with partial.open("x", newline="") as stream:
        fields = [
            "subset",
            "id",
            "reference",
            "output",
            "normalized_reference",
            "normalized_output",
            "reference_words",
            "substitutions",
            "deletions",
            "insertions",
            "errors",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            samples = _load_waveform(row)
            # Match the existing all-speech fp16/fp32 WER protocol's CPU log-mel.
            encoded = model.encode(log_mel_spectrogram(samples, n_mels=model.config.n_mels))
            decoded = model.decode_short(encoded, [], settings)
            score = word_errors(row["reference"], decoded.text, normalizer=normalizer)
            writer.writerow(
                {
                    "subset": row["subset"],
                    "id": row["id"],
                    "reference": row["reference"],
                    "output": decoded.text,
                    **score,
                }
            )
            stream.flush()
            total = totals[row["subset"]]
            total["utterances"] += 1
            total["reference_words"] += score["reference_words"]
            total["errors"] += score["errors"]
            if index == 1 or index % 100 == 0:
                print(
                    f"WER {index}/{len(rows)} elapsed={time.perf_counter() - start:.1f}s {totals}",
                    flush=True,
                )
    for split, words in (("test-clean", 52349), ("test-other", 52281)):
        if totals[split]["reference_words"] != words:
            raise ValueError(f"reference word count changed: {split}")
        totals[split]["wer_percent"] = 100 * totals[split]["errors"] / words
    partial.rename(output / "wer.csv")
    return {"subsets": totals}


def smoke(model, manifest, *, capture=False):
    """Return real transcription and an optional native trace without external profilers."""
    from scripts.profile_latency import DecoderProfiler, select_utterances

    results = []
    with DecoderProfiler(model) as profiler:
        for row in select_utterances(manifest, per_band=1):
            wave = _load_waveform(row)
            model.transcribe(wave, beam_size=1, temperature=0, best_of=1)
            observed = profiler.measure(row, wave)
            result = {
                "id": row["id"],
                "output": observed["output"],
                "tokens": observed["tokens"],
                **word_errors(row["reference"], observed["output"]),
            }
            if capture:
                result["native_operations"] = observed["native_operations"]
            results.append(result)
    return {"utterances": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "wer", "latency", "kernels"))
    parser.add_argument("--precision", choices=("fp8", "fp16", "fp32"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-band", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--quiet-interval", help="identifier supplied by the host coordinator")
    args = parser.parse_args()
    if args.rounds < 1 or args.iterations < 1:
        parser.error("rounds and iterations must be positive")
    if args.mode == "latency" and not args.quiet_interval:
        parser.error("latency requires --quiet-interval after external resource coordination")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "report.json"
    if target.exists():
        raise FileExistsError(target)
    metadata = {
        "started_at": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "precision": args.precision,
        "quiet_interval": args.quiet_interval,
        "checkpoint_sha256": checkpoint_hashes(args.checkpoint_dir),
        "python": platform.python_version(),
        "os": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("tensorrt-cu13", "tensoracle", "numpy", "cuda-python")
        },
        "cache": os.environ.get("XDG_CACHE_HOME"),
        "gpu_before": resource_sample(0),
    }
    start = time.perf_counter()
    with MeasuredWhisper(args.checkpoint_dir, precision=args.precision) as model:
        metadata["model_prepare_seconds"] = time.perf_counter() - start
        metadata["engine_prepare_seconds"] = model.build_seconds
        metadata["engine_sha256"] = _cache_engine_hashes()
        if args.mode == "latency":
            rounds = []
            for index in range(1, args.rounds + 1):
                directory = args.output_dir / f"round{index}"
                directory.mkdir(exist_ok=False)
                measured = latency(
                    model,
                    args.manifest,
                    directory,
                    per_band=args.per_band,
                    iterations=args.iterations,
                )
                (directory / "report.json").write_text(json.dumps(measured, indent=2) + "\n")
                rounds.append(measured)
            result = {"rounds": rounds}
        elif args.mode == "wer":
            result = evaluate(model, args.manifest, args.output_dir)
        else:
            result = smoke(model, args.manifest, capture=args.mode == "kernels")
        metadata["engine_sha256_after"] = _cache_engine_hashes()
    metadata.update(
        finished_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.perf_counter() - start,
        gpu_after=resource_sample(0),
    )
    target.write_text(json.dumps({**metadata, **result}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
