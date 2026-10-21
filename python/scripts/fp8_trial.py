"""Measure product fp8, fp16 and fp32 recognition and latency.

Use a fresh disposable XDG_CACHE_HOME for each run. The latency mode requires an
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
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from cuda.bindings import runtime as cudart
from scripts.benchmark_latency import _cache_engine_hashes, _gpu_sample, _load_waveform
from scripts.evaluate_wer import EXPECTED_INCLUDED, checkpoint_hashes, read_csv, word_errors
from scripts.profile_latency import DecoderProfiler, select_utterances, summarize
from scripts.profile_layers import decoder_buckets

from listenflow.fastest_whisper._audio import log_mel_spectrogram
from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._decoding import decoding_settings
from listenflow.fastest_whisper._inference import _cuda, _DeviceTensor, _release_tensors
from listenflow.fastest_whisper._precision import PRECISIONS


class MeasuredWhisper(CachedWhisper):
    """Measure preparation without changing product precision or graph selection."""

    def _open_engine(self, checkpoint, *, decoder):
        start = time.perf_counter()
        result = super()._open_engine(checkpoint, decoder=decoder)
        self.build_seconds["decoder" if decoder else "encoder"] = time.perf_counter() - start
        return result

    def _open_mel(self, checkpoint):
        start = time.perf_counter()
        result = super()._open_mel(checkpoint)
        self.build_seconds["mel"] = time.perf_counter() - start
        return result


def resource_sample(completed):
    sample = _gpu_sample(completed)
    sample["load_average"] = list(os.getloadavg())
    return sample


def completed_run(io, inputs):
    values = io.run(inputs, device_outputs=io.outputs)
    try:
        _cuda(cudart.cudaStreamSynchronize(io.stream), "trial stream synchronize")
    finally:
        _release_tensors(values)


def gpu_times(io, inputs, *, iterations, warmup=5):
    """CUDA events around warmed product replay; transfers and host waiting are excluded."""
    for _ in range(warmup):
        completed_run(io, inputs)
    start = _cuda(cudart.cudaEventCreate(), "trial start event")
    stop = _cuda(cudart.cudaEventCreate(), "trial stop event")
    times = []
    original = io._enqueue

    def enqueue(addresses, *, stream):
        _cuda(cudart.cudaEventRecord(start, stream), "trial event record")
        result = original(addresses, stream=stream)
        _cuda(cudart.cudaEventRecord(stop, stream), "trial event record")
        return result

    io._enqueue = enqueue
    try:
        for _ in range(iterations):
            values = io.run(inputs, device_outputs=io.outputs)
            try:
                _cuda(cudart.cudaEventSynchronize(stop), "trial event synchronize")
                times.append(_cuda(cudart.cudaEventElapsedTime(start, stop), "trial event time"))
            finally:
                _release_tensors(values)
    finally:
        del io._enqueue
        _cuda(cudart.cudaEventDestroy(stop), "trial event destroy")
        _cuda(cudart.cudaEventDestroy(start), "trial event destroy")
    return {
        "samples_ms": times,
        "median_ms": statistics.median(times),
        "p95_ms": float(np.percentile(times, 95)),
    }


def isolated_timings(model, samples, *, iterations):
    """Reuse the same real cross KV and zero self KV for every product cache bucket."""
    mel = model._mel_spectrogram(samples)
    encoded = model.encode(mel)
    # Encoder is enqueued normally by the product; decoder uses its captured CUDA graphs.
    encoder = gpu_times(model._encoder, {"mel": mel}, iterations=iterations)
    decoder = {}
    c = model.config
    dtype = np.dtype(PRECISIONS[model.precision].dtype)
    for bucket in decoder_buckets():
        owned = {}
        try:
            shape = (c.text_layers, 1, c.text_heads, bucket, c.audio_state // c.text_heads)
            for name in ("self_keys", "self_values"):
                size = int(np.prod(shape)) * np.dtype(dtype).itemsize
                pointer = _cuda(cudart.cudaMalloc(size), "trial cache allocation")
                owned[name] = _DeviceTensor(pointer, shape, dtype, 0)
                _cuda(cudart.cudaMemset(pointer, 0, size), "trial cache zero")
            mask = np.zeros((1, 1, 1, bucket + 1), dtype=np.float32)
            mask.flat[0] = -np.inf
            inputs = {
                "tokens": np.array([[model.initial_tokens[0]]], dtype=np.int32),
                "positions": np.array([[bucket - 1]], dtype=np.int32),
                "mask": mask,
                **owned,
                "cross_keys": encoded.tensors["cross_keys"],
                "cross_values": encoded.tensors["cross_values"],
                "suppression": np.zeros((1, 1, c.vocab_size), dtype=np.float32),
            }
            decoder[str(bucket)] = gpu_times(model._decoder, inputs, iterations=iterations)
            print(
                f"decoder bucket {bucket}: {decoder[str(bucket)]['median_ms']:.3f} ms", flush=True
            )
        finally:
            _release_tensors(owned)
    return {"encoder": encoder, "decoder": decoder}


def latency(model, manifest, output, *, per_band, iterations):
    selected = select_utterances(manifest, per_band=per_band)
    # Preload waveforms so disk I/O cannot affect the three precision comparisons.
    waves = [_load_waveform(row) for row in selected]
    for row in select_utterances(manifest, per_band=1):
        model.transcribe(_load_waveform(row), beam_size=1, temperature=0, best_of=1)
    _cuda(cudart.cudaDeviceSynchronize(), "trial warmup complete")
    observations = [resource_sample(0)]
    records = []
    with DecoderProfiler(model, cudart) as profiler:
        for index, (row, samples) in enumerate(zip(selected, waves, strict=True), 1):
            measured = profiler.measure(row, samples)
            records.append(measured)
            if index % 50 == 0:
                observations.append(resource_sample(index))
                print(f"latency {index}/{len(selected)}", flush=True)
    with (output / "utterances.jsonl").open("x") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    result = {"utterances": summarize(records), "ids": [row["id"] for row in selected]}
    result["gpu_observations"] = observations
    result["isolated"] = isolated_timings(model, waves[0], iterations=iterations)
    return result


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
    selected = select_utterances(manifest, per_band=1)
    waves = [_load_waveform(row) for row in selected]
    for wave in waves:
        model.transcribe(wave, beam_size=1, temperature=0, best_of=1)
    _cuda(cudart.cudaDeviceSynchronize(), "trial smoke warmup")
    if capture:
        _cuda(cudart.cudaProfilerStart(), "trial profiler start")
    try:
        results = []
        for row, wave in zip(selected, waves, strict=True):
            text = " ".join(
                segment.text
                for segment in model.transcribe(wave, beam_size=1, temperature=0, best_of=1)
            )
            results.append({"id": row["id"], "output": text, **word_errors(row["reference"], text)})
            print(row["id"], repr(text), flush=True)
        _cuda(cudart.cudaDeviceSynchronize(), "trial smoke complete")
    finally:
        if capture:
            _cuda(cudart.cudaProfilerStop(), "trial profiler stop")
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
