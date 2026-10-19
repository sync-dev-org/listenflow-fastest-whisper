"""Capture warmed product transcription for Nsight Systems CUDA API tracing.

Launch under nsys with --capture-range=cudaProfilerApi --cuda-graph-trace=node.
The trace brackets real CachedWhisper.transcribe calls, without per-layer events
or DecoderProfiler hooks. CUDA API callbacks, transfers and node tracing themselves
add overhead; this separate trace is diagnostic rather than a latency benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path


def capture_transcriptions(model, samples, cudart):
    """Return product text and timings while guaranteeing capture closure."""

    def check(status, name):
        if status[0] != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{name} failed")

    check(cudart.cudaProfilerStart(), "cudaProfilerStart")
    results = []
    try:
        for utterance, waveform in samples:
            begin = time.perf_counter()
            text = " ".join(
                segment.text
                for segment in model.transcribe(waveform, beam_size=1, temperature=0, best_of=1)
            )
            check(cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
            elapsed = time.perf_counter() - begin
            results.append(
                {
                    "id": utterance["id"],
                    "frames": int(utterance["frames"]),
                    "wall_seconds": elapsed,
                    "output": text,
                }
            )
    finally:
        check(cudart.cudaProfilerStop(), "cudaProfilerStop")
    return results


def run(checkpoint_dir, manifest, output_dir):
    from cuda.bindings import runtime as cudart
    from scripts.benchmark_latency import _cache_engine_hashes, _gpu_sample, _load_waveform
    from scripts.profile_latency import select_utterances

    from listenflow.fastest_whisper._cached_model import CachedWhisper

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "timeline.json"
    if target.exists():
        raise FileExistsError(target)
    samples = [(row, _load_waveform(row)) for row in select_utterances(manifest, per_band=1)]
    report = {"started_at": datetime.now(UTC).isoformat(), "gpu_before": _gpu_sample(0)}
    with CachedWhisper(checkpoint_dir, precision="fp16", device=0) as model:
        # Visit all measured utterances before capture so their graph shapes are warm.
        baseline = []
        for row, audio in samples:
            started = time.perf_counter()
            segments = model.transcribe(audio, beam_size=1, temperature=0, best_of=1)
            baseline.append(
                {
                    "id": row["id"],
                    "wall_seconds": time.perf_counter() - started,
                    "output": " ".join(segment.text for segment in segments),
                }
            )
        report["warmup"] = baseline
        report["engine_sha256"] = _cache_engine_hashes()
        report["utterances"] = capture_transcriptions(model, samples, cudart)
    report["gpu_after"] = _gpu_sample(0)
    report["finished_at"] = datetime.now(UTC).isoformat()
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.checkpoint_dir, args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
