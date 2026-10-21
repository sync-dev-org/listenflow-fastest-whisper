"""Measure fused layers in the product's cached fp16 TensorRT engines.

This is a separate intrusive diagnostic: per-layer CUDA events are enabled and
CUDA graph replay is disabled. Never interpret its wall time as product latency.
Use profile_latency for utterances and Nsight Systems for graph/copy timelines.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
import textwrap
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


def decoder_buckets(*, source=None):
    """Read the product's single definition instead of duplicating the bucket list."""
    if source is None:
        from listenflow.fastest_whisper._inference import _CACHE_BOUNDARIES

        return _CACHE_BOUNDARIES
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "boundaries" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("decoder cache boundaries are unavailable")


def layer_kind(name):
    """Infer source operation families from names retained by the product engine."""
    types = sorted(set(re.findall(r"\[([A-Za-z][A-Za-z0-9 _]*)\]", name)))
    joined = " ".join(types).lower()
    if "_mha_" in name:
        kind = "attention"
    elif "__myl_Fc_" in name:
        kind = "matmul"
    elif "MeanSubMulMeanAddSqrt" in name:
        kind = "normalization"
    elif "SubExp" in name:
        kind = "softmax"
    elif "Conc" in name:
        kind = "concat"
    elif "__myl_Tran" in name:
        kind = "shuffle"
    elif "matrix multiply" in joined or "convolution" in joined:
        kind = "matmul" if "matrix multiply" in joined else "convolution"
    elif "normalization" in joined:
        kind = "normalization"
    elif "softmax" in joined or "softmax" in name.lower():
        kind = "softmax"
    elif "reduce" in joined:
        kind = "reduce"
    elif "pwn" in name.lower() or any(
        op in joined for op in ("elementwise", "unary", "activation")
    ):
        kind = "pointwise"
    elif "concatenation" in joined:
        kind = "concat"
    elif "shuffle" in joined:
        kind = "shuffle"
    elif "reformat" in name.lower() or "copynode" in name.lower():
        kind = "reformat"
    else:
        kind = "other"
    return kind, types


def summarize_layers(samples, *, iterations):
    """Retain fused names and rank per-inference mean CUDA time, counting each once."""
    if iterations < 1 or not samples:
        raise ValueError("a layer profile requires measured iterations")
    timings = defaultdict(list)
    for name, milliseconds in samples:
        if not name or not math.isfinite(milliseconds) or milliseconds < 0:
            raise ValueError("invalid layer timing")
        timings[name].append(milliseconds)
    if any(len(values) != iterations for values in timings.values()):
        raise ValueError("incomplete layer callbacks")
    total = sum(sum(values) / iterations for values in timings.values())
    if total <= 0:
        raise ValueError("layer GPU total must be positive")
    layers, kinds = [], defaultdict(float)
    for name, values in timings.items():
        mean = statistics.mean(values)
        kind, types = layer_kind(name)
        kinds[kind] += mean
        layers.append(
            {
                "name": name,
                "kind": kind,
                "operator_types": types,
                "mean_ms": mean,
                "median_ms": statistics.median(values),
                "share": mean / total,
                "samples_ms": values,
            }
        )
    return {
        "iterations": iterations,
        "total_mean_ms": total,
        "kinds_ms": dict(sorted(kinds.items(), key=lambda item: item[1], reverse=True)),
        "layers": sorted(layers, key=lambda layer: layer["mean_ms"], reverse=True),
    }


def profile_io(io, inputs, *, iterations):
    """Profile completed enqueues of the same engine, separately from normal replay."""
    import tensorrt as trt

    from listenflow.fastest_whisper._inference import _release_tensors

    class Profiler(trt.IProfiler):
        def __init__(self):
            trt.IProfiler.__init__(self)
            self.samples = []

        def report_layer_time(self, layer_name, ms):
            self.samples.append((layer_name, ms))

    original_execution = io.execution
    execution = original_execution._handle.create_execution()
    context = execution._context
    profiler = Profiler()
    graphs, io._graphs = io._graphs, None
    io.execution = execution
    try:
        # Warm shape/tactics before attaching intrusive per-layer events.
        for _ in range(3):
            _release_tensors(io.run(inputs, device_outputs=io.outputs))
        context.profiler = profiler
        context.enqueue_emits_profile = False
        for _ in range(iterations):
            outputs = io.run(inputs, device_outputs=io.outputs)
            try:
                if not context.report_to_profiler():
                    raise RuntimeError("TensorRT did not report layer timings")
            finally:
                _release_tensors(outputs)
        result = summarize_layers(profiler.samples, iterations=iterations)
        inspector = io.execution._handle.engine.create_engine_inspector()
        inspector.execution_context = context
        result["inspector"] = json.loads(
            inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        )
        return result
    finally:
        io.execution, io._graphs = original_execution, graphs
        # TensorRT cannot detach a profiler by setting it to None. Release its
        # dedicated context while the Python callback object is still alive.
        context = None
        execution.close()


def run(checkpoint_dir, manifest, output_dir, *, iterations=20, buckets=None):
    import numpy as np
    from cuda.bindings import runtime as cudart
    from scripts.benchmark_latency import _cache_engine_hashes, _gpu_sample, _load_waveform
    from scripts.profile_latency import select_utterances

    from listenflow.fastest_whisper._cached_model import CachedWhisper
    from listenflow.fastest_whisper._inference import _cuda, _DeviceTensor, _release_tensors

    if iterations < 1:
        raise ValueError("iterations must be positive")
    buckets = decoder_buckets() if buckets is None else tuple(buckets)
    if not buckets or not set(buckets) <= set(decoder_buckets()):
        raise ValueError("select existing decoder cache buckets")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "layers.json"
    if target.exists():
        raise FileExistsError(target)
    report = {"started_at": datetime.now(UTC).isoformat(), "gpu_before": _gpu_sample(0)}
    with CachedWhisper(checkpoint_dir, precision="fp16", device=0) as model:
        selected = select_utterances(manifest, per_band=1)
        samples = _load_waveform(selected[0])
        mel = model._mel_spectrogram(samples)
        encoded = model.encode(mel)
        report["engine_sha256"] = _cache_engine_hashes()
        report["encoder"] = profile_io(model._encoder, {"mel": mel}, iterations=iterations)
        report["decoder"] = {}
        c = model.config
        for bucket in buckets:
            owned = {}
            try:
                shape = (c.text_layers, 1, c.text_heads, bucket, c.audio_state // c.text_heads)
                for name in ("self_keys", "self_values"):
                    size = int(np.prod(shape)) * 2
                    pointer = _cuda(cudart.cudaMalloc(size), "cudaMalloc layer cache")
                    owned[name] = _DeviceTensor(pointer, shape, np.float16, 0)
                    _cuda(cudart.cudaMemset(pointer, 0, size), "cudaMemset layer cache")
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
                report["decoder"][str(bucket)] = profile_io(
                    model._decoder, inputs, iterations=iterations
                )
                print(f"layers: decoder bucket {bucket}", flush=True)
            finally:
                _release_tensors(owned)
    report["gpu_after"] = _gpu_sample(0)
    report["finished_at"] = datetime.now(UTC).isoformat()
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--buckets", type=int, nargs="+")
    args = parser.parse_args(argv)
    run(
        args.checkpoint_dir,
        args.manifest,
        args.output_dir,
        iterations=args.iterations,
        buckets=args.buckets,
    )


if __name__ == "__main__":
    main()
