"""Observe product fp16 TensorRT layers with Nsight Systems 2026.6.1.

Mel and encoder use ordinary enqueue. Decoder includes the initial enqueue,
graph construction and replay, including prefix preparation. GPU projections
are diagnostic observations; use profile_latency to measure product latency.
An external GPU reservation is required before running this script.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sqlite3
import statistics
import subprocess
import sys
import textwrap
from collections import defaultdict
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

# Domain names may be absent when a domain was created before collection.
# Scope all operations to an enclosing TensorRT enqueue on the same OS thread.
_RANGES_SQL = """
    WITH annotations AS (
        SELECT n.rowid AS range_id, n.*, coalesce(s.value, n.text, '') AS name
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
    ), domains AS (
        SELECT DISTINCT domainId, (globalTid & 0xFFFFFFFFFF000000) AS globalPid, name
        FROM annotations WHERE eventType = 75
    ), trt_ranges AS (
        SELECT n.* FROM annotations n JOIN domains d
          ON d.domainId = n.domainId
         AND d.globalPid = (n.globalTid & 0xFFFFFFFFFF000000)
        WHERE d.name = 'TensorRT' AND n.eventType IN (59, 60, 70, 71)
    ), scoped_ranges AS (
        SELECT n.*, coalesce(d.name, '') AS domain,
               (SELECT p.name FROM trt_ranges p
                WHERE p.globalTid = n.globalTid AND p.start <= n.start AND p.end >= n.end
                  AND p.name != 'ExecutionContext::enqueueV3' AND p.range_id != n.range_id
                ORDER BY (p.end - p.start), p.range_id LIMIT 1) AS parent_layer
        FROM annotations n LEFT JOIN domains d
          ON d.domainId = n.domainId
         AND d.globalPid = (n.globalTid & 0xFFFFFFFFFF000000)
        WHERE n.eventType IN (59, 60, 70, 71) AND n.end IS NOT NULL
          AND n.name != 'ExecutionContext::enqueueV3'
          AND EXISTS (
              SELECT 1 FROM trt_ranges e
              WHERE e.name = 'ExecutionContext::enqueueV3' AND e.globalTid = n.globalTid
                AND e.start <= n.start AND e.end >= n.end
          )
    )
"""


@contextmanager
def _trace_database(path):
    with closing(
        sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    ) as connection:
        connection.row_factory = sqlite3.Row
        yield connection


def read_enqueue_trace(path, sqlite_path):
    """Read the innermost scoped operation's nsys GPU projection (nanoseconds)."""
    fields = {
        "Name",
        "Projected Start (ns)",
        "Projected Duration (ns)",
        "PID",
        "NumGPUOps",
        "RangeId",
    }
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not fields <= set(reader.fieldnames or []):
            raise ValueError("nsys GPU projection columns are unavailable")
        projected = list(reader)
    with _trace_database(sqlite_path) as connection:
        scoped = {
            row["range_id"]: dict(row)
            for row in connection.execute(_RANGES_SQL + "SELECT * FROM scoped_ranges")
        }
    candidates = []
    for row in projected:
        source = scoped.get(int(row["RangeId"]))
        if source is None or int(row["PID"]) != ((source["globalTid"] >> 24) & 0x00FFFFFF):
            continue
        duration, count = int(row["Projected Duration (ns)"]), int(row["NumGPUOps"])
        if not source["name"] or duration <= 0 or count < 1:
            raise ValueError("invalid TensorRT GPU projection")
        candidates.append(
            (
                source,
                {
                    "pid": int(row["PID"]),
                    "range_id": source["range_id"],
                    "layer": source["name"],
                    "domain": source["domain"],
                    "parent_layer": source["parent_layer"],
                    "gpu_start_ns": int(row["Projected Start (ns)"]),
                    "gpu_span_ns": duration,
                    "gpu_operation_count": count,
                },
            )
        )
    result, leaves = [], []
    for source, observation in sorted(
        candidates, key=lambda item: (item[0]["end"] - item[0]["start"], item[0]["range_id"])
    ):
        if any(
            child["globalTid"] == source["globalTid"]
            and source["start"] <= child["start"]
            and source["end"] >= child["end"]
            for child in leaves
        ):
            continue
        leaves.append(source)
        result.append(observation)
    return sorted(result, key=lambda item: item["gpu_start_ns"])


def read_graph_trace(path):
    """Project innermost captured operations across domains onto replay kernels."""
    query = (
        _RANGES_SQL
        + """
        , candidates AS (
            SELECT k.rowid AS kernel_id, k.globalPid, k.correlationId,
                   k.start, k.end, n.range_id, n.name AS layer, n.domain, n.parent_layer,
                   row_number() OVER (
                       PARTITION BY k.rowid ORDER BY (n.end - n.start), n.range_id
                   ) AS nesting_rank
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN CUDA_GRAPH_NODE_EVENTS g
              ON g.graphNodeId = k.sourceGraphNodeId AND g.graphId = k.sourceGraphId
            JOIN scoped_ranges n
              ON n.globalTid = g.globalTid AND n.start <= g.start AND n.end >= g.end
             AND (n.globalTid & 0xFFFFFFFFFF000000) = k.globalPid
            WHERE k.sourceGraphNodeId IS NOT NULL AND k.sourceGraphNodeId != 0
        )
        SELECT ((globalPid >> 24) & 0x00FFFFFF) AS pid, correlationId AS launch_id,
               layer, domain, parent_layer, count(*) AS gpu_kernel_count,
               min(start) AS gpu_start_ns, max(end) - min(start) AS gpu_span_ns,
               sum(end - start) AS gpu_kernel_sum_ns
        FROM candidates WHERE nesting_rank = 1
        GROUP BY globalPid, correlationId, range_id, layer, domain, parent_layer
        ORDER BY gpu_start_ns
    """
    )
    with _trace_database(path) as connection:
        return [dict(row) for row in connection.execute(query)]


def decoder_buckets(*, source=None):
    """Read the product's single definition instead of duplicating the bucket list."""
    if source is None:
        header = Path(__file__).resolve().parents[2] / "cpp/include/listenflow/whisper_network.hpp"
        definition = re.search(r"cache_buckets\s*\{([^}]+)\}", header.read_text())
        if definition is None:
            raise ValueError("native decoder cache boundaries are unavailable")
        return ast.literal_eval("(" + definition.group(1) + ")")
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


def _cuda(result):
    if result[0] != 0:
        raise RuntimeError(f"CUDA layer observation failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


@contextmanager
def capture_range(driver):
    """Trace graph construction too, draining only immediately before stopping."""
    _cuda(driver.cuProfilerStart())
    try:
        yield
    finally:
        try:
            _cuda(driver.cuCtxSynchronize())
        finally:
            _cuda(driver.cuProfilerStop())


def capture_product(checkpoint_dir, manifest, *, component, iterations, buckets):
    """Run existing product calls without a replacement executor or profiler API."""
    from cuda.bindings import driver
    from scripts.benchmark_latency import _load_waveform
    from scripts.fp8_trial import _prefix_probe

    from listenflow.fastest_whisper._cached_model import CachedWhisper

    with manifest.open(newline="") as stream:
        selected = next(
            (
                row
                for row in csv.DictReader(stream)
                if row["subset"] == "test-clean" and 0 < int(row["frames"]) <= 480_000
            ),
            None,
        )
    if selected is None:
        raise ValueError("manifest requires a test-clean waveform of at most 30 seconds")
    samples = _load_waveform(selected)
    _cuda(driver.cuInit(0))
    device = _cuda(driver.cuDeviceGet(0))
    context = _cuda(driver.cuDevicePrimaryCtxRetain(device))
    try:
        _cuda(driver.cuCtxPushCurrent(context))
        try:
            with CachedWhisper(checkpoint_dir, precision="fp16", device=0) as model:
                # Preparation does not execute a decoder or capture any of its graphs.
                mel = model._mel_spectrogram(samples) if component != "mel" else None
                encoded = model.encode(mel) if component == "decoder" else None
                with capture_range(driver):
                    if component == "mel":
                        for _ in range(iterations):
                            model._mel_spectrogram(samples)
                    elif component == "encoder":
                        for _ in range(iterations):
                            model.encode(mel)
                    else:
                        for bucket in buckets:
                            for _ in range(iterations):
                                with _prefix_probe(model, encoded, bucket) as (call, _prefix):
                                    call()
                return {
                    "utterance_id": selected["id"],
                    "engine_paths": model._native_model.info["engine_paths"],
                }
        finally:
            _cuda(driver.cuCtxPopCurrent())
    finally:
        _cuda(driver.cuDevicePrimaryCtxRelease(device))


def summarize_observations(rows):
    """Retain per-range GPU spans with actual sample counts and fused names."""
    timings = defaultdict(list)
    for row in rows:
        timings[(row["layer"], row["domain"], row["parent_layer"])].append(row["gpu_span_ns"] / 1e6)
    layers = []
    for (name, domain, parent), values in timings.items():
        kind, types = layer_kind(name)
        layers.append(
            {
                "name": name,
                "domain": domain,
                "parent_layer": parent,
                "kind": kind,
                "operator_types": types,
                "count": len(values),
                "mean_ms": statistics.mean(values),
                "median_ms": statistics.median(values),
                "samples_ms": values,
            }
        )
    return {
        "observations": rows,
        "layers": sorted(layers, key=lambda item: item["mean_ms"], reverse=True),
    }


def run(checkpoint_dir, manifest, output_dir, *, iterations=20, buckets=None, nsys="nsys"):
    """Profile each product component in a separate trace for unambiguous attribution."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    buckets = decoder_buckets() if buckets is None else tuple(buckets)
    if (
        not buckets
        or len(set(buckets)) != len(buckets)
        or not set(buckets) <= set(decoder_buckets())
    ):
        raise ValueError("select distinct existing decoder cache buckets")
    output_dir = output_dir.resolve()
    temporary_root = Path(__file__).resolve().parents[2] / ".nf/tmp"
    if not output_dir.is_relative_to(temporary_root):
        raise ValueError("Nsight Systems output must be inside the checkout's .nf/tmp/")
    output_dir.mkdir(parents=True, exist_ok=False)
    version = subprocess.run(
        [str(nsys), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if "2026.6.1" not in version:
        raise ValueError("use Nsight Systems CLI 2026.6.1")
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "nsys_version": version,
        "iterations": iterations,
        "decoder_buckets": buckets,
        "precision": "fp16",
        "timing": "GPU span from first to last enclosed operation, including gaps; not CPU NVTX duration or product latency",
        "decoder_inputs": "product_prefix; includes prefix preparation and first-enqueue warmup",
    }
    for component in ("mel", "encoder", "decoder"):
        prefix = output_dir / component
        command = [
            str(nsys),
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--cuda-graph-trace=node",
            "--output",
            str(prefix),
            sys.executable,
            "-m",
            "scripts.profile_layers",
            "--checkpoint-dir",
            str(checkpoint_dir.resolve()),
            "--manifest",
            str(manifest.resolve()),
            "--output-dir",
            str(output_dir),
            "--iterations",
            str(iterations),
            "--buckets",
            *map(str, buckets),
            "--capture-component",
            component,
        ]
        with (output_dir / f"{component}.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(
                [
                    str(nsys),
                    "export",
                    "--type=sqlite",
                    "--output",
                    str(prefix.with_suffix(".sqlite")),
                    str(prefix.with_suffix(".nsys-rep")),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
            subprocess.run(
                [
                    str(nsys),
                    "stats",
                    "--report",
                    "nvtx_gpu_proj_trace",
                    "--format",
                    "csv",
                    "--timeunit",
                    "ns",
                    "--output",
                    str(prefix),
                    str(prefix.with_suffix(".sqlite")),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        enqueues = read_enqueue_trace(
            output_dir / f"{component}_nvtx_gpu_proj_trace.csv", prefix.with_suffix(".sqlite")
        )
        if not enqueues:
            raise RuntimeError(f"{component}: no TensorRT GPU projections; inspect {component}.log")
        observed = {"enqueue": summarize_observations(enqueues)}
        if component == "decoder":
            replay = read_graph_trace(prefix.with_suffix(".sqlite"))
            if not replay:
                raise RuntimeError(
                    "decoder: no graph layer projections; graph construction must be traced"
                )
            observed["graph_replay"] = summarize_observations(replay)
        observed["product"] = json.loads((output_dir / f"{component}-product.json").read_text())
        report[component] = observed
    report["finished_at"] = datetime.now(UTC).isoformat()
    (output_dir / "layers.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--buckets", type=int, nargs="+")
    parser.add_argument("--nsys", default="nsys", help="Nsight Systems CLI 2026.6.1 executable")
    parser.add_argument(
        "--capture-component", choices=("mel", "encoder", "decoder"), help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.capture_component:
        metadata = capture_product(
            args.checkpoint_dir,
            args.manifest,
            component=args.capture_component,
            iterations=args.iterations,
            buckets=args.buckets,
        )
        (args.output_dir / f"{args.capture_component}-product.json").write_text(
            json.dumps(metadata) + "\n"
        )
        return
    run(
        args.checkpoint_dir,
        args.manifest,
        args.output_dir,
        iterations=args.iterations,
        buckets=args.buckets,
        nsys=args.nsys,
    )


if __name__ == "__main__":
    main()
