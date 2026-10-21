"""Recompute issue 40 tables from raw CSV and read-only Nsight SQLite exports."""

import argparse
import csv
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.decoder_host_profile import SECTIONS, distribution, partition_error, summarize


def compare_tokens(runs):
    reference = runs[0]["tokens"]
    for run in runs[1:]:
        if reference.keys() != run["tokens"].keys():
            raise ValueError(f"different samples in {run['run_id']}")
        if reference != run["tokens"]:
            raise ValueError(f"different tokens in {run['run_id']}")
    return {"equal": True, "samples": len(reference), "runs": len(runs)}


def _union(intervals):
    total, right = 0, -1
    for start, end in sorted(intervals):
        total += max(0, end - max(start, right))
        right = max(right, end)
    return total


def trace_graphs(path):
    """Join NVTX to CUDA launches, then to kernels by launch correlation ID."""
    result = []
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        ranges = connection.execute(
            "SELECT start,end,text,globalTid FROM NVTX_EVENTS WHERE text LIKE 'issue40/%'"
        ).fetchall()
        connection.execute(
            "CREATE TEMP VIEW launches AS SELECT r.start,r.end,r.globalTid,r.correlationId "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
            "WHERE s.value LIKE 'cudaGraphLaunch%'"
        )
        # Nsight's export normally indexes these joins. A temporary in-memory
        # grouping avoids rescanning millions of kernel rows for every graph.
        kernels = defaultdict(list)
        for start, end, correlation in connection.execute(
            "SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            kernels[correlation].append((start, end))
        for begin, finish, name, thread in ranges:
            _, workload, sample_id = name.split("/", 2)
            launches = connection.execute(
                "SELECT start,end,correlationId FROM launches "
                "WHERE start>=? AND end<=? AND globalTid=? ORDER BY start",
                (begin, finish, thread),
            ).fetchall()
            if not launches:
                raise ValueError(f"no graph launches in measured range {name}")
            spans, busy, api = [], [], []
            count = 0
            for start, end, correlation in launches:
                intervals = kernels[correlation]
                if not intervals:
                    raise ValueError(f"no kernels for graph launch {correlation}")
                spans.append(max(stop for _, stop in intervals) - min(at for at, _ in intervals))
                busy.append(_union(intervals))
                api.append(end - start)
                count += len(intervals)
            result.append(
                {
                    "sample_id": sample_id,
                    "workload": workload,
                    "graph_count": len(launches),
                    "kernel_count": count,
                    "graph_span_median_ns": statistics.median(spans),
                    "graph_busy_median_ns": statistics.median(busy),
                    "launch_median_ns": statistics.median(api),
                }
            )
    return result


def trace_tails(path):
    """Partition graph end to stream-sync return using actual CUPTI copy activity.

    This is a separate traced execution, never a correction to untraced timing.
    Gaps between copies include host enqueue delay and are distinct from DMA.
    """
    result = []
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        kernels, copies = defaultdict(list), defaultdict(list)
        for start, end, correlation in connection.execute(
            "SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            kernels[correlation].append((start, end))
        for start, end, correlation, size in connection.execute(
            "SELECT start,end,correlationId,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=2"
        ):
            copies[correlation].append((start, end, size))
        ranges = connection.execute(
            "SELECT start,end,text,globalTid FROM NVTX_EVENTS WHERE text LIKE 'issue40/%'"
        ).fetchall()
        for begin, finish, name, thread in ranges:
            _, workload, sample_id = name.split("/", 2)
            runtime = connection.execute(
                "SELECT r.start,r.end,r.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN StringIds s ON r.nameId=s.id WHERE r.start>=? AND r.end<=? "
                "AND r.globalTid=? ORDER BY r.start",
                (begin, finish, thread),
            ).fetchall()
            launches = [i for i, row in enumerate(runtime) if row[3].startswith("cudaGraphLaunch")]
            if not launches:
                raise ValueError(f"no graph launches in {name}")
            for step, index in enumerate(launches):
                boundary = launches[step + 1] if step + 1 < len(launches) else len(runtime)
                launch = runtime[index]
                if not kernels[launch[2]]:
                    raise ValueError("graph has no correlated kernels")
                graph_end = max(end for _, end in kernels[launch[2]])
                calls = runtime[index + 1 : boundary]
                syncs = [row for row in calls if row[3].startswith("cudaStreamSynchronize")]
                if len(syncs) != 1:
                    raise ValueError("expected one stream synchronization per graph")
                transfers = []
                for call in calls:
                    if call[3].startswith("cudaMemcpyAsync"):
                        # H2D for the following step may precede its launch;
                        # only D2H activities belong to this graph's output.
                        transfers.extend(copies[call[2]])
                if not transfers:
                    raise ValueError("graph has no correlated output copy")
                start = min(item[0] for item in transfers)
                end = max(item[1] for item in transfers)
                returned = syncs[0][1]
                if not graph_end <= start <= end <= returned:
                    raise ValueError("copy/graph/synchronization timestamps are not ordered")
                active = _union([(at, stop) for at, stop, _ in transfers])
                result.append(
                    {
                        "sample_id": sample_id,
                        "workload": workload,
                        "step": step,
                        "graph_end_ns": graph_end,
                        "copy_start_ns": start,
                        "copy_end_ns": end,
                        "sync_return_ns": returned,
                        "tail_ns": returned - graph_end,
                        "before_copy_ns": start - graph_end,
                        "copy_active_ns": active,
                        "between_copies_ns": end - start - active,
                        "host_return_ns": returned - end,
                        "copy_bytes": sum(size for _, _, size in transfers),
                    }
                )
    return result


def _rows(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name, value in row.items():
            if name.endswith("_ns") or name in ("step", "bucket", "input_token", "output_token"):
                row[name] = int(value)
            elif name == "partition_error":
                row[name] = float(value)
    return rows


def clock_sensitivity(runs, rows, *, max_width_ns):
    anchors = {}
    widths = defaultdict(list)
    for run in runs:
        for anchor in run["anchors"]:
            width = anchor["upper_ns"] - anchor["lower_ns"]
            anchors[(run["run_id"], anchor["sample_id"])] = width
            widths[run["condition"]].append(width)
    origins, bounded, kept = defaultdict(list), defaultdict(list), []
    for original in rows:
        row = dict(original)
        sample = row["sample_id"]
        if row["workload"] == "controlled":
            sample = sample.split("-pair-")[0]
        width = anchors[(row["run_id"], sample)]
        row["after_graph_lower_ns"] = row["d2h_sync_ns"] - row["gpu_pending_upper_ns"]
        row["after_graph_upper_ns"] = row["d2h_sync_ns"] - row["gpu_pending_lower_ns"]
        origins[(row["run_id"], row["condition"], row["workload"], sample, width)].append(row)
        bounded[(row["condition"], row["workload"], row["bucket"])].append(row)
        if width <= max_width_ns:
            kept.append(row)
    names = (
        "gpu_pending",
        "gpu_pending_lower",
        "gpu_pending_upper",
        "after_graph",
        "after_graph_lower",
        "after_graph_upper",
    )

    def values(group):
        return {name: distribution([row[f"{name}_ns"] for row in group]) for name in names}

    return {
        "threshold_ns": max_width_ns,
        "excluded_steps": len(rows) - len(kept),
        "widths": {
            condition: {**distribution(samples), "max_ns": max(samples), "count": len(samples)}
            for condition, samples in widths.items()
        },
        "filtered_groups": summarize(kept),
        "per_origin": [
            {
                "run_id": key[0],
                "condition": key[1],
                "workload": key[2],
                "sample_id": key[3],
                "origin_width_ns": key[4],
                "count": len(group),
                "excluded_from_sensitivity": key[4] > max_width_ns,
                **values(group),
            }
            for key, group in origins.items()
        ],
        "bounds": [
            {"condition": key[0], "workload": key[1], "bucket": key[2], **values(group)}
            for key, group in sorted(bounded.items())
        ],
    }


def analyze(root):
    runs, failures = [], []
    for resource_path in sorted(root.glob("*-resource.json")):
        resource = json.loads(resource_path.read_text())
        run_id = resource_path.name.removesuffix("-resource.json")
        if resource["exit_code"]:
            failures.append({"run_id": run_id, "exit_code": resource["exit_code"]})
            continue
        run = json.loads((root / run_id / "run.json").read_text())
        if run["mode"] != "build":
            run["resource"] = resource
            runs.append(run)
    sections = [run for run in runs if run["mode"] == "sections"]
    if [run["condition"] for run in sections] != ["L0", "L1", "L2"] * 2:
        raise ValueError("incomplete or reordered six section runs")
    for mode in ("wall", "trace"):
        if [run["condition"] for run in runs if run["mode"] == mode] != ["L0", "L1", "L2"]:
            raise ValueError(f"incomplete {mode} controls")
    tokens = compare_tokens(runs)
    if tokens["samples"] != 112:
        raise ValueError("expected 72 controlled input labels and 40 utterances")
    if any(run["engines"] != runs[0]["engines"] for run in runs):
        raise ValueError("saved engines changed across processes")
    rows = [row for run in sections for row in _rows(root / run["run_id"] / "steps.csv")]
    groups = summarize(rows)
    rankings = []
    candidates = (*SECTIONS[:5], "after_graph", "python_after")
    for group in groups:
        if group["condition"] != "L1":
            continue
        peers = {
            item["condition"]: item
            for item in groups
            if (item["workload"], item["bucket"]) == (group["workload"], group["bucket"])
        }
        for metric in ("median_ns", "p95_ns"):
            orders = {
                condition: sorted(
                    candidates,
                    key=lambda name, item=item: item["sections"][name][metric],
                    reverse=True,
                )
                for condition, item in peers.items()
            }
            rankings.append(
                {
                    "workload": group["workload"],
                    "bucket": group["bucket"],
                    "metric": metric,
                    "orders": orders,
                    "changes_L0": orders.get("L0") != orders["L1"],
                    "changes_L2": orders.get("L2") != orders["L1"],
                }
            )
    controls = []
    wall_rows = [
        row
        for run in runs
        if run["mode"] == "wall"
        for row in _rows(root / run["run_id"] / "steps.csv")
    ]
    for group in groups:
        key = (group["condition"], group["workload"], group["bucket"])
        matched = [
            row["total_ns"]
            for row in wall_rows
            if (row["condition"], row["workload"], row["bucket"]) == key
        ]
        wall = distribution(matched)
        controls.append(
            {
                "condition": key[0],
                "workload": key[1],
                "bucket": key[2],
                "sections": group["total"],
                "wall": wall,
                "median_difference_ns": group["total"]["median_ns"] - wall["median_ns"],
            }
        )
    traced = []
    for run in runs:
        if run["mode"] == "trace":
            graphs = trace_graphs(root / f"{run['run_id']}-trace.sqlite")
            if len(graphs) != 112:
                raise ValueError("trace lacks measured ranges")
            traced.append(
                {"condition": run["condition"], "run_id": run["run_id"], "ranges": graphs}
            )
    return {
        "tokens": tokens,
        "groups": groups,
        "rankings": rankings,
        "partition_max_fraction": max(map(partition_error, rows)),
        "event_record_overhead": distribution([row["event_record_ns"] for row in rows]),
        "controls": controls,
        "traces": traced,
        "clock_sensitivity": clock_sensitivity(sections, rows, max_width_ns=100_000),
        "failures": failures,
        "runs": [
            {
                name: run[name]
                for name in (
                    "run_id",
                    "mode",
                    "condition",
                    "load",
                    "anchors",
                    "clock_checks",
                    "resource",
                    "started_at",
                    "finished_at",
                    "step_count",
                )
            }
            for run in runs
        ],
    }


def report(result):
    candidates = (*SECTIONS[:5], "after_graph", "python_after")
    controlled = [
        group
        for group in result["groups"]
        if group["condition"] == "L1" and group["workload"] == "controlled"
    ]

    def extent(name):
        values = [group["sections"][name]["median_ns"] / 1e6 for group in controlled]
        return f"{min(values):.3f}–{max(values):.3f}"

    lines = [
        "# Issue 40 stage 0: decoder host timing",
        "",
        (
            f"Under L1, the largest median host candidate is Python before enqueue "
            f"({extent('python_before')} ms in controlled buckets 8/64/128), followed by "
            f"the part of section 6 after graph completion ({extent('after_graph')} ms) "
            f"and cache padding ({extent('pad_cache')} ms). This selects Python input/buffer/shape "
            "handling as the first subject to investigate. These intervals include observer work "
            "and overlapping device work; they do not establish a recoverable speedup. "
            "Some after-graph/padding orderings lie within the calibration bounds."
        ),
        "",
        (
            "All values below are milliseconds per step; cells show median / p95. "
            "The six untraced section runs are pooled by condition and bucket. "
            "Host intervals include descheduling and observer work; they are not CPU busy time. "
            "See PROTOCOL.md for interval ownership, fixed inputs, clock calibration and resources."
        ),
        "",
        (
            f"Token equality: {result['tokens']}. Maximum partition error: "
            f"{result['partition_max_fraction']:.9%}. No primary samples were discarded. "
            "Quantiles of individual sections are not additive; accounting is checked per raw step."
        ),
        "",
        "## Seven exclusive sections",
        "",
        (
            "| Load | Workload | Bucket | n | Whole | Python before | Pad | H2D | Graph launch | "
            "Compact | D2H + sync | Python after |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def cell(value):
        return f"{value['median_ns'] / 1e6:.4f} / {value['p95_ns'] / 1e6:.4f}"

    for group in result["groups"]:
        values = [
            group["condition"],
            group["workload"],
            str(group["bucket"]),
            str(group["count"]),
            cell(group["total"]),
            *(cell(group["sections"][name]) for name in SECTIONS),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines += [
        "",
        "## Section 6 and graph events",
        "",
        (
            "After graph includes device cache compaction/D2H and host resumption. "
            "It is an opportunity estimate, not pure CPU work. GPU/host calibration bounds "
            "and checks are retained below and in analysis.json."
        ),
        "",
        "| Load | Workload | Bucket | Graph event | Pending in section 6 | After graph in section 6 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group in result["groups"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    group["condition"],
                    group["workload"],
                    str(group["bucket"]),
                    *(
                        cell(group["sections"][name])
                        for name in ("graph_gpu", "gpu_pending", "after_graph")
                    ),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Candidates ordered under L1",
        "",
        (
            "Section 6 uses its after-graph part for this ranking. Raw section 6 also includes "
            "ongoing graph computation and is shown in full above. These are measured times, "
            "not recoverable speedup guarantees."
        ),
        "",
        "| Workload | Bucket | Metric | L1 order | Different in L0 | Different in L2 |",
        "|---|---:|---|---|---|---|",
    ]
    for item in result["rankings"]:
        lines.append(
            f"| {item['workload']} | {item['bucket']} | {item['metric']} | "
            + " > ".join(item["orders"]["L1"])
            + " | "
            + (" > ".join(item["orders"]["L0"]) if item["changes_L0"] else "same")
            + " | "
            + (" > ".join(item["orders"]["L2"]) if item["changes_L2"] else "same")
            + " |"
        )
    lines += [
        "",
        "## Event-free whole-step controls",
        "",
        (
            "Controls include one fresh process per condition. Differences include normal run variation "
            "and all observer work, not just event records. They are diagnostic, not a correction applied "
            "to samples."
        ),
        "",
        "| Load | Workload | Bucket | Sections whole | Event-free whole | Median difference |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in result["controls"]:
        lines.append(
            f"| {item['condition']} | {item['workload']} | {item['bucket']} | "
            f"{cell(item['sections'])} | {cell(item['wall'])} | "
            f"{item['median_difference_ns'] / 1e6:.4f} |"
        )
    lines += [
        "",
        "## Resources, load and clock validation",
        "",
        (
            "L0 adds no synthetic workers; the host retains approximately 2–2.5 busy cores of "
            "desk/hook/tmux background activity. Host cores below cover the entire invocation, "
            "including uninstrumented warmup; worker cores cover the active load interval."
        ),
        "",
        "| Run | Start | End | Worker cores | Host cores | Start load1 | Clock checks outside bounds | Max origin width ms |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        bad = sum(not check["within_uncertainty"] for check in run["clock_checks"])
        width = max((item["upper_ns"] - item["lower_ns"] for item in run["anchors"]), default=0)
        lines.append(
            f"| {run['run_id']} | {run['started_at']} | {run['finished_at']} | "
            f"{run['load']['cores']:.4f} | {run['resource']['host_cpu_cores']:.4f} | "
            f"{run['resource']['before']['load_average'][0]:.2f} | {bad} | {width / 1e6:.4f} |"
        )
    sensitivity = result["clock_sensitivity"]
    lines += [
        "",
        "## Origin uncertainty and sensitivity",
        "",
        (
            "The supplementary analysis excludes a whole controlled group or utterance when its "
            "origin host bracket exceeds 100,000 ns (0.1 ms). This rule was fixed in message 20137 "
            "before sensitivity aggregation. Primary tables retain all samples. The origin midpoint "
            "half-width is at most 0.05 ms in the supplementary subset. Seven-section accounting "
            "uses only host timestamps and is independent of GPU/host clock mapping."
        ),
        "",
        (
            f"Supplementary excluded steps: {sensitivity['excluded_steps']}. "
            "Every group/utterance's width and split median/p95 bounds are in "
            "[origin-groups.csv](origin-groups.csv). Bounds include origin uncertainty, CUDA event "
            "float representation error and the 1 microsecond event-resolution allowance. "
            "That allowance is an analysis choice, not a formal CUDA accuracy guarantee. "
            "NVIDIA describes approximately 0.5 microsecond elapsed-time resolution and asynchronous "
            "event recording in its [CUDA event reference]"
            "(https://docs.nvidia.com/cuda/archive/13.0.3/cuda-runtime-api/group__CUDART__EVENT.html)."
        ),
        "",
        "| Load | Origin groups | Width median ms | Width p95 ms | Width max ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, value in sorted(sensitivity["widths"].items()):
        lines.append(
            f"| {condition} | {value['count']} | {value['median_ns'] / 1e6:.4f} | "
            f"{value['p95_ns'] / 1e6:.4f} | {value['max_ns'] / 1e6:.4f} |"
        )
    lines += [
        "",
        "Pooled split bounds (median lower–upper; p95 lower–upper), in ms:",
        "",
        "| Load | Workload | Bucket | Pending median bounds | Pending p95 bounds | After graph median bounds | After graph p95 bounds |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in sensitivity["bounds"]:
        cells = []
        for name in ("gpu_pending", "after_graph"):
            for metric in ("median_ns", "p95_ns"):
                cells.append(
                    f"{group[name + '_lower'][metric] / 1e6:.4f}–"
                    f"{group[name + '_upper'][metric] / 1e6:.4f}"
                )
        lines.append(
            "| "
            + " | ".join([group["condition"], group["workload"], str(group["bucket"]), *cells])
            + " |"
        )
    lines += [
        "",
        "Inclusive versus supplementary after-graph split:",
        "",
        "| Load | Workload | Bucket | All n | Retained n | All after graph median / p95 | Retained after graph median / p95 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    retained = {
        (group["condition"], group["workload"], group["bucket"]): group
        for group in sensitivity["filtered_groups"]
    }
    for group in result["groups"]:
        keep = retained.get((group["condition"], group["workload"], group["bucket"]))
        lines.append(
            f"| {group['condition']} | {group['workload']} | {group['bucket']} | "
            f"{group['count']} | {keep['count'] if keep else 0} | "
            f"{cell(group['sections']['after_graph'])} | "
            + (cell(keep["sections"]["after_graph"]) if keep else "no retained samples")
            + " |"
        )
    lines += [
        "",
        "L1 ranking sensitivity (same seven candidates as the primary ranking):",
        "",
        "| Workload | Bucket | Metric | Retained order | Same as primary |",
        "|---|---:|---|---|---|",
    ]
    for ranking in result["rankings"]:
        keep = retained.get(("L1", ranking["workload"], ranking["bucket"]))
        if keep is None:
            continue
        order = sorted(
            candidates, key=lambda name: keep["sections"][name][ranking["metric"]], reverse=True
        )
        lines.append(
            f"| {ranking['workload']} | {ranking['bucket']} | {ranking['metric']} | "
            + " > ".join(order)
            + f" | {order == ranking['orders']['L1']} |"
        )
    lines += [
        "",
        "## Separate CUDA traces",
        "",
        (
            "Controlled graph spans/busy times and launch API times below come only from the "
            "separate Nsight executions. They are not subtracted from ordinary host times. "
            "Cells show median / p95 across 24 controlled graphs."
        ),
        "",
        "| Load | Bucket | Graph span | Busy union | Launch API |",
        "|---|---:|---:|---:|---:|",
    ]
    for trace in result["traces"]:
        for bucket in (8, 64, 128):
            samples = [
                item
                for item in trace["ranges"]
                if item["sample_id"].startswith(f"bucket-{bucket}-pair-")
            ]
            cells = [
                cell(distribution([item[field] for item in samples]))
                for field in ("graph_span_median_ns", "graph_busy_median_ns", "launch_median_ns")
            ]
            lines.append("| " + " | ".join([trace["condition"], str(bucket), *cells]) + " |")
    lines += [
        "",
        (
            "Exact commands, root disk, starting load averages and other compute process checks "
            "are saved in each `*-resource.json`; worker CPU snapshots are in each run's cpu.csv. "
            "Nsight trace ranges are joined to kernels by launch correlation, including asynchronous "
            "tails. Detailed graph spans/busy intervals are in analysis.json. "
            "Tests, failures, cleanup and artifact hashes are listed in completion.md and inventory.json."
        ),
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = analyze(args.record_dir)
    output = args.output_dir or args.record_dir
    output.mkdir(exist_ok=True, parents=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    flat = []
    for group in result["clock_sensitivity"]["per_origin"]:
        row = {}
        for name, value in group.items():
            if isinstance(value, dict):
                row.update(
                    {f"{name}_{metric}": measurement for metric, measurement in value.items()}
                )
            else:
                row[name] = value
        flat.append(row)
    with (output / "origin-groups.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (output / "report.md").write_text(report(result))


if __name__ == "__main__":
    main()
