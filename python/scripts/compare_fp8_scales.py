"""Compare completed FLEURS FP8 calibrations against the saved fp32 results.

The previous-selection JSON maps each language code to its selected result
directory. No recognition runs are launched. Pairwise temperature-zero cohorts,
their complements, and a common three-run cohort are all retained in the JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from scripts.evaluate_fleurs import CONFIGS, _validated_rows, error_unit


def load_run(directory: Path, language: str, precision: str) -> tuple[list[dict], dict]:
    """Read a complete, unmodified measurement with its original provenance."""
    path = Path(directory) / f"{language}-{precision}.json"
    metadata = json.loads(path.read_text())
    csv_path = path.with_suffix(".csv")
    if metadata["status"] != "complete":
        raise ValueError(f"incomplete run: {path}")
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != metadata["csv_sha256"]:
        raise ValueError(f"changed result CSV: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if (
        not rows
        or len(rows) != metadata["expected_utterances"]
        or len(rows) != metadata["completed_utterances"]
    ):
        raise ValueError(f"inconsistent recording counts: {path}")
    provenance = metadata["provenance"]
    if provenance["language"] != language or provenance["precision"] != precision:
        raise ValueError(f"incorrect language or precision: {path}")
    return rows, provenance


def _zero(row):
    values = row["temperatures"]
    values = json.loads(values) if isinstance(values, str) else values
    if (
        not values
        or values[0] != 0
        or any(not math.isfinite(value) or value < 0 for value in values)
    ):
        raise ValueError(f"invalid temperature trials: {row['id']}")
    return all(value == 0 for value in values)


def _rates(baseline, candidate, ids):
    ids = sorted(ids)
    units = sum(baseline[key]["reference_units"] for key in ids)
    left = sum(baseline[key]["errors"] for key in ids)
    right = sum(candidate[key]["errors"] for key in ids)
    return {
        "utterances": len(ids),
        "ids": ids,
        "reference_units": units,
        "fp32_errors": left,
        "fp8_errors": right,
        "fp32_rate_percent": 100 * left / units if units else None,
        "fp8_rate_percent": 100 * right / units if units else None,
        "difference_points": 100 * (right - left) / units if units else None,
    }


def compare_language(language, fp32, old_fp8, new_fp8):
    """Re-score matched recordings, excluding every retry from deterministic comparisons."""
    baseline, old, new = (_validated_rows(rows, language) for rows in (fp32, old_fp8, new_fp8))
    if not baseline or set(baseline) != set(old) or set(baseline) != set(new):
        raise ValueError("runs contain different recording ids")
    all_ids = set(baseline)
    zero = []
    for rows in (baseline, old, new):
        if any(
            row["reference"] != baseline[key]["reference"] or row["seed"] != baseline[key]["seed"]
            for key, row in rows.items()
        ):
            raise ValueError("runs contain different references or seeds")
        zero.append({key for key, row in rows.items() if _zero(row)})
    result = {"error_unit": error_unit(language), "utterances": len(baseline)}
    for label, rows, eligible in (("old", old, zero[1]), ("new", new, zero[2])):
        paired = zero[0] & eligible
        result[label] = {
            "temperature_zero": _rates(baseline, rows, paired),
            "other": _rates(baseline, rows, all_ids - paired),
        }
    common = set.intersection(*zero)
    result["common_temperature_zero"] = {
        label: _rates(baseline, rows, common) for label, rows in (("old", old), ("new", new))
    }
    return result


def validate_provenance(baseline, old, new, scale_sha256):
    """Require matching audio, checkpoint, decoding and scoring conditions."""
    for key in (
        "language",
        "config",
        "dataset_revision",
        "split",
        "dataset_sha256",
        "checkpoint_sha256",
        "settings",
        "initial_prompt",
        "normalizer",
        "error_unit",
        "seed_algorithm",
    ):
        if baseline[key] != old[key] or baseline[key] != new[key]:
            raise ValueError(f"paired provenance differs: {key}")
    if new["scale_sha256"] != scale_sha256 or old["scale_sha256"] == scale_sha256:
        raise ValueError("FP8 scale hashes do not identify the old and new calibration")
    if old["fp8_graph_version"] != new["fp8_graph_version"]:
        raise ValueError("FP8 graph version changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-selection", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--new-scale-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.previous_selection.read_text())
    report = {"languages": {}, "inputs": {}}
    lines = [
        "| Language | Unit | Old t=0 N | Old delta (pp) | New t=0 N | New delta (pp) | Common N | Common old delta | Common new delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def number(value):
        return "n/a" if value is None else f"{value:+.6f}"

    for language in CONFIGS:
        directory = Path(selection[language])
        fp32, baseline = load_run(directory, language, "fp32")
        old, old_provenance = load_run(directory, language, "fp8")
        new, new_provenance = load_run(args.new_dir, language, "fp8")
        validate_provenance(baseline, old_provenance, new_provenance, args.new_scale_sha256)
        result = compare_language(language, fp32, old, new)
        report["languages"][language] = result
        report["inputs"][language] = {
            "previous_directory": str(directory),
            "new_directory": str(args.new_dir),
            "provenance": {"fp32": baseline, "old_fp8": old_provenance, "new_fp8": new_provenance},
        }
        left, right = (result[label]["temperature_zero"] for label in ("old", "new"))
        common = result["common_temperature_zero"]
        lines.append(
            f"| {language} | {result['error_unit']} | {left['utterances']} | "
            f"{number(left['difference_points'])} | {right['utterances']} | "
            f"{number(right['difference_points'])} | {common['old']['utterances']} | "
            f"{number(common['old']['difference_points'])} | "
            f"{number(common['new']['difference_points'])} |"
        )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    markdown = "\n".join(lines) + "\n"
    args.output.with_suffix(".md").write_text(markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
