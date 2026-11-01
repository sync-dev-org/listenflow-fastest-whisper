"""Measure resumable FLEURS test-split error rates with the product decoding defaults.

Run one explicit language and precision at a time, then use ``report`` to pair
completed runs. GPU runs require an externally granted resource reservation.
The reader, normalization and scoring functions do not require a GPU or an engine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import resource
import signal
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from fractions import Fraction
from functools import cache
from pathlib import Path
from unittest.mock import patch

import numpy as np

CONFIGS = {
    "en": "en_us",
    "ja": "ja_jp",
    "zh": "cmn_hans_cn",
    "yue": "yue_hant_hk",
    "ko": "ko_kr",
    "ru": "ru_ru",
    "ar": "ar_eg",
    "hi": "hi_in",
    "th": "th_th",
}
PRECISIONS = ("fp32", "fp16", "fp8")
CHARACTER_LANGUAGES = frozenset(("zh", "yue", "ja", "ko", "th"))
SEED_ALGORITHM = "first 8 bytes of SHA-256(config + '/' + original filename), unsigned big-endian"
CACHE_GROWTH_LIMIT_BYTES = 10_000_000_000
FIELDS = (
    "id",
    "fleurs_id",
    "language",
    "precision",
    "reference",
    "output",
    "seed",
    "duration",
    "declared_samples",
    "decoded_samples",
    "seconds",
    "temperatures",
    "attempts",
    "normalized_reference",
    "normalized_output",
    "reference_units",
    "substitutions",
    "deletions",
    "insertions",
    "errors",
)


def error_unit(language: str) -> str:
    """Return the required error unit for a representative Whisper language code."""
    if language not in CONFIGS:
        raise ValueError(f"unknown representative language: {language}")
    return "CER" if language in CHARACTER_LANGUAGES else "WER"


@cache
def _normalizer(language):
    from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

    error_unit(language)
    return EnglishTextNormalizer() if language == "en" else BasicTextNormalizer()


def normalize_text(text: str, language: str) -> str:
    """Apply the official normalizer and remove whitespace for character scoring."""
    normalized = _normalizer(language)(text)
    return "".join(normalized.split()) if error_unit(language) == "CER" else normalized


def score_text(reference: str, output: str, language: str) -> dict:
    """Count actual substitutions, deletions and insertions on the selected units."""
    import jiwer

    left, right = normalize_text(reference, language), normalize_text(output, language)
    measure = jiwer.process_characters if error_unit(language) == "CER" else jiwer.process_words
    result = measure(left, right)
    return {
        "normalized_reference": left,
        "normalized_output": right,
        "reference_units": result.hits + result.substitutions + result.deletions,
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
        "errors": result.substitutions + result.deletions + result.insertions,
    }


def _shard_path(data_dir: Path, language: str) -> Path:
    error_unit(language)
    return Path(data_dir) / CONFIGS[language] / "test-00000-of-00001.parquet"


def _manifest(data_dir: Path, language: str) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(
        _shard_path(data_dir, language),
        columns=["id", "path", "transcription", "num_samples"],
        use_threads=False,
    ).to_pylist()
    result = [
        {
            "id": f"{CONFIGS[language]}/{Path(row['path']).name}",
            "fleurs_id": int(row["id"]),
            "reference": row["transcription"],
            "frames": int(row["num_samples"]),
        }
        for row in rows
    ]
    if not result or len({row["id"] for row in result}) != len(result):
        raise ValueError("empty split or duplicate FLEURS recording id")
    return result


def read_fleurs(data_dir: Path, language: str, *, skip_ids=()):
    """Yield recording ids, references and exact mono 16 kHz float32 waveforms.

    Repeated sentence ids remain distinct through config/original-filename keys.
    Skipped recordings are not decoded from their embedded WAV bytes.
    Decoded WAV samples determine the audio and duration; both sample counts are
    retained so measurements can record discrepancies in the declared length.
    """
    import pyarrow.parquet as pq
    import soundfile as sf

    seen = set()
    skip_ids = frozenset(skip_ids)
    columns = ["id", "path", "transcription", "num_samples", "audio"]
    for batch in pq.ParquetFile(_shard_path(data_dir, language)).iter_batches(
        batch_size=1, columns=columns, use_threads=False
    ):
        for row in batch.to_pylist():
            filename = Path(row["path"]).name
            key = f"{CONFIGS[language]}/{filename}"
            if key in seen:
                raise ValueError(f"duplicate recording: {key}")
            seen.add(key)
            if key in skip_ids:
                continue
            audio = row["audio"]
            if audio["path"] and Path(audio["path"]).name != filename:
                raise ValueError(f"audio identity differs: {key}")
            samples, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            if rate != 16000 or samples.ndim != 1 or not np.isfinite(samples).all():
                raise ValueError(f"{key} must be finite mono 16 kHz audio")
            yield {
                "id": key,
                "fleurs_id": int(row["id"]),
                "reference": row["transcription"],
                "audio": samples,
                "duration": len(samples) / 16000,
                "declared_samples": int(row["num_samples"]),
                "decoded_samples": len(samples),
            }


def _temperatures(row):
    values = row.get("temperatures", [])
    return json.loads(values) if isinstance(values, str) else list(values)


def _validated_rows(rows, language):
    result = {}
    for row in rows:
        key = row["id"]
        if key in result:
            raise ValueError(f"duplicate recording: {key}")
        score = score_text(row["reference"], row["output"], language)
        for field, value in score.items():
            if field in row and str(row[field]) != str(value):
                raise ValueError(f"stored {field} differs for {key}")
        result[key] = {**row, **score}
    return result


def summarize_language(language: str, runs: dict[str, list[dict]]) -> dict:
    """Pair the same recordings and apply the fp16 0.05-point gate per language."""
    selected = {precision: _validated_rows(runs[precision], language) for precision in PRECISIONS}
    baseline = selected["fp32"]
    if not baseline or any(set(rows) != set(baseline) for rows in selected.values()):
        raise ValueError("precision runs contain different recording ids")
    for rows in selected.values():
        if any(rows[key]["reference"] != baseline[key]["reference"] for key in baseline):
            raise ValueError("precision runs contain different references")
    units = sum(row["reference_units"] for row in baseline.values())
    if units <= 0:
        raise ValueError("language has no normalized reference units")
    errors = {
        precision: sum(row["errors"] for row in rows.values())
        for precision, rows in selected.items()
    }
    result = {
        "error_unit": error_unit(language),
        "utterances": len(baseline),
        "reference_units": units,
        "precisions": {
            precision: {
                "errors": count,
                "rate_percent": 100 * count / units,
                "difference_points": 100 * (count - errors["fp32"]) / units,
            }
            for precision, count in errors.items()
        },
        "fp16_gate": Fraction(100 * (errors["fp16"] - errors["fp32"]), units) <= Fraction("0.05"),
        "top_regressions": {},
        "skipped_fp32_fallbacks": {},
    }
    for precision in ("fp16", "fp8"):
        regressions = []
        skipped = 0
        for key, reference in baseline.items():
            row = selected[precision][key]
            missing_temperatures = sorted(
                {t for t in _temperatures(reference) if t > 0} - set(_temperatures(row))
            )
            omitted = bool(missing_temperatures)
            skipped += omitted
            difference = row["errors"] - reference["errors"]
            if difference > 0:
                regressions.append(
                    {
                        "id": key,
                        "reference": reference["reference"],
                        "fp32_output": reference["output"],
                        "output": row["output"],
                        "fp32_errors": reference["errors"],
                        "errors": row["errors"],
                        "error_increase": difference,
                        "skipped_fp32_fallback": omitted,
                        "skipped_fp32_temperatures": missing_temperatures,
                        "fp32_temperatures": _temperatures(reference),
                        "temperatures": _temperatures(row),
                    }
                )
        result["top_regressions"][precision] = sorted(
            regressions, key=lambda row: (-row["error_increase"], row["id"])
        )[:5]
        result["skipped_fp32_fallbacks"][precision] = skipped
    return result


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _cache_snapshot():
    root = Path.home() / ".cache/tensoracle"
    files = {path.name: path.stat().st_size for path in root.glob("*.engine")}
    return {"bytes": sum(files.values()), "files": files}


def _open_model(checkpoint_dir, precision):
    from listenflow.fastest_whisper._cached_model import CachedWhisper

    return CachedWhisper(checkpoint_dir, precision=precision)


def _provenance(data_dir, checkpoint_dir, language, precision):
    from dataclasses import asdict

    from listenflow.fastest_whisper._decoding import DecodingSettings

    weight_files = sorted(Path(checkpoint_dir).glob("*.safetensors"))
    if not weight_files:
        raise ValueError("checkpoint has no safetensors weights")
    files = [*weight_files, *sorted(Path(checkpoint_dir).glob("*.json"))]
    result = {
        "language": language,
        "config": CONFIGS[language],
        "precision": precision,
        "dataset_revision": Path(data_dir).name,
        "split": "test",
        "dataset_sha256": _sha256(_shard_path(data_dir, language)),
        "checkpoint_sha256": {path.name: _sha256(path) for path in files},
        "settings": json.loads(json.dumps(asdict(DecodingSettings()))),
        "initial_prompt": None,
        "normalizer": "EnglishTextNormalizer()" if language == "en" else "BasicTextNormalizer()",
        "error_unit": error_unit(language),
        "seed_algorithm": SEED_ALGORITHM,
        "script_sha256": _sha256(Path(__file__)),
        "versions": {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        },
    }
    if precision == "fp8":
        from listenflow.fastest_whisper import _fp8

        result["scale_sha256"] = _sha256(_fp8._SCALES)
        result["fp8_graph_version"] = _fp8._GRAPH_VERSION
    return result


@contextmanager
def _stop_signals(control):
    def stop(signum, _frame):
        control["stop"] = signal.Signals(signum).name

    previous = {signum: signal.signal(signum, stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_evaluation(
    data_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    language: str,
    precision: str,
    *,
    stop_file: Path | None = None,
) -> int:
    """Append durable per-recording results and resume only matching unfinished runs."""
    from scripts.evaluate_fp8_product import seeded_utterance

    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision: {precision}")
    manifest = _manifest(data_dir, language)
    expected = {row["id"]: row for row in manifest}
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / f"{language}-{precision}.json"
    csv_path = metadata_path.with_suffix(".csv")
    provenance = _provenance(data_dir, checkpoint_dir, language, precision)
    previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if previous and previous["provenance"] != provenance:
        raise ValueError("resume provenance differs from the saved run")
    completed = {}
    if csv_path.exists():
        if not previous:
            raise ValueError("CSV has no provenance record")
        if previous.get("csv_sha256") and _sha256(csv_path) != previous["csv_sha256"]:
            raise ValueError("saved result CSV changed")
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError("saved CSV fields differ")
            completed = _validated_rows(list(reader), language)
        for key, row in completed.items():
            if (
                key not in expected
                or row["reference"] != expected[key]["reference"]
                or row["language"] != language
                or row["precision"] != precision
            ):
                raise ValueError("completed recordings do not match the input split")
    if previous and previous["status"] == "complete":
        if len(completed) != len(manifest):
            raise ValueError("completed run is missing recordings")
        print(f"{language}-{precision}: already complete", flush=True)
        return 0
    baseline_path = output_dir / "cache-baseline.json"
    before = _cache_snapshot()
    if not baseline_path.exists():
        _write_json(baseline_path, before)
    cache_baseline = json.loads(baseline_path.read_text())
    tick = time.perf_counter()
    started = datetime.now(UTC).isoformat()
    metadata = {
        "provenance": provenance,
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "os": platform.platform(),
        "started_at": previous["started_at"] if previous else started,
        "status": "running",
        "expected_utterances": len(manifest),
        "completed_utterances": len(completed),
        "sessions": previous["sessions"] if previous else [],
        "cache_before": before,
    }
    _write_json(metadata_path, metadata)
    control = {"stop": None}
    exit_status = 1
    try:
        with (
            _stop_signals(control),
            csv_path.open("a" if completed else "w", newline="", encoding="utf-8") as stream,
        ):
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if not completed:
                writer.writeheader()
                stream.flush()
            if stop_file and stop_file.exists():
                control["stop"] = "stop file"
            if not control["stop"]:
                with _open_model(checkpoint_dir, precision) as model:
                    metadata["prepare_seconds"] = time.perf_counter() - tick
                    metadata["engine_prepare_seconds"] = model.build_seconds
                    prepared_cache = _cache_snapshot()
                    metadata["cache_after_prepare"] = prepared_cache
                    if prepared_cache["bytes"] - cache_baseline["bytes"] > CACHE_GROWTH_LIMIT_BYTES:
                        raise RuntimeError(
                            "shared engine cache growth exceeded 10 GB; request a decision"
                        )
                    for row in read_fleurs(data_dir, language, skip_ids=completed):
                        if control["stop"] or (stop_file and stop_file.exists()):
                            control["stop"] = control["stop"] or "stop file"
                            break
                        attempts = []
                        original = model._decode_attempt

                        def attempt(*args, _original=original, _attempts=attempts, **kwargs):
                            result = _original(*args, **kwargs)
                            _attempts.append(
                                {
                                    name: getattr(result, name, None)
                                    for name in (
                                        "temperature",
                                        "avg_logprob",
                                        "no_speech_prob",
                                        "compression_ratio",
                                        "text",
                                    )
                                }
                            )
                            return result

                        decode_tick = time.perf_counter()
                        with (
                            seeded_utterance(row["id"]) as seed,
                            patch.object(model, "_decode_attempt", attempt),
                        ):
                            text = " ".join(
                                segment.text
                                for segment in model.transcribe(row["audio"], language=language)
                            )
                        record = {
                            "id": row["id"],
                            "fleurs_id": row["fleurs_id"],
                            "language": language,
                            "precision": precision,
                            "reference": row["reference"],
                            "output": text,
                            "seed": seed,
                            "duration": row["duration"],
                            "declared_samples": row["declared_samples"],
                            "decoded_samples": row["decoded_samples"],
                            "seconds": time.perf_counter() - decode_tick,
                            "temperatures": json.dumps([item["temperature"] for item in attempts]),
                            "attempts": json.dumps(attempts, ensure_ascii=False, allow_nan=False),
                            **score_text(row["reference"], text, language),
                        }
                        writer.writerow(record)
                        stream.flush()
                        os.fsync(stream.fileno())
                        completed[row["id"]] = record
                        metadata["completed_utterances"] = len(completed)
                        if len(completed) % 25 == 0 or len(completed) == len(manifest):
                            print(
                                f"{language}-{precision}: {len(completed)}/{len(manifest)} in {time.perf_counter() - tick:.1f}s",
                                flush=True,
                            )
            exit_status = 0 if len(completed) == len(manifest) else 75
            metadata["status"] = "complete" if exit_status == 0 else "stopped"
            metadata["stop_reason"] = control["stop"]
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        session = {
            "started_at": started,
            "finished_at": datetime.now(UTC).isoformat(),
            "exit_status": exit_status,
            "elapsed_seconds": time.perf_counter() - tick,
            "completed_utterances": len(completed),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        metadata["sessions"].append(session)
        metadata.update(
            finished_at=session["finished_at"],
            exit_status=exit_status,
            elapsed_seconds=sum(item["elapsed_seconds"] for item in metadata["sessions"]),
            cache_after=_cache_snapshot(),
        )
        if csv_path.exists():
            metadata["csv_sha256"] = _sha256(csv_path)
        _write_json(metadata_path, metadata)
    return exit_status


def build_report(output_dir: Path) -> dict:
    """Write complete paired comparisons and explicitly list unfinished runs."""
    report = {"languages": {}, "remaining": []}
    for language in CONFIGS:
        runs, identities = {}, {}
        for precision in PRECISIONS:
            metadata_path = output_dir / f"{language}-{precision}.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            if metadata.get("status") != "complete":
                report["remaining"].append(
                    {
                        "language": language,
                        "precision": precision,
                        "reason": metadata.get("status", "not run"),
                        "completed_utterances": metadata.get("completed_utterances", 0),
                    }
                )
                continue
            csv_path = metadata_path.with_suffix(".csv")
            if _sha256(csv_path) != metadata["csv_sha256"]:
                raise ValueError(f"result CSV changed: {csv_path}")
            with csv_path.open(newline="", encoding="utf-8") as stream:
                runs[precision] = list(csv.DictReader(stream))
            identity = dict(metadata["provenance"])
            for field in ("precision", "scale_sha256", "fp8_graph_version"):
                identity.pop(field, None)
            identities[precision] = identity
        if "fp32" in runs:
            if any(identity != identities["fp32"] for identity in identities.values()):
                raise ValueError(f"paired provenance differs: {language}")
            paired = {precision: runs.get(precision, runs["fp32"]) for precision in PRECISIONS}
            summary = summarize_language(language, paired)
            summary["status"] = "complete" if len(runs) == len(PRECISIONS) else "incomplete"
            for precision in PRECISIONS:
                if precision not in runs:
                    summary["precisions"][precision] = None
                    summary["top_regressions"][precision] = []
                    summary["skipped_fp32_fallbacks"][precision] = None
            if "fp16" not in runs:
                summary["fp16_gate"] = None
            report["languages"][language] = summary
        else:
            report["languages"][language] = {
                "error_unit": error_unit(language),
                "status": "incomplete",
                "reference_units": None,
                "precisions": dict.fromkeys(PRECISIONS),
                "fp16_gate": None,
            }
    _write_json(output_dir / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--checkpoint-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--language", choices=CONFIGS, required=True)
    run.add_argument("--precision", choices=PRECISIONS, required=True)
    run.add_argument("--stop-file", type=Path)
    report = commands.add_parser("report")
    report.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "report":
        print(json.dumps(build_report(args.output_dir), ensure_ascii=False, indent=2))
        return 0
    return run_evaluation(
        args.data_dir,
        args.checkpoint_dir,
        args.output_dir,
        args.language,
        args.precision,
        stop_file=args.stop_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
