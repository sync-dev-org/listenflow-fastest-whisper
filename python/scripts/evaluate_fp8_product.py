"""Compare shipped fp8/fp32 on both historical greedy and product default decoding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scripts.benchmark_latency import _cache_engine_hashes, _load_waveform
from scripts.evaluate_wer import EXPECTED_INCLUDED, checkpoint_hashes, read_csv, word_errors
from scripts.fp8_trial import MeasuredWhisper

from listenflow.fastest_whisper import _fp8
from listenflow.fastest_whisper._audio import log_mel_spectrogram
from listenflow.fastest_whisper._decoding import DecodingSettings, decoding_settings


def condition_options(condition):
    if condition == 1:
        return decoding_settings(temperature=0.0, best_of=1), None
    if condition == 2:
        return DecodingSettings(), "yakult yakult man"
    raise ValueError("condition must be 1 or 2")


def precision_metadata(precision):
    """Record calibration and graph details associated with the selected precision."""
    if precision != "fp8":
        return {}
    return {
        "scale_sha256": hashlib.sha256(_fp8._SCALES.read_bytes()).hexdigest(),
        "graph_version": _fp8._GRAPH_VERSION,
    }


@contextmanager
def seeded_utterance(utterance_id):
    """Supply entropy internally, without adding a seed option to the product API."""
    seed = int.from_bytes(hashlib.sha256(utterance_id.encode()).digest()[:8], "big")
    generator = np.random.default_rng(seed)
    with patch.object(np.random, "default_rng", lambda: generator):
        yield seed


def run(args):
    from whisper.normalizers import EnglishTextNormalizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_csv(args.manifest) if int(row["frames"]) <= 480_000]
    if len({row["id"] for row in rows}) != sum(EXPECTED_INCLUDED.values()) or any(
        sum(row["subset"] == split for row in rows) != count
        for split, count in EXPECTED_INCLUDED.items()
    ):
        raise ValueError("manifest must contain all 5,543 evaluation utterances exactly once")
    name = f"condition{args.condition}-{args.precision}"
    target = args.output_dir / f"{name}.json"
    partial = args.output_dir / f"{name}.csv.partial"
    final = partial.with_suffix("")
    if target.exists() or partial.exists() or final.exists():
        raise FileExistsError(name)
    settings, hint = condition_options(args.condition)
    normalizer = EnglishTextNormalizer()
    totals = {
        split: {"utterances": 0, "reference_words": 0, "errors": 0} for split in EXPECTED_INCLUDED
    }
    retries = []
    started = time.perf_counter()
    metadata = {
        "started_at": datetime.now(UTC).isoformat(),
        "condition": args.condition,
        "precision": args.precision,
        "initial_prompt": hint,
        "settings": asdict(settings),
        "seed_algorithm": "first 8 bytes of SHA-256(utterance ID), unsigned big-endian",
        "checkpoint_sha256": checkpoint_hashes(args.checkpoint_dir),
        **precision_metadata(args.precision),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "versions": {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        },
    }
    fields = [
        "subset",
        "id",
        "reference",
        "output",
        "seed",
        "tokens",
        "attempts",
        "normalized_reference",
        "normalized_output",
        "reference_words",
        "substitutions",
        "deletions",
        "insertions",
        "errors",
    ]
    with MeasuredWhisper(args.checkpoint_dir, precision=args.precision) as model:
        metadata["prepare_seconds"] = time.perf_counter() - started
        metadata["engine_prepare_seconds"] = model.build_seconds
        metadata["engine_sha256"] = _cache_engine_hashes()
        with partial.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(rows, 1):
                samples = _load_waveform(row)
                attempts, decoded = [], []
                original_attempt, original_decode = model._decode_attempt, model.decode_short

                def attempt(*values, _original=original_attempt, _records=attempts, **options):
                    result = _original(*values, **options)
                    _records.append(asdict(result))
                    return result

                def decode(*values, _original=original_decode, _records=decoded, **options):
                    result = _original(*values, **options)
                    _records.append(result)
                    return result

                with (
                    seeded_utterance(row["id"]) as seed,
                    patch.object(model, "_decode_attempt", attempt),
                    patch.object(model, "decode_short", decode),
                ):
                    if args.condition == 1:
                        mel = log_mel_spectrogram(samples, n_mels=model.config.n_mels)
                        text = model.decode_short(model.encode(mel), [], settings).text
                    else:
                        # Leave all decoding options at the product's actual defaults.
                        text = " ".join(
                            segment.text
                            for segment in model.transcribe(samples, initial_prompt=hint)
                        )
                if len(decoded) != 1:
                    raise RuntimeError("one short utterance must produce one decoded window")
                score = word_errors(row["reference"], text, normalizer=normalizer)
                record = {
                    "subset": row["subset"],
                    "id": row["id"],
                    "reference": row["reference"],
                    "output": text,
                    "seed": seed,
                    "tokens": json.dumps(decoded[0].tokens),
                    "attempts": json.dumps(attempts),
                    **score,
                }
                writer.writerow(record)
                stream.flush()
                if len(attempts) > 1:
                    retries.append(
                        {
                            "subset": row["subset"],
                            "id": row["id"],
                            "seed": seed,
                            "attempts": attempts,
                            "output": text,
                        }
                    )
                total = totals[row["subset"]]
                total["utterances"] += 1
                total["reference_words"] += score["reference_words"]
                total["errors"] += score["errors"]
                if index == 1 or index % 100 == 0:
                    print(
                        f"{name}: {index}/{len(rows)} {time.perf_counter() - started:.1f}s {totals}",
                        flush=True,
                    )
        metadata["engine_sha256_after"] = _cache_engine_hashes()
    for split, words in (("test-clean", 52349), ("test-other", 52281)):
        if totals[split]["reference_words"] != words:
            raise ValueError(f"reference word count changed: {split}")
        totals[split]["wer_percent"] = 100 * totals[split]["errors"] / words
    metadata.update(
        finished_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.perf_counter() - started,
        subsets=totals,
        retried_utterances=len(retries),
        retries=retries,
    )
    partial.rename(final)
    target.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("fp8", "fp32"), required=True)
    parser.add_argument("--condition", choices=(1, 2), type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
