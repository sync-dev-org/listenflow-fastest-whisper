"""Evaluate this repository's fp16 and fp32 large-v3 paths on LibriSpeech.

Run prepare, fp32, fp16, and report in that order. Each inference phase builds its
engines once and writes one result file after all included utterances finish.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from functools import wraps
from pathlib import Path

from listenflow.fastest_whisper._decoding import DecodingSettings, decoding_settings

SUBSETS = ("test-clean", "test-other")
SAMPLE_RATE = 16_000
MAX_FRAMES = 480_000
TOLERANCE_POINTS = 0.05
EXPECTED_INCLUDED = {"test-clean": 2611, "test-other": 2932}
# Saved precision and latency comparisons use greedy without temperature retries.
GREEDY_SETTINGS = decoding_settings(temperature=0.0, best_of=1)


class DecodeRetryRecorder(AbstractContextManager):
    """Observe real decoder attempts without changing their arguments or results."""

    def __init__(self, unit_method: str):
        self.unit_method = unit_method
        self.records: list[dict] = []
        self._current: dict | None = None
        self._restore = ExitStack()

    def __enter__(self):
        from listenflow.fastest_whisper._inference import TensorRTWhisper

        original_unit = getattr(TensorRTWhisper, self.unit_method)

        @wraps(original_unit)
        def unit(model, encoder, prompt, settings):
            record = {
                "attempts": 0,
                "temperatures": [],
                "prompt_limit": model.config.text_positions // 2 - 1,
                "prompt_tokens": list(prompt[-(model.config.text_positions // 2 - 1) :]),
            }
            self.records.append(record)
            previous = self._current
            self._current = record
            try:
                result = original_unit(model, encoder, prompt, settings)
                record["tokens"] = list(result.tokens)
                return result
            finally:
                self._current = previous

        def attempt_wrapper(original):
            @wraps(original)
            def attempt(*args, **kwargs):
                if self._current is not None:
                    self._current["attempts"] += 1
                    self._current["temperatures"].append(kwargs.get("temperature", 0.0))
                return original(*args, **kwargs)

            return attempt

        try:
            self._restore.callback(setattr, TensorRTWhisper, self.unit_method, original_unit)
            setattr(TensorRTWhisper, self.unit_method, unit)
            for name in ("_decode_attempt", "greedy_decode", "timestamp_decode"):
                original = getattr(TensorRTWhisper, name)
                self._restore.callback(setattr, TensorRTWhisper, name, original)
                setattr(TensorRTWhisper, name, attempt_wrapper(original))
        except BaseException:
            self._restore.close()
            raise
        return self

    def __exit__(self, *_exc):
        self._restore.close()

    def summary(self) -> dict:
        return {
            "units": len(self.records),
            "retried_units": sum(record["attempts"] > 1 for record in self.records),
            "attempts": sum(record["attempts"] for record in self.records),
            "records": self.records,
        }


def _setting_name(value: float | None) -> str:
    if value is None:
        return "none"
    if value == 0:
        return "0"
    return repr(value).removesuffix(".0").replace(".", "p").replace("-", "m")


def mode_name(settings: DecodingSettings, precision: str = "fp16") -> str:
    """Keep every decoding configuration in its own file, preserving default names."""
    defaults = GREEDY_SETTINGS
    if settings == defaults:
        return precision
    temperatures = "_".join(_setting_name(value) for value in settings.temperatures)
    name = f"{precision}-b{settings.beam_size}-t{temperatures}"
    for field, suffix in (
        ("best_of", "best"),
        ("compression_ratio_threshold", "cr"),
        ("logprob_threshold", "lp"),
        ("no_speech_threshold", "ns"),
    ):
        value = getattr(settings, field)
        if value != getattr(defaults, field):
            name += f"-{suffix}{_setting_name(value)}"
    return name


def _optional_float(value: str) -> float | None:
    if value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("threshold must be a number or none") from error


def add_decoding_options(parser: argparse.ArgumentParser) -> None:
    """Keep historical greedy measurements and share option names between commands."""
    defaults = GREEDY_SETTINGS
    parser.add_argument("--beam-size", type=int, default=defaults.beam_size)
    parser.add_argument("--temperature", default="0", help="comma-separated decoding temperatures")
    parser.add_argument("--best-of", type=int, default=defaults.best_of)
    parser.add_argument(
        "--record-retries",
        action="store_true",
        help="record real decoding attempts in inference JSON",
    )
    for field in ("compression_ratio_threshold", "logprob_threshold", "no_speech_threshold"):
        parser.add_argument(
            "--" + field.replace("_", "-"),
            type=_optional_float,
            default=getattr(defaults, field),
            help="decoding threshold; use none to disable the check",
        )


def settings_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> DecodingSettings:
    """Validate options before any waveform is read or inference begins."""
    try:
        return decoding_settings(
            beam_size=args.beam_size,
            temperature=tuple(float(value) for value in args.temperature.split(",")),
            best_of=args.best_of,
            compression_ratio_threshold=args.compression_ratio_threshold,
            logprob_threshold=args.logprob_threshold,
            no_speech_threshold=args.no_speech_threshold,
        )
    except ValueError as error:
        parser.error(str(error))


@dataclass(frozen=True)
class PairedResult:
    subset: str
    utterance_id: str
    reference_words: int
    fp16_errors: int
    fp32_errors: int
    fp16_output: str
    fp32_output: str


@dataclass(frozen=True)
class SubsetScore:
    utterances: int
    reference_words: int
    fp16_errors: int
    fp32_errors: int
    fp16_wer: float
    fp32_wer: float
    difference_points: float
    changed_outputs: int
    passed: bool


def included_utterances(
    rows: list[tuple[str, str, int]],
) -> tuple[list[tuple[str, str, int]], dict[str, list[str]]]:
    """Separate at most 30-second utterances and record excluded IDs by subset."""
    included: list[tuple[str, str, int]] = []
    excluded: dict[str, list[str]] = {subset: [] for subset in SUBSETS}
    for row in rows:
        subset, utterance_id, frames = row
        if subset not in excluded or frames < 0:
            raise ValueError(f"unexpected subset or frame count: {row}")
        if frames <= MAX_FRAMES:
            included.append(row)
        else:
            excluded[subset].append(utterance_id)
    return included, excluded


def score_subsets(rows: list[PairedResult]) -> dict[str, SubsetScore]:
    """Compare word-weighted WER within each subset using the fixed 0.05-point gate."""
    result = {}
    for subset in SUBSETS:
        selected = [row for row in rows if row.subset == subset]
        words = sum(row.reference_words for row in selected)
        if not selected or words <= 0:
            raise ValueError(f"missing reference words in {subset}")
        fp16_errors = sum(row.fp16_errors for row in selected)
        fp32_errors = sum(row.fp32_errors for row in selected)
        difference = fp16_errors - fp32_errors
        result[subset] = SubsetScore(
            utterances=len(selected),
            reference_words=words,
            fp16_errors=fp16_errors,
            fp32_errors=fp32_errors,
            fp16_wer=fp16_errors / words,
            fp32_wer=fp32_errors / words,
            difference_points=100 * difference / words,
            changed_outputs=sum(row.fp16_output != row.fp32_output for row in selected),
            passed=Fraction(100 * difference, words) <= Fraction(str(TOLERANCE_POINTS)),
        )
    if len(rows) != sum(score.utterances for score in result.values()):
        raise ValueError("unexpected subset in paired results")
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(librispeech_dir: Path, output_dir: Path) -> None:
    import soundfile as sf

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for subset in SUBSETS:
        root = librispeech_dir / subset
        transcripts = {}
        for path in sorted(root.rglob("*.trans.txt")):
            for line in path.read_text(encoding="utf-8").splitlines():
                utterance_id, reference = line.split(" ", 1)
                if utterance_id in transcripts:
                    raise ValueError(f"duplicate transcript: {utterance_id}")
                transcripts[utterance_id] = reference
        found = set()
        for path in sorted(root.rglob("*.flac")):
            utterance_id = path.stem
            if utterance_id in seen or utterance_id not in transcripts:
                raise ValueError(f"duplicate or missing transcript: {utterance_id}")
            seen.add(utterance_id)
            found.add(utterance_id)
            info = sf.info(str(path))
            if (info.samplerate, info.channels, info.subtype) != (SAMPLE_RATE, 1, "PCM_16"):
                raise ValueError(f"unexpected FLAC format: {path}")
            rows.append(
                {
                    "subset": subset,
                    "id": utterance_id,
                    "audio": str(path),
                    "reference": transcripts[utterance_id],
                    "frames": info.frames,
                }
            )
        if found != set(transcripts):
            raise ValueError(f"audio missing for transcript IDs in {subset}")
    included, excluded = included_utterances(
        [(row["subset"], row["id"], row["frames"]) for row in rows]
    )
    for subset in SUBSETS:
        count = sum(row[0] == subset for row in included)
        if count != EXPECTED_INCLUDED[subset]:
            raise ValueError(f"unexpected included count in {subset}: {count}")
        print(f"{subset}: included={count} excluded={len(excluded[subset])}", flush=True)
    write_csv(output_dir / "manifest.csv", rows, list(rows[0]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_hashes(checkpoint_dir: Path) -> dict[str, str]:
    files = sorted(checkpoint_dir.glob("*.safetensors"))
    files.append(checkpoint_dir / "model.safetensors.index.fp32.json")
    return {path.name: _sha256(path) for path in files if path.exists()}


def validate_mono_waveform(audio, *, sample_rate: int):
    """Check the waveform used by both short and long evaluations."""
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32)
    if sample_rate != SAMPLE_RATE or samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("audio must be a finite 16 kHz mono waveform")
    return samples


def word_errors(reference: str, output: str, *, normalizer=None) -> dict:
    """Normalize and count word errors once for both evaluation scripts."""
    import jiwer
    from whisper.normalizers import EnglishTextNormalizer

    normalizer = normalizer or EnglishTextNormalizer()
    normalized_reference = normalizer(reference)
    normalized_output = normalizer(output)
    measured = jiwer.process_words(normalized_reference, normalized_output)
    return {
        "normalized_reference": normalized_reference,
        "normalized_output": normalized_output,
        "reference_words": measured.hits + measured.substitutions + measured.deletions,
        "substitutions": measured.substitutions,
        "deletions": measured.deletions,
        "insertions": measured.insertions,
        "errors": measured.substitutions + measured.deletions + measured.insertions,
    }


def run(
    precision: str,
    checkpoint_dir: Path,
    output_dir: Path,
    *,
    settings: DecodingSettings | None = None,
    record_retries: bool = False,
) -> None:
    import numpy as np
    import soundfile as sf
    import tensorrt as trt
    from whisper.normalizers import EnglishTextNormalizer

    from listenflow.fastest_whisper._audio import log_mel_spectrogram
    from listenflow.fastest_whisper._inference import TensorRTWhisper

    settings = settings or GREEDY_SETTINGS
    name = mode_name(settings, precision)
    manifest = read_csv(output_dir / "manifest.csv")
    included, _ = included_utterances(
        [(row["subset"], row["id"], int(row["frames"])) for row in manifest]
    )
    selected = {(subset, utterance_id) for subset, utterance_id, _ in included}
    if any(
        sum(row[0] == subset for row in included) != EXPECTED_INCLUDED[subset] for subset in SUBSETS
    ):
        raise ValueError("manifest has an unexpected number of included utterances")
    final_path = output_dir / f"{name}.csv"
    partial_path = output_dir / f"{name}.csv.partial"
    metadata_path = output_dir / f"{name}.json"
    if final_path.exists() or partial_path.exists() or metadata_path.exists():
        raise FileExistsError(f"an output already exists for {name}")
    normalizer = EnglishTextNormalizer()
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
    started_at = datetime.now(UTC).isoformat()
    start = time.monotonic()
    recorder = DecodeRetryRecorder("decode_short") if record_retries else None
    with (
        recorder if recorder is not None else nullcontext(),
        TensorRTWhisper(checkpoint_dir, precision=precision) as model,
        partial_path.open("x", newline="", encoding="utf-8") as stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(manifest, 1):
            if (row["subset"], row["id"]) not in selected:
                continue
            samples, sample_rate = sf.read(row["audio"], dtype="int16")
            if (
                sample_rate != SAMPLE_RATE
                or samples.ndim != 1
                or len(samples) != int(row["frames"])
            ):
                raise ValueError(f"audio changed since manifest: {row['audio']}")
            waveform = validate_mono_waveform(
                samples.astype(np.float32) / np.float32(32768.0), sample_rate=sample_rate
            )
            mel = log_mel_spectrogram(waveform, n_mels=model.config.n_mels)
            encoder = model.encode(mel)
            transcription = model.decode_short(encoder, [], settings)
            measurement = word_errors(row["reference"], transcription.text, normalizer=normalizer)
            writer.writerow(
                {
                    "subset": row["subset"],
                    "id": row["id"],
                    "reference": row["reference"],
                    "output": transcription.text,
                    **measurement,
                }
            )
            stream.flush()
            if index == 1 or index % 100 == 0 or index == len(manifest):
                print(
                    f"{name}: manifest row {index}/{len(manifest)} elapsed={time.monotonic() - start:.1f}s",
                    flush=True,
                )
    build_seconds = model.build_seconds
    partial_path.rename(final_path)
    metadata = {
        "precision": precision,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - start,
        "engine_build_seconds": build_seconds,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_sha256": checkpoint_hashes(checkpoint_dir),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).splitlines()[0],
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).splitlines()[0],
        "tensorrt": trt.__version__,
        "python": platform.python_version(),
    }
    if settings != GREEDY_SETTINGS:
        metadata["decode_options"] = asdict(settings)
    if recorder is not None:
        metadata["retry_statistics"] = recorder.summary()
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: completed {len(selected)} utterances", flush=True)


def _read_results(path: Path, expected: set[tuple[str, str]]) -> dict:
    rows = read_csv(path)
    keyed = {(row["subset"], row["id"]): row for row in rows}
    if len(keyed) != len(rows) or set(keyed) != expected:
        raise ValueError(f"{path.stem} results do not match the manifest")
    return keyed


def _paired_results(included: list, manifest_by_key: dict, first: dict, second: dict) -> list:
    paired = []
    for subset, utterance_id, _ in included:
        key = (subset, utterance_id)
        left, right = first[key], second[key]
        if (
            left["reference"] != manifest_by_key[key]["reference"]
            or right["reference"] != left["reference"]
            or left["reference_words"] != right["reference_words"]
        ):
            raise ValueError(f"paired reference mismatch: {utterance_id}")
        paired.append(
            PairedResult(
                subset,
                utterance_id,
                int(left["reference_words"]),
                int(left["errors"]),
                int(right["errors"]),
                left["output"],
                right["output"],
            )
        )
    return paired


def _normalized_results(rows: dict, manifest_by_key: dict, normalizer) -> dict:
    normalized = {}
    for key, row in rows.items():
        if row["reference"] != manifest_by_key[key]["reference"]:
            raise ValueError(f"reference mismatch: {key[1]}")
        measurement = word_errors(row["reference"], row["output"], normalizer=normalizer)
        for field in ("errors", "reference_words"):
            if int(row[field]) != measurement[field]:
                raise ValueError(f"{field} mismatch: {key[1]}")
        normalized[key] = {**row, **measurement}
    return normalized


def _report_settings(
    output_dir: Path,
    settings: DecodingSettings,
    manifest_by_key: dict,
    included: list,
    excluded: dict,
) -> None:
    from whisper.normalizers import EnglishTextNormalizer

    normalizer = EnglishTextNormalizer()
    expected = {(subset, utterance_id) for subset, utterance_id, _ in included}
    comparisons = {}
    selected_runs = {}
    for precision in ("fp16", "fp32"):
        name = mode_name(settings, precision)
        selected_path = output_dir / f"{name}.csv"
        if not selected_path.exists():
            if (output_dir / f"{name}.json").exists() or selected_path.with_suffix(
                ".csv.partial"
            ).exists():
                raise ValueError(f"{name} results do not match the manifest: run is incomplete")
            continue
        selected = _normalized_results(
            _read_results(selected_path, expected), manifest_by_key, normalizer
        )
        baseline = _normalized_results(
            _read_results(output_dir / f"{precision}.csv", expected), manifest_by_key, normalizer
        )
        metadata = json.loads((output_dir / f"{name}.json").read_text(encoding="utf-8"))
        baseline_metadata = json.loads(
            (output_dir / f"{precision}.json").read_text(encoding="utf-8")
        )
        if metadata["checkpoint_sha256"] != baseline_metadata["checkpoint_sha256"]:
            raise ValueError(f"{name} and {precision} used different checkpoints")
        scores = score_subsets(_paired_results(included, manifest_by_key, selected, baseline))
        comparisons[precision] = {
            "subsets": {
                subset: {
                    "utterances": score.utterances,
                    "reference_words": score.reference_words,
                    "baseline_errors": score.fp32_errors,
                    "errors": score.fp16_errors,
                    "baseline_wer": score.fp32_wer,
                    "wer": score.fp16_wer,
                    "error_difference": score.fp16_errors - score.fp32_errors,
                    "difference_points": score.difference_points,
                    "changed_outputs": score.changed_outputs,
                }
                for subset, score in scores.items()
            },
            "baseline": baseline_metadata,
            "run": metadata,
        }
        selected_runs[precision] = selected
    if not comparisons:
        raise ValueError("no completed results for the requested decoding settings")
    result = {
        "decode_options": asdict(settings),
        "comparisons": comparisons,
        "excluded_ids": excluded,
    }
    if len(selected_runs) == 2:
        if (
            comparisons["fp16"]["run"]["checkpoint_sha256"]
            != comparisons["fp32"]["run"]["checkpoint_sha256"]
        ):
            raise ValueError("selected fp16 and fp32 used different checkpoints")
        scores = score_subsets(
            _paired_results(included, manifest_by_key, selected_runs["fp16"], selected_runs["fp32"])
        )
        result["subsets"] = {subset: asdict(score) for subset, score in scores.items()}
        result["passed"] = all(score.passed for score in scores.values())
    target = output_dir / f"{mode_name(settings, 'report')}.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def report(output_dir: Path, *, settings: DecodingSettings | None = None) -> None:
    from whisper.normalizers import EnglishTextNormalizer

    manifest = read_csv(output_dir / "manifest.csv")
    manifest_by_key = {(row["subset"], row["id"]): row for row in manifest}
    if len(manifest_by_key) != len(manifest):
        raise ValueError("duplicate utterance in manifest")
    included, excluded = included_utterances(
        [(row["subset"], row["id"], int(row["frames"])) for row in manifest]
    )
    settings = settings or GREEDY_SETTINGS
    if settings != GREEDY_SETTINGS:
        _report_settings(output_dir, settings, manifest_by_key, included, excluded)
        return
    expected = {(subset, utterance_id) for subset, utterance_id, _ in included}
    normalizer = EnglishTextNormalizer()
    by_precision = {
        precision: _normalized_results(
            _read_results(output_dir / f"{precision}.csv", expected), manifest_by_key, normalizer
        )
        for precision in ("fp16", "fp32")
    }
    paired = _paired_results(included, manifest_by_key, by_precision["fp16"], by_precision["fp32"])
    scores = score_subsets(paired)
    result = {
        "subsets": {subset: asdict(score) for subset, score in scores.items()},
        "passed": all(score.passed for score in scores.values()),
        "excluded_ids": excluded,
        "fp16": json.loads((output_dir / "fp16.json").read_text(encoding="utf-8")),
        "fp32": json.loads((output_dir / "fp32.json").read_text(encoding="utf-8")),
    }
    (output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fp32", "fp16", "report"))
    parser.add_argument("--librispeech-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_decoding_options(parser)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.librispeech_dir is None:
            parser.error("prepare requires --librispeech-dir")
        prepare(args.librispeech_dir, args.output_dir)
    elif args.phase == "report":
        report(args.output_dir, settings=settings_from_args(args, parser))
    else:
        if args.checkpoint_dir is None:
            parser.error(f"{args.phase} requires --checkpoint-dir")
        options = {"settings": settings_from_args(args, parser)}
        if args.record_retries:
            options["record_retries"] = True
        run(
            args.phase,
            args.checkpoint_dir,
            args.output_dir,
            **options,
        )


if __name__ == "__main__":
    main()
