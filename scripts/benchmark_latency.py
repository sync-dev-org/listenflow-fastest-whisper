"""Compare single-utterance fp16 latency with faster-whisper on LibriSpeech.

Run convert, fp16, faster-whisper, and report in order. Invoke as a module so the
shared WER evaluator is available from a standalone checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scripts.evaluate_wer import (
    EXPECTED_INCLUDED,
    GREEDY_SETTINGS,
    MAX_FRAMES,
    SAMPLE_RATE,
    checkpoint_hashes,
    included_utterances,
    mode_name,
    read_csv,
    validate_mono_waveform,
    word_errors,
)

from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._decoding import DecodingSettings, decoding_settings


class _TimedCachedWhisper(CachedWhisper):
    """Time the product's build-or-load calls without changing their cache policy."""

    def _open_engine(self, checkpoint, *, decoder):
        started = time.monotonic()
        handle = super()._open_engine(checkpoint, decoder=decoder)
        self.build_seconds["decoder" if decoder else "encoder"] = time.monotonic() - started
        return handle

    def _open_mel(self, checkpoint):
        started = time.monotonic()
        handle = super()._open_mel(checkpoint)
        self.build_seconds["mel"] = time.monotonic() - started
        return handle


def _cache_engine_hashes() -> dict[str, str]:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    hashes = {}
    for path in sorted((cache_root / "tensoracle").glob("*.engine")):
        with path.open("rb") as stream:
            hashes[str(path.resolve())] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


@dataclass(frozen=True)
class Latency:
    utterance_id: str
    frames: int
    seconds: float


def summarize_latencies(
    rows: list[Latency], expected_ids: set[str]
) -> dict[str, dict[str, int | float]]:
    """Summarize one observation per utterance in half-open duration bands."""
    if len(rows) != len(expected_ids) or {row.utterance_id for row in rows} != expected_ids:
        raise ValueError("latency rows do not match the expected utterance IDs")
    bands: dict[str, list[float]] = {name: [] for name in ("0-5", "5-10", "10-20", "20-30")}
    for row in rows:
        if not (0 <= row.frames <= MAX_FRAMES) or not (
            math.isfinite(row.seconds) and row.seconds >= 0
        ):
            raise ValueError(f"invalid latency row: {row.utterance_id}")
        duration = row.frames / SAMPLE_RATE
        if duration < 5:
            band = "0-5"
        elif duration < 10:
            band = "5-10"
        elif duration < 20:
            band = "10-20"
        else:
            band = "20-30"
        bands[band].append(row.seconds)
    return {
        band: {
            "count": len(values),
            "median_seconds": statistics.median(values) if values else math.nan,
            "p95_seconds": float(np.percentile(values, 95)) if values else math.nan,
        }
        for band, values in bands.items()
    }


def clean_manifest(wer_output_dir: Path) -> list[dict[str, str]]:
    """Reuse the WER evaluator's included-utterance rule and exact clean subset."""
    manifest = read_csv(wer_output_dir / "manifest.csv")
    included, _ = included_utterances(
        [(row["subset"], row["id"], int(row["frames"])) for row in manifest]
    )
    selected = {utterance_id for subset, utterance_id, _ in included if subset == "test-clean"}
    rows = [row for row in manifest if row["subset"] == "test-clean" and row["id"] in selected]
    if len(selected) != EXPECTED_INCLUDED["test-clean"] or len(rows) != len(selected):
        raise ValueError("manifest does not contain the expected test-clean utterance IDs")
    return rows


def _load_waveform(row: dict[str, str]) -> np.ndarray:
    import soundfile as sf

    samples, sample_rate = sf.read(row["audio"], dtype="int16")
    if len(samples) != int(row["frames"]):
        raise ValueError(f"audio changed since manifest: {row['audio']}")
    return validate_mono_waveform(
        samples.astype(np.float32) / np.float32(32768), sample_rate=sample_rate
    )


def _gpu_field(name: str) -> str:
    return (
        subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={name}", "--format=csv,noheader,nounits"], text=True
        )
        .splitlines()[0]
        .strip()
    )


def _gpu_sample(completed: int) -> dict:
    process_ids = [
        int(value)
        for value in subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).splitlines()
        if value.strip()
    ]
    others = [pid for pid in process_ids if pid != os.getpid()]
    if others:
        raise RuntimeError(f"other GPU compute processes overlap the comparison: {others}")
    fields = ("clocks.sm", "utilization.gpu", "memory.used")
    values = (
        subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            text=True,
        )
        .splitlines()[0]
        .split(",")
    )
    return {
        "at": datetime.now(UTC).isoformat(),
        "completed_utterances": completed,
        "compute_process_pids": process_ids,
        **dict(zip(fields, (value.strip() for value in values), strict=True)),
    }


def _synchronize() -> None:
    from cuda.bindings import runtime as cudart

    result = cudart.cudaDeviceSynchronize()
    if result[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaDeviceSynchronize failed: {result[0]}")


def convert(checkpoint_dir: Path, model_dir: Path) -> None:
    """Convert the exact local checkpoint to a float16 CTranslate2 model."""
    import ctranslate2

    if model_dir.exists():
        raise FileExistsError(f"converted model already exists: {model_dir}")
    source_hashes = checkpoint_hashes(checkpoint_dir)
    if "model.safetensors" not in source_hashes:
        raise ValueError("checkpoint lacks model.safetensors")
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    converter = ctranslate2.converters.TransformersConverter(
        str(checkpoint_dir),
        copy_files=["tokenizer.json", "preprocessor_config.json"],
        load_as_float16=True,
    )
    converter.convert(str(model_dir), quantization="float16")
    metadata = {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_sha256": source_hashes,
        "quantization": "float16",
        "load_as_float16": True,
        "copy_files": ["tokenizer.json", "preprocessor_config.json"],
        "ctranslate2": ctranslate2.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "conversion_seconds": time.monotonic() - started,
    }
    (model_dir / "conversion.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


def _transcribe_tensor_rt(
    model, samples: np.ndarray, settings: DecodingSettings | None = None
) -> str:
    settings = settings or GREEDY_SETTINGS
    mel = model._mel_spectrogram(samples)
    encoder = model.encode(mel)
    if settings.greedy_only:
        return model.greedy_decode(encoder).text
    return model.decode_short(encoder, [], settings).text


def _transcribe_faster_whisper(model, samples: np.ndarray, options: dict) -> str:
    segments, _ = model.transcribe(
        samples, **{**options, "suppress_tokens": list(options["suppress_tokens"])}
    )
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip())


def _faster_whisper_options(checkpoint_dir: Path) -> dict:
    from tokenizers import Tokenizer

    from listenflow.fastest_whisper._inference import _suppressed_tokens

    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    return {
        "language": "en",
        "task": "transcribe",
        "beam_size": 1,
        "best_of": 1,
        "temperature": 0.0,
        "compression_ratio_threshold": None,
        "log_prob_threshold": None,
        "no_speech_threshold": None,
        "condition_on_previous_text": False,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": sorted(_suppressed_tokens(tokenizer)),
        "without_timestamps": True,
        "word_timestamps": False,
        "multilingual": False,
        "vad_filter": False,
        "hallucination_silence_threshold": None,
        "hotwords": None,
    }


def run(
    route: str,
    checkpoint_dir: Path,
    wer_output_dir: Path,
    output_dir: Path,
    model_dir: Path | None,
    *,
    settings: DecodingSettings | None = None,
) -> None:
    """Measure one warmed, loaded model on every selected utterance, batch one."""
    import ctranslate2
    import faster_whisper
    import tensorrt as trt

    settings = settings or GREEDY_SETTINGS
    name = mode_name(settings) if route == "fp16" else route
    rows = clean_manifest(wer_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_csv = output_dir / f"{name}.csv"
    partial_csv = output_dir / f"{name}.csv.partial"
    final_json = output_dir / f"{name}.json"
    if any(path.exists() for path in (final_csv, partial_csv, final_json)):
        raise FileExistsError(f"an output already exists for {name}")
    source_hashes = checkpoint_hashes(checkpoint_dir)
    conversion = None
    if route == "faster-whisper":
        if model_dir is None:
            raise ValueError("faster-whisper requires --model-dir")
        conversion = json.loads((model_dir / "conversion.json").read_text(encoding="utf-8"))
        if (
            conversion["checkpoint_sha256"] != source_hashes
            or conversion["quantization"] != "float16"
        ):
            raise ValueError("converted model does not match the float16 checkpoint")
    started_at = datetime.now(UTC).isoformat()
    observations = [_gpu_sample(0)]
    engine_hashes_before = _cache_engine_hashes() if route == "fp16" else None
    load_started = time.monotonic()
    options = None
    if route == "fp16":
        model = _TimedCachedWhisper(checkpoint_dir, precision="fp16", device=0, force_rebuild=False)
        effective_compute_type = "fp16 TensorRT network"
    else:
        assert model_dir is not None
        options = _faster_whisper_options(checkpoint_dir)
        model = faster_whisper.WhisperModel(
            str(model_dir),
            device="cuda",
            device_index=0,
            compute_type="float16",
            cpu_threads=1,
            num_workers=1,
            local_files_only=True,
        )
        build_seconds = None
        effective_compute_type = model.model.compute_type
        if model.model.device != "cuda" or effective_compute_type != "float16":
            raise RuntimeError(
                f"faster-whisper requires cuda float16, got {model.model.device} {effective_compute_type}"
            )
    load_seconds = time.monotonic() - load_started
    if route == "fp16":
        engine_hashes_loaded = _cache_engine_hashes()
        engine_reused = {
            name: str(handle.path.resolve()) in engine_hashes_before
            for name, handle in (
                ("encoder", model._encoder_handle),
                ("decoder", model._decoder_handle),
                ("mel", model._mel_handle),
            )
        }
        build_seconds = {
            name: seconds if not engine_reused[name] else 0.0
            for name, seconds in model.build_seconds.items()
        }
    first_waveform = _load_waveform(rows[0])
    _synchronize()
    warmup_started = time.monotonic()
    if route == "fp16":
        _transcribe_tensor_rt(model, first_waveform, settings)
    else:
        assert options is not None
        _transcribe_faster_whisper(model, first_waveform, options)
    _synchronize()
    warmup_seconds = time.monotonic() - warmup_started
    observations.append(_gpu_sample(0))
    with partial_csv.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subset", "id", "frames", "seconds", "output"])
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            samples = first_waveform if index == 1 else _load_waveform(row)
            _synchronize()
            started = time.perf_counter_ns()
            if route == "fp16":
                text = _transcribe_tensor_rt(model, samples, settings)
            else:
                assert options is not None
                text = _transcribe_faster_whisper(model, samples, options)
            _synchronize()
            seconds = (time.perf_counter_ns() - started) / 1e9
            writer.writerow(
                {
                    "subset": "test-clean",
                    "id": row["id"],
                    "frames": row["frames"],
                    "seconds": seconds,
                    "output": text,
                }
            )
            if index % 100 == 0 or index == len(rows):
                stream.flush()
                observations.append(_gpu_sample(index))
                print(f"{name}: {index}/{len(rows)}", flush=True)
    partial_csv.rename(final_csv)
    if route == "fp16":
        engine_hashes_after = _cache_engine_hashes()
        if engine_hashes_after != engine_hashes_loaded:
            raise RuntimeError("saved TensorRT engines changed during the comparison")
        model.__exit__(None, None, None)
    metadata = {
        "route": name,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "execution_order": "manifest order",
        "repetitions_per_utterance": 1,
        "warmup_utterance_id": rows[0]["id"],
        "warmup_seconds": warmup_seconds,
        "load_seconds": load_seconds,
        "engine_build_seconds": build_seconds,
        "tensor_rt_path": {
            "model": "CachedWhisper",
            "features": "GPU log-mel",
            "force_rebuild": False,
            "engine_reused": engine_reused,
            "engine_open_seconds": model.build_seconds,
            "engine_sha256_before": engine_hashes_before,
            "engine_sha256_after_load": engine_hashes_loaded,
            "engine_sha256_after_run": engine_hashes_after,
        }
        if route == "fp16"
        else None,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_sha256": source_hashes,
        "converted_model_dir": str(model_dir.resolve()) if model_dir else None,
        "conversion": conversion,
        "requested_compute_type": "float16",
        "effective_compute_type": effective_compute_type,
        "decode_options": options
        or {
            "language": "en",
            "task": "transcribe",
            "beam_size": settings.beam_size,
            "temperature": list(settings.temperatures),
            "best_of": settings.best_of,
            "compression_ratio_threshold": settings.compression_ratio_threshold,
            "logprob_threshold": settings.logprob_threshold,
            "without_timestamps": True,
            "suppressed_tokens": sorted(model.suppressed),
            "begin_suppressed_tokens": sorted(model.begin_suppressed),
            "no_speech_threshold": settings.no_speech_threshold,
            "condition_on_previous_text": False,
        },
        "gpu": _gpu_field("name"),
        "driver": _gpu_field("driver_version"),
        "tensorrt": trt.__version__,
        "ctranslate2": ctranslate2.__version__,
        "faster_whisper": faster_whisper.__version__,
        "gpu_observations": observations,
    }
    final_json.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{name}: completed {len(rows)} utterances", flush=True)


def _latencies(rows: list[dict[str, str]], manifest: list[dict[str, str]]) -> list[Latency]:
    expected = {row["id"]: int(row["frames"]) for row in manifest}
    if len(expected) != len(manifest):
        raise ValueError("duplicate utterance IDs in manifest")
    result = []
    for row in rows:
        utterance_id = row["id"]
        if (
            row["subset"] != "test-clean"
            or utterance_id not in expected
            or int(row["frames"]) != expected[utterance_id]
        ):
            raise ValueError(f"result does not match manifest: {utterance_id}")
        result.append(Latency(utterance_id, expected[utterance_id], float(row["seconds"])))
    return result


def report(wer_output_dir: Path, output_dir: Path) -> None:
    """Validate both complete runs and write duration bands and text diagnostics."""
    from whisper.normalizers import EnglishTextNormalizer

    manifest = clean_manifest(wer_output_dir)
    expected_ids = {row["id"] for row in manifest}
    by_id = {row["id"]: row for row in manifest}
    route_rows = {
        route: read_csv(output_dir / f"{route}.csv") for route in ("fp16", "faster-whisper")
    }
    metadata = {
        route: json.loads((output_dir / f"{route}.json").read_text(encoding="utf-8"))
        for route in route_rows
    }
    if metadata["fp16"]["checkpoint_sha256"] != metadata["faster-whisper"]["checkpoint_sha256"]:
        raise ValueError("routes used different checkpoints")
    summaries = {
        route: summarize_latencies(_latencies(rows, manifest), expected_ids)
        for route, rows in route_rows.items()
    }
    keyed = {route: {row["id"]: row for row in rows} for route, rows in route_rows.items()}
    prior_rows = [
        row for row in read_csv(wer_output_dir / "fp16.csv") if row["subset"] == "test-clean"
    ]
    prior = {row["id"]: row for row in prior_rows}
    if len(prior) != len(expected_ids) or set(prior) != expected_ids:
        raise ValueError("prior fp16 outputs do not match the manifest")
    if any(
        prior[utterance_id]["reference"] != by_id[utterance_id]["reference"]
        for utterance_id in expected_ids
    ):
        raise ValueError("prior fp16 reference does not match the manifest")
    normalizer = EnglishTextNormalizer()
    differences = []
    fw_errors = 0
    fw_words = 0
    for utterance_id in sorted(expected_ids):
        reference = by_id[utterance_id]["reference"]
        trt_text = keyed["fp16"][utterance_id]["output"]
        fw_text = keyed["faster-whisper"][utterance_id]["output"]
        old_text = prior[utterance_id]["output"]
        if trt_text != old_text:
            old_score = word_errors(reference, old_text, normalizer=normalizer)
            new_score = word_errors(reference, trt_text, normalizer=normalizer)
            differences.append(
                {
                    "id": utterance_id,
                    "prior_fp16": old_text,
                    "fp16": trt_text,
                    "faster_whisper": fw_text,
                    "reference": reference,
                    "prior_errors": old_score["errors"],
                    "new_errors": new_score["errors"],
                }
            )
        score = word_errors(reference, fw_text, normalizer=normalizer)
        fw_errors += int(score["errors"])
        fw_words += int(score["reference_words"])
    result = {
        "utterances": len(expected_ids),
        "duration_bands": summaries,
        "prior_fp16_matching": len(expected_ids) - len(differences),
        "prior_fp16_differences": differences,
        "faster_whisper_reference_words": fw_words,
        "faster_whisper_errors": fw_errors,
        "faster_whisper_wer": fw_errors / fw_words,
        "runs": metadata,
    }
    target = output_dir / "report.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("convert", "fp16", "faster-whisper", "report"))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--wer-output-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--temperature", default="0")
    parser.add_argument("--best-of", type=int, default=1)
    args = parser.parse_args()
    if args.phase == "convert":
        if args.checkpoint_dir is None:
            parser.error("convert requires --checkpoint-dir")
        convert(args.checkpoint_dir, args.output_dir)
    elif args.phase == "report":
        if args.wer_output_dir is None:
            parser.error("report requires --wer-output-dir")
        report(args.wer_output_dir, args.output_dir)
    else:
        if args.checkpoint_dir is None or args.wer_output_dir is None:
            parser.error(f"{args.phase} requires --checkpoint-dir and --wer-output-dir")
        if args.phase == "fp16":
            try:
                temperatures = tuple(float(value) for value in args.temperature.split(","))
                settings = decoding_settings(
                    beam_size=args.beam_size, temperature=temperatures, best_of=args.best_of
                )
            except ValueError as error:
                parser.error(str(error))
            run(
                args.phase,
                args.checkpoint_dir,
                args.wer_output_dir,
                args.output_dir,
                args.model_dir,
                settings=settings,
            )
        else:
            run(
                args.phase,
                args.checkpoint_dir,
                args.wer_output_dir,
                args.output_dir,
                args.model_dir,
            )


if __name__ == "__main__":
    main()
