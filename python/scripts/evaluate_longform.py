"""Prepare, run, and compare the fixed Earnings-21 long-form fp32 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scripts.evaluate_wer import (
    GREEDY_SETTINGS,
    DecodeRetryRecorder,
    add_decoding_options,
    mode_name,
    settings_from_args,
    validate_mono_waveform,
    word_errors,
)

from listenflow.fastest_whisper._decoding import DecodingSettings

SOURCE_COMMIT = "c05ab6fd8b4b627d123c922a22a39e993dd37635"
SOURCE_PATH = "earnings21/media/4386541.mp3"
MP3_SHA256 = "8ba0f8013f8d3eb7a1aabc390479ebd2c7db895797e79709264b2af71f3c77f2"
WAVEFORM_SHA256 = "48c6f01b21d5c6122b26a7635d77f29152be1399c9b8e9292fd07626cfa4d7c4"
SAMPLES = 17_553_600


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_audio(output_dir: Path) -> np.ndarray:
    audio = validate_mono_waveform(np.load(output_dir / "audio.npy"), sample_rate=16_000)
    if len(audio) != SAMPLES or _hash_bytes(audio.tobytes()) != WAVEFORM_SHA256:
        raise ValueError("prepared Earnings-21 waveform does not match the fixed input")
    return audio


def prepare(audio_path: Path, output_dir: Path) -> None:
    import whisper

    if _hash_file(audio_path) != MP3_SHA256:
        raise ValueError("Earnings-21 MP3 does not match the fixed source")
    audio = validate_mono_waveform(whisper.load_audio(str(audio_path)), sample_rate=16_000)
    if len(audio) != SAMPLES or _hash_bytes(audio.tobytes()) != WAVEFORM_SHA256:
        raise ValueError("decoded Earnings-21 waveform does not match the fixed input")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "audio.npy", audio)
    manifest = {
        "source_repository": "revdotcom/speech-datasets",
        "source_commit": SOURCE_COMMIT,
        "source_path": SOURCE_PATH,
        "mp3_sha256": MP3_SHA256,
        "waveform_sha256": WAVEFORM_SHA256,
        "sample_rate": 16_000,
        "samples": SAMPLES,
        "seconds": SAMPLES / 16_000,
        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def oracle(
    checkpoint: Path, output_dir: Path, *, initial_prompt: str | None = None, beam_size: int = 1
) -> None:
    import torch
    import whisper

    audio = _read_audio(output_dir)
    torch.set_num_threads(8)
    start = time.perf_counter()
    model = whisper.load_model(str(checkpoint), device="cpu")
    loaded = time.perf_counter()
    calls = []
    original_decode = model.decode

    def record_decode(mel, options=None, **kwargs):
        result = original_decode(mel, options, **kwargs)
        calls.append(
            {
                "tokens": result.tokens,
                "prompt": list(options.prompt) if options and options.prompt else [],
                "avg_logprob": result.avg_logprob,
                "no_speech_prob": result.no_speech_prob,
            }
        )
        print(f"oracle window {len(calls)} completed", flush=True)
        return result

    model.decode = record_decode
    decode_options = {"beam_size": beam_size} if beam_size > 1 else {}
    result = model.transcribe(
        audio,
        fp16=False,
        language="en",
        task="transcribe",
        temperature=0,
        verbose=None,
        initial_prompt=initial_prompt,
        **decode_options,
    )
    result["decode_calls"] = calls
    result["initial_prompt"] = initial_prompt
    result["beam_size"] = beam_size
    result["runtime"] = {
        "load_seconds": loaded - start,
        "decode_seconds": time.perf_counter() - loaded,
        "python": platform.python_version(),
        "openai_whisper": "20250625",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _hash_file(checkpoint),
    }
    (output_dir / "oracle.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"oracle completed: {len(calls)} windows, {len(result['segments'])} segments")


def summarize_result(document: dict) -> dict:
    """Summarize the recorded text and traces without another inference run."""
    common = Counter(segment["text"] for segment in document["segments"]).most_common(1)
    return {
        "segments": len(document["segments"]),
        "text_characters": len(document["text"]),
        "most_common_segment": common[0][0] if common else None,
        "most_common_segment_count": common[0][1] if common else 0,
        "windows": len(document["windows"]),
    }


def compare_unretried_windows(baseline: dict, result: dict) -> dict:
    """Require baseline tokens only for unretried windows with the same input context."""
    records = result["retry_statistics"]["records"]
    if baseline.get("initial_prompt") != result.get("initial_prompt"):
        raise ValueError("initial prompts differ between the baseline and measured recording")
    all_tokens = (
        list(records[0]["prompt_tokens"]) if records and baseline.get("initial_prompt") else []
    )
    segments_by_seek = {}
    for segment in baseline["segments"]:
        segments_by_seek.setdefault(segment["seek"], []).extend(segment["tokens"])
    windows_by_seek = {}
    for window in baseline["windows"]:
        windows_by_seek[window["seek"]] = (list(all_tokens), window["tokens"])
        all_tokens.extend(segments_by_seek.get(window["seek"], []))
    comparison = {
        "passed": True,
        "checked_windows": 0,
        "input_different_windows": 0,
        "retried_windows": 0,
        "changed_windows": 0,
        "first_difference": None,
    }
    for index, (window, record) in enumerate(zip(result["windows"], records, strict=True)):
        if record["attempts"] > 1:
            comparison["retried_windows"] += 1
            continue
        expected = windows_by_seek.get(window["seek"])
        if expected is None or expected[0][-record["prompt_limit"] :] != record["prompt_tokens"]:
            comparison["input_different_windows"] += 1
            continue
        comparison["checked_windows"] += 1
        if window["tokens"] != expected[1]:
            comparison["changed_windows"] += 1
            if comparison["first_difference"] is None:
                comparison["first_difference"] = {
                    "window_index": index,
                    "seek": window["seek"],
                    "baseline_tokens": expected[1],
                    "tokens": window["tokens"],
                }
    comparison["passed"] = comparison["changed_windows"] == 0
    return comparison


def fp32(
    checkpoint: Path,
    output_dir: Path,
    *,
    initial_prompt: str | None = None,
    beam_size: int = 1,
    settings: DecodingSettings | None = None,
    record_retries: bool = False,
) -> None:
    from listenflow.fastest_whisper._inference import _transcribe_long_fp32

    settings = settings or replace(GREEDY_SETTINGS, beam_size=beam_size)
    audio = _read_audio(output_dir)
    start = time.perf_counter()
    windows = []
    segment_traces = []
    recorder = DecodeRetryRecorder("decode_long") if record_retries else None
    with recorder if recorder is not None else nullcontext():
        result = _transcribe_long_fp32(
            checkpoint,
            audio,
            initial_prompt=initial_prompt,
            settings=settings,
            trace=windows.append,
            segment_trace=segment_traces.append,
        )
    document = asdict(result)
    document["initial_prompt"] = initial_prompt
    document["beam_size"] = settings.beam_size
    document["segments"] = [asdict(segment) for segment in segment_traces]
    document["windows"] = [asdict(window) for window in windows]
    document["runtime"] = {
        "seconds": time.perf_counter() - start,
        "python": platform.python_version(),
        "checkpoint": str(checkpoint.resolve()),
    }
    if settings != GREEDY_SETTINGS:
        document["decode_options"] = asdict(settings)
        document["summary"] = summarize_result(document)
    if recorder is not None:
        document["retry_statistics"] = recorder.summary()
    (output_dir / f"{mode_name(settings, 'fp32')}.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"fp32 completed: {len(windows)} windows, {len(result.segments)} segments")


def compare_results(oracle_result: dict, fp32_result: dict) -> dict:
    """Apply the exact transcript and segment acceptance criterion."""
    expected_segments = oracle_result["segments"]
    actual_segments = fp32_result["segments"]
    segment_differences = []
    for index, (expected, actual) in enumerate(zip(expected_segments, actual_segments)):
        for field in ("text", "start", "end"):
            if expected[field] != actual[field]:
                segment_differences.append(
                    {
                        "field": field,
                        "segment_index": index,
                        "oracle": expected[field],
                        "fp32": actual[field],
                        "oracle_seek": expected.get("seek"),
                        "fp32_seek": actual.get("seek"),
                    }
                )
                break
    if len(expected_segments) != len(actual_segments):
        segment_differences.append(
            {
                "field": "segment_count",
                "segment_index": min(len(expected_segments), len(actual_segments)),
                "oracle": len(expected_segments),
                "fp32": len(actual_segments),
            }
        )

    if oracle_result["text"] != fp32_result["text"]:
        first_difference = {
            "field": "text",
            "oracle": oracle_result["text"],
            "fp32": fp32_result["text"],
        }
    else:
        first_difference = segment_differences[0] if segment_differences else None

    calls = oracle_result.get("decode_calls", [])
    windows = fp32_result.get("windows", [])
    first_token_difference = None
    for window_index in range(max(len(calls), len(windows))):
        expected = calls[window_index]["tokens"] if window_index < len(calls) else []
        actual = windows[window_index]["tokens"] if window_index < len(windows) else []
        for token_index in range(max(len(expected), len(actual))):
            expected_token = expected[token_index] if token_index < len(expected) else None
            actual_token = actual[token_index] if token_index < len(actual) else None
            if expected_token != actual_token:
                first_token_difference = {
                    "window_index": window_index,
                    "seek_frame": windows[window_index]["seek"]
                    if window_index < len(windows)
                    else None,
                    "token_index": token_index,
                    "oracle": expected_token,
                    "fp32": actual_token,
                }
                break
        if first_token_difference is not None:
            break

    return {
        "passed": first_difference is None,
        "oracle_segments": len(expected_segments),
        "fp32_segments": len(actual_segments),
        "oracle_windows": len(calls),
        "fp32_windows": len(windows),
        "first_difference": first_difference,
        "first_segment_difference": segment_differences[0] if segment_differences else None,
        "segment_differences_after_first": max(0, len(segment_differences) - 1),
        "first_token_difference": first_token_difference,
    }


def report(output_dir: Path) -> None:
    oracle_result = json.loads((output_dir / "oracle.json").read_text(encoding="utf-8"))
    fp32_result = json.loads((output_dir / "fp32.json").read_text(encoding="utf-8"))
    comparison = compare_results(oracle_result, fp32_result)
    comparison["word_error_diagnostic"] = word_errors(oracle_result["text"], fp32_result["text"])
    comparison["reported_at"] = datetime.now(UTC).isoformat()
    (output_dir / "report.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    if not comparison["passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "oracle", "fp32", "report"))
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--initial-prompt")
    add_decoding_options(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.beam_size < 1:
        parser.error("--beam-size must be positive")
    if args.phase == "prepare":
        if args.audio is None:
            parser.error("prepare requires --audio")
        prepare(args.audio, args.output_dir)
    elif args.phase == "oracle":
        if args.checkpoint is None:
            parser.error("oracle requires --checkpoint")
        options = {"initial_prompt": args.initial_prompt}
        if args.beam_size > 1:
            options["beam_size"] = args.beam_size
        oracle(args.checkpoint, args.output_dir, **options)
    elif args.phase == "fp32":
        if args.checkpoint is None:
            parser.error("fp32 requires --checkpoint")
        settings = settings_from_args(args, parser)
        options = {"initial_prompt": args.initial_prompt}
        if settings == replace(GREEDY_SETTINGS, beam_size=args.beam_size):
            if args.beam_size > 1:
                options["beam_size"] = args.beam_size
        else:
            options["settings"] = settings
        if args.record_retries:
            options["record_retries"] = True
        fp32(args.checkpoint, args.output_dir, **options)
    else:
        report(args.output_dir)


if __name__ == "__main__":
    main()
