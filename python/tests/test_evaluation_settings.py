"""CPU-only checks for evaluation of independently recorded decode settings."""

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from math import log
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from scripts import evaluate_longform, evaluate_wer
from scripts.benchmark_latency import mode_name
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from listenflow.fastest_whisper import (
    _inference,
    transcribe_fp16,
    transcribe_fp32,
    transcribe_long_fp32,
)
from listenflow.fastest_whisper._audio import log_mel_spectrogram
from listenflow.fastest_whisper._decoding import DecodingSettings
from listenflow.fastest_whisper._inference import TensorRTWhisper
from listenflow.fastest_whisper._longform import (
    LongTranscription,
    SegmentTrace,
    TimedSegment,
    WindowTrace,
)

SETTINGS = DecodingSettings(
    beam_size=2,
    temperatures=(0.0, 0.4, 0.8),
    best_of=3,
    compression_ratio_threshold=None,
    logprob_threshold=-2.0,
    no_speech_threshold=0.8,
)
OPTIONS = [
    "--beam-size",
    "2",
    "--temperature",
    "0,0.4,0.8",
    "--best-of",
    "3",
    "--compression-ratio-threshold",
    "none",
    "--logprob-threshold",
    "-2",
    "--no-speech-threshold",
    "0.8",
]


def _json_settings(settings):
    return json.loads(json.dumps(asdict(settings)))


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("entry", [transcribe_fp16, transcribe_fp32, transcribe_long_fp32])
def test_omitted_decoding_options_pass_the_selected_retry_defaults(entry, tmp_path, cpu_tensor_rt):
    """Omitted options use greedy followed by five sampled candidates at increasing temperatures."""
    entry(tmp_path / "model", np.zeros(16_000, dtype=np.float32))
    settings = (
        cpu_tensor_rt.window_settings[0][1]
        if entry is transcribe_long_fp32
        else cpu_tensor_rt.short_calls[0][2]
    )
    assert settings.beam_size == 1
    assert settings.temperatures == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert settings.best_of == 5
    assert settings.compression_ratio_threshold == 2.4
    assert settings.logprob_threshold == -1.0
    assert settings.no_speech_threshold == 0.6


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_measurements_keep_historical_greedy_names_and_separate_the_new_engine_defaults():
    """Historical files still mean greedy without retries, and engine defaults get a distinct name."""
    historical = DecodingSettings(temperatures=(0.0,), best_of=1)
    assert mode_name(historical) == "fp16"
    assert mode_name(DecodingSettings()) == "fp16-b1-t0_0p2_0p4_0p6_0p8_1-best5"


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_evaluation_options_explicitly_keep_the_historical_greedy_measurement():
    """Omitted evaluation options retain greedy without retries for saved-result comparisons."""
    parser = argparse.ArgumentParser()
    evaluate_wer.add_decoding_options(parser)
    settings = evaluate_wer.settings_from_args(parser.parse_args([]), parser)
    assert settings == DecodingSettings(temperatures=(0.0,), best_of=1)


@pytest.fixture
def cpu_tensor_rt(tmp_path, monkeypatch):
    """Replace GPU engine execution while retaining every CPU decoding operation."""
    vocabulary = {
        "Hello, world!": 0,
        " Echo.": 1,
        " End.": 2,
        " ": 3,
        "earnings": 4,
        "<unk>": 5,
        "<|endoftext|>": 6,
        "<|startoftranscript|>": 7,
        "<|en|>": 8,
        "<|transcribe|>": 9,
        "<|notimestamps|>": 10,
        "<|startofprev|>": 11,
        "<|nospeech|>": 12,
    }
    timestamp_begin = len(vocabulary)
    vocabulary.update({f"<|{index * 0.02:.2f}|>": timestamp_begin + index for index in range(1501)})
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.Fuse()
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    tokenizer.save(str(checkpoint / "tokenizer.json"))
    sequences = [
        [timestamp_begin, 1, timestamp_begin + 50, timestamp_begin + 50, 3, timestamp_begin + 60],
        [timestamp_begin, 1, timestamp_begin + 50, timestamp_begin + 50, 2, timestamp_begin + 100],
    ]
    observed = SimpleNamespace(
        long_calls=[],
        short_calls=[],
        window_settings=[],
        window_thresholds=[],
        greedy_calls=[],
        fallback_calls=[],
        attempt_calls=[],
        gpu_prefixes=[],
        closed_engines=[],
        no_speech_probability=0.1,
        sequences=sequences,
    )

    class MelExecution:
        def run(self, inputs, *, device_outputs):
            mel = log_mel_spectrogram(inputs["audio"][200:-200], n_mels=128)
            return {"mel": mel, "log_mel": mel * 4 - 4}

        def close(self):
            observed.closed_engines.append("mel")

    class EncoderExecution:
        def __init__(self):
            self.window = -1

        def run(self, inputs, *, device_outputs):
            mel = inputs["mel"]
            assert mel.shape == (1, 128, 3000)
            assert mel.dtype == np.float32 and np.isfinite(mel).all()
            self.window += 1
            return {
                "encoder": np.array([[[self.window]]], dtype=np.float32),
                "cross_keys": np.full((1, 1, 1, 1, 1), self.window, dtype=np.float32),
                "cross_values": np.full((1, 1, 1, 1, 1), self.window, dtype=np.float32),
            }

        def close(self):
            observed.closed_engines.append("encoder")

    class DecoderExecution:
        outputs = ("token", "logits", "present_keys", "present_values")

        def __init__(self, precision):
            self.precision = precision

        def run(self, inputs, *, device_outputs, reuse_cache=False):
            prefix = inputs["self_keys"].flatten()[1:].astype(int).tolist()
            prefix += inputs["tokens"][0].tolist()
            window = int(inputs["cross_keys"].flat[0])
            observed.gpu_prefixes.append((self.precision, window, prefix))
            sot = (
                prefix.index(vocabulary["<|startoftranscript|>"])
                if vocabulary["<|startoftranscript|>"] in prefix
                else len(prefix)
            )
            short = vocabulary["<|notimestamps|>"] in prefix
            generated = prefix[sot + (4 if short else 3) :]
            sequence = [0] if short else sequences[window]
            assert generated == sequence[: len(generated)]
            next_token = sequence[len(generated)] if len(generated) < len(sequence) else 6
            logits = np.full((1, len(vocabulary)), -np.inf, dtype=np.float32)
            probability = observed.no_speech_probability
            if len(prefix) - 1 == sot:
                logits[0, 0] = log(1 - probability)
                logits[0, vocabulary["<|nospeech|>"]] = log(probability)
            else:
                logits[0, next_token] = 0.0
                # The CPU decoder must suppress this otherwise dominant unknown token.
                logits[-1, vocabulary["<unk>"]] = 20.0
            cache = np.array([0, *prefix], dtype=np.float32).reshape(1, 1, 1, -1, 1)
            token = np.argmax(logits[None] + inputs["suppression"], axis=-1).astype(np.int32)[
                ..., None
            ]
            filtered = logits[None] + inputs["suppression"]
            logprob = filtered.max(axis=-1, keepdims=True) - np.logaddexp.reduce(
                filtered, axis=-1, keepdims=True
            )
            silence = np.exp(logits[0, vocabulary["<|nospeech|>"]] - np.logaddexp.reduce(logits[0]))
            return {
                "logits": logits[None],
                "present_keys": cache,
                "present_values": cache.copy(),
                "token": token,
                "greedy": np.concatenate(
                    (
                        token.astype(np.float32),
                        logprob,
                        np.full((1, 1, 1), silence, dtype=np.float32),
                    ),
                    axis=-1,
                ),
            }

        def close(self):
            observed.closed_engines.append("decoder")

    def initialize(model, checkpoint_dir, *, precision="fp32", device=0):
        model.checkpoint_dir = Path(checkpoint_dir)
        model.precision = precision
        model.device = device
        model.tokenizer = Tokenizer.from_file(str(model.checkpoint_dir / "tokenizer.json"))
        model.config = SimpleNamespace(
            n_mels=128,
            text_positions=448,
            audio_positions=1500,
            audio_state=1,
            text_layers=1,
            text_heads=1,
            vocab_size=len(vocabulary),
        )
        model.initial_tokens = [
            vocabulary[token]
            for token in (
                "<|startoftranscript|>",
                "<|en|>",
                "<|transcribe|>",
                "<|notimestamps|>",
            )
        ]
        model.eot = vocabulary["<|endoftext|>"]
        model.suppressed = _inference._suppressed_tokens(model.tokenizer)
        model.begin_suppressed = set(model.tokenizer.encode(" ", add_special_tokens=False).ids)
        model.begin_suppressed.add(model.eot)
        model.build_seconds = 0.25
        model._encoder = EncoderExecution()
        model._decoder = DecoderExecution(precision)
        model._mel = MelExecution()

    # Only initialization of GPU engines and their execution are replaced. The
    # original context manager, encode, logits, tokenizer and decoders all run.
    monkeypatch.setattr(TensorRTWhisper, "__init__", initialize)
    profiled_files = {
        _inference.__file__,
        _inference.transcribe_mel.__code__.co_filename,
        _inference.decode_with_fallback.__code__.co_filename,
    }

    def profile(frame, event, _argument):
        if event != "call":
            return
        values = frame.f_locals
        name = frame.f_code.co_name
        if frame.f_code.co_filename not in profiled_files:
            return
        if name == "_transcribe_long_fp32":
            observed.long_calls.append(
                (values["checkpoint_dir"], values["initial_prompt"], values["settings"])
            )
        elif name == "decode_short":
            observed.short_calls.append(
                (values["self"].precision, values["prompt"].copy(), values["settings"])
            )
        elif name == "decode_long":
            observed.window_settings.append((values["prompt"].copy(), values["settings"]))
        elif name == "transcribe_mel":
            observed.window_thresholds.append(
                (values["no_speech_threshold"], values["logprob_threshold"])
            )
        elif name == "greedy_decode":
            observed.greedy_calls.append(values["self"].precision)
        elif name == "decode_with_fallback":
            observed.fallback_calls.append(
                {field: values[field] for field in asdict(DecodingSettings())}
            )
        elif name == "_decode_attempt":
            observed.attempt_calls.append(
                (values["temperature"], values["beam_size"], values["best_of"])
            )

    # Profiling observes real calls and arguments; it never replaces a CPU
    # function or supplies a CPU function's result.
    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        yield observed
    finally:
        sys.setprofile(previous_profile)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_output_names_distinguish_every_decode_setting_and_keep_existing_names():
    """A measurement cannot replace the result of a different decoding configuration."""
    settings = [
        DecodingSettings(temperatures=(0.0,), best_of=1),
        DecodingSettings(beam_size=2, temperatures=(0.0,), best_of=1),
        SETTINGS,
        replace(SETTINGS, beam_size=3),
        replace(SETTINGS, temperatures=(0.0, 0.4000001, 0.8)),
        replace(SETTINGS, temperatures=(0.0, 0.4000002, 0.8)),
        replace(SETTINGS, best_of=4),
        replace(SETTINGS, best_of=10**17 + 1),
        replace(SETTINGS, best_of=10**17 + 2),
        replace(SETTINGS, compression_ratio_threshold=2.4),
        replace(SETTINGS, logprob_threshold=None),
        replace(SETTINGS, no_speech_threshold=None),
        DecodingSettings(best_of=2),
        DecodingSettings(no_speech_threshold=None),
    ]

    names = [mode_name(setting) for setting in settings]

    assert len(set(names)) == len(settings)
    assert names[0] == "fp16"
    assert names[1] == "fp16-b2-t0"
    assert (
        mode_name(DecodingSettings(beam_size=2, temperatures=(0.0, 0.4), best_of=1))
        == "fp16-b2-t0_0p4"
    )


@pytest.fixture
def longform_input(tmp_path, monkeypatch, cpu_tensor_rt):
    # A real 32-second recording exercises two windows and trailing padding.
    audio = np.zeros(32 * 16_000, dtype=np.float32)
    monkeypatch.setattr(evaluate_longform, "SAMPLES", len(audio))
    monkeypatch.setattr(
        evaluate_longform, "WAVEFORM_SHA256", hashlib.sha256(audio.tobytes()).hexdigest()
    )
    np.save(tmp_path / "audio.npy", audio)
    segments = [
        TimedSegment(0.0, 1.0, " Echo."),
        TimedSegment(1.0, 1.2, ""),
        TimedSegment(30.0, 31.0, " Echo."),
        TimedSegment(31.0, 32.0, " End."),
    ]
    result = LongTranscription(" Echo. Echo. End.", segments)
    first, second = cpu_tensor_rt.sequences
    traces = [
        SegmentTrace(0.0, 1.0, " Echo.", first[:3], 0),
        SegmentTrace(1.0, 1.2, "", [], 0),
        SegmentTrace(30.0, 31.0, " Echo.", second[:3], 3000),
        SegmentTrace(31.0, 32.0, " End.", second[3:], 3000),
    ]
    windows = [WindowTrace(0, 3000, first), WindowTrace(3000, 3200, second)]
    return cpu_tensor_rt.long_calls, result, traces, windows


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("configured", [False, True])
def test_longform_cli_passes_settings_and_preserves_the_default_document(
    configured, longform_input, tmp_path, monkeypatch, cpu_tensor_rt
):
    """The chosen settings reach inference, while the default record keeps its historical bytes."""
    seen, result, traces, windows = longform_input
    baseline = tmp_path / "fp32.json"
    if configured:
        baseline.write_bytes(b"existing default result\n")
    monkeypatch.setattr(evaluate_longform.time, "perf_counter", iter([10.0, 13.5]).__next__)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_longform",
            "fp32",
            "--checkpoint",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path),
            "--initial-prompt",
            "earnings",
            *(OPTIONS if configured else []),
        ],
    )

    evaluate_longform.main()

    selected = SETTINGS if configured else DecodingSettings(temperatures=(0.0,), best_of=1)
    assert seen == [(tmp_path / "model", "earnings", selected)]
    assert cpu_tensor_rt.window_settings == [
        ([4], selected),
        ([4, *traces[0].tokens], selected),
    ]
    assert cpu_tensor_rt.window_thresholds == [
        (selected.no_speech_threshold, selected.logprob_threshold)
    ]
    assert cpu_tensor_rt.closed_engines == ["decoder", "encoder", "mel"]
    if configured:
        assert cpu_tensor_rt.fallback_calls == [asdict(selected)] * 2
    legacy = {
        "text": result.text,
        "segments": [asdict(trace) for trace in traces],
        "initial_prompt": "earnings",
        "beam_size": selected.beam_size,
        "windows": [asdict(window) for window in windows],
        "runtime": {
            "seconds": 3.5,
            "python": platform.python_version(),
            "checkpoint": str((tmp_path / "model").resolve()),
        },
    }
    if configured:
        assert baseline.read_bytes() == b"existing default result\n"
        files = list(tmp_path.glob("fp32-*.json"))
        assert len(files) == 1
        document = json.loads(files[0].read_text())
        assert document["decode_options"] == _json_settings(SETTINGS)
        assert document["summary"] == {
            "segments": 4,
            "text_characters": len(result.text),
            "most_common_segment": " Echo.",
            "most_common_segment_count": 2,
            "windows": 2,
        }
        assert document["runtime"] == legacy["runtime"]
    else:
        assert (
            baseline.read_bytes()
            == (json.dumps(legacy, ensure_ascii=False, indent=2) + "\n").encode()
        )
        assert not list(tmp_path.glob("fp32-*.json"))


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_longform_selected_silence_threshold_controls_which_windows_are_kept(
    longform_input, tmp_path, cpu_tensor_rt
):
    """The selected silence threshold retains windows that the default threshold would skip."""
    cpu_tensor_rt.no_speech_probability = 0.7
    settings = DecodingSettings(no_speech_threshold=0.8, logprob_threshold=None)

    evaluate_longform.fp32(tmp_path / "model", tmp_path, settings=settings)

    result = json.loads((tmp_path / f"{mode_name(settings, 'fp32')}.json").read_text())
    assert result["text"] == " Echo. Echo. End."
    assert result["summary"]["segments"] == 4
    assert result["summary"]["windows"] == 2
    assert cpu_tensor_rt.window_thresholds == [(0.8, None)]


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_longform_summary_uses_existing_records_and_handles_no_segments():
    """The default recording can be summarized without rerunning inference or changing its file."""
    document = {
        "text": "Repeat. Repeat.",
        "segments": [{"text": "Repeat."}, {"text": "Repeat."}, {"text": ""}],
        "windows": [{}, {}],
    }
    assert evaluate_longform.summarize_result(document) == {
        "segments": 3,
        "text_characters": 15,
        "most_common_segment": "Repeat.",
        "most_common_segment_count": 2,
        "windows": 2,
    }
    assert evaluate_longform.summarize_result({"text": "", "segments": [], "windows": []}) == {
        "segments": 0,
        "text_characters": 0,
        "most_common_segment": None,
        "most_common_segment_count": 0,
        "windows": 0,
    }


@pytest.fixture
def short_audio(tmp_path, monkeypatch, cpu_tensor_rt):
    audio_path = tmp_path / "audio.flac"
    sf.write(audio_path, np.zeros(16_000, dtype=np.int16), 16_000, subtype="PCM_16")
    manifest = [
        {
            "subset": subset,
            "id": subset,
            "audio": str(audio_path),
            "reference": "HELLO WORLD",
            "frames": 16_000,
        }
        for subset in evaluate_wer.SUBSETS
    ]
    evaluate_wer.write_csv(tmp_path / "manifest.csv", manifest, list(manifest[0]))
    monkeypatch.setattr(evaluate_wer, "EXPECTED_INCLUDED", dict.fromkeys(evaluate_wer.SUBSETS, 1))
    monkeypatch.setattr(
        evaluate_wer.subprocess, "check_output", lambda *_args, **_kwargs: "cpu test\n"
    )
    return cpu_tensor_rt.short_calls


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("configured", [False, True])
def test_wer_cli_passes_settings_and_keeps_default_csv_and_metadata(
    precision, configured, short_audio, tmp_path, monkeypatch, cpu_tensor_rt
):
    """Both precisions use the selected decoding settings and keep separate evaluation records."""
    baseline = tmp_path / f"{precision}.csv"
    if configured:
        baseline.write_bytes(b"existing default result\n")
        (tmp_path / f"{precision}.json").write_bytes(b"existing metadata\n")

    class Clock:
        @staticmethod
        def now(_timezone):
            return datetime(2026, 10, 11, tzinfo=UTC)

    monkeypatch.setattr(evaluate_wer, "datetime", Clock)
    monkeypatch.setattr(evaluate_wer.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_wer",
            precision,
            "--checkpoint-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path),
            *(OPTIONS if configured else []),
        ],
    )

    evaluate_wer.main()

    settings = SETTINGS if configured else DecodingSettings(temperatures=(0.0,), best_of=1)
    assert short_audio == [(precision, [], settings)] * 2
    assert cpu_tensor_rt.closed_engines == ["decoder", "encoder", "mel"]
    if configured:
        assert cpu_tensor_rt.greedy_calls == []
        assert cpu_tensor_rt.fallback_calls == [asdict(settings)] * 2
        assert cpu_tensor_rt.attempt_calls == [(0.0, 2, None)] * 2
    else:
        assert cpu_tensor_rt.greedy_calls == [precision] * 2
        assert cpu_tensor_rt.fallback_calls == []
        assert cpu_tensor_rt.attempt_calls == []
    if configured:
        assert baseline.read_bytes() == b"existing default result\n"
        assert (tmp_path / f"{precision}.json").read_bytes() == b"existing metadata\n"
        files = list(tmp_path.glob(f"{precision}-*.csv"))
        assert len(files) == 1
        path = files[0]
    else:
        path = baseline
    expected = (
        "subset,id,reference,output,normalized_reference,normalized_output,reference_words,"
        "substitutions,deletions,insertions,errors\r\n"
        'test-clean,test-clean,HELLO WORLD,"Hello, world!",hello world,hello world,2,0,0,0,0\r\n'
        'test-other,test-other,HELLO WORLD,"Hello, world!",hello world,hello world,2,0,0,0,0\r\n'
    )
    assert path.read_bytes() == expected.encode()
    metadata = json.loads(path.with_suffix(".json").read_text())
    legacy_keys = {
        "precision",
        "started_at",
        "finished_at",
        "elapsed_seconds",
        "engine_build_seconds",
        "checkpoint_dir",
        "checkpoint_sha256",
        "gpu",
        "driver",
        "tensorrt",
        "python",
    }
    assert set(metadata) == legacy_keys | ({"decode_options"} if configured else set())
    assert metadata["precision"] == precision
    assert metadata["engine_build_seconds"] == 0.25
    if configured:
        assert metadata["decode_options"] == _json_settings(SETTINGS)
    else:
        expected_metadata = {
            "precision": precision,
            "started_at": "2026-10-11T00:00:00+00:00",
            "finished_at": "2026-10-11T00:00:00+00:00",
            "elapsed_seconds": 0.0,
            "engine_build_seconds": 0.25,
            "checkpoint_dir": str((tmp_path / "model").resolve()),
            "checkpoint_sha256": {},
            "gpu": "cpu test",
            "driver": "cpu test",
            "tensorrt": _inference.trt.__version__,
            "python": platform.python_version(),
        }
        assert (
            path.with_suffix(".json").read_bytes()
            == (json.dumps(expected_metadata, indent=2) + "\n").encode()
        )


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_wer_passes_changed_thresholds_even_when_temperature_and_beam_are_default(
    precision, short_audio, tmp_path, monkeypatch
):
    """A threshold-only configuration reaches the decoder and is recorded separately."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_wer",
            precision,
            "--checkpoint-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path),
            "--no-speech-threshold",
            "none",
        ],
    )

    evaluate_wer.main()

    selected = DecodingSettings(temperatures=(0.0,), best_of=1, no_speech_threshold=None)
    assert short_audio == [(precision, [], selected)] * 2
    assert not (tmp_path / f"{precision}.csv").exists()
    metadata = json.loads(
        (tmp_path / f"{evaluate_wer.mode_name(selected, precision)}.json").read_text()
    )
    assert metadata["decode_options"] == _json_settings(selected)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_wer_retries_failed_windows_with_each_requested_temperature_and_best_of(
    short_audio, tmp_path, monkeypatch, cpu_tensor_rt
):
    """Failed quality checks retry each utterance at the requested temperatures and candidate count."""
    options = OPTIONS.copy()
    options[options.index("--compression-ratio-threshold") + 1] = "0"
    settings = replace(SETTINGS, compression_ratio_threshold=0.0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_wer",
            "fp16",
            "--checkpoint-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path),
            *options,
        ],
    )

    evaluate_wer.main()

    assert short_audio == [("fp16", [], settings)] * 2
    assert cpu_tensor_rt.fallback_calls == [asdict(settings)] * 2
    assert cpu_tensor_rt.attempt_calls == [(0.0, 2, None), (0.4, None, 3), (0.8, None, 3)] * 2
    # Identical sampled prefixes share their cached scores within each attempt.
    # The beam and two temperature attempts each execute the first generated word.
    for window in (0, 1):
        assert (
            sum(
                observed_window == window and prefix[-1] == 0
                for _precision, observed_window, prefix in cpu_tensor_rt.gpu_prefixes
            )
            == 3
        )
    rows = evaluate_wer.read_csv(tmp_path / f"{mode_name(settings)}.csv")
    assert [row["output"] for row in rows] == ["Hello, world!"] * 2
    assert [row["errors"] for row in rows] == ["0", "0"]


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_longform_retry_option_records_windows_and_their_decoder_context(
    longform_input, tmp_path, monkeypatch
):
    """A measured recording identifies retried windows and preserves the context of each window."""
    options = OPTIONS.copy()
    options[options.index("--compression-ratio-threshold") + 1] = "0"
    settings = replace(SETTINGS, compression_ratio_threshold=0.0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_longform",
            "fp32",
            "--checkpoint",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path),
            "--initial-prompt",
            "earnings",
            *options,
            "--record-retries",
        ],
    )

    evaluate_longform.main()

    document = json.loads((tmp_path / f"{mode_name(settings, 'fp32')}.json").read_text())
    measured = document["retry_statistics"]
    assert measured["units"] == 2
    assert measured["retried_units"] == 2
    assert measured["attempts"] == 6
    assert [record["attempts"] for record in measured["records"]] == [3, 3]
    assert [record["temperatures"] for record in measured["records"]] == [[0.0, 0.4, 0.8]] * 2
    assert [record["prompt_tokens"] for record in measured["records"]] == [[4], []]
    assert [record["tokens"] for record in measured["records"]] == [
        window["tokens"] for window in document["windows"]
    ]
    assert document["text"] == " Echo. Echo. End."
    assert document["summary"]["segments"] == 4


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("temperatures,attempts,retried", [("0,0.4,0.8", 3, 2), ("0.4", 1, 0)])
def test_wer_retry_option_counts_additional_attempts_per_utterance(
    temperatures, attempts, retried, short_audio, tmp_path, monkeypatch
):
    """A report counts repeated decoding, while a single positive-temperature attempt is not a retry."""
    options = OPTIONS.copy()
    options[options.index("--compression-ratio-threshold") + 1] = "0"
    options[options.index("--temperature") + 1] = temperatures
    settings = replace(
        SETTINGS,
        temperatures=tuple(float(value) for value in temperatures.split(",")),
        compression_ratio_threshold=0.0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_wer",
            "fp16",
            "--checkpoint-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path),
            *options,
            "--record-retries",
        ],
    )

    evaluate_wer.main()

    name = mode_name(settings)
    metadata = json.loads((tmp_path / f"{name}.json").read_text())
    measured = metadata["retry_statistics"]
    assert measured["units"] == 2
    assert measured["retried_units"] == retried
    assert measured["attempts"] == 2 * attempts
    assert [record["attempts"] for record in measured["records"]] == [attempts] * 2
    assert [record["temperatures"] for record in measured["records"]] == [
        list(settings.temperatures)
    ] * 2
    assert [record["prompt_tokens"] for record in measured["records"]] == [[], []]
    assert [row["output"] for row in evaluate_wer.read_csv(tmp_path / f"{name}.csv")] == [
        "Hello, world!",
        "Hello, world!",
    ]


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("difference", ["none", "tokens", "seek", "prompt", "retry"])
def test_unretried_longform_windows_are_compared_only_with_the_same_decoder_input(difference):
    """An unchanged window input requires the baseline tokens, while changed inputs and retries are separate."""
    baseline = {
        "initial_prompt": None,
        "segments": [{"seek": 0, "tokens": [10, 1, 20]}, {"seek": 0, "tokens": []}],
        "windows": [{"seek": 0, "tokens": [10, 1, 20]}, {"seek": 3000, "tokens": [10, 2, 30]}],
    }
    result = {
        "initial_prompt": None,
        "windows": [{"seek": 0, "tokens": [10, 1, 20]}, {"seek": 3000, "tokens": [10, 2, 30]}],
        "retry_statistics": {
            "records": [
                {"attempts": 1, "prompt_limit": 2, "prompt_tokens": []},
                {"attempts": 1, "prompt_limit": 2, "prompt_tokens": [1, 20]},
            ]
        },
    }
    if difference != "none":
        result["windows"][1]["tokens"] = [10, 3, 30]
    if difference == "seek":
        result["windows"][1]["seek"] = 3200
    elif difference == "prompt":
        result["retry_statistics"]["records"][1]["prompt_tokens"] = [3, 20]
    elif difference == "retry":
        result["retry_statistics"]["records"][1]["attempts"] = 2

    compared = evaluate_longform.compare_unretried_windows(baseline, result)

    assert compared["passed"] == (difference != "tokens")
    assert compared["checked_windows"] == (2 if difference in {"none", "tokens"} else 1)
    assert compared["input_different_windows"] == int(difference in {"seek", "prompt"})
    assert compared["retried_windows"] == int(difference == "retry")
    assert compared["changed_windows"] == int(difference == "tokens")
    if difference == "tokens":
        assert compared["first_difference"] == {
            "window_index": 1,
            "seek": 3000,
            "baseline_tokens": [10, 2, 30],
            "tokens": [10, 3, 30],
        }
    else:
        assert compared["first_difference"] is None


def _report_inputs(tmp_path, precisions=("fp16",), settings=SETTINGS):
    manifest = [
        {"subset": "test-clean", "id": "a", "frames": 1, "reference": "HELLO WORLD"},
        {"subset": "test-clean", "id": "b", "frames": 1, "reference": "SAME WORDS"},
        {"subset": "test-other", "id": "c", "frames": 1, "reference": "GOOD MORNING"},
        {"subset": "test-clean", "id": "excluded", "frames": 480_001, "reference": "IGNORED"},
    ]
    evaluate_wer.write_csv(tmp_path / "manifest.csv", manifest, list(manifest[0]))
    default_outputs = ["hello there", "same words", "good morning"]
    selected_outputs = ["Hello, world!", "SAME WORDS", "good"]
    for precision in ("fp16", "fp32"):
        rows = [
            {
                "subset": row["subset"],
                "id": row["id"],
                "reference": row["reference"],
                "output": output,
                **evaluate_wer.word_errors(row["reference"], output),
            }
            for row, output in zip(manifest, default_outputs)
        ]
        evaluate_wer.write_csv(tmp_path / f"{precision}.csv", rows, list(rows[0]))
        (tmp_path / f"{precision}.json").write_text(
            json.dumps({"checkpoint_sha256": {"model": "same"}})
        )
        if precision in precisions:
            configured = [
                dict(
                    row,
                    output=output,
                    **evaluate_wer.word_errors(row["reference"], output),
                )
                for row, output in zip(rows, selected_outputs)
            ]
            name = evaluate_wer.mode_name(settings, precision)
            evaluate_wer.write_csv(tmp_path / f"{name}.csv", configured, list(configured[0]))
            (tmp_path / f"{name}.json").write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": {"model": "same"},
                        "decode_options": _json_settings(settings),
                    }
                )
            )


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("precisions", [("fp16",), ("fp32",), ("fp16", "fp32")])
def test_wer_report_compares_available_settings_to_each_default_on_the_same_utterances(
    precisions, tmp_path, monkeypatch
):
    """The report recomputes normalized WER, counts changed transcripts, and preserves the default report."""
    _report_inputs(tmp_path, precisions)
    evaluate_wer.report(tmp_path)
    baseline_report = (tmp_path / "report.json").read_bytes()
    baseline = json.loads(baseline_report)
    assert baseline["passed"]
    assert baseline["subsets"]["test-clean"]["fp16_errors"] == 1
    assert baseline["subsets"]["test-other"]["fp32_errors"] == 0
    expected_default = {
        "subsets": {
            "test-clean": {
                "utterances": 2,
                "reference_words": 4,
                "fp16_errors": 1,
                "fp32_errors": 1,
                "fp16_wer": 0.25,
                "fp32_wer": 0.25,
                "difference_points": 0.0,
                "changed_outputs": 0,
                "passed": True,
            },
            "test-other": {
                "utterances": 1,
                "reference_words": 2,
                "fp16_errors": 0,
                "fp32_errors": 0,
                "fp16_wer": 0.0,
                "fp32_wer": 0.0,
                "difference_points": 0.0,
                "changed_outputs": 0,
                "passed": True,
            },
        },
        "passed": True,
        "excluded_ids": {"test-clean": ["excluded"], "test-other": []},
        "fp16": {"checkpoint_sha256": {"model": "same"}},
        "fp32": {"checkpoint_sha256": {"model": "same"}},
    }
    assert baseline_report == (json.dumps(expected_default, indent=2) + "\n").encode()
    monkeypatch.setattr(
        sys, "argv", ["evaluate_wer", "report", "--output-dir", str(tmp_path), *OPTIONS]
    )

    evaluate_wer.main()

    assert (tmp_path / "report.json").read_bytes() == baseline_report
    path = tmp_path / f"{evaluate_wer.mode_name(SETTINGS, 'report')}.json"
    report = json.loads(path.read_text())
    assert report["decode_options"] == _json_settings(SETTINGS)
    assert report["excluded_ids"] == {"test-clean": ["excluded"], "test-other": []}
    assert set(report["comparisons"]) == set(precisions)
    for precision in precisions:
        scores = report["comparisons"][precision]["subsets"]
        assert scores["test-clean"] == {
            "utterances": 2,
            "reference_words": 4,
            "baseline_errors": 1,
            "errors": 0,
            "baseline_wer": 0.25,
            "wer": 0.0,
            "error_difference": -1,
            "difference_points": -25.0,
            "changed_outputs": 2,
        }
        assert scores["test-other"] == {
            "utterances": 1,
            "reference_words": 2,
            "baseline_errors": 0,
            "errors": 1,
            "baseline_wer": 0.0,
            "wer": 0.5,
            "error_difference": 1,
            "difference_points": 50.0,
            "changed_outputs": 1,
        }
    assert ("subsets" in report) == (len(precisions) == 2)
    if len(precisions) == 2:
        assert report["passed"]
        assert report["subsets"]["test-other"]["fp16_errors"] == 1


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("defect", ["missing", "duplicate", "excluded", "reference", "checkpoint"])
def test_configured_wer_report_rejects_a_run_that_cannot_be_compared_to_the_default(
    defect, tmp_path
):
    """A setting comparison requires the same included utterances, references, and checkpoint."""
    _report_inputs(tmp_path)
    path = tmp_path / f"{evaluate_wer.mode_name(SETTINGS)}.csv"
    rows = evaluate_wer.read_csv(path)
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(rows[0])
    elif defect == "excluded":
        rows.append(dict(rows[0], id="excluded"))
    elif defect == "reference":
        rows[0]["reference"] = "CHANGED"
    else:
        path.with_suffix(".json").write_text(
            json.dumps({"checkpoint_sha256": {"model": "different"}})
        )
    path.unlink()
    evaluate_wer.write_csv(path, rows, list(rows[0]))

    with pytest.raises(ValueError, match="manifest|reference|checkpoint"):
        evaluate_wer.report(tmp_path, settings=SETTINGS)
    assert not list(tmp_path.glob("report-*.json"))


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("script,phase", [(evaluate_longform, "fp32"), (evaluate_wer, "fp16")])
@pytest.mark.parametrize(
    "option,value",
    [
        ("--beam-size", "0"),
        ("--best-of", "0"),
        ("--temperature", "0,nan"),
        ("--compression-ratio-threshold", "inf"),
        ("--logprob-threshold", "bad"),
        ("--no-speech-threshold", "nan"),
    ],
)
def test_evaluation_cli_rejects_invalid_decode_settings_before_reading_audio(
    script, phase, option, value, tmp_path, monkeypatch, capsys
):
    """Invalid decoding values fail as option errors before inference begins."""
    checkpoint_option = "--checkpoint" if script is evaluate_longform else "--checkpoint-dir"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            phase,
            checkpoint_option,
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path),
            option,
            value,
        ],
    )
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
    assert "unrecognized arguments" not in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
