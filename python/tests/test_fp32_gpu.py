"""Opt-in comparisons against the fixed official fp32 large-v3 oracle."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scripts.generate_oracle import read_pcm16_mono

from listenflow.fastest_whisper import _inference, transcribe_fp32, transcribe_long_fp32
from listenflow.fastest_whisper._audio import log_mel_spectrogram
from listenflow.fastest_whisper._decoding import DecodingSettings
from listenflow.fastest_whisper._inference import TensorRTWhisper

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def model():
    checkpoint = os.environ.get("LFFW_CHECKPOINT_DIR")
    if os.environ.get("LFFW_USE_GPU") != "1" or not checkpoint:
        pytest.skip("set LFFW_USE_GPU=1 and LFFW_CHECKPOINT_DIR to run GPU comparisons")
    with TensorRTWhisper(Path(checkpoint)) as instance:
        yield instance


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("name", ["small_boat_16khz_mono_s16le", "silence_16khz_mono_s16le"])
def test_fp32_engine_matches_official_intermediates_and_greedy_tokens(model, name):
    """TensorRT reproduces the official log-mel, encoder, raw logits, and token sequence."""
    samples = read_pcm16_mono(FIXTURES / f"{name}.wav")
    with np.load(FIXTURES / "oracle" / f"{name}.npz") as oracle:
        expected_mel = oracle["mel"]
        expected_encoder = oracle["encoder"]
        expected_logits = oracle["logits"]
        initial = oracle["initial_tokens"].tolist()
        expected_tokens = oracle["greedy_tokens"].tolist()

    mel = log_mel_spectrogram(samples, n_mels=model.config.n_mels)
    encoder = model.encode(mel)
    raw_logits = model.logits(initial + expected_tokens[:-1], encoder)
    greedy = model.greedy_decode(encoder)

    np.testing.assert_allclose(mel, expected_mel, atol=3e-5, rtol=3e-5)
    # CPU/GPU fp32 reduction differences accumulate, especially in the padded silence.
    np.testing.assert_allclose(encoder, expected_encoder, atol=2e-2, rtol=5e-3)
    np.testing.assert_allclose(
        raw_logits[len(initial) - 1 :], expected_logits, atol=2e-2, rtol=2e-2
    )
    assert greedy.tokens == expected_tokens


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_public_fp32_entry_point_returns_text_and_token_ids():
    """A caller can transcribe a supplied 16 kHz waveform with a supplied checkpoint."""
    checkpoint = os.environ.get("LFFW_CHECKPOINT_DIR")
    if os.environ.get("LFFW_USE_GPU") != "1" or not checkpoint:
        pytest.skip("set LFFW_USE_GPU=1 and LFFW_CHECKPOINT_DIR to run GPU comparisons")
    samples = read_pcm16_mono(FIXTURES / "silence_16khz_mono_s16le.wav")
    with np.load(FIXTURES / "oracle" / "silence_16khz_mono_s16le.npz") as oracle:
        expected_tokens = oracle["greedy_tokens"].tolist()

    result = transcribe_fp32(Path(checkpoint), samples, temperature=0.0, best_of=1)

    assert result.tokens == expected_tokens
    assert result.text == "you"


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-009")
@pytest.mark.parametrize(
    "case",
    [
        "long_hint/small_boat_16khz_mono_s16le",
        "long_hint/silence_16khz_mono_s16le",
        "one_word_hint/small_boat_16khz_mono_s16le",
        "one_word_hint/silence_16khz_mono_s16le",
    ],
)
def test_prompted_short_decode_matches_official_tokens(model, case):
    """Long and one-word hints reproduce the official fp32 tokens for each input."""
    oracle = json.loads((FIXTURES / "oracle" / "initial_prompt.json").read_text())["cases"][case]
    name = case.split("/")[1]
    samples = read_pcm16_mono(FIXTURES / f"{name}.wav")
    mel = log_mel_spectrogram(samples, n_mels=model.config.n_mels)

    actual = model.greedy_decode(model.encode(mel), model.prompt_tokens(oracle["prompt"]))

    assert actual.tokens == oracle["tokens"]
    assert actual.text == oracle["text"]


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_public_short_entry_passes_initial_prompt_to_decoder(model, monkeypatch):
    """The public fp32 entry uses the hint while retaining the same token-level oracle."""
    from contextlib import nullcontext

    case = json.loads((FIXTURES / "oracle" / "initial_prompt.json").read_text())["cases"][
        "long_hint/silence_16khz_mono_s16le"
    ]
    samples = read_pcm16_mono(FIXTURES / "silence_16khz_mono_s16le.wav")
    monkeypatch.setattr(_inference, "TensorRTWhisper", lambda *_args, **_kwargs: nullcontext(model))

    actual = transcribe_fp32(
        model.checkpoint_dir, samples, initial_prompt=case["prompt"], temperature=0.0, best_of=1
    )

    assert actual.tokens == case["tokens"]


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_public_long_entry_with_hint_matches_official_segments(model, monkeypatch):
    """The public long-form entry passes the hint across windows and excludes it from text."""
    from contextlib import nullcontext

    oracle = json.loads(
        (FIXTURES / "oracle" / "initial_prompt_long_small_boat_repeat.json").read_text()
    )
    samples = np.tile(read_pcm16_mono(FIXTURES / "small_boat_16khz_mono_s16le.wav"), 3)
    monkeypatch.setattr(_inference, "TensorRTWhisper", lambda *_args, **_kwargs: nullcontext(model))

    actual = transcribe_long_fp32(
        model.checkpoint_dir, samples, initial_prompt=oracle["prompt"], temperature=0.0, best_of=1
    )

    assert actual.text == oracle["text"]
    assert [
        {"start": segment.start, "end": segment.end, "text": segment.text}
        for segment in actual.segments
    ] == oracle["segments"]


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("name", ["small_boat_16khz_mono_s16le", "silence_16khz_mono_s16le"])
def test_fp32_beam_two_matches_official_generated_tokens(model, name):
    """A width-two beam reproduces the official CPU fp32 token sequence for both fixtures."""
    oracle = json.loads((FIXTURES / "oracle" / "beam_size_2.json").read_text())["fixtures"][name]
    samples = read_pcm16_mono(FIXTURES / f"{name}.wav")
    mel = log_mel_spectrogram(samples, n_mels=model.config.n_mels)

    actual = model.decode_short(
        model.encode(mel), [], DecodingSettings(beam_size=2, temperatures=(0.0,), best_of=1)
    )

    assert actual.tokens == oracle["tokens_without_eot"] + [oracle["eot"]]
    assert actual.text == oracle["text"]


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.req("SP-LFFASTESTWHISPER-009")
@pytest.mark.parametrize("name", ["small_boat_16khz_mono_s16le", "silence_16khz_mono_s16le"])
def test_fp32_beam_with_hint_matches_official_generated_tokens(model, name):
    """A hinted beam uses the official previous-text context and matches its fp32 tokens."""
    oracle = json.loads((FIXTURES / "oracle" / "beam_with_prompt.json").read_text())
    case = oracle["fixtures"][name]
    samples = read_pcm16_mono(FIXTURES / f"{name}.wav")
    mel = log_mel_spectrogram(samples, n_mels=model.config.n_mels)

    actual = model.decode_short(
        model.encode(mel),
        model.prompt_tokens(oracle["decode"]["initial_prompt"]),
        DecodingSettings(beam_size=2, temperatures=(0.0,), best_of=1),
    )

    assert actual.tokens == case["tokens_without_eot"] + [case["eot"]]
    assert actual.text == case["text"]


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_prompted_long_beam_matches_official_segments_and_window_tokens(model, monkeypatch):
    """A hinted beam carries the official previous text into each long-form window."""
    from contextlib import nullcontext

    oracle = json.loads((FIXTURES / "oracle" / "beam_long_with_prompt.json").read_text())
    samples = np.tile(read_pcm16_mono(FIXTURES / "small_boat_16khz_mono_s16le.wav"), 3)
    monkeypatch.setattr(_inference, "TensorRTWhisper", lambda *_args, **_kwargs: nullcontext(model))
    windows = []
    prompts = []
    original_decode_long = model.decode_long

    def record_decode_long(encoder, prompt, settings):
        prompts.append(prompt.copy())
        return original_decode_long(encoder, prompt, settings)

    monkeypatch.setattr(model, "decode_long", record_decode_long)

    actual = _inference._transcribe_long_fp32(
        model.checkpoint_dir,
        samples,
        initial_prompt=oracle["prompt"],
        settings=DecodingSettings(beam_size=2, temperatures=(0.0,), best_of=1),
        trace=windows.append,
    )

    assert actual.text == oracle["text"]
    assert [
        {"start": segment.start, "end": segment.end, "text": segment.text}
        for segment in actual.segments
    ] == oracle["segments"]
    assert [window.tokens for window in windows] == [
        call["tokens"] for call in oracle["decode_calls"]
    ]
    assert prompts == [call["prompt"] for call in oracle["decode_calls"]]
