"""Bounded decoder graphs preserve one engine's history across calls and audio windows."""

import os
from pathlib import Path

import numpy as np
import pytest
from scripts.generate_oracle import read_pcm16_mono

from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._decoding import DecodingSettings

FIXTURES = Path(__file__).parent / "fixtures"
WAVEFORM = "small_boat_16khz_mono_s16le.wav"
HINT = " acoustic interactive content recognition" * 100


@pytest.fixture(scope="module")
def model():
    checkpoint = os.environ.get("LFFW_CHECKPOINT_DIR")
    if os.environ.get("LFFW_USE_GPU") != "1" or not checkpoint:
        pytest.skip("set LFFW_USE_GPU=1 and LFFW_CHECKPOINT_DIR for GPU checks")
    with CachedWhisper(Path(checkpoint), precision="fp16", device=0) as model:
        yield model


def encoded(model):
    samples = read_pcm16_mono(FIXTURES / WAVEFORM)
    return model.encode(model._mel_spectrogram(samples))


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_boundary_lengths_preserve_previous_tokens_with_at_most_sixteen_graphs(model):
    """Prefixes before and after every tested interval agree with the same engine's complete prefix."""
    encoder = encoded(model)
    prefix = (model.initial_tokens + [198] * 448)[:448]
    expected = model.logits(prefix, encoder).copy()
    encoder = encoded(model)
    for length in (1, 7, 8, 9, 15, 16, 17, 447, 448):
        logits = model.logits(prefix[:length], encoder)
        np.testing.assert_array_equal(logits[-1], expected[length - 1])
        assert len(model._decoder._graphs.entries) <= 16


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_four_hundred_token_hint_preserves_previous_tokens_and_bounded_graphs(model):
    """A 400-token hint agrees with its retained recent context with bounded work."""
    prompt = model.prompt_tokens(HINT)
    assert len(prompt) == 400
    actual = model.greedy_decode(encoded(model), prompt)
    recent = prompt[-(model.config.text_positions // 2 - 1) :]
    expected = model.greedy_decode(encoded(model), recent)
    assert actual.tokens == expected.tokens
    assert len(model._decoder._graphs.entries) <= 16


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_beam_paths_preserve_previous_tokens_and_bounded_graphs(model):
    """Width-two beam histories keep their tokens after a different audio window uses shared work."""
    actual = model.decode_short(encoded(model), [], DecodingSettings(beam_size=2))
    model.greedy_decode(model.encode(model._mel_spectrogram(np.zeros(16_000, np.float32))))
    expected = model.decode_short(encoded(model), [], DecodingSettings(beam_size=2))
    assert actual.tokens == expected.tokens
    assert len(model._decoder._graphs.entries) <= 16


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_temperature_retry_preserves_previous_tokens_and_bounded_graphs(model, monkeypatch):
    """A seeded retry at temperature 0.2 reproduces its sampled tokens with reused work."""
    factory = np.random.default_rng
    # Replace only OS entropy; sampling still uses the real NumPy Generator.
    monkeypatch.setattr(np.random, "default_rng", lambda: factory(42))
    actual = model.decode_short(
        encoded(model),
        [],
        DecodingSettings(
            temperatures=(0.0, 0.2),
            logprob_threshold=100,
            no_speech_threshold=None,
            compression_ratio_threshold=None,
        ),
    )
    expected = model.decode_short(
        encoded(model),
        [],
        DecodingSettings(
            temperatures=(0.0, 0.2),
            logprob_threshold=100,
            no_speech_threshold=None,
            compression_ratio_threshold=None,
        ),
    )
    assert actual.tokens == expected.tokens
    assert len(model._decoder._graphs.entries) <= 16


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_long_recording_reuses_bounded_graphs_and_preserves_each_windows_tokens(model, monkeypatch):
    """Successive windows reproduce timestamp tokens and segment boundaries across calls."""
    factory = np.random.default_rng
    # Keep the retry policy real while comparing the same sampled candidates.
    monkeypatch.setattr(np.random, "default_rng", lambda: factory(42))
    samples = np.tile(read_pcm16_mono(FIXTURES / WAVEFORM), 3)
    assert samples.size > 480_000
    windows = []
    original = model.decode_long

    def decode(encoder, prompt, settings):
        result = original(encoder, prompt, settings)
        windows.append(result.tokens)
        assert len(model._decoder._graphs.entries) <= 16
        return result

    model.decode_long = decode
    try:
        segments = model.transcribe(samples)
    finally:
        del model.decode_long
    assert len(windows) >= 2
    expected_windows = []

    def repeat(encoder, prompt, settings):
        result = original(encoder, prompt, settings)
        expected_windows.append(result.tokens)
        return result

    model.decode_long = repeat
    try:
        expected = model.transcribe(samples)
    finally:
        del model.decode_long
    assert windows == expected_windows
    assert segments == expected
