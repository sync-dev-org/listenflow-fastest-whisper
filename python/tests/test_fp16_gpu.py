"""Opt-in inference through the fp16 TensorRT entry."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scripts.generate_oracle import read_pcm16_mono

from listenflow.fastest_whisper import transcribe_fp16
from listenflow.fastest_whisper._audio import log_mel_spectrogram
from listenflow.fastest_whisper._decoding import DecodingSettings
from listenflow.fastest_whisper._inference import TensorRTWhisper


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-004")
def test_fp16_entry_transcribes_a_supplied_waveform_with_a_supplied_checkpoint():
    """The fp16 route builds on the caller's GPU and returns decoded text and tokens."""
    checkpoint = os.environ.get("LFFW_CHECKPOINT_DIR")
    if os.environ.get("LFFW_USE_GPU") != "1" or not checkpoint:
        pytest.skip("set LFFW_USE_GPU=1 and LFFW_CHECKPOINT_DIR to run GPU comparisons")
    samples = read_pcm16_mono(Path(__file__).parent / "fixtures" / "silence_16khz_mono_s16le.wav")

    result = transcribe_fp16(Path(checkpoint), samples)

    assert result.text == "you"
    assert result.tokens


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_fp16_model_runs_beam_and_sampled_settings_from_one_loaded_checkpoint():
    """Both selected fp16 decode modes finish on the same loaded model without a quality gate."""
    checkpoint = os.environ.get("LFFW_CHECKPOINT_DIR")
    if os.environ.get("LFFW_USE_GPU") != "1" or not checkpoint:
        pytest.skip("set LFFW_USE_GPU=1 and LFFW_CHECKPOINT_DIR to run GPU comparisons")
    samples = read_pcm16_mono(Path(__file__).parent / "fixtures" / "silence_16khz_mono_s16le.wav")

    with TensorRTWhisper(Path(checkpoint), precision="fp16") as model:
        mel = log_mel_spectrogram(samples, n_mels=model.config.n_mels)
        encoder = model.encode(mel)
        beam = model.decode_short(encoder, [], DecodingSettings(beam_size=2))
        sampled = model.decode_short(encoder, [], DecodingSettings(temperatures=(0.2,), best_of=2))

    assert beam.tokens[-1] == model.eot
    assert sampled.tokens[-1] == model.eot


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_default_first_attempt_matches_greedy_in_the_same_engine_and_scores_its_tokens(precision):
    """An accepted default attempt preserves greedy tokens and computes retry scores on the device."""
    checkpoint = os.environ.get("LFFW_CHECKPOINT_DIR")
    if os.environ.get("LFFW_USE_GPU") != "1" or not checkpoint:
        pytest.skip("set LFFW_USE_GPU=1 and LFFW_CHECKPOINT_DIR to run GPU comparisons")
    samples = read_pcm16_mono(
        Path(__file__).parent / "fixtures" / "small_boat_16khz_mono_s16le.wav"
    )
    attempts = []

    def observe(frame, event, _argument):
        if event == "call" and frame.f_code is TensorRTWhisper._decode_attempt.__code__:
            attempts.append(frame.f_locals["temperature"])

    with TensorRTWhisper(Path(checkpoint), precision=precision) as model:
        encoder = model.encode(model._mel_spectrogram(samples))
        expected = model.decode_short(encoder, [], DecodingSettings(temperatures=(0.0,), best_of=1))
        previous_profile = sys.getprofile()
        sys.setprofile(observe)
        try:
            actual = model.decode_short(encoder, [], DecodingSettings())
        finally:
            sys.setprofile(previous_profile)
        assert actual == expected
        assert attempts == [0.0]
        measured = model._decode_attempt(
            encoder, [], with_timestamps=False, temperature=0.0, beam_size=None, best_of=None
        )
        # A sequential float32 logaddexp reduction loses about 1e-3 across
        # Whisper's 51k vocabulary. Use float64 for the mathematical reference.
        raw = model.logits(model.initial_tokens + actual.tokens[:-1], encoder).astype(np.float64)
        speech = raw[0]
        no_speech = model.tokenizer.token_to_id("<|nospeech|>")
        expected_silence = np.exp(speech[no_speech] - np.logaddexp.reduce(speech))
        score = 0.0
        for index, token in enumerate(actual.tokens):
            logits = raw[len(model.initial_tokens) - 1 + index].copy()
            logits[list(model.suppressed)] = -np.inf
            if index == 0:
                logits[list(model.begin_suppressed)] = -np.inf
            score += logits[token] - np.logaddexp.reduce(logits)
        assert measured.avg_logprob == pytest.approx(float(score) / len(actual.tokens), abs=2e-4)
        assert measured.no_speech_prob == pytest.approx(float(expected_silence), abs=2e-5)
