"""CPU checks of the official long-form window and timestamp behavior."""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from listenflow.fastest_whisper import LongTranscription, TimedSegment, transcribe_long_fp32
from listenflow.fastest_whisper._audio import log_mel_spectrogram_long
from listenflow.fastest_whisper._longform import (
    DecodedWindow,
    apply_timestamp_rules,
    decoder_prefix,
    transcribe_mel,
)


def _text(tokens):
    return "".join({1: " A", 2: " B", 3: " C", 4: " D"}.get(token, "") for token in tokens)


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_public_long_result_contains_only_text_and_timed_segments():
    """Window trace data stays inside correctness evaluation rather than the public result."""
    assert [field.name for field in fields(LongTranscription)] == ["text", "segments"]
    assert [field.name for field in fields(TimedSegment)] == ["start", "end", "text"]


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("audio", [np.zeros((2, 2)), np.array([np.nan])])
def test_long_entry_rejects_invalid_waveforms_before_loading_a_checkpoint(audio):
    """The long transcription entry rejects non-mono and nonfinite audio."""
    with pytest.raises(ValueError, match="audio must be"):
        transcribe_long_fp32("missing-checkpoint", audio)


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_long_mel_uses_one_global_loudness_scale_and_30_seconds_of_padding():
    """The entire waveform, including a later loud sound, sets the mel scale for early frames."""
    samples = np.zeros(480_160, dtype=np.float32)
    frequency = np.float32(2 * np.pi * 440 / 16_000)
    samples[:1_600] = 0.01 * np.sin(np.arange(1_600, dtype=np.float32) * frequency)
    samples[-160:] = 0.8 * np.sin(np.arange(160, dtype=np.float32) * frequency)

    mel = log_mel_spectrogram_long(samples, n_mels=128)

    # Fixed from openai-whisper 20250625 log_mel_spectrogram(..., padding=480000).
    assert mel.shape == (128, 6_001)
    for band, frame, expected in [
        (0, 0, 0.0580345392),
        (10, 10, -0.0141891241),
        (64, 10, -0.5606595278),
        (100, 3000, 0.0468112230),
        (64, 3001, 0.5720999241),
        (10, 5999, -0.5606595278),
    ]:
        assert mel[band, frame] == pytest.approx(expected, abs=3e-5)


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("audio", [np.zeros((2, 2)), np.array([np.nan])])
def test_long_mel_rejects_non_mono_or_nonfinite_input(audio):
    """Long audio must be a finite mono waveform before feature extraction."""
    with pytest.raises(ValueError, match="audio must be"):
        log_mel_spectrogram_long(audio, n_mels=128)


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_unfinished_window_restarts_at_last_confirmed_timestamp_and_keeps_prompt():
    """Three windows retain confirmed text and timestamps in the next window's prompt."""
    mel = np.broadcast_to(np.arange(10_000, dtype=np.float32), (1, 10_000))
    decoded = [
        DecodedWindow([100, 1, 1500, 1500, 2], -0.2, 0.1),
        DecodedWindow([100, 3, 200], -0.2, 0.1),
        DecodedWindow([100, 4, 150], -0.2, 0.1),
    ]
    seen = []
    traces = []

    def decode(window, prompt):
        seen.append((int(window[0, 0]), prompt.copy()))
        return decoded[len(seen) - 1]

    result = transcribe_mel(
        mel,
        content_frames=7_000,
        decode=decode,
        trace=traces.append,
        text=_text,
        timestamp_begin=100,
        eot=90,
        audio_positions=1_500,
    )

    assert seen == [
        (0, []),
        (2_800, [100, 1, 1500]),
        (5_800, [100, 1, 1500, 100, 3, 200]),
    ]
    assert result.text == " A C D"
    assert [(s.start, s.end, s.text) for s in result.segments] == [
        (0.0, 28.0, " A"),
        (28.0, 30.0, " C"),
        (58.0, 59.0, " D"),
    ]
    assert [(trace.seek, trace.next_seek) for trace in traces] == [
        (0, 2_800),
        (2_800, 5_800),
        (5_800, 7_000),
    ]


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_no_speech_threshold_is_strict_and_good_logprob_overrides_it():
    """The official no-speech gate skips only high probability with low log probability."""
    mel = np.zeros((1, 9_001), dtype=np.float32)
    answers = [
        DecodedWindow([100, 1, 150], -1.0, 0.6),
        DecodedWindow([100, 2, 150], -1.0, 0.6001),
        DecodedWindow([100, 3, 150], -0.9, 0.6001),
    ]
    prompts = []

    def decode(_window, prompt):
        prompts.append(prompt.copy())
        return answers[len(prompts) - 1]

    result = transcribe_mel(
        mel,
        content_frames=6_001,
        decode=decode,
        text=_text,
        timestamp_begin=100,
        eot=90,
        audio_positions=1_500,
    )

    assert result.text == " A C"
    assert [(s.start, s.end) for s in result.segments] == [(0.0, 1.0), (60.0, 61.0)]
    assert prompts == [[], [100, 1, 150], [100, 1, 150]]


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_empty_segments_remain_but_do_not_become_prompt():
    """Empty segments keep their positions and are excluded from the next prompt."""
    mel = np.zeros((1, 6_001), dtype=np.float32)
    answers = [
        DecodedWindow([100], -0.2, 0.1),
        DecodedWindow([100, 2, 150], -0.2, 0.1),
    ]
    prompts = []
    segment_traces = []

    def decode(_window, prompt):
        prompts.append(prompt.copy())
        return answers[len(prompts) - 1]

    result = transcribe_mel(
        mel,
        content_frames=3_001,
        decode=decode,
        segment_trace=segment_traces.append,
        text=_text,
        timestamp_begin=100,
        eot=90,
        audio_positions=1_500,
    )

    assert [(s.start, s.end, s.text) for s in result.segments] == [
        (0.0, 30.0, ""),
        (30.0, 31.0, " B"),
    ]
    assert [trace.tokens for trace in segment_traces] == [[], [100, 2, 150]]
    assert prompts == [[], []]
    assert result.text == " B"


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_timestamp_logits_start_within_one_second_and_pair_without_going_backwards():
    """Timestamp decoding starts by 1 s, pairs timestamps, and cannot move backward."""
    logits = np.arange(200, dtype=np.float32)
    first = apply_timestamp_rules(logits.copy(), [], timestamp_begin=100, eot=90)
    assert np.isneginf(first[:100]).all()
    assert np.isneginf(first[151:]).all()
    assert np.isfinite(first[100:151]).all()

    after_text = apply_timestamp_rules(logits.copy(), [100, 10], timestamp_begin=100, eot=90)
    assert np.isneginf(after_text[:90]).all()
    assert np.isneginf(after_text[100:101]).all()
    assert np.isfinite(after_text[101:]).all()

    after_pair = apply_timestamp_rules(
        logits.copy(), [100, 10, 110, 110, 1], timestamp_begin=100, eot=90
    )
    assert np.isneginf(after_pair[100:111]).all()
    assert np.isfinite(after_pair[111:]).all()


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_timestamp_mass_can_outweigh_the_best_text_token():
    """A set of likely timestamp tokens wins even when each is weaker than one text token."""
    logits = np.full(200, -100.0, dtype=np.float32)
    logits[10] = 1.0
    logits[120] = 0.5
    logits[121] = 0.5
    actual = apply_timestamp_rules(logits, [100, 1], timestamp_begin=100, eot=90)
    assert np.isneginf(actual[:100]).all()
    assert np.isfinite(actual[120:122]).all()


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_prompt_keeps_recent_tokens_including_timestamps_within_half_context():
    """The next window uses the newest 223 previous tokens and their timestamp IDs."""
    prompt = list(range(400)) + [50364]
    prefix = decoder_prefix(
        prompt,
        initial_tokens=[50258, 50259, 50359],
        previous_token=50361,
        text_positions=448,
    )
    assert prefix == [50361] + prompt[-223:] + [50258, 50259, 50359]
    assert decoder_prefix(
        [], initial_tokens=[50258, 50259, 50359], previous_token=50361, text_positions=448
    ) == [50258, 50259, 50359]


@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_initial_prompt_precedes_recognized_text_without_becoming_transcription():
    """The first window gets the hint, later windows get the hint before confirmed speech."""
    mel = np.zeros((1, 9_001), dtype=np.float32)
    answers = [
        DecodedWindow([100, 1, 150], -0.2, 0.1),
        DecodedWindow([100, 2, 150], -0.2, 0.1),
        DecodedWindow([100, 3, 150], -0.2, 0.1),
    ]
    prompts = []

    def decode(_window, prompt):
        prompts.append(prompt.copy())
        return answers[len(prompts) - 1]

    result = transcribe_mel(
        mel,
        content_frames=6_001,
        decode=decode,
        text=_text,
        timestamp_begin=100,
        eot=90,
        audio_positions=1_500,
        initial_prompt_tokens=[4, 4],
    )

    assert prompts == [
        [4, 4],
        [4, 4, 100, 1, 150],
        [4, 4, 100, 1, 150, 100, 2, 150],
    ]
    assert result.text == " A B C"
    assert [segment.text for segment in result.segments] == [" A", " B", " C"]


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_hot_retry_clears_prior_text_and_hint_before_the_next_window():
    """A window accepted above temperature 0.5 resets the following decoder prompt."""
    mel = np.zeros((1, 9_001), dtype=np.float32)
    answers = [
        DecodedWindow([100, 1, 150], -0.2, 0.1, temperature=0.6),
        DecodedWindow([100, 2, 150], -0.2, 0.1),
        DecodedWindow([100, 3, 150], -0.2, 0.1),
    ]
    prompts = []

    def decode(_window, prompt):
        prompts.append(prompt.copy())
        return answers[len(prompts) - 1]

    result = transcribe_mel(
        mel,
        content_frames=6_001,
        decode=decode,
        text=_text,
        timestamp_begin=100,
        eot=90,
        audio_positions=1_500,
        initial_prompt_tokens=[4],
    )

    assert prompts == [[4], [], [100, 2, 150]]
    assert result.text == " A B C"
    assert [segment.text for segment in result.segments] == [" A", " B", " C"]


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_skipped_silent_window_does_not_enter_the_next_prompt():
    """A skipped window keeps its predecessor's context and contributes no segment."""
    mel = np.zeros((1, 9_001), dtype=np.float32)
    answers = [
        DecodedWindow([100, 1, 150], -0.2, 0.1),
        DecodedWindow([100, 2, 150], -1.1, 0.7, temperature=0.4),
        DecodedWindow([100, 3, 150], -0.2, 0.1),
    ]
    prompts = []

    def decode(_window, prompt):
        prompts.append(prompt.copy())
        return answers[len(prompts) - 1]

    result = transcribe_mel(
        mel,
        content_frames=6_001,
        decode=decode,
        text=_text,
        timestamp_begin=100,
        eot=90,
        audio_positions=1_500,
        initial_prompt_tokens=[4],
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
    )

    assert prompts == [[4], [4, 100, 1, 150], [4, 100, 1, 150]]
    assert result.text == " A C"
    assert [segment.text for segment in result.segments] == [" A", " C"]


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_no_speech_gate_can_be_disabled_or_used_without_logprob_threshold():
    """A caller can disable silence skipping or use no-speech probability alone."""
    mel = np.zeros((1, 3_100), dtype=np.float32)
    answer = DecodedWindow([100, 1, 150], -0.1, 0.7)
    options = {
        "content_frames": 100,
        "decode": lambda _window, _prompt: answer,
        "text": _text,
        "timestamp_begin": 100,
        "eot": 90,
        "audio_positions": 1_500,
    }

    kept = transcribe_mel(mel, no_speech_threshold=None, **options)
    skipped = transcribe_mel(mel, no_speech_threshold=0.6, logprob_threshold=None, **options)

    assert kept.text == " A"
    assert [segment.text for segment in kept.segments] == [" A"]
    assert skipped.text == ""
    assert skipped.segments == []


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_hinted_beam_oracle_tokens_carry_the_same_prompt_across_long_windows():
    """Confirmed text and timestamps after a hinted beam become the next window's prompt."""
    fixture = Path(__file__).parent / "fixtures" / "oracle" / "beam_long_with_prompt.json"
    oracle = json.loads(fixture.read_text(encoding="utf-8"))
    content_frames = oracle["audio_samples"] // 160
    mel = np.zeros((1, content_frames + 3_000), dtype=np.float32)
    prompts = []

    def decode(_window, prompt):
        prompts.append(prompt.copy())
        return DecodedWindow(oracle["decode_calls"][len(prompts) - 1]["tokens"], -0.2, 0.1)

    result = transcribe_mel(
        mel,
        content_frames=content_frames,
        decode=decode,
        text=lambda tokens: " A" if tokens else "",
        timestamp_begin=50_365,
        eot=50_257,
        audio_positions=1_500,
        initial_prompt_tokens=oracle["decode_calls"][0]["prompt"],
    )

    assert prompts == [call["prompt"] for call in oracle["decode_calls"]]
    assert [(part.start, part.end) for part in result.segments] == [
        (part["start"], part["end"]) for part in oracle["segments"]
    ]
