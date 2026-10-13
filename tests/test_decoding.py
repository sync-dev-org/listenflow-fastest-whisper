"""GPU-free behavior checks for alternative Whisper decoding modes."""

from math import log
from types import SimpleNamespace

import numpy as np
import pytest

from listenflow.fastest_whisper import (
    _decoding,
    _inference,
    transcribe_fp16,
    transcribe_fp32,
    transcribe_long_fp32,
)
from listenflow.fastest_whisper._audio import log_mel_spectrogram
from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._decoding import (
    DecodeAttempt,
    decode_with_fallback,
    search_candidates,
)
from listenflow.fastest_whisper._inference import TensorRTWhisper, Transcription
from listenflow.fastest_whisper._longform import DecodedWindow


class _NativeMel:
    def run(self, inputs, *, device_outputs):
        mel = log_mel_spectrogram(inputs["audio"][200:-200], n_mels=128)
        return {"mel": mel, "log_mel": mel * 4 - 4}

    def close(self):
        pass


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_beam_keeps_two_live_paths_until_completion():
    """Beam search follows the two best live paths until sentences finish."""
    seen = []

    def logits(prefix):
        generated = prefix[1:]
        seen.append(generated)
        probabilities = {
            (): (0.6, 0.4, 0.0, 0.0),
            (0,): (0.0, 0.0, 0.6, 0.4),
            (1,): (0.0, 0.0, 0.1, 0.9),
        }[tuple(generated)]
        return np.array([log(p) if p else -np.inf for p in probabilities], dtype=np.float32)

    chosen = search_candidates([9], logits, eot=3, max_tokens=8, context_length=16, beam_size=2)

    assert chosen.tokens == [1]
    assert chosen.sum_logprob == pytest.approx(log(0.4 * 0.9), abs=1e-6)
    assert seen.count([]) == 1
    assert [0] in seen and [1] in seen


def _length_tradeoff_logits(prefix):
    """A short sentence has the better total score; a long one has the better mean."""
    generated = tuple(prefix[1:])
    probabilities = {
        (): (0.55, 0.45, 0.0),
        (0,): (0.05, 0.05, 0.9),
        (1,): (0.05, 0.9, 0.05),
        (1, 1): (0.005, 0.005, 0.99),
    }.get(generated, (0.0, 0.0, 1.0))
    return np.array([log(p) if p else -np.inf for p in probabilities], dtype=np.float32)


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_beam_selects_longer_sentence_by_mean_logprobability():
    """A longer completed sentence wins when its probability per token is higher."""
    seen = []

    def logits(prefix):
        seen.append(prefix[1:])
        return _length_tradeoff_logits(prefix)

    chosen = search_candidates([9], logits, eot=2, max_tokens=3, context_length=16, beam_size=2)

    assert chosen.tokens == [1, 1]
    assert chosen.sum_logprob == pytest.approx(log(0.45 * 0.9 * 0.99), abs=1e-6)
    assert [0] in seen and [1, 1] in seen


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_beam_ignores_end_tokens_below_two_retained_live_paths():
    """A lower-ranked EOT cannot fill the finished list after the beam is full."""
    seen = []

    def logits(prefix):
        seen.append(prefix.copy())
        return np.array([2.0, 1.0, -np.inf, -8.0], dtype=np.float32)

    chosen = search_candidates([9], logits, eot=3, max_tokens=3, context_length=16, beam_size=2)

    assert chosen.tokens == [0, 0, 0]
    assert [9, 0, 0] in seen


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_positive_temperature_samples_large_vocabulary_without_probability_sum_errors():
    """Sampling a Whisper vocabulary succeeds and returns an unsuppressed token."""
    logits = np.full(51_866, -2.0, dtype=np.float32)
    logits[:1_000] = 0.0
    logits[-2:] = -np.inf

    chosen = search_candidates(
        [],
        lambda _prefix: logits,
        eot=51_865,
        max_tokens=1,
        context_length=1,
        temperature=0.2,
        best_of=5,
    )

    assert len(chosen.tokens) == 1
    assert 0 <= chosen.tokens[0] < 51_864
    assert np.isfinite(chosen.sum_logprob)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_positive_temperature_passes_stable_softmax_probabilities_to_sampler():
    """Large logit offsets preserve temperature-scaled probabilities and suppressed tokens stay zero."""
    logits = np.array([1_000_000.0, 999_999.9375, 999_999.875, -np.inf], dtype=np.float32)
    observed = []

    def sample(probabilities):
        observed.append(probabilities.copy())
        return 1

    chosen = search_candidates(
        [],
        lambda _prefix: logits,
        eot=3,
        max_tokens=1,
        context_length=1,
        temperature=0.2,
        sample_token=sample,
    )

    weights = np.array([1.0, np.exp(-0.3125), np.exp(-0.625), 0.0])
    expected = weights / weights.sum()
    assert chosen.tokens == [1]
    assert len(observed) == 1
    assert observed[0].sum() == pytest.approx(1.0, abs=1e-14, rel=0)
    assert observed[0][3] == 0.0
    np.testing.assert_allclose(observed[0], expected, rtol=1e-14, atol=0)


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_best_of_samples_independent_sentences_and_selects_highest_mean_score():
    """Positive temperature ranks independently sampled sentences using unscaled log probabilities."""
    starts = iter((1, 0, 1))
    seen = []

    def logits(prefix):
        generated = prefix[1:]
        seen.append(generated)
        if generated:
            return np.array([-np.inf, -np.inf, 0.0], dtype=np.float32)
        return np.array([log(0.7), log(0.3), -np.inf], dtype=np.float32)

    def sample(probabilities):
        if probabilities[2] == 1:
            return 2
        return next(starts)

    chosen = search_candidates(
        [9],
        logits,
        eot=2,
        max_tokens=8,
        context_length=16,
        temperature=0.4,
        best_of=3,
        sample_token=sample,
    )

    assert chosen.tokens == [0]
    assert chosen.sum_logprob == pytest.approx(log(0.7), abs=1e-6)
    assert seen.count([]) == 3


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_best_of_selects_longer_sentence_by_mean_logprobability():
    """Several sampled sentences are ranked by probability per generated token."""
    choices = iter((0, 2, 1, 1, 2))
    seen = []

    def logits(prefix):
        seen.append(prefix[1:])
        return _length_tradeoff_logits(prefix)

    chosen = search_candidates(
        [9],
        logits,
        eot=2,
        max_tokens=3,
        context_length=16,
        temperature=0.4,
        best_of=2,
        sample_token=lambda _probabilities: next(choices),
    )

    assert chosen.tokens == [1, 1]
    assert chosen.sum_logprob == pytest.approx(log(0.45 * 0.9 * 0.99), abs=1e-6)
    assert seen.count([]) == 2


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_fallback_tries_temperatures_in_order_switches_off_beam_and_keeps_last_result():
    """Each failed attempt advances once; positive temperature uses best-of sampling."""
    calls = []
    attempts = [
        DecodeAttempt([1], "repeat", -1.1, 0.1, 2.5, 0.0),
        DecodeAttempt([2], "still bad", -1.2, 0.1, 2.5, 0.4),
        DecodeAttempt([3], "last", -1.3, 0.1, 2.5, 0.8),
    ]

    def decode(temperature, beam_size, best_of):
        calls.append((temperature, beam_size, best_of))
        return attempts[len(calls) - 1]

    chosen = decode_with_fallback(
        decode,
        temperatures=(0.0, 0.4, 0.8),
        beam_size=3,
        best_of=5,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )

    assert chosen is attempts[-1]
    assert calls == [(0.0, 3, None), (0.4, None, 5), (0.8, None, 5)]


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize(
    ("avg_logprob", "compression_ratio", "should_retry"),
    [(-1.1, 2.4, True), (-1.0, 2.4, False), (-1.0, 2.5, True)],
    ids=("low_logprob_only", "at_both_thresholds", "high_compression_only"),
)
def test_fallback_retries_for_each_failed_threshold(avg_logprob, compression_ratio, should_retry):
    """A window is retried when either quality limit fails, with exact limits accepted."""
    calls = []
    attempts = [
        DecodeAttempt([1], "first", avg_logprob, 0.1, compression_ratio, 0.0),
        DecodeAttempt([2], "retried", -0.4, 0.1, 1.1, 0.2),
    ]

    def decode(temperature, beam_size, best_of):
        calls.append((temperature, beam_size, best_of))
        return attempts[len(calls) - 1]

    chosen = decode_with_fallback(
        decode,
        temperatures=(0.0, 0.2),
        beam_size=2,
        best_of=3,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )

    assert chosen is attempts[int(should_retry)]
    assert calls == ([(0.0, 2, None), (0.2, None, 3)] if should_retry else [(0.0, 2, None)])


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_fallback_uses_strict_thresholds_and_stops_for_silence():
    """Exact threshold values pass, while high no-speech probability ends a failed attempt."""
    calls = []

    def decode(temperature, _beam_size, _best_of):
        calls.append(temperature)
        return DecodeAttempt([1], "text", -1.0, 0.6, 2.4, temperature)

    decode_with_fallback(
        decode,
        temperatures=(0.0, 0.2),
        beam_size=2,
        best_of=3,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    assert calls == [0.0]

    calls.clear()

    def silence(temperature, _beam_size, _best_of):
        calls.append(temperature)
        return DecodeAttempt([1], "text", -1.1, 0.61, 2.5, temperature)

    decode_with_fallback(
        silence,
        temperatures=(0.0, 0.2),
        beam_size=2,
        best_of=3,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    assert calls == [0.0]


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_tensor_rt_beam_decodes_filtered_logits_and_returns_only_generated_text():
    """A decoder prompt does not enter the result, and both beam paths use filtered logits."""
    model = TensorRTWhisper.__new__(TensorRTWhisper)
    model.config = SimpleNamespace(text_positions=16)
    model.initial_tokens = [5, 6, 7, 8]
    model.eot = 3
    model.suppressed = {4}
    model.begin_suppressed = set()
    model.tokenizer = SimpleNamespace(
        decode=lambda tokens, **_kwargs: "".join(
            {0: "A", 1: "B", 2: "C"}.get(t, "") for t in tokens
        ),
        token_to_id={"<|startofprev|>": 10, "<|nospeech|>": 4}.get,
    )
    prefixes = []

    def logits(prefix, _encoder):
        prefixes.append(prefix.copy())
        generated = prefix[6:]
        probabilities = {
            (): (0.6, 0.4, 0.0, 0.0),
            (0,): (0.0, 0.0, 0.6, 0.4),
            (1,): (0.0, 0.0, 0.1, 0.9),
        }[tuple(generated)]
        result = np.full((len(prefix), 12), -np.inf, dtype=np.float32)
        result[2, 4] = -2.0
        result[2, 0] = 0.0
        result[-1, :4] = [log(p) if p else -np.inf for p in probabilities]
        result[-1, 4] = 20.0  # The suppressed token would otherwise always win.
        return result

    model.logits = logits
    result = model._decode_attempt(
        np.empty((0,), dtype=np.float32),
        [42],
        with_timestamps=False,
        temperature=0.0,
        beam_size=2,
        best_of=None,
    )

    assert result.tokens == [1]
    assert result.text == "B"
    assert result.avg_logprob == pytest.approx(log(0.4 * 0.9) / 2, abs=1e-6)
    assert prefixes[0] == [10, 42, 5, 6, 7, 8]
    assert [10, 42, 5, 6, 7, 8, 0] in prefixes
    assert [10, 42, 5, 6, 7, 8, 1] in prefixes


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_hint_is_present_in_beam_and_sampled_retry_prefixes(monkeypatch):
    """A hinted window retries from the same decoder context after switching away from beam."""
    model = TensorRTWhisper.__new__(TensorRTWhisper)
    model.config = SimpleNamespace(text_positions=16)
    model.initial_tokens = [5, 6, 7, 8]
    model.eot = 3
    model.suppressed = {4}
    model.begin_suppressed = set()
    model.tokenizer = SimpleNamespace(
        decode=lambda tokens, **_kwargs: "".join({0: "A", 1: "B"}.get(t, "") for t in tokens),
        token_to_id={"<|startofprev|>": 10, "<|nospeech|>": 4}.get,
    )
    prefixes = []

    def logits(prefix, _encoder):
        prefixes.append(prefix.copy())
        result = np.full((len(prefix), 12), -np.inf, dtype=np.float32)
        result[2, 0] = 0.0
        result[2, 4] = -5.0
        if len(prefix) == 6:
            result[-1, 0] = 2.0
            result[-1, 1] = 1.0
        else:
            result[-1, 3] = 0.0
        return result

    model.logits = logits
    starts = iter((1, 0))

    class Generator:
        def choice(self, _count, p):
            return 3 if p[3] == 1 else next(starts)

    monkeypatch.setattr(_decoding.np.random, "default_rng", Generator)
    settings = _decoding.DecodingSettings(
        beam_size=2,
        temperatures=(0.0, 0.4),
        best_of=2,
        compression_ratio_threshold=0.01,
    )

    result = model.decode_short(np.empty((0,), dtype=np.float32), [42], settings)

    assert result.tokens == [0, 3]
    assert prefixes.count([10, 42, 5, 6, 7, 8]) == 2
    assert all(prefix[:6] == [10, 42, 5, 6, 7, 8] for prefix in prefixes)


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_omitted_best_of_samples_five_sentences_at_positive_temperature(monkeypatch):
    """The selected default makes five stochastic candidates when best_of is omitted."""
    model = TensorRTWhisper.__new__(TensorRTWhisper)
    model.config = SimpleNamespace(text_positions=16)
    model.initial_tokens = [5, 6, 7, 8]
    model.eot = 3
    model.suppressed = {4}
    model.begin_suppressed = set()
    model.tokenizer = SimpleNamespace(
        decode=lambda tokens, **_kwargs: "".join("A" for token in tokens if token == 0),
        token_to_id={"<|nospeech|>": 4}.get,
    )

    def logits(prefix, _encoder):
        result = np.full((len(prefix), 12), -np.inf, dtype=np.float32)
        result[0, 0] = 0.0
        result[0, 4] = -5.0
        if len(prefix) == 4:
            result[-1, 0] = 2.0
            result[-1, 1] = 1.0
        else:
            result[-1, 3] = 0.0
        return result

    model.logits = logits
    choices = []

    class Generator:
        def choice(self, _count, p):
            token = 3 if p[3] == 1 else 0
            choices.append(token)
            return token

    monkeypatch.setattr(_decoding.np.random, "default_rng", Generator)

    actual = model.decode_short(
        np.empty((0,), dtype=np.float32), [], _decoding.decoding_settings(temperature=0.4)
    )

    assert actual.tokens == [0, 3]
    assert choices.count(0) == 5


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("entry", [transcribe_fp32, transcribe_fp16])
def test_short_entries_accept_beam_and_retry_settings_without_loading_a_gpu(entry, monkeypatch):
    """Both precision routes pass caller-selected decoding settings to the model."""
    seen = []

    class Model(TensorRTWhisper):
        config = SimpleNamespace(n_mels=128)
        _mel = _NativeMel()

        def __init__(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def prompt_tokens(self, hint):
            return [42] if hint else []

        def encode(self, _mel):
            return np.empty((0,), dtype=np.float32)

        def decode_short(self, _encoder, prompt, settings):
            seen.append((prompt, settings))
            return Transcription("hinted", [1, 3])

    monkeypatch.setattr(_inference, "TensorRTWhisper", lambda *_args, **_kwargs: Model())
    result = entry(
        "unused",
        np.zeros(16_000, dtype=np.float32),
        initial_prompt="hint",
        beam_size=2,
        temperature=(0.0, 0.4),
        best_of=3,
        compression_ratio_threshold=2.1,
        logprob_threshold=-0.8,
        no_speech_threshold=0.7,
    )

    assert result == Transcription("hinted", [1, 3])
    assert seen[0][0] == [42]
    settings = seen[0][1]
    assert (settings.beam_size, settings.temperatures, settings.best_of) == (2, (0.0, 0.4), 3)
    assert (
        settings.compression_ratio_threshold,
        settings.logprob_threshold,
        settings.no_speech_threshold,
    ) == (2.1, -0.8, 0.7)

    entry("unused", np.zeros(16_000, dtype=np.float32), temperature=0.4)
    assert seen[1][1].best_of == 5


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_long_entry_applies_settings_with_hint_to_its_decoded_window(monkeypatch):
    """The long route sends the hint and settings to the timestamp decoder for each window."""
    seen = []

    class Model(TensorRTWhisper):
        config = SimpleNamespace(n_mels=128, audio_positions=1_500)
        _mel = _NativeMel()

        def __init__(self):
            pass

        eot = 90
        tokenizer = SimpleNamespace(
            token_to_id=lambda _token: 100,
            decode=lambda tokens, **_kwargs: "".join(" B" for token in tokens if token == 1),
        )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def prompt_tokens(self, hint):
            return [42] if hint else []

        def encode(self, _mel):
            return np.empty((0,), dtype=np.float32)

        def decode_long(self, _encoder, prompt, settings):
            seen.append((prompt, settings))
            return DecodedWindow([100, 1, 150], -0.2, 0.1)

    monkeypatch.setattr(_inference, "TensorRTWhisper", lambda *_args, **_kwargs: Model())
    result = transcribe_long_fp32(
        "unused",
        np.zeros(16_000, dtype=np.float32),
        initial_prompt="hint",
        beam_size=3,
        temperature=(0.0, 0.2),
        best_of=4,
    )

    assert result.text == " B"
    assert len(seen) == 1
    assert seen[0][0] == [42]
    assert (seen[0][1].beam_size, seen[0][1].temperatures, seen[0][1].best_of) == (3, (0.0, 0.2), 4)


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_cached_runtime_accepts_beam_settings_for_short_audio():
    """The reusable runtime can decode one utterance with a selected beam width."""
    model = CachedWhisper.__new__(CachedWhisper)
    model._mel = _NativeMel()
    model.config = SimpleNamespace(n_mels=128)
    model.prompt_tokens = lambda hint: [42] if hint else []
    model.encode = lambda _mel: np.empty((0,), dtype=np.float32)
    seen = []

    def decode_short(_encoder, prompt, settings):
        seen.append((prompt, settings.beam_size))
        return Transcription("hinted", [1, 3])

    model.decode_short = decode_short

    segments = model.transcribe(
        np.zeros(16_000, dtype=np.float32), initial_prompt="hint", beam_size=2
    )

    assert [(segment.start, segment.end, segment.text) for segment in segments] == [
        (0, 1, "hinted")
    ]
    assert seen == [([42], 2)]


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_cached_runtime_omits_best_of_as_five_samples():
    """The reusable runtime selects five candidates when the caller omits best_of."""
    model = CachedWhisper.__new__(CachedWhisper)
    model._mel = _NativeMel()
    model.config = SimpleNamespace(n_mels=128)
    model.prompt_tokens = lambda _hint: []
    model.encode = lambda _mel: np.empty((0,), dtype=np.float32)
    observed = []

    def decode_short(_encoder, _prompt, settings):
        observed.append(settings.best_of)
        return Transcription("sampled", [1, 3])

    model.decode_short = decode_short
    result = model.transcribe(np.zeros(16_000, dtype=np.float32), temperature=0.4)

    assert [segment.text for segment in result] == ["sampled"]
    assert observed == [5]
