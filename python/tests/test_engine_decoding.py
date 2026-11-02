"""Stream decoding with real selection rules and a replaced native GPU boundary."""

import asyncio
from itertools import cycle
from math import log
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from listenflow import (
    ConfigurationError,
    Final,
    OfflineModelDirectorySource,
    UtteranceBoundarySettings,
)
from listenflow.fastest_whisper import FastestWhisperEngine
from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._decoding import decoding_settings

DEFAULTS = {
    "beam_size": 1,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "best_of": 5,
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}


def _engine(tmp_path, **options):
    return FastestWhisperEngine(
        model_source=OfflineModelDirectorySource(str(tmp_path)),
        precision=options.pop("precision", "fp32"),
        device=0,
        language=options.pop("language", "en"),
        boundary_settings=UtteranceBoundarySettings(preroll_s=0),
        **options,
    )


@pytest.fixture
def runtime(monkeypatch):
    from listenflow.fastest_whisper import _cached_model
    from listenflow.fastest_whisper import _engine as engine_module

    monkeypatch.setattr(engine_module, "_cuda_device_info", lambda _device: (1, (12, 0)))

    def install(
        *,
        repeated=False,
        no_speech=0.1,
        starts=(1,),
        by_temperature=False,
        detected_languages=("en",),
        language_scores=None,
        prefixes=None,
        first_probabilities=None,
        sequence=None,
        word=None,
    ):
        if word is None:
            word = "A" * 100 if repeated else "A"
        vocab = {
            word: 0,
            "B": 1,
            "C": 2,
            "<|endoftext|>": 3,
            "<|nospeech|>": 4,
            "<|startoftranscript|>": 5,
            "<|en|>": 6,
            "<|transcribe|>": 7,
            "<|notimestamps|>": 8,
            "<|startofprev|>": 9,
            "hint": 10,
            **{f"<|{index * 0.02:.2f}|>": 11 + index for index in range(51)},
            **{
                f"<|{code}|>": 62 + index
                for index, code in enumerate(("ja", "zh", "yue", "ko", "ru", "ar", "hi", "th"))
            },
            "<|xyz|>": 70,
        }
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="hint"))
        tokenizer.pre_tokenizer = Whitespace()

        class NativeModel(CachedWhisper):
            """Replace GPU loading, features, and logits; retain transcription and search."""

            def __init__(self, _checkpoint, *, precision, **_options):
                self.precision = precision
                self._settings = _options.get("settings", decoding_settings())
                self.config = SimpleNamespace(n_mels=1, audio_positions=1500, text_positions=32)
                self.tokenizer = tokenizer
                self.initial_tokens = [5, 6, 7, 8]
                self.eot = 3
                self.suppressed = set(range(4, 11))
                self.begin_suppressed = set()
                self.encoded_windows = 0

            def warmup(self):
                pass

            def close(self):
                pass

            def _mel_spectrogram(self, audio):
                return np.zeros((1, 3000), dtype=np.float32)

            def _mel_spectrogram_long(self, audio):
                return np.zeros((1, audio.size // 160 + 3000), dtype=np.float32)

            def encode(self, _mel):
                self.encoded_windows += 1
                return np.array([self.encoded_windows - 1], dtype=np.float32)

            def logits(self, prefix, encoder):
                if prefixes is not None:
                    prefixes.append(prefix.copy())
                if prefix == [5]:
                    output = np.full((1, len(vocab)), -np.inf, dtype=np.float32)
                    scores = (
                        [{language: 1} for language in detected_languages]
                        if language_scores is None
                        else language_scores
                    )
                    index = min(int(encoder[0]), len(scores) - 1)
                    for language, score in scores[index].items():
                        output[0, vocab[f"<|{language}|>"]] = score
                    output[0, vocab["<|xyz|>"]] = 10
                    return output
                task = len(prefix) - 1 - prefix[::-1].index(7)
                long = len(prefix) <= task + 1 or prefix[task + 1] != 8
                generated = prefix[task + (1 if long else 2) :]
                output = np.full((len(prefix), len(vocab)), -np.inf, dtype=np.float32)
                output[task - 2, 4] = log(no_speech)
                output[task - 2, 0] = log(1 - no_speech)
                if sequence is not None:
                    token = sequence[len(generated)] if len(generated) < len(sequence) else 3
                    output[-1, token] = 0
                    return output
                if long and not generated:
                    output[-1, 11] = 0
                    return output
                if long:
                    generated = generated[1:]
                end = 61 if long else 3
                if generated and generated[-1] == 61:
                    output[-1, 3] = 0
                    return output
                probabilities = {
                    (): {0: 0.55, 1: 0.45},
                    (0,): {2: 0.51, end: 0.49},
                    (1,): {2: 0.01, end: 0.99},
                }.get(tuple(generated), {end: 1.0})
                if first_probabilities is not None:
                    probabilities = first_probabilities if not generated else {end: 1.0}
                for token, probability in probabilities.items():
                    output[-1, token] = log(probability)
                return output

        choices = cycle(starts)

        class Entropy:
            def choice(self, _count, p):
                if p[0] and p[1]:
                    return (0 if p[0] > 0.65 else 1) if by_temperature else next(choices)
                return int(np.argmax(p))

        # Random entropy and native computation are the only replaced boundaries.
        monkeypatch.setattr(np.random, "default_rng", Entropy)
        monkeypatch.setattr(_cached_model, "CachedWhisper", NativeModel)
        return word

    return install


@pytest.mark.req("SP-LFFASTESTWHISPER-039")
@pytest.mark.req("SP-LFFASTESTWHISPER-042")
@pytest.mark.parametrize("language", ["en", "ja", "zh", "yue", "ko", "ru", "ar", "hi", "th"])
@pytest.mark.parametrize("seconds", [1, 31])
@pytest.mark.parametrize("as_list", [False, True], ids=["code", "singleton-list"])
def test_explicit_language_is_used_in_short_and_long_decoder_prompts(
    tmp_path, runtime, language, seconds, as_list
):
    """One code, as a string or singleton list, controls each prompt without language detection."""
    prefixes = []
    runtime(prefixes=prefixes)
    engine = _engine(tmp_path, language=[language] if as_list else language, temperature=0.0)
    with asyncio.Runner() as runner:
        assert _texts(runner, engine, seconds=seconds) == ["A C"] * (2 if seconds > 30 else 1)
        expected = engine._model.tokenizer.token_to_id(f"<|{language}|>")
        assert prefixes and all(prefix != [5] for prefix in prefixes)
        assert {prefix[prefix.index(5) + 1] for prefix in prefixes} == {expected}
        runner.run(engine.aclose())


@pytest.mark.req("SP-LFFASTESTWHISPER-039")
@pytest.mark.req("SP-LFFASTESTWHISPER-041")
@pytest.mark.parametrize("precision", ["fp32", "fp16", "fp8"])
@pytest.mark.parametrize(("seconds", "utterances"), [(1, 2), (31, 1)])
def test_language_list_restricts_detection_for_each_utterance_and_long_audio_window(
    tmp_path, runtime, precision, seconds, utterances
):
    """Each utterance or long-audio window selects its best listed language, excluding Chinese."""
    prefixes = []
    runtime(
        language_scores=({"zh": 3, "ja": 2, "en": 1}, {"zh": 3, "ja": 1, "en": 2}),
        prefixes=prefixes,
    )
    engine = _engine(tmp_path, precision=precision, language=["en", "ja"], temperature=0.0)
    with asyncio.Runner() as runner:
        assert _texts(runner, engine, seconds=seconds, utterances=utterances) == ["A C", "A C"]
        assert prefixes.count([5]) == 2
        prompts = [prefix for prefix in prefixes if prefix != [5]]
        languages = [prefix[prefix.index(5) + 1] for prefix in prompts]
        ja = engine._model.tokenizer.token_to_id("<|ja|>")
        en = engine._model.tokenizer.token_to_id("<|en|>")
        assert languages[0] == ja and languages[-1] == en
        assert set(languages) == {ja, en}
        runner.run(engine.aclose())


@pytest.mark.req("SP-LFFASTESTWHISPER-039")
def test_automatic_language_is_redetected_and_selected_for_each_long_audio_window(
    tmp_path, runtime
):
    """Each long-audio window uses its own detected language token, including Cantonese."""
    prefixes = []
    runtime(detected_languages=("ja", "yue"), prefixes=prefixes)
    engine = _engine(tmp_path, language=None, temperature=0.0)
    with asyncio.Runner() as runner:
        assert _texts(runner, engine, seconds=31) == ["A C", "A C"]
        assert prefixes.count([5]) == 2
        prompts = [prefix for prefix in prefixes if prefix != [5]]
        languages = [prefix[prefix.index(5) + 1] for prefix in prompts]
        ja = engine._model.tokenizer.token_to_id("<|ja|>")
        yue = engine._model.tokenizer.token_to_id("<|yue|>")
        assert languages[0] == ja and languages[-1] == yue
        assert set(languages) == {ja, yue}
        runner.run(engine.aclose())


@pytest.mark.req("SP-LFFASTESTWHISPER-039")
@pytest.mark.parametrize("language", ["ja", "yue"])
def test_automatic_short_decode_uses_the_detected_language(tmp_path, runtime, language):
    """A short utterance transcribes with the detected language rather than defaulting to English."""
    prefixes = []
    runtime(detected_languages=(language,), prefixes=prefixes)
    engine = _engine(tmp_path, language=None, temperature=0.0)
    with asyncio.Runner() as runner:
        assert _texts(runner, engine) == ["A C"]
        assert prefixes.count([5]) == 1
        expected = engine._model.tokenizer.token_to_id(f"<|{language}|>")
        assert {prefix[1] for prefix in prefixes if prefix != [5]} == {expected}
        runner.run(engine.aclose())


def _texts(runner, engine, *, seconds=1, utterances=1):
    pcm = np.full(seconds * 16000, 1000, dtype="<i2").tobytes()

    async def chunks():
        for index in range(utterances):
            yield pcm
            if index + 1 < utterances:
                yield np.zeros(16000, dtype="<i2").tobytes()

    async def collect():
        await engine.prepare()
        return [
            event.text
            async for event in engine.stream(chunks())
            if isinstance(event, Final) and event.text
        ]

    return runner.run(collect())


@pytest.mark.req("SP-LFFASTESTWHISPER-047")
@pytest.mark.req("SP-LFFASTESTWHISPER-049")
@pytest.mark.parametrize("seconds", [1, 31])
@pytest.mark.parametrize("temperature", [0.0, 0.4, (0.0, 0.4)])
@pytest.mark.parametrize("no_speech_threshold", [None, 0.6])
@pytest.mark.parametrize("mode", ["pass", "drop"])
def test_phrase_exhaustion_controls_short_and_timestamped_window_outputs(
    tmp_path, runtime, seconds, temperature, no_speech_threshold, mode
):
    """The constructor's immutable phrase policy reaches every window and decoding mode."""
    runtime(starts=(0,))
    engine = _engine(
        tmp_path,
        temperature=temperature,
        reject_phrases=["A"],
        no_speech_threshold=no_speech_threshold,
        on_rejected_exhausted=mode,
    )
    with asyncio.Runner() as runner:
        assert _texts(runner, engine, seconds=seconds) == (
            ["A C"] * (2 if seconds > 30 else 1) if mode == "pass" else []
        )
        runner.run(engine.aclose())


@pytest.mark.req("SP-LFFASTESTWHISPER-047")
@pytest.mark.parametrize("beam_size", [1, 2])
@pytest.mark.parametrize("seconds", [1, 31])
def test_losing_phrase_candidate_does_not_advance_temperature_in_stream_output(
    tmp_path, runtime, beam_size, seconds
):
    """An unselected rejected beam or sample leaves the clean winning sentence unchanged."""
    prefixes = []
    runtime(starts=(0, 1), prefixes=prefixes, first_probabilities={0: 0.3, 1: 0.7})
    engine = _engine(
        tmp_path,
        beam_size=beam_size,
        best_of=2,
        temperature=(0.4, 0.8) if beam_size == 1 else (0.0, 0.4),
        reject_phrases=["A"],
        on_rejected_exhausted="drop",
    )
    with asyncio.Runner() as runner:
        assert _texts(runner, engine, seconds=seconds) == ["B"] * (2 if seconds > 30 else 1)
        # Each trial shares its first logits across candidates.
        starts = [prefix for prefix in prefixes if prefix[-1] in (7, 8)]
        assert len(starts) == (2 if seconds > 30 else 1)
        runner.run(engine.aclose())


@pytest.mark.req("SP-LFFASTESTWHISPER-050")
@pytest.mark.parametrize("seconds", [1, 31])
@pytest.mark.parametrize("enabled", [False, True])
def test_phrase_silence_suppresses_high_confidence_text_only_when_requested(
    tmp_path, runtime, seconds, enabled
):
    """High no-speech probability suppresses a phrase hit even when its logprob would pass."""
    runtime(no_speech=0.8)
    engine = _engine(tmp_path, temperature=0.0, reject_phrases=["A"], phrase_hit_no_speech=enabled)
    with asyncio.Runner() as runner:
        assert _texts(runner, engine, seconds=seconds) == (
            [] if enabled else ["A C"] * (2 if seconds > 30 else 1)
        )
        runner.run(engine.aclose())


@pytest.mark.req("SP-LFFASTESTWHISPER-049")
@pytest.mark.parametrize("sequence", [[11, 0, 61, 61, 1, 2], [36, 1, 2]])
def test_pass_keeps_an_unfinished_timestamp_sentence_until_the_window_end_without_redecoding(
    tmp_path, runtime, sequence
):
    """Rejected final text retains its opening timestamp and consumes each full audio window once."""
    prefixes = []
    runtime(sequence=sequence, prefixes=prefixes)
    engine = _engine(tmp_path, temperature=0.0, reject_phrases=["B"])
    with asyncio.Runner() as runner:
        runner.run(engine.prepare())
        segments = engine._model.transcribe(np.zeros(60 * 16_000), temperature=0.0)
        tail = [segment for segment in segments if segment.text.strip() == "B C"]
        start = 1.0 if sequence[0] == 11 else 0.5
        assert [(segment.start, segment.end) for segment in tail] == [
            (start, 30.0),
            (30.0 + start, 60.0),
        ]
        assert [segment.text.strip() for segment in segments] == (
            ["A", "B C", "A", "B C"] if sequence[0] == 11 else ["B C", "B C"]
        )
        assert len([prefix for prefix in prefixes if prefix[-1] == 7]) == 2
        runner.run(engine.aclose())


# Ten equally likely initial tokens followed by certain EOT give avg_logprob < -1.
LOW_CONFIDENCE_TOKENS = {token: 0.1 for token in (0, 1, 2, 3, *range(11, 17))}


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize(
    ("no_speech", "first_probabilities", "temperature", "thresholds", "expected"),
    [
        pytest.param(0.8, LOW_CONFIDENCE_TOKENS, (0.0, 0.4), {}, [], id="silent-final-attempt"),
        pytest.param(0.8, LOW_CONFIDENCE_TOKENS, 0.4, {}, [], id="single-temperature"),
        pytest.param(0.8, None, (0.0, 0.4), {}, ["A C"], id="good-logprob-keeps-text"),
        pytest.param(
            0.5,
            LOW_CONFIDENCE_TOKENS,
            0.4,
            {"no_speech_threshold": 0.5},
            ["A"],
            id="equal-no-speech-prob-keeps-text",
        ),
        pytest.param(
            0.8,
            {0: 1.0},
            0.4,
            {"logprob_threshold": 0.0},
            [],
            id="equal-logprob-skips-text",
        ),
        pytest.param(0.8, None, 0.4, {"logprob_threshold": None}, [], id="no-logprob-override"),
        pytest.param(
            0.8,
            LOW_CONFIDENCE_TOKENS,
            0.4,
            {"no_speech_threshold": None},
            ["A"],
            id="disabled-no-speech-threshold",
        ),
    ],
)
def test_short_silence_follows_official_probability_thresholds(
    tmp_path, runtime, no_speech, first_probabilities, temperature, thresholds, expected
):
    """Short utterances skip silent text unless speech probability or log probability allows it."""
    runtime(no_speech=no_speech, starts=(0,), first_probabilities=first_probabilities)
    with asyncio.Runner() as runner:
        engine = _engine(tmp_path, temperature=temperature, **thresholds)
        assert _texts(runner, engine) == expected
        runner.run(engine.aclose())


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_greedy_silence_skips_short_window_without_changing_decoded_tokens(tmp_path, runtime):
    """Greedy short utterances suppress silent text while preserving the selected tokens."""
    from listenflow.fastest_whisper import _cached_model

    runtime(no_speech=0.8, first_probabilities=LOW_CONFIDENCE_TOKENS)
    model = _cached_model.CachedWhisper(tmp_path, precision="fp32")
    audio = np.zeros(16000, dtype=np.float32)
    encoder = model.encode(model._mel_spectrogram(audio))

    unscored = model.greedy_decode(encoder, [10])
    result = model.decode_short(encoder, [10], decoding_settings(temperature=0.0))

    assert unscored.text == "A"
    assert result.tokens == unscored.tokens == [0, 3]
    assert result.text == ""
    assert model.transcribe(audio, initial_prompt="hint", temperature=0.0) == []


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize(
    "changed",
    [
        {},
        *[
            {name: value}
            for name, value in {
                "beam_size": 2,
                "temperature": 0.4,
                "best_of": 1,
                "compression_ratio_threshold": None,
                "logprob_threshold": 0.0,
                "no_speech_threshold": None,
            }.items()
        ],
    ],
)
def test_omitted_decoding_options_keep_the_existing_defaults(tmp_path, runtime, changed):
    """Omitted settings retain the established retry policy, including with one override."""
    runtime(repeated=True, starts=(1, 0))
    with asyncio.Runner() as runner:
        implicit = _engine(tmp_path, **changed)
        actual = _texts(runner, implicit)
        runner.run(implicit.aclose())
        runtime(repeated=True, starts=(1, 0))
        explicit = _engine(tmp_path, **(DEFAULTS | changed))
        assert _texts(runner, explicit) == actual
        assert actual
        runner.run(explicit.aclose())


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize(
    ("precision", "seconds", "prompt"),
    [
        ("fp32", 1, None),
        ("fp16", 1, "hint"),
        ("fp32", 31, "hint"),
    ],
)
def test_selected_beam_changes_stream_text(tmp_path, runtime, precision, seconds, prompt):
    """Both precisions and long hinted utterances use the selected zero-temperature beam."""
    runtime()
    with asyncio.Runner() as runner:
        greedy = _engine(tmp_path, precision=precision, initial_prompt=prompt, temperature=0.0)
        beam = _engine(
            tmp_path, precision=precision, initial_prompt=prompt, temperature=0.0, beam_size=2
        )
        assert _texts(runner, greedy, seconds=seconds) == ["A C"] * (2 if seconds > 30 else 1)
        assert _texts(runner, beam, seconds=seconds) == ["B"] * (2 if seconds > 30 else 1)
        runner.run(greedy.aclose())
        runner.run(beam.aclose())


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize(("best_of", "expected"), [(1, "B"), (2, "A C")])
def test_positive_temperature_selects_the_requested_number_of_candidates(
    tmp_path, runtime, best_of, expected
):
    """Positive temperature ranks the requested sampled sentences regardless of beam width."""
    runtime(starts=(1, 0))
    with asyncio.Runner() as runner:
        engine = _engine(tmp_path, temperature=0.4, best_of=best_of, beam_size=3)
        assert _texts(runner, engine) == [expected]
        runner.run(engine.aclose())


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize(
    ("temperature", "expected"),
    [
        ((0.2, 0.8), "B"),
        ((0.8, 0.2), "A C"),
        (0.2, "A C"),
    ],
)
def test_failed_quality_retries_in_the_selected_temperature_order(
    tmp_path, runtime, temperature, expected
):
    """A failed window tries the chosen order, and a scalar temperature has no later retry."""
    runtime(by_temperature=True)
    with asyncio.Runner() as runner:
        engine = _engine(tmp_path, temperature=temperature, compression_ratio_threshold=0.0)
        assert _texts(runner, engine) == [expected]
        runner.run(engine.aclose())


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize(
    ("threshold", "value", "expected"),
    [
        ("compression_ratio_threshold", 2.4, "B"),
        ("compression_ratio_threshold", 100.0, "repeated"),
        ("compression_ratio_threshold", None, "repeated"),
        ("logprob_threshold", -1.0, "A C"),
        ("logprob_threshold", 0.0, "B"),
        ("logprob_threshold", None, "A C"),
        ("no_speech_threshold", 0.6, None),
        ("no_speech_threshold", 0.9, "B"),
        ("no_speech_threshold", None, "B"),
    ],
)
def test_each_threshold_changes_or_disables_its_check_in_stream_results(
    tmp_path, runtime, threshold, value, expected
):
    """Compression and probability limits control retries, and silence limits can skip windows."""
    silence = threshold == "no_speech_threshold"
    word = runtime(
        repeated=threshold == "compression_ratio_threshold", no_speech=0.8 if silence else 0.1
    )
    options = {"temperature": (0.0, 0.4), threshold: value}
    if silence:
        options["logprob_threshold"] = 0.0
    with asyncio.Runner() as runner:
        engine = _engine(tmp_path, **options)
        actual = _texts(runner, engine, seconds=31 if silence else 1)
        text = f"{word} C" if expected == "repeated" else expected
        assert actual == ([] if text is None else [text] * (2 if silence else 1))
        runner.run(engine.aclose())


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_temperature_list_is_copied_for_all_utterances_and_streams(tmp_path, runtime):
    """Mutating the caller's list cannot change the engine's later utterances or streams."""
    runtime(by_temperature=True)
    temperatures = [0.2, 0.8]
    engine = _engine(tmp_path, temperature=temperatures, compression_ratio_threshold=0.0)
    temperatures[:] = [0.8, 0.2]
    with asyncio.Runner() as runner:
        assert _texts(runner, engine, utterances=2) == ["B", "B"]
        temperatures.clear()
        assert _texts(runner, engine) == ["B"]
        runner.run(engine.aclose())


INVALID = [
    *[
        (name, value)
        for name in ("beam_size", "best_of")
        for value in (True, False, 0, -1, 1.5, "2", None)
    ],
    *[
        ("temperature", value)
        for value in (
            None,
            object(),
            "warm",
            True,
            -0.1,
            [],
            [False],
            ["0"],
            [0.0, -0.1],
            float("inf"),
            float("-inf"),
            float("nan"),
            [0.0, float("nan")],
        )
    ],
    *[
        (name, value)
        for name in ("compression_ratio_threshold", "logprob_threshold", "no_speech_threshold")
        for value in (True, False, "1", object(), float("inf"), float("-inf"), float("nan"))
    ],
]


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize(("name", "value"), INVALID)
def test_invalid_decoding_values_fail_when_constructing_the_engine(tmp_path, name, value):
    """Invalid decoding settings raise ConfigurationError without preparation or GPU access."""
    with pytest.raises(ConfigurationError, match=name):
        _engine(tmp_path, **{name: value})


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize("temperature", [0, 0.4, 2.0, [0, 0.4], (0.4, 0.0)])
def test_valid_decoding_values_have_no_additional_numeric_limits(tmp_path, temperature):
    """Scalar and sequence temperatures and disabled thresholds are accepted without a GPU."""
    _engine(
        tmp_path,
        beam_size=100,
        best_of=100,
        temperature=temperature,
        compression_ratio_threshold=None,
        logprob_threshold=None,
        no_speech_threshold=None,
    )
    _engine(
        tmp_path,
        temperature=temperature,
        compression_ratio_threshold=-5,
        logprob_threshold=5,
        no_speech_threshold=5,
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize(
    "entry_name", ["transcribe_fp32", "transcribe_fp16", "transcribe_long_fp32"]
)
@pytest.mark.parametrize(
    ("name", "value", "error_type"),
    [
        ("beam_size", True, ValueError),
        ("best_of", 0, ValueError),
        ("temperature", [], ValueError),
        ("temperature", None, TypeError),
        ("compression_ratio_threshold", True, ValueError),
        ("logprob_threshold", float("nan"), ValueError),
        ("no_speech_threshold", "0.6", ValueError),
    ],
)
def test_numerical_entries_keep_their_decoding_error_types(entry_name, name, value, error_type):
    """Numerical entries retain their validation error types before loading a GPU model."""
    import listenflow.fastest_whisper as module

    with pytest.raises(error_type):
        getattr(module, entry_name)("unused", np.zeros(16000, dtype=np.float32), **{name: value})


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize(
    "entry_name",
    [
        "engine",
        "transcribe_fp32",
        "transcribe_fp16",
        "transcribe_long_fp32",
    ],
)
@pytest.mark.parametrize("kind", ["str", "bytes", "bytearray", "dict", "set", "generator"])
def test_temperature_rejects_text_and_iterables_without_a_numeric_sequence(
    tmp_path, monkeypatch, entry_name, kind
):
    """Temperature accepts numerical scalars and sequences, excluding text and other iterables."""
    import listenflow.fastest_whisper as module
    from listenflow.fastest_whisper import _inference

    def load_gpu(*_args, **_kwargs):
        raise AssertionError("invalid temperature reached GPU model loading")

    monkeypatch.setattr(_inference, "TensorRTWhisper", load_gpu)

    factories = {
        "str": lambda: "0",
        "bytes": lambda: b"0",
        "bytearray": lambda: bytearray(b"0"),
        "dict": lambda: {0.0: "first", 0.4: "retry"},
        "set": lambda: {0.0, 0.4},
        "generator": lambda: (value for value in (0.0, 0.4)),
    }
    temperature = factories[kind]()
    error_type = ValueError if kind in ("str", "bytes", "bytearray") else TypeError
    if entry_name == "engine":
        with pytest.raises(ConfigurationError, match="temperature"):
            _engine(tmp_path, temperature=temperature)
    else:
        with pytest.raises(error_type, match="temperature"):
            getattr(module, entry_name)("unused", np.zeros(16000), temperature=temperature)
