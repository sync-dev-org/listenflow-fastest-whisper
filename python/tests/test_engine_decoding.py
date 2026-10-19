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
        language="en",
        boundary_settings=UtteranceBoundarySettings(preroll_s=0),
        **options,
    )


@pytest.fixture
def runtime(monkeypatch):
    from listenflow.fastest_whisper import _cached_model
    from listenflow.fastest_whisper import _engine as engine_module

    monkeypatch.setattr(engine_module, "_cuda_device_info", lambda _device: (1, (12, 0)))

    def install(*, repeated=False, no_speech=0.1, starts=(1,), by_temperature=False):
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
        }
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="hint"))
        tokenizer.pre_tokenizer = Whitespace()

        class NativeModel(CachedWhisper):
            """Replace GPU loading, features, and logits; retain transcription and search."""

            def __init__(self, _checkpoint, *, precision, **_options):
                self.precision = precision
                self.config = SimpleNamespace(n_mels=1, audio_positions=1500, text_positions=32)
                self.tokenizer = tokenizer
                self.initial_tokens = [5, 6, 7, 8]
                self.eot = 3
                self.suppressed = set(range(4, 11))
                self.begin_suppressed = set()

            def warmup(self):
                pass

            def close(self):
                pass

            def _mel_spectrogram(self, audio):
                return np.zeros((1, 3000), dtype=np.float32)

            def _mel_spectrogram_long(self, audio):
                return np.zeros((1, audio.size // 160 + 3000), dtype=np.float32)

            def encode(self, _mel):
                return np.empty(0, dtype=np.float32)

            def logits(self, prefix, _encoder):
                task = len(prefix) - 1 - prefix[::-1].index(7)
                long = len(prefix) <= task + 1 or prefix[task + 1] != 8
                generated = prefix[task + (1 if long else 2) :]
                output = np.full((len(prefix), 62), -np.inf, dtype=np.float32)
                output[task - 2, 4] = log(no_speech)
                output[task - 2, 0] = log(1 - no_speech)
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


def _texts(runner, engine, *, seconds=1, utterances=1):
    pcm = np.full(seconds * 16000, 1000, dtype="<i2").tobytes()

    async def chunks():
        for index in range(utterances):
            yield pcm
            if index + 1 < utterances:
                yield np.zeros(16000, dtype="<i2").tobytes()

    async def collect():
        await engine.prepare()
        return [event.text async for event in engine.stream(chunks()) if isinstance(event, Final)]

    return runner.run(collect())


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
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


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
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


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
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


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
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


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
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


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
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


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize(("name", "value"), INVALID)
def test_invalid_decoding_values_fail_when_constructing_the_engine(tmp_path, name, value):
    """Invalid decoding settings raise ConfigurationError without preparation or GPU access."""
    with pytest.raises(ConfigurationError, match=name):
        _engine(tmp_path, **{name: value})


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
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
