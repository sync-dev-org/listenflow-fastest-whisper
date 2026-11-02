"""Caller-selected phrase and repetition rejection without a CUDA device."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from listenflow import ConfigurationError, OfflineModelDirectorySource
from listenflow.fastest_whisper import FastestWhisperEngine
from listenflow.fastest_whisper._decoding import (
    DecodeAttempt,
    DecodingSettings,
    decode_with_fallback,
    search_candidates,
)
from listenflow.fastest_whisper._inference import TensorRTWhisper


def policy(**options):
    from listenflow.fastest_whisper._rejection import rejection_settings

    return rejection_settings(**options)


@pytest.mark.req("SP-LFFASTESTWHISPER-046")
@pytest.mark.parametrize(
    ("phrase", "text", "final", "matches"),
    [
        ("don't forget to subscribe", "  DON’T\tforget to subscribe!", True, True),
        ("foobar", "foo,bar", True, True),
        ("bar", "foo,bar", True, False),
        ("go", "going", True, False),
        ("go", "go", False, False),
        ("go", "go,", False, False),
        ("go", "go ", False, True),
        ("go", "go+", False, True),
        ("go", "go\u0301", True, False),
        ("go", "go\u200d", False, True),
        ("watch", "watch_thing", True, False),
        ("谢谢", "谢谢观看", True, False),
        ("谢谢", "谢谢", True, True),
    ],
)
def test_phrases_match_normalized_whole_words_after_the_right_boundary_is_known(
    phrase, text, final, matches
):
    """Punctuation vanishes and unfinished words cannot reject a candidate prematurely."""
    rejection = policy(reject_phrases=[phrase])
    assert (rejection.reason([], text, final=final) == "phrase") is matches


@pytest.mark.req("SP-LFFASTESTWHISPER-045")
@pytest.mark.req("SP-LFFASTESTWHISPER-048")
@pytest.mark.req("SP-LFFASTESTWHISPER-049")
@pytest.mark.req("SP-LFFASTESTWHISPER-050")
@pytest.mark.parametrize(
    "options",
    [
        {"reject_phrases": "watch"},
        {"reject_phrases": ("watch",)},
        {"reject_phrases": None},
        {"reject_phrases": [1]},
        {"reject_phrases": [""]},
        {"reject_phrases": [" …\t!"]},
        {"reject_repetition": 1},
        {"reject_repetition": None},
        {"phrase_hit_no_speech": "yes"},
        {"on_rejected_exhausted": "skip"},
        {"on_rejected_exhausted": []},
    ],
)
def test_constructor_rejects_invalid_rejection_configuration_before_loading(tmp_path, options):
    """Invalid lists, empty normalized phrases, flags and exhaustion modes are configuration errors."""
    with pytest.raises(ConfigurationError):
        FastestWhisperEngine(
            model_source=OfflineModelDirectorySource(str(tmp_path)),
            precision="fp16",
            device=0,
            language="en",
            **options,
        )


@pytest.mark.req("SP-LFFASTESTWHISPER-045")
def test_constructor_copies_the_phrase_list_and_keeps_rejection_disabled_by_default(tmp_path):
    """Changing the caller's list cannot change an initialized engine's rejection rules."""
    phrases = ["watch"]
    engine = FastestWhisperEngine(
        model_source=OfflineModelDirectorySource(str(tmp_path)),
        precision="fp16",
        device=0,
        language="en",
        reject_phrases=phrases,
    )
    phrases.clear()
    assert engine._decoding_settings.rejection.reason([], "watch", final=True) == "phrase"
    default = FastestWhisperEngine(
        model_source=OfflineModelDirectorySource(str(tmp_path)),
        precision="fp16",
        device=0,
        language="en",
    )._decoding_settings.rejection
    assert not default.enabled
    assert default.on_exhausted == "pass"
    assert default.phrase_hit_no_speech is True


@pytest.mark.req("SP-LFFASTESTWHISPER-048")
@pytest.mark.parametrize("period", [1, 2, 15])
def test_repetition_rejects_three_copies_of_a_tail_period_only_when_enabled(period):
    """Three copies of a 1–15 token suffix stop generation; shorter or broken tails continue."""
    unit = list(range(period))
    rejection = policy(reject_repetition=True)
    assert rejection.reason(unit * 2, "", final=False) is None
    assert rejection.reason(unit * 3, "", final=False) == "repetition"
    assert rejection.reason(unit * 3 + [99], "", final=False) is None
    assert policy().reason(unit * 3, "", final=False) is None
    assert rejection.reason(list(range(16)) * 3, "", final=False) is None


def fallback(attempts, rejection, *, temperatures=(0.0, 0.4, 0.8), **thresholds):
    seen = []

    def decode(temperature, beam, best_of):
        seen.append((temperature, beam, best_of))
        return replace(attempts[min(len(seen) - 1, len(attempts) - 1)], temperature=temperature)

    result = decode_with_fallback(
        decode,
        temperatures=temperatures,
        beam_size=2,
        best_of=3,
        compression_ratio_threshold=thresholds.get("compression_ratio_threshold", 2.4),
        logprob_threshold=thresholds.get("logprob_threshold", -1.0),
        no_speech_threshold=thresholds.get("no_speech_threshold", 0.6),
        rejection=rejection,
    )
    return result, seen


def attempt(*, reason=None, text="watch", probability=0.1, logprob=-0.2, ratio=1.0):
    return DecodeAttempt([0], text, logprob, probability, ratio, 0.0, rejection=reason)


@pytest.mark.req("SP-LFFASTESTWHISPER-047")
def test_rejected_selected_candidate_advances_once_then_accepts_the_next_temperature():
    """A rejected winning candidate retries, while a later successful winner ends the window."""
    result, seen = fallback(
        [attempt(reason="phrase"), attempt(text="accepted")], policy(reject_phrases=["watch"])
    )
    assert result.text == "accepted"
    assert seen == [(0.0, 2, None), (0.4, None, 3)]


@pytest.mark.req("SP-LFFASTESTWHISPER-047")
@pytest.mark.req("SP-LFFASTESTWHISPER-049")
@pytest.mark.parametrize("reason", ["phrase", "repetition"])
@pytest.mark.parametrize("temperatures", [(0.0,), (0.0, 0.4, 0.8)])
@pytest.mark.parametrize("mode", ["pass", "drop"])
def test_exhaustion_applies_only_after_the_last_rejected_attempt(reason, temperatures, mode):
    """Retries never exceed the supplied temperature count and pass/drop uses the last text."""
    rejection = policy(on_rejected_exhausted=mode)
    result, seen = fallback([attempt(reason=reason)], rejection, temperatures=temperatures)
    assert len(seen) == len(temperatures)
    assert result.text == ("watch" if mode == "pass" else "")
    assert result.tokens == ([0] if mode == "pass" else [])


@pytest.mark.req("SP-LFFASTESTWHISPER-049")
@pytest.mark.parametrize("failed", [{"ratio": 4.0}, {"logprob": -2.0}])
def test_quality_failure_without_rejection_keeps_official_last_attempt_even_in_drop_mode(failed):
    """An earlier phrase rejection cannot make a later quality-only failure drop its text."""
    result, seen = fallback(
        [attempt(reason="phrase"), attempt(text="quality only", **failed)],
        policy(on_rejected_exhausted="drop"),
        temperatures=(0.0, 0.4),
    )
    assert result.text == "quality only"
    assert len(seen) == 2


@pytest.mark.req("SP-LFFASTESTWHISPER-047")
@pytest.mark.req("SP-LFFASTESTWHISPER-050")
@pytest.mark.parametrize(
    ("reason", "enabled", "threshold", "logprob", "log_threshold", "silent"),
    [
        ("phrase", True, 0.6, -0.2, -1.0, True),
        ("phrase", False, 0.6, -0.2, -1.0, False),
        ("phrase", True, None, -2.0, -1.0, False),
        ("phrase", False, 0.6, -2.0, -1.0, True),
        ("phrase", False, 0.6, -1.0, -1.0, True),
        ("phrase", False, 0.6, -0.2, None, True),
        ("repetition", True, 0.6, -0.2, -1.0, False),
        ("repetition", True, 0.6, -2.0, -1.0, True),
        (None, True, 0.6, -0.2, -1.0, False),
    ],
)
def test_rejected_attempts_apply_silence_before_retry_and_phrase_override_is_scoped(
    reason, enabled, threshold, logprob, log_threshold, silent
):
    """Phrase-only early silence honors its flag and nullable thresholds before consuming a retry."""
    result, seen = fallback(
        [attempt(reason=reason, probability=0.8, logprob=logprob), attempt(text="next")],
        policy(phrase_hit_no_speech=enabled),
        no_speech_threshold=threshold,
        logprob_threshold=log_threshold,
    )
    assert (result.text == "") is silent
    assert len(seen) == (1 if silent or reason is None else 2)


@pytest.mark.req("SP-LFFASTESTWHISPER-047")
@pytest.mark.parametrize("beam", [None, 2])
def test_stopped_losing_candidates_remain_ranked_without_rejecting_the_winner(beam):
    """The rejected lower-scoring path finishes early while the ordinary winner still completes."""
    seen = []

    def logits(prefix):
        generated = prefix[1:]
        seen.append(generated)
        probabilities = [0.7, 0.3, 0.0] if not generated else [0.0, 0.0, 1.0]
        with np.errstate(divide="ignore"):
            return np.log(np.array(probabilities, dtype=np.float32))

    choices = iter((1, 0, 2))
    chosen = search_candidates(
        [9],
        logits,
        eot=2,
        max_tokens=8,
        context_length=16,
        beam_size=beam,
        temperature=0.4 if beam is None else 0.0,
        best_of=2,
        sample_token=lambda _: next(choices),
        stop_candidate=lambda tokens, final: tokens == [1],
    )
    assert chosen.tokens == [0]
    assert [1] not in seen
    assert [0] in seen


@pytest.mark.req("SP-LFFASTESTWHISPER-047")
@pytest.mark.req("SP-LFFASTESTWHISPER-048")
@pytest.mark.parametrize("enabled", [False, True])
def test_device_selected_greedy_tokens_stop_on_repetition_and_release_the_cache(enabled):
    """The native token-selection route stops only when enabled and frees its K/V cache."""
    calls = []

    class Decoder:
        outputs = ("token",)

        def run(self, inputs, **_):
            calls.append(int(inputs["positions"][0, 0]))
            return {
                "greedy": np.array([0, -0.2, 0.1], dtype=np.float32),
                "token": np.array([0], dtype=np.int32),
                "logits": np.zeros(1),
                "present_keys": np.zeros(1),
                "present_values": np.zeros(1),
            }

    model = TensorRTWhisper.__new__(TensorRTWhisper)
    model.config = SimpleNamespace(
        text_positions=16, text_layers=1, text_heads=1, audio_state=2, vocab_size=6
    )
    model.precision = "fp32"
    model.initial_tokens = [4]
    model.eot = 3
    model.suppressed = model.begin_suppressed = set()
    model._prefix_states = {}
    model._decoder = Decoder()
    model.tokenizer = SimpleNamespace(
        token_to_id={"<|nospeech|>": 5}.get,
        decode=lambda tokens, **_: "".join("a" for _ in tokens),
    )
    encoder = SimpleNamespace(tensors={"cross_keys": np.zeros(1), "cross_values": np.zeros(1)})
    model._encoded_audio = encoder
    settings = DecodingSettings(
        temperatures=(0.0,),
        no_speech_threshold=None,
        rejection=policy(reject_repetition=enabled),
    )
    actual = model.decode_short(encoder, [], settings)
    assert actual.text == "a" * (3 if enabled else 8)
    assert len(calls) == (3 if enabled else 8)
    assert model._greedy_cache == {}


@pytest.fixture
def byte_level_model():
    """Use the real byte-level decoder, replacing only native token/logit execution."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    merged = ["go", "+à", "à¤"]
    special = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        "<|en|>",
        "<|transcribe|>",
        "<|notimestamps|>",
        "<|nospeech|>",
    ]
    vocabulary = {token: index for index, token in enumerate(alphabet + merged + special)}
    tokenizer = Tokenizer(models.BPE(vocabulary, [("g", "o"), ("+", "à"), ("à", "¤")]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(special)

    def make(tokens, route):
        model = TensorRTWhisper.__new__(TensorRTWhisper)
        model.tokenizer = tokenizer
        model.eot = vocabulary[special[0]]
        model.initial_tokens = [vocabulary[token] for token in special[1:5]]
        model.config = SimpleNamespace(
            text_positions=64,
            text_layers=1,
            text_heads=1,
            audio_state=2,
            vocab_size=len(vocabulary),
        )
        model.precision = "fp32"
        model.suppressed = model.begin_suppressed = set()
        observed = []

        def next_token(index):
            token = tokens[index] if index < len(tokens) else model.eot
            observed.append(token)
            return token

        def logits(prefix, _encoder):
            token = next_token(len(prefix) - len(model.initial_tokens))
            values = np.full((len(prefix), len(vocabulary)), -np.inf, np.float32)
            values[:, token] = 0
            return values

        class NativeSamples:
            def run(self, inputs, **_):
                position = int(inputs["positions"][0, 0])
                index = position + 1 - len(model.initial_tokens)
                token = next_token(index) if index >= 0 else tokens[0]
                return {
                    "sampled": np.array([[[token, 0, 0]], [[token, 0, 0]]], np.float32),
                    "logits": np.zeros(1),
                    "present_keys": inputs["self_keys"],
                    "present_values": inputs["self_values"],
                }

        model.logits = logits
        if route == "sampled-batch":
            model._sampled_batch = 2
            model._sampled_decoder = NativeSamples()
        return model, observed

    return tokenizer, make


@pytest.mark.req("SP-LFFASTESTWHISPER-046")
@pytest.mark.parametrize("route", ["greedy", "beam", "sampled", "sampled-batch"])
@pytest.mark.parametrize("sentence", ["goअ", "go\u0315"])
def test_split_unicode_letters_and_combining_marks_do_not_reject_an_unfinished_word(
    byte_level_model, route, sentence
):
    """An incomplete character after a phrase cannot turn that phrase into a whole word."""
    tokenizer, make = byte_level_model
    tokens = tokenizer.encode(sentence, add_special_tokens=False).ids
    assert "\ufffd" in tokenizer.decode(tokens[:-1], skip_special_tokens=False)
    model, observed = make(tokens, route)
    rejection = policy(reject_phrases=["go"], on_rejected_exhausted="drop")
    settings = DecodingSettings(
        temperatures=(0.4 if route.startswith("sampled") else 0.0,),
        beam_size=2 if route == "beam" else 1,
        best_of=2,
        no_speech_threshold=None,
        rejection=rejection,
    )
    encoder = SimpleNamespace(tensors={"cross_keys": np.zeros(1), "cross_values": np.zeros(1)})
    result = model.decode_short(encoder, [], settings)
    assert result.text == sentence
    assert result.tokens == tokens + [model.eot]
    assert model.eot in observed


@pytest.mark.req("SP-LFFASTESTWHISPER-046")
@pytest.mark.parametrize(
    ("sentence", "stop_prefix"),
    [("go+अtail", "go+à"), ("go\ufffdtail", "go\ufffd"), ("go tail", "go ")],
)
def test_byte_level_stop_uses_confirmed_boundaries_without_waiting_for_later_tokens(
    byte_level_model, sentence, stop_prefix
):
    """A complete replacement character or real delimiter confirms the boundary immediately."""
    tokenizer, make = byte_level_model
    tokens = tokenizer.encode(sentence, add_special_tokens=False).ids
    model, observed = make(tokens, "greedy")
    # The first token after '+' includes the first byte of an unfinished अ.
    if sentence.startswith("go+"):
        expected = [tokenizer.token_to_id("go"), tokenizer.token_to_id("+à")]
    else:
        expected = tokenizer.encode(stop_prefix, add_special_tokens=False).ids
    result = model._greedy_attempt(None, [], rejection=policy(reject_phrases=["go"]))
    assert result.tokens == expected
    assert result.rejection == "phrase"
    assert model.eot not in observed


@pytest.mark.req("SP-LFFASTESTWHISPER-046")
@pytest.mark.parametrize("incomplete", [False, True])
def test_actual_eot_confirms_a_boundary_even_after_unfinished_utf8(byte_level_model, incomplete):
    """Only the actual end of generation finalizes a pending byte sequence and a trailing word."""
    tokenizer, make = byte_level_model
    tokens = tokenizer.encode("goअ" if incomplete else "go", add_special_tokens=False).ids
    if incomplete:
        tokens = tokens[:-1]
    model, observed = make(tokens, "greedy")
    rejection = policy(reject_phrases=["go"], on_rejected_exhausted="drop")
    attempt = model._greedy_attempt(None, [], rejection=rejection)
    assert attempt.tokens == tokens
    assert model.eot in observed
    result, seen = fallback([attempt], rejection, temperatures=(0.0,))
    assert attempt.rejection == "phrase"
    assert result.text == "" and len(seen) == 1
