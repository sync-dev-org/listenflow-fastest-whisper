"""GPU-free checks of the official initial-prompt decoder rules."""

from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models

from listenflow.fastest_whisper._inference import TensorRTWhisper


@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_hint_text_uses_one_leading_space_after_trimming_the_edges():
    """The decoder tokenizes the same text as the official initial_prompt path."""
    tokenizer = Tokenizer(
        models.WordLevel(
            vocab={"[UNK]": 0, " comet delta": 11, "comet delta": 12}, unk_token="[UNK]"
        )
    )
    model = TensorRTWhisper.__new__(TensorRTWhisper)
    model.tokenizer = tokenizer

    assert model.prompt_tokens("  comet delta \n") == [11]
    assert model.prompt_tokens(None) == []


@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_short_decode_places_hint_before_start_of_transcript():
    """A short utterance receives recent hint tokens as previous text before transcription."""
    model = TensorRTWhisper.__new__(TensorRTWhisper)
    model.config = SimpleNamespace(text_positions=448)
    model.initial_tokens = [1, 2, 3, 4]
    model.eot = 5
    model.suppressed = set()
    model.begin_suppressed = set()
    model.tokenizer = SimpleNamespace(
        decode=lambda *_args, **_kwargs: "", token_to_id=lambda _token: 7
    )
    prefixes = []

    def logits(prefix, _encoder):
        prefixes.append(prefix.copy())
        result = np.zeros((len(prefix), 10), dtype=np.float32)
        result[-1, 5] = 10.0
        return result

    model.logits = logits
    encoder = np.empty((0,), dtype=np.float32)

    assert model.greedy_decode(encoder, prompt=list(range(300))).tokens == [5]
    assert prefixes == [[7] + list(range(300))[-223:] + [1, 2, 3, 4]]
    prefixes.clear()
    assert model.greedy_decode(encoder).tokens == [5]
    assert prefixes == [[1, 2, 3, 4]]


@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_long_decode_reads_no_speech_score_from_start_token_after_hint():
    """A hint containing a start token does not move the no-speech logit lookup."""
    model = TensorRTWhisper.__new__(TensorRTWhisper)
    model.config = SimpleNamespace(text_positions=448)
    model.initial_tokens = [1, 2, 3, 4]
    model.eot = 5
    model.suppressed = set()
    model.begin_suppressed = set()
    ids = {"<|0.00|>": 8, "<|startofprev|>": 7, "<|nospeech|>": 6, "<|notimestamps|>": 4}
    model.tokenizer = SimpleNamespace(token_to_id=ids.get)
    calls = 0

    def logits(prefix, _encoder):
        nonlocal calls
        calls += 1
        result = np.zeros((len(prefix), 12), dtype=np.float32)
        if calls == 1:
            result[1, 6] = 20.0  # The start token inside the hint.
            result[2, 6] = -20.0  # The actual start token after the hint.
            result[-1, 8] = 20.0
        else:
            result[-1, 5] = 20.0
        return result

    model.logits = logits
    result = model.timestamp_decode(np.empty((0,), dtype=np.float32), prompt=[1])

    assert result.no_speech_prob < 0.01
    assert calls == 2


@pytest.mark.req("SP-LFFASTESTWHISPER-009")
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize(
    ("prompt", "expected_steps", "initial_length"),
    [(list(range(300)), 221, 228), (None, 224, 4)],
)
def test_short_decode_finalizes_when_the_decoder_context_is_full(
    prompt, expected_steps, initial_length
):
    """A long hint stops after the last allowed logits call and keeps the last sampled token."""
    model = TensorRTWhisper.__new__(TensorRTWhisper)
    model.config = SimpleNamespace(text_positions=448)
    model.initial_tokens = [1, 2, 3, 4]
    model.eot = 5
    model.suppressed = set()
    model.begin_suppressed = set()
    model.tokenizer = SimpleNamespace(
        decode=lambda *_args, **_kwargs: "decoded", token_to_id=lambda _token: 7
    )
    lengths = []

    def logits(prefix, _encoder):
        length = len(prefix)
        if length > 448:
            raise ValueError("decoder prefix exceeds checkpoint context length")
        lengths.append(length)
        result = np.zeros((length, 10), dtype=np.float32)
        result[-1, 9] = 10.0
        return result

    model.logits = logits

    actual = model.greedy_decode(np.empty((0,), dtype=np.float32), prompt=prompt)

    assert actual.tokens == [9] * expected_steps + [5]
    assert lengths == list(range(initial_length, initial_length + expected_steps))
