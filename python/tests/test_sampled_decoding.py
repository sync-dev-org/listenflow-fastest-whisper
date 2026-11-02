"""Candidate batches retain the serial decoder's scoring and completion rules."""

from types import SimpleNamespace

import numpy as np
import pytest

from listenflow.fastest_whisper._decoding import search_candidates
from listenflow.fastest_whisper._inference import TensorRTWhisper


def sampled_model(rows, *, context=16):
    """Replace only the external decoder execution; keep sequence handling real."""
    calls = []
    remaining = iter(rows)

    class NativeDecoder:
        def run(self, inputs, *, device_outputs, reuse_cache):
            calls.append({name: value.copy() for name, value in inputs.items()})
            assert reuse_cache
            assert "logits" in device_outputs
            result = next(remaining)
            return {
                "sampled": np.asarray(result, dtype=np.float32).reshape(2, 1, 3),
                "logits": np.zeros((2, 1, 6), np.float32),
                "present_keys": inputs["self_keys"],
                "present_values": inputs["self_values"],
            }

    model = TensorRTWhisper.__new__(TensorRTWhisper)
    model.config = SimpleNamespace(
        text_positions=context, text_layers=1, text_heads=1, audio_state=2, vocab_size=6
    )
    model.precision = "fp32"
    model.initial_tokens = [4]
    model.eot = 3
    model.suppressed = {5}
    model.begin_suppressed = {3}
    model._sampled_decoder = NativeDecoder()
    model._sampled_batch = 2
    model.tokenizer = SimpleNamespace(token_to_id={"<|nospeech|>": 5}.get)
    encoder = SimpleNamespace(tensors={"cross_keys": np.zeros(1), "cross_values": np.zeros(1)})
    return model, encoder, calls


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_sampled_batch_matches_serial_candidate_ranking_and_scores_eot_once():
    """Finished candidates retain their scores while longer candidates finish in the same batch."""
    probabilities = {
        (): (0.55, 0.45, 0.0, 0.0),
        (0,): (0.0, 0.0, 0.1, 0.9),
        (1,): (0.0, 0.9, 0.0, 0.1),
        (1, 1): (0.0, 0.0, 0.0, 1.0),
    }

    def logits(prefix):
        with np.errstate(divide="ignore"):
            return np.log(np.asarray(probabilities[tuple(prefix[1:])], np.float32))

    choices = iter((0, 3, 1, 1, 3))
    expected = search_candidates(
        [4],
        logits,
        eot=3,
        max_tokens=8,
        context_length=16,
        temperature=0.4,
        best_of=2,
        sample_token=lambda _: next(choices),
    )
    model, encoder, calls = sampled_model(
        [
            [(0, np.log(0.55), 0.2), (1, np.log(0.45), 0.2)],
            [(3, np.log(0.9), 0.2), (1, np.log(0.9), 0.2)],
            [(3, -99, 0.2), (3, 0, 0.2)],
        ]
    )

    actual, silence = model._sampled_candidates(
        encoder, [4], temperature=0.4, with_timestamps=False
    )

    assert actual.tokens == expected.tokens == [1, 1]
    assert actual.sum_logprob == pytest.approx(expected.sum_logprob, abs=1e-6)
    assert silence == pytest.approx(0.2)
    assert [call["tokens"].reshape(-1).tolist() for call in calls] == [[4, 4], [0, 1], [3, 1]]
    assert calls[0]["suppression"][0, 0, 3] == -np.inf
    assert calls[1]["suppression"][0, 0, 3] == 0
    assert all(call["suppression"][0, 0, 5] == -np.inf for call in calls)


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_sampled_batch_keeps_first_candidate_on_equal_ranking():
    """Equal mean scores retain the first sampled candidate, including a zero-length sentence."""
    model, encoder, calls = sampled_model(
        [
            [(3, -0.5, 0.1), (1, -0.25, 0.1)],
            [(3, -20, 0.1), (3, -0.25, 0.1)],
        ]
    )
    actual, _ = model._sampled_candidates(encoder, [4], temperature=1.0, with_timestamps=False)
    assert actual.tokens == []
    assert actual.sum_logprob == -0.5
    assert len(calls) == 2


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_sampled_batch_keeps_context_and_token_limits_without_truncating_other_candidates():
    """A context-full prefix generates its last permitted token without an extra decoder step."""
    model, encoder, calls = sampled_model(
        [
            [(0, -1, 0.1), (1, -2, 0.1)],
            [(0, -1, 0.1), (1, -2, 0.1)],
            [(0, -1, 0.1), (1, -2, 0.1)],
            [(0, -1, 0.1), (1, -2, 0.1)],
        ],
        context=4,
    )
    actual, _ = model._sampled_candidates(
        encoder, [4, 2, 2, 2], temperature=0.2, with_timestamps=False
    )
    assert actual.tokens == [0]
    assert len(calls) == 4


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_sampled_batch_uses_seeded_entropy_without_probability_sampling(monkeypatch):
    """The same utterance seed supplies the same independent uniforms to every batch step."""
    factory = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda: factory(421))
    observed = []
    for _ in range(2):
        model, encoder, calls = sampled_model(
            [
                [(0, -0.5, 0.1), (1, -0.5, 0.1)],
                [(3, -0.1, 0.1), (3, -0.1, 0.1)],
            ]
        )
        model._sampled_candidates(encoder, [4], temperature=0.2, with_timestamps=False)
        observed.append([call["uniform"] for call in calls])
    np.testing.assert_array_equal(observed[0], observed[1])
    assert observed[0][0].shape == (2, 1, 1)
    assert not np.array_equal(observed[0][0], observed[0][1])


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_prepared_retry_uses_its_batch_without_building_or_requesting_host_logits():
    """A matching prepared retry returns its chosen text without building or reading full logits."""
    model, encoder, _ = sampled_model(
        [
            [(0, -1.0, 0.1), (1, -0.5, 0.1)],
            [(3, -0.1, 0.1), (3, -0.1, 0.1)],
        ]
    )
    model.tokenizer.decode = lambda tokens, **_: " ".join(map(str, tokens))

    def forbidden(*_args, **_kwargs):
        pytest.fail("a prepared retry rebuilt an engine or requested host logits")

    model.logits = model._open_sampled_engine = forbidden
    actual = model._decode_attempt(
        encoder, [], with_timestamps=False, temperature=0.4, beam_size=None, best_of=2
    )
    assert actual.tokens == [1]
    assert actual.text == "1"
    assert actual.avg_logprob == pytest.approx(-0.6 / 2)


@pytest.mark.req("SP-LFFASTESTWHISPER-047")
def test_sampled_batch_stops_a_phrase_candidate_without_stopping_or_rescoring_other_paths():
    """A rejected sample freezes at its confirmed phrase while the better sample finishes."""
    model, encoder, calls = sampled_model(
        [
            [(0, -1.0, 0.1), (1, -0.1, 0.1)],
            [(2, -1.0, 0.1), (1, -0.1, 0.1)],
            [(0, -99, 0.1), (3, -0.1, 0.1)],
        ]
    )
    model.tokenizer.decode = lambda tokens, **_: "".join(
        {0: "watch", 1: "ok", 2: " "}[t] for t in tokens
    )
    from listenflow.fastest_whisper._rejection import rejection_settings

    rejection = rejection_settings(reject_phrases=["watch"])
    actual = model._decode_attempt(
        encoder,
        [],
        with_timestamps=False,
        temperature=0.4,
        beam_size=None,
        best_of=2,
        rejection=rejection,
    )
    assert actual.text == "okok"
    assert actual.rejection is None
    assert actual.avg_logprob == pytest.approx(-0.3 / 3)
    assert len(calls) == 3
    assert calls[-1]["tokens"].reshape(-1).tolist() == [3, 1]


@pytest.mark.req("SP-LFFASTESTWHISPER-048")
def test_sampled_batch_stops_repeating_paths_at_three_tokens_and_keeps_their_scores():
    """Repeated samples finish at the first periodic tail rather than running to the token limit."""
    model, encoder, calls = sampled_model([[(0, -0.1, 0.1), (1, -0.2, 0.1)]] * 3)
    model.tokenizer.decode = lambda tokens, **_: "".join(map(str, tokens))
    from listenflow.fastest_whisper._rejection import rejection_settings

    actual = model._decode_attempt(
        encoder,
        [],
        with_timestamps=False,
        temperature=0.4,
        beam_size=None,
        best_of=2,
        rejection=rejection_settings(reject_repetition=True),
    )
    assert actual.tokens == [0, 0, 0]
    assert actual.rejection == "repetition"
    assert actual.avg_logprob == pytest.approx(-0.3 / 4)
    assert len(calls) == 3
