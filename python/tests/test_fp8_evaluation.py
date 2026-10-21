"""Product FP8 comparisons use defaults and stable per-utterance sampling."""

import numpy as np
import pytest

from listenflow.fastest_whisper._decoding import DecodingSettings, search_candidates


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_fp32_evaluation_record_does_not_claim_fp8_calibration_or_graph():
    """An fp32 comparison record omits the FP8-only calibration and graph identity."""
    from scripts.evaluate_fp8_product import precision_metadata

    assert not {"scale_sha256", "graph_version"}.intersection(precision_metadata("fp32"))
    selected = precision_metadata("fp8")
    assert len(selected["scale_sha256"]) == 64
    assert selected["graph_version"] == "fp8-1"


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_product_condition_retains_default_retries_and_hint():
    """Condition two retains all six temperatures, five candidates and the requested hint."""
    from scripts.evaluate_fp8_product import condition_options

    settings, hint = condition_options(2)
    assert settings == DecodingSettings()
    assert settings.temperatures == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert settings.best_of == 5
    assert hint == "yakult yakult man"
    greedy, hint = condition_options(1)
    assert greedy.temperatures == (0.0,) and hint is None


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_utterance_seed_repeats_real_sampling_independently_of_other_utterances():
    """Repeated evaluation of one ID samples the same tokens regardless of intervening IDs."""
    from scripts.evaluate_fp8_product import seeded_utterance

    def sample(utterance):
        with seeded_utterance(utterance) as seed:
            candidate = search_candidates(
                [0],
                lambda tokens: np.array([0.0, 0.1, 0.2, -1.0], np.float32),
                eot=3,
                max_tokens=12,
                context_length=20,
                temperature=0.8,
                best_of=5,
            )
        return seed, candidate

    first = sample("1089-134686-0000")
    other = sample("1089-134686-0001")
    assert sample("1089-134686-0000") == first
    assert first[0] != other[0]
    assert first[1].tokens
