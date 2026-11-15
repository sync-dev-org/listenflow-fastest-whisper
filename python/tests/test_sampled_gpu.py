"""GPU sampling frequencies through the real native model's decoder interface."""

import numpy as np
import pytest
from scripts.evaluate_fp8_product import seeded_utterance

from listenflow.fastest_whisper import _native
from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._decoding import DecodingSettings
from native_runtime_fixture import encoded, record_engines

pytestmark = [pytest.mark.gpu, pytest.mark.req("SP-LFFASTESTWHISPER-002")]
COUNT, BATCH = 40000, 100


@pytest.fixture(scope="module")
def sampler(native_checkpoint):
    with CachedWhisper(
        native_checkpoint, settings=DecodingSettings(temperatures=(0.4,), best_of=BATCH)
    ) as model:
        record_engines(native_checkpoint, model)
        yield model


@pytest.mark.parametrize("temperature", [0.2, 1.0])
def test_gpu_sampling_matches_softmax_with_six_sigma_bound_and_same_seed(sampler, temperature):
    """Native inverse-CDF draws match masked decoder probabilities and unscaled scores."""
    audio = encoded(sampler)
    token = sampler.initial_tokens[0]
    raw = sampler.logits([token], audio)[-1].astype(np.float64)
    kept = np.array([1, 2, 15])
    suppression = np.full((BATCH, 16), -np.inf, np.float32)
    suppression[:, kept] = 0
    samples, scores = [], []
    random = np.random.default_rng(61120)
    for _ in range(COUNT // BATCH):
        chain = audio.decoder(_native.SAMPLED, BATCH)
        try:
            selected, logprob, _ = chain.sampled_step(
                np.full(BATCH, token, np.int32),
                suppression,
                temperature,
                random.random(BATCH, dtype=np.float32),
                False,
            )
            samples.extend(selected)
            scores.extend(logprob)
        finally:
            chain.close()
    tokens = np.asarray(samples)
    centered = raw[kept] - raw[kept].max()
    expected = np.exp(centered / temperature)
    expected /= expected.sum()
    counts = np.array([(tokens == value).sum() for value in kept])
    assert counts.sum() == COUNT
    assert np.all(abs(counts - COUNT * expected) < 6 * np.sqrt(COUNT * expected * (1 - expected)))
    logprobs = centered - np.log(np.exp(centered).sum())
    observed = np.array([logprobs[np.flatnonzero(kept == value)[0]] for value in tokens])
    np.testing.assert_allclose(scores, observed, atol=3e-6)
    chain = audio.decoder(_native.SAMPLED, BATCH)
    first_uniform = np.random.default_rng(61120).random(BATCH, dtype=np.float32)
    try:
        repeated, _, _ = chain.sampled_step(
            np.full(BATCH, token, np.int32), suppression, temperature, first_uniform, False
        )
    finally:
        chain.close()
    np.testing.assert_array_equal(repeated, tokens[:BATCH])


def test_prepared_gpu_batch_repeats_utterance_seed_without_logit_or_prefix_transfers(
    native_checkpoint,
):
    """A prepared retry uses its resident audio and returns the same seeded token sequence."""
    settings = DecodingSettings(
        temperatures=(0.4,),
        best_of=2,
        no_speech_threshold=None,
        compression_ratio_threshold=None,
        logprob_threshold=None,
    )
    with CachedWhisper(native_checkpoint, settings=settings) as model:
        record_engines(native_checkpoint, model)
        audio = encoded(model)

        def forbidden(*args, **kwargs):
            pytest.fail("prepared sampling requested raw host logits")

        model.logits = forbidden
        values = []
        for _ in range(2):
            with seeded_utterance("native-repeat"):
                values.append(model.decode_short(audio, [], settings).tokens)
        assert values[0] == values[1]
