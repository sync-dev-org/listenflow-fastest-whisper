"""The native decoder preserves token results when its cache enters new buckets."""

import numpy as np
import pytest

from listenflow.fastest_whisper import _native
from native_runtime_fixture import encoded

pytestmark = [pytest.mark.gpu]


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_boundary_lengths_keep_mutable_and_immutable_results(native_model):
    """Greedy graph replay yields the same choices as independent immutable prefixes."""
    audio = encoded(native_model)
    tokens = native_model.initial_tokens + [1, 2] * 27
    decoder = audio.decoder(_native.GREEDY)
    try:
        for position, token in enumerate(tokens):
            first = position < len(native_model.initial_tokens)
            selected, _, _ = decoder.greedy_step(token, first, False)
            raw = native_model.logits(tokens[: position + 1], audio)[-1].astype(np.float32)
            raw[list(native_model.suppressed)] = -np.inf
            if first:
                raw[list(native_model.begin_suppressed)] = -np.inf
            assert selected == int(np.argmax(raw))
    finally:
        decoder.close()


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_invalid_continuation_does_not_poison_native_chain(native_model):
    """An invalid token is rejected before state changes and a valid token still runs."""
    audio = encoded(native_model)
    decoder = audio.decoder(_native.GREEDY)
    try:
        with pytest.raises(ValueError, match="vocabulary"):
            decoder.greedy_step(-1, True, False)
        selected, _, _ = decoder.greedy_step(native_model.initial_tokens[0], True, False)
        assert 0 <= selected < 16
    finally:
        decoder.close()
