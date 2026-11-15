"""Immutable native prefixes preserve independent decoder branches."""

import numpy as np
import pytest

from native_runtime_fixture import encoded

pytestmark = [pytest.mark.gpu]


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_prefix_extension_and_repeated_prefix_retain_rows(native_model):
    """Incremental extension agrees with a fresh prefix and repeated rows remain owned."""
    audio = encoded(native_model)
    prefix = native_model.initial_tokens + [1, 2, 1]
    first = native_model.logits(prefix[:4], audio).copy()
    actual = native_model.logits(prefix, audio).copy()
    np.testing.assert_array_equal(actual[:4], first)
    np.testing.assert_array_equal(native_model.logits(prefix, audio), actual)
    native_model._clear_prefix_states()
    np.testing.assert_array_equal(native_model.logits(prefix, audio), actual)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_beam_reordering_uses_selected_parent_cache(native_model):
    """Sibling extensions cannot overwrite each other's immutable K/V state."""
    audio = encoded(native_model)
    root = native_model.initial_tokens
    branches = [root + [1, 2], root + [2, 1], root + [1, 1]]
    values = [native_model.logits(path, audio).copy() for path in branches]
    for path, expected in zip(reversed(branches), reversed(values), strict=True):
        np.testing.assert_array_equal(native_model.logits(path, audio), expected)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_new_audio_replaces_cross_attention_and_prefix_cache(native_model):
    """Old audio is rejected and the next window has a fresh prefix cache."""
    old = encoded(native_model)
    native_model.logits(native_model.initial_tokens, old)
    current = native_model.encode(np.ones((80, 3000), np.float32))
    assert not native_model._prefix_states
    with pytest.raises(ValueError, match="current"):
        native_model.logits(native_model.initial_tokens, old)
    assert np.isfinite(native_model.logits(native_model.initial_tokens, current)).all()
