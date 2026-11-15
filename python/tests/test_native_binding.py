"""The model binding is usable without initializing CUDA at import time."""

import numpy as np
import pytest
from safetensors.numpy import load_file

from listenflow.fastest_whisper import _native
from listenflow.fastest_whisper._audio import _FILTERS


@pytest.mark.req("SP-LFFASTESTWHISPER-018")
@pytest.mark.req("SP-LFFASTESTWHISPER-033")
def test_native_binding_reports_its_c_abi_without_a_model():
    """Import and ABI introspection require no model or CUDA device."""
    assert _native.abi_version() == (1, 0, 0)


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_native_model_rejects_invalid_dimensions_before_cuda():
    """Malformed configuration is rejected at the binding, before GPU setup."""
    with pytest.raises(ValueError, match="dimension"):
        _native.Model(
            {
                "config": dict.fromkeys(
                    (
                        "n_mels",
                        "audio_positions",
                        "audio_state",
                        "audio_heads",
                        "audio_ffn",
                        "audio_layers",
                        "text_heads",
                        "text_ffn",
                        "text_layers",
                        "text_positions",
                        "vocab_size",
                    ),
                    0,
                ),
                "weights": [],
            },
            "fp32",
            0,
            0,
            [],
            [],
        )


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_native_mel_asset_keeps_every_original_filter_value():
    """The native safetensors input contains the same filters as the CPU oracle."""
    actual = load_file(_FILTERS.with_suffix(".safetensors"))
    with np.load(_FILTERS) as expected:
        assert set(actual) == set(expected.files)
        for name in expected.files:
            assert actual[name].dtype == expected[name].dtype
            np.testing.assert_array_equal(actual[name], expected[name])
