"""The selected product precision controls preparation and packaged quantization."""

import asyncio
import json
from importlib.resources import files

import pytest

from listenflow import ConfigurationError, EnvironmentUnavailableError, OfflineModelDirectorySource
from listenflow.fastest_whisper import FastestWhisperEngine


def engine(tmp_path, precision="fp8"):
    return FastestWhisperEngine(
        model_source=OfflineModelDirectorySource(str(tmp_path)),
        precision=precision,
        device=0,
        language="en",
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_fp8_is_an_explicit_precision_and_unknown_values_are_rejected(tmp_path):
    """Selecting fp8 retains fp8 and all default decoding options; unknown values fail."""
    selected = engine(tmp_path)
    assert selected.precision == "fp8"
    assert selected._decoding_settings == engine(tmp_path, "fp16")._decoding_settings
    for invalid in ("fp4", "int8", None, 8):
        with pytest.raises(ConfigurationError, match="fp8"):
            engine(tmp_path, invalid)


@pytest.mark.req("SP-LFFASTESTWHISPER-013")
def test_prepare_rejects_fp8_on_sm86_before_loading_or_warming(tmp_path, monkeypatch):
    """sm86 preparation reports unsupported fp8 and offers fp16/fp32 without loading engines."""
    from listenflow.fastest_whisper import _engine

    monkeypatch.setattr(_engine, "_cuda_device_info", lambda device: (1, (8, 6)))
    selected = engine(tmp_path)
    monkeypatch.setattr(selected, "_load_model", lambda: pytest.fail("must not build on sm86"))
    with pytest.raises(EnvironmentUnavailableError, match="fp8") as failure:
        asyncio.run(selected.prepare())
    assert "fp16" in str(failure.value) and "fp32" in str(failure.value)
    assert "sm86" in str(failure.value)
    assert selected.precision == "fp8"
    assert selected._model is None


@pytest.mark.req("SP-LFFASTESTWHISPER-004")
def test_packaged_scales_cover_every_large_v3_linear_input():
    """The package contains all 512 positive activation scales used for large-v3."""
    import math

    artifact = files("listenflow.fastest_whisper").joinpath("assets/fp8-scales.json")
    scales = json.loads(artifact.read_text())
    expected = {
        f"model.{part}.layers.{layer}.{operation}"
        for part in ("encoder", "decoder")
        for layer in range(32)
        for operation in (
            "fc1",
            "fc2",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            *(
                (
                    "encoder_attn.q_proj",
                    "encoder_attn.k_proj",
                    "encoder_attn.v_proj",
                    "encoder_attn.out_proj",
                )
                if part == "decoder"
                else ()
            ),
        )
    }
    assert scales.keys() == expected
    assert all(math.isfinite(value) and value > 0 for value in scales.values())
