"""The shipped FP8 route recognizes speech and rebuilds changed saved engines."""

import json
from pathlib import Path

import numpy as np
import pytest
from scripts.evaluate_wer import DecodeRetryRecorder

from listenflow import Final
from listenflow.fastest_whisper import _inference
from listenflow.fastest_whisper._assets import FP8_SCALES
from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._precision import PRECISIONS
from test_engine_gpu import (
    _engine,
    _engine_identity,
    _events,
    _pcm,
)

pytest_plugins = ("test_engine_gpu",)


def _device_capability(device):
    from cuda.bindings import runtime as cudart

    status, properties = cudart.cudaGetDeviceProperties(device)
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA device {device} capability query failed: {status}")
    return int(properties.major), int(properties.minor)


def _require_fp8_device(device=0):
    capability = _device_capability(device)
    if capability not in PRECISIONS["fp8"].capabilities:
        pytest.skip(
            f"fp8 GPU tests require sm89 or sm120; selected sm{capability[0]}{capability[1]}"
        )


@pytest.fixture(autouse=True)
def fp8_capable_device(gpu):
    _require_fp8_device()


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-004")
@pytest.mark.req("SP-LFFASTESTWHISPER-010")
@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize(
    "options,repeats",
    [
        ({"language": None}, 1),
        ({"beam_size": 2, "temperature": 0.0, "initial_prompt": "the"}, 1),
        (
            {
                "temperature": (0.0, 0.8),
                "best_of": 2,
                "compression_ratio_threshold": 0.0,
                "logprob_threshold": None,
                "no_speech_threshold": None,
            },
            1,
        ),
        ({"temperature": 0.0, "initial_prompt": "the"}, 3),
    ],
)
def test_fp8_stream_retains_auto_language_beam_sampling_and_long_audio(
    gpu, engine_loop, options, repeats
):
    """Explicit fp8 completes speech through each existing public decoding configuration."""
    selected = _engine(gpu, precision="fp8", **options)
    try:
        with DecodeRetryRecorder("decode_short" if repeats == 1 else "decode_long") as recorder:
            events = _events(
                engine_loop, selected, _pcm("small_boat_16khz_mono_s16le.wav") * repeats
            )
        if options.get("compression_ratio_threshold") == 0.0:
            assert any(record["temperatures"] == [0.0, 0.8] for record in recorder.records)
            print("forced retry observed:", recorder.summary())
        texts = [event.text for event in events if isinstance(event, Final)]
        assert texts and any("boat" in text.lower() for text in texts)
        assert selected.precision == selected._model.precision == "fp8"
        info = selected._model._native_model.info
        assert info["encoder_dtype"] == info["decoder_dtype"] == "float16"
    finally:
        engine_loop.run(selected.aclose())


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-008")
def test_fp8_saved_engines_reuse_and_separate_changed_calibration(gpu, tmp_path, monkeypatch):
    """Native plans are reused for the same calibration and separated when its bytes change."""
    data = tmp_path / "scales.json"
    data.write_bytes(FP8_SCALES.read_bytes())
    monkeypatch.setattr(_inference, "FP8_SCALES", data)

    def recognize():
        with CachedWhisper(gpu, precision="fp8") as model:
            samples = np.frombuffer(_pcm("small_boat_16khz_mono_s16le.wav"), "<i2")
            segments = model.transcribe(samples.astype(np.float32) / 32768, temperature=0.0)
            assert "boat" in " ".join(segment.text for segment in segments).lower()
            return dict(model._native_model.info["engine_paths"])

    paths = recognize()
    assert set(paths) == {"mel", "encoder", "decoder"}
    before = {name: _engine_identity(Path(path)) for name, path in paths.items()}
    assert recognize() == paths
    assert before == {name: _engine_identity(Path(path)) for name, path in paths.items()}
    scales = json.loads(data.read_text())
    scales[next(iter(scales))] *= 1.0001
    data.write_text(json.dumps(scales))
    changed = recognize()
    assert changed["mel"] == paths["mel"]
    assert all(changed[name] != paths[name] for name in ("encoder", "decoder"))
    assert len(set(paths.values()) | set(changed.values())) == 5
    assert before == {name: _engine_identity(Path(path)) for name, path in paths.items()}
    rebuilt = {name: _engine_identity(Path(path)) for name, path in changed.items()}
    assert recognize() == changed
    assert rebuilt == {name: _engine_identity(Path(path)) for name, path in changed.items()}
