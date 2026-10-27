"""The shipped FP8 route recognizes speech and rebuilds changed saved engines."""

import json

import numpy as np
import pytest
import tensorrt as trt
from scripts.evaluate_wer import DecodeRetryRecorder

from listenflow import Final
from listenflow.fastest_whisper import _fp8
from listenflow.fastest_whisper._cached_model import CachedWhisper
from test_engine_gpu import (
    _engine,
    _engine_files,
    _engine_hash,
    _events,
    _pcm,
)

pytest_plugins = ("test_engine_gpu",)


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
        for name in ("self_keys", "self_values", "cross_keys", "cross_values", "logits"):
            assert selected._model._decoder.execution.get_tensor_dtype(name) == trt.float16
    finally:
        engine_loop.run(selected.aclose())


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-008")
def test_fp8_saved_engines_reuse_and_rebuild_for_changed_scales_and_graph(
    gpu, tmp_path, monkeypatch
):
    """Real TensorRT builds recur only when scale bytes or the FP8 graph version change."""
    import tensoracle._cache as cache

    data = tmp_path / "scales.json"
    data.write_bytes(_fp8._SCALES.read_bytes())
    monkeypatch.setattr(_fp8, "_SCALES", data)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "engine-cache"))
    builds = []
    original_build = cache._build_serialized_engine

    def observe_build(graph, declaration, **kwargs):
        result = original_build(graph, declaration, **kwargs)
        builds.append(tuple(declaration.inputs))
        return result

    monkeypatch.setattr(cache, "_build_serialized_engine", observe_build)

    def recognize():
        with CachedWhisper(gpu, precision="fp8") as model:
            samples = np.frombuffer(_pcm("small_boat_16khz_mono_s16le.wav"), "<i2")
            segments = model.transcribe(samples.astype(np.float32) / 32768, temperature=0.0)
            assert "boat" in " ".join(segment.text for segment in segments).lower()

    recognize()
    assert len(builds) == 3
    files = _engine_files(tmp_path / "engine-cache")
    before = {path: (path.stat().st_mtime_ns, _engine_hash(path)) for path in files}
    recognize()
    assert len(builds) == 3
    assert before == {path: (path.stat().st_mtime_ns, _engine_hash(path)) for path in files}
    scales = json.loads(data.read_text())
    scales[next(iter(scales))] *= 1.0001
    data.write_text(json.dumps(scales))
    recognize()
    assert len(builds) == 5, "scale bytes must rebuild encoder and decoder"
    recognize()
    assert len(builds) == 5
    monkeypatch.setattr(_fp8, "_GRAPH_VERSION", _fp8._GRAPH_VERSION + "-next")
    recognize()
    assert len(builds) == 7, "graph version must rebuild encoder and decoder"
    recognize()
    assert len(builds) == 7
    print("actual TensorRT builds: initial=3, reuse=0, changed-scales=2, changed-version=2")
