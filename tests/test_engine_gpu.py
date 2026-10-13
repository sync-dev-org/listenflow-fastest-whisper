"""Opt-in integration checks for the listenflow stream and cached TensorRT engines."""

import asyncio
import gc
import hashlib
import json
import os
import subprocess
import sys
import wave
import weakref
from pathlib import Path

import numpy as np
import pytest

from listenflow import (
    EngineState,
    Final,
    OfflineModelDirectorySource,
    Partial,
    UtteranceBoundarySettings,
    UtteranceEnd,
    UtteranceStart,
)
from listenflow.fastest_whisper import (
    FastestWhisperEngine,
    UnsupportedLanguageError,
    transcribe_fp16,
    transcribe_long_fp32,
)
from listenflow.fastest_whisper._engine import _cuda_device_info

FIXTURES = Path(__file__).parent / "fixtures"
# A fresh TensorRT build takes several minutes on sm86; the limit only stops a hung process.
PREPARE_TIMEOUT_SECONDS = 1800


@pytest.fixture(scope="module")
def gpu_cache(tmp_path_factory):
    if configured := os.environ.get("LFFW_TEST_GPU_CACHE_HOME"):
        return Path(configured)
    return tmp_path_factory.mktemp("shared-gpu-cache")


@pytest.fixture
def gpu(gpu_cache, monkeypatch):
    checkpoint = os.environ.get("LFFW_CHECKPOINT_DIR")
    if os.environ.get("LFFW_USE_GPU") != "1" or not checkpoint:
        pytest.skip("set LFFW_USE_GPU=1 and LFFW_CHECKPOINT_DIR for GPU checks")
    monkeypatch.setenv("XDG_CACHE_HOME", str(gpu_cache))
    yield Path(checkpoint)


def _pcm(name):
    with wave.open(str(FIXTURES / name), "rb") as source:
        assert (source.getframerate(), source.getnchannels(), source.getsampwidth()) == (
            16_000,
            1,
            2,
        )
        return source.readframes(source.getnframes())


def _engine(
    checkpoint,
    *,
    precision="fp32",
    language="en",
    device=0,
    force_rebuild=False,
    initial_prompt=None,
    **decoding_options,
):
    return FastestWhisperEngine(
        model_source=OfflineModelDirectorySource(str(checkpoint)),
        precision=precision,
        device=device,
        language=language,
        boundary_settings=UtteranceBoundarySettings(preroll_s=0),
        force_rebuild=force_rebuild,
        initial_prompt=initial_prompt,
        **decoding_options,
    )


async def _chunks(pcm):
    yield pcm


@pytest.fixture
def engine_loop():
    with asyncio.Runner() as runner:
        yield runner


def _events(engine_loop, engine, pcm):
    async def collect():
        await engine.prepare()
        return [event async for event in engine.stream(_chunks(pcm))]

    return engine_loop.run(collect())


def _engine_files(cache):
    return sorted((cache / "tensoracle").glob("*.engine"))


def _engine_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _engine_name(path):
    with path.open("rb") as stream:
        stream.read(len(b"TENSORACLE\x01"))
        header_size = int.from_bytes(stream.read(8), "big")
        return json.loads(stream.read(header_size))["conditions"]["name"]


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_fp32_engine_stream_matches_official_text(gpu, engine_loop):
    """The fp32 engine emits the official greedy text through the core event stream."""
    pcm = _pcm("small_boat_16khz_mono_s16le.wav")
    metadata = json.loads((FIXTURES / "oracle" / "metadata.json").read_text())
    expected = metadata["fixtures"]["small_boat_16khz_mono_s16le"]["text"]
    events = _events(engine_loop, _engine(gpu), pcm)
    assert [type(event) for event in events] == [UtteranceStart, Partial, Final, UtteranceEnd]
    assert events[2].text == expected


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("prompt", [None, "the"])
def test_fp32_engine_stream_uses_selected_beam_with_or_without_a_hint(gpu, engine_loop, prompt):
    """The selected beam emits the official text, including when a hint supplies context."""
    name = "beam_with_prompt.json" if prompt else "beam_size_2.json"
    oracle = json.loads((FIXTURES / "oracle" / name).read_text())
    engine = _engine(gpu, beam_size=2, temperature=0.0, initial_prompt=prompt)
    try:
        actual = [
            event.text
            for event in _events(engine_loop, engine, _pcm("small_boat_16khz_mono_s16le.wav"))
            if isinstance(event, Final)
        ]
        assert actual == [oracle["fixtures"]["small_boat_16khz_mono_s16le"]["text"]]
    finally:
        engine_loop.run(engine.aclose())


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_long_engine_stream_uses_selected_beam_and_hint(gpu, engine_loop):
    """Long hinted utterances use the chosen beam and retain official text and offsets."""
    oracle = json.loads((FIXTURES / "oracle" / "beam_long_with_prompt.json").read_text())
    engine = _engine(gpu, beam_size=2, temperature=0.0, initial_prompt=oracle["prompt"])
    try:
        actual = [
            event
            for event in _events(engine_loop, engine, _pcm("small_boat_16khz_mono_s16le.wav") * 3)
            if isinstance(event, Final)
        ]
        assert [(event.text, event.ts_offset_s) for event in actual] == [
            (segment["text"], segment["start"]) for segment in oracle["segments"]
        ]
    finally:
        engine_loop.run(engine.aclose())


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_fp16_engine_stream_uses_selected_sampling_and_retry_settings(
    gpu, engine_loop, monkeypatch
):
    """The fp16 stream applies the same hinted retry and sampling policy as its numerical entry."""
    factory = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda: factory(42))
    pcm = _pcm("small_boat_16khz_mono_s16le.wav")
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    settings = {
        "initial_prompt": "the",
        "beam_size": 2,
        "temperature": (0.0, 0.8),
        "best_of": 2,
        "compression_ratio_threshold": 0.0,
        "logprob_threshold": None,
        "no_speech_threshold": None,
    }
    expected = transcribe_fp16(gpu, samples, **settings).text
    assert expected
    engine = _engine(gpu, precision="fp16", **settings)
    try:
        assert [
            event.text for event in _events(engine_loop, engine, pcm) if isinstance(event, Final)
        ] == [expected]
    finally:
        engine_loop.run(engine.aclose())


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_engine_stream_uses_hint_and_reports_its_application(gpu, engine_loop):
    """A configured hint changes recognized text and health reports its use."""
    pcm = _pcm("small_boat_16khz_mono_s16le.wav")
    oracle = json.loads((FIXTURES / "oracle" / "initial_prompt.json").read_text())["cases"][
        "long_hint/small_boat_16khz_mono_s16le"
    ]
    engine = _engine(gpu, initial_prompt=oracle["prompt"])

    actual = [event.text for event in _events(engine_loop, engine, pcm) if isinstance(event, Final)]

    assert actual == [oracle["text"]]
    bias = engine.health().bias_application
    assert (bias.kind, bias.prompt, bias.applied) == ("prompt", oracle["prompt"], True)


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_fp16_engine_stream_matches_fp16_entry(gpu, engine_loop):
    """The fp16 engine presents the same text as the fp16 correctness entry."""
    pcm = _pcm("small_boat_16khz_mono_s16le.wav")
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    expected = transcribe_fp16(gpu, samples).text
    events = _events(engine_loop, _engine(gpu, precision="fp16"), pcm)
    assert [event.text for event in events if isinstance(event, Final)] == [expected]


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_fp16_engine_accepts_a_hint_and_completes_transcription(gpu, engine_loop):
    """The fp16 stream can decode with a configured hint without a quality assertion."""
    engine = _engine(gpu, precision="fp16", initial_prompt="the")

    events = _events(engine_loop, engine, _pcm("small_boat_16khz_mono_s16le.wav"))

    assert [event.text for event in events if isinstance(event, Final)]
    assert engine.health().bias_application.applied


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_long_utterance_matches_long_fp32_segments(gpu, engine_loop, monkeypatch):
    """A continuous utterance over 30 seconds keeps long-form text and source offsets."""
    factory = np.random.default_rng
    # Compare the two routes under the same entropy; retries use real sampling.
    monkeypatch.setattr(np.random, "default_rng", lambda: factory(42))
    pcm = _pcm("small_boat_16khz_mono_s16le.wav") * 3
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    expected = transcribe_long_fp32(gpu, samples)
    actual = [
        event for event in _events(engine_loop, _engine(gpu), pcm) if isinstance(event, Final)
    ]
    segments = [segment for segment in expected.segments if segment.text.strip()]
    assert [(event.text, event.ts_offset_s) for event in actual] == [
        (segment.text, segment.start) for segment in segments
    ]


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_auto_detects_english_and_rejects_japanese(gpu, engine_loop):
    """Automatic detection transcribes English and reports the official Japanese code."""
    english = _pcm("small_boat_16khz_mono_s16le.wav")
    automatic = _engine(gpu, language=None)
    fixed = _engine(gpu, language="en")
    assert [
        event.text for event in _events(engine_loop, automatic, english) if isinstance(event, Final)
    ] == [event.text for event in _events(engine_loop, fixed, english) if isinstance(event, Final)]
    japanese_path = FIXTURES / "ja_pronunciation_practice5_25s_16khz_mono_s16le.wav"
    japanese = _pcm(japanese_path.name)
    with pytest.raises(UnsupportedLanguageError) as failure:
        _events(engine_loop, automatic, japanese)
    assert failure.value.language == "ja"


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_nonzero_device_is_used_when_visible(gpu, monkeypatch, engine_loop):
    """The selected nonzero CUDA device builds and transcribes when present."""
    count, _ = _cuda_device_info(1)
    if count < 2:
        pytest.skip("this host exposes only one CUDA device")
    from cuda.bindings import runtime as cudart

    selected_devices = []
    set_device = cudart.cudaSetDevice

    def observe_device(device):
        result = set_device(device)
        assert result[0] == cudart.cudaError_t.cudaSuccess
        status, selected = cudart.cudaGetDevice()
        assert status == cudart.cudaError_t.cudaSuccess
        selected_devices.append(selected)
        return result

    monkeypatch.setattr(cudart, "cudaSetDevice", observe_device)
    engine = _engine(gpu, device=1)
    assert [
        event
        for event in _events(engine_loop, engine, _pcm("small_boat_16khz_mono_s16le.wav"))
        if isinstance(event, Final)
    ]
    assert len(selected_devices) >= 4
    assert set(selected_devices) == {1}


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-008")
def test_cached_engines_reuse_across_processes_and_require_force_for_corruption(
    gpu, tmp_path, monkeypatch
):
    """Separate processes reuse matching precision and reject damaged encoder or decoder files."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    code = """
import asyncio, sys
from listenflow import OfflineModelDirectorySource
from listenflow.fastest_whisper import FastestWhisperEngine
engine = FastestWhisperEngine(
    model_source=OfflineModelDirectorySource(sys.argv[1]),
    precision=sys.argv[3], device=0, language='en', force_rebuild=sys.argv[2] == '1'
)
asyncio.run(engine.prepare())
"""

    def prepare(*, force_rebuild=False, precision="fp32"):
        return subprocess.run(
            [sys.executable, "-c", code, str(gpu), "1" if force_rebuild else "0", precision],
            capture_output=True,
            text=True,
            timeout=PREPARE_TIMEOUT_SECONDS,
            check=False,
        )

    first = prepare()
    assert first.returncode == 0, first.stdout + first.stderr
    files = _engine_files(tmp_path)
    assert len(files) == 3
    before = {path.name: (path.stat().st_mtime_ns, _engine_hash(path)) for path in files}
    second = prepare()
    assert second.returncode == 0, second.stdout + second.stderr
    assert {path.name: (path.stat().st_mtime_ns, _engine_hash(path)) for path in files} == before

    fp16 = prepare(precision="fp16")
    assert fp16.returncode == 0, fp16.stdout + fp16.stderr
    precision_files = _engine_files(tmp_path)
    assert len(precision_files) == 6
    added = [path for path in precision_files if path not in files]
    assert len(added) == 3
    assert sorted(_engine_name(path) for path in added) == [
        "listenflow-fastest-whisper-decoder",
        "listenflow-fastest-whisper-encoder",
        "listenflow-fastest-whisper-mel",
    ]
    assert {path.name: (path.stat().st_mtime_ns, _engine_hash(path)) for path in files} == before

    for name in (
        "listenflow-fastest-whisper-encoder",
        "listenflow-fastest-whisper-decoder",
        "listenflow-fastest-whisper-mel",
    ):
        target = next(path for path in files if _engine_name(path) == name)
        target.write_bytes(b"truncated")
        failure = prepare()
        assert failure.returncode != 0
        assert "CorruptEngineError" in failure.stderr
        assert "EnvironmentUnavailableError" in failure.stderr
        assert str(target) in failure.stderr
        assert "force_rebuild=True" in failure.stderr
        rebuilt = prepare(force_rebuild=True)
        assert rebuilt.returncode == 0, rebuilt.stdout + rebuilt.stderr
        reused = prepare()
        assert reused.returncode == 0, reused.stdout + reused.stderr


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("release", ["aclose", "async_with", "drop_prepared", "drop_streamed"])
def test_engine_returns_gpu_memory_without_cyclic_collection(gpu, engine_loop, release):
    """Closing or dropping an engine after preparation or streaming returns GPU memory without gc."""
    from cuda.bindings import runtime as cudart

    def free_bytes():
        status, free, _total = cudart.cudaMemGetInfo()
        assert status == cudart.cudaError_t.cudaSuccess
        return free

    assert cudart.cudaSetDevice(0)[0] == cudart.cudaError_t.cudaSuccess
    assert cudart.cudaFree(0)[0] == cudart.cudaError_t.cudaSuccess
    baseline = free_bytes()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        engine = _engine(gpu, precision="fp16")
        reference = weakref.ref(engine)

        pcm = _pcm("small_boat_16khz_mono_s16le.wav")
        if release == "async_with":

            async def recognize_in_context(owner):
                async with owner:
                    model_reference = weakref.ref(owner._model)
                    loaded = free_bytes()
                    events = [event async for event in owner.stream(_chunks(pcm))]
                    assert isinstance(events[-1], UtteranceEnd)
                return model_reference, loaded

            model_reference, loaded = engine_loop.run(recognize_in_context(engine))
        else:
            engine_loop.run(engine.prepare())
            model_reference = weakref.ref(engine._model)
            loaded = free_bytes()
            if release in ("aclose", "drop_streamed"):
                assert isinstance(_events(engine_loop, engine, pcm)[-1], UtteranceEnd)
            if release == "aclose":
                engine_loop.run(engine.aclose())
        assert baseline - loaded > 512 * 1024**2, "preparation did not allocate model GPU memory"
        if release in ("aclose", "async_with"):
            assert engine.health().state is EngineState.CLOSED
            assert model_reference() is None
        del engine
        if release in ("drop_prepared", "drop_streamed"):
            assert reference() is None, "the engine retained a reference cycle"
        assert model_reference() is None, "native runtime still has an owner"
        returned = free_bytes()
        print({"release": release, "baseline": baseline, "loaded": loaded, "returned": returned})
        assert returned >= baseline - 64 * 1024**2, (baseline, loaded, returned)
    finally:
        if was_enabled:
            gc.enable()
