"""Opt-in integration checks for the listenflow stream and cached TensorRT engines."""

import asyncio
import gc
import hashlib
import json
import mmap
import os
import re
import subprocess
import sys
import wave
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory

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
    initial_prompt=None,
    **decoding_options,
):
    return FastestWhisperEngine(
        model_source=OfflineModelDirectorySource(str(checkpoint)),
        precision=precision,
        device=device,
        language=language,
        boundary_settings=UtteranceBoundarySettings(preroll_s=0),
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


def _damage_saved_plugin(path, damage):
    """Keep the envelope valid while making a serialized creator unavailable."""
    magic = b"TENSORACLE\x01"
    name = b"LffwFp16Fc1"
    namespace = b"listenflow.fastest_whisper"
    with path.open("r+b") as stream:
        assert stream.read(len(magic)) == magic
        size = int.from_bytes(stream.read(8), "big")
        header = json.loads(stream.read(size))
        if not header["conditions"]["name"].endswith("-decoder"):
            return False
        plan_offset = stream.tell()
        with mmap.mmap(stream.fileno(), 0) as contents:
            count, offset = 0, plan_offset
            while (offset := contents.find(name, offset)) >= 0:
                if damage == "missing-plugin":
                    # Creator names are length-prefixed, rather than NUL-terminated.
                    # The plan also contains NUL-terminated diagnostic strings.
                    contents[offset : offset + len(name)] = b"LffwNoneFc1"
                    count += 1
                else:
                    start = offset + len(name)
                    end = contents.find(namespace, start, start + 128)
                    if end >= 0:
                        record = contents[start:end]
                        # Between the creator name and namespace, version 1 is the
                        # only ASCII '1'. Binary field lengths/tags use byte 0x01.
                        if record.count(b"1") == 1:
                            version = start + record.index(b"1")
                            contents[version : version + 1] = b"9"
                            count += 1
                offset += len(name)
            assert count > 0, "the saved decoder did not contain its required creator"
            header["plan_sha256"] = hashlib.sha256(memoryview(contents)[plan_offset:]).hexdigest()
            encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
            assert len(encoded) == size
            contents[len(magic) + 8 : plan_offset] = encoded
    return True


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
def test_cached_engines_reuse_and_automatically_recover_across_processes(
    gpu, tmp_path, monkeypatch
):
    """Saved graphs recover from corruption, changed conditions or unavailable plugins and recognize speech."""
    code = """
import asyncio, json, sys, wave
from listenflow import Final, OfflineModelDirectorySource
from listenflow.fastest_whisper import FastestWhisperEngine
async def run():
    async with FastestWhisperEngine(
        model_source=OfflineModelDirectorySource(sys.argv[1]),
        precision=sys.argv[2], device=0, language='en', temperature=0.0,
    ) as engine:
        with wave.open(sys.argv[3], 'rb') as source:
            pcm = source.readframes(source.getnframes())
        async def chunks():
            yield pcm
        print(json.dumps([event.text async for event in engine.stream(chunks())
                          if isinstance(event, Final)]))
asyncio.run(run())
"""

    def recognize(precision, unavailable=None):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(gpu),
                precision,
                str(FIXTURES / "small_boat_16khz_mono_s16le.wav"),
            ],
            capture_output=True,
            text=True,
            timeout=PREPARE_TIMEOUT_SECONDS,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        if unavailable is not None:
            diagnostics = result.stdout + result.stderr
            assert "IPluginRegistry" in diagnostics, diagnostics
            if unavailable == "missing-plugin":
                assert "LffwNoneFc1" in diagnostics, diagnostics
            else:
                assert "LffwFp16Fc1" in diagnostics, diagnostics
                assert re.search(r"\bversion\b[\s:=\"']*9\b", diagnostics), diagnostics
        text = json.loads(result.stdout.splitlines()[-1])
        assert text and all(text)
        return text

    identities = []
    for precision in ("fp32", "fp16"):
        # Keep one precision on disk at a time, including when an assertion fails.
        with TemporaryDirectory(dir=tmp_path, prefix=f"{precision}-") as directory:
            cache = Path(directory)
            monkeypatch.setenv("XDG_CACHE_HOME", directory)
            expected = recognize(precision)
            files = _engine_files(cache)
            assert len(files) == 3
            identities.append({path.name for path in files})
            before = {path.name: (path.stat().st_mtime_ns, _engine_hash(path)) for path in files}
            assert recognize(precision) == expected
            assert {
                path.name: (path.stat().st_mtime_ns, _engine_hash(path)) for path in files
            } == before

            for damage in ("corrupt", "incompatible", "missing-plugin", "plugin-version"):
                if precision != "fp16" and damage in ("missing-plugin", "plugin-version"):
                    continue
                for target in files:
                    unchanged = {path: path.stat().st_mtime_ns for path in files if path != target}
                    if damage == "corrupt":
                        target.write_bytes(b"truncated")
                    elif damage == "incompatible":
                        with target.open("r+b") as stream:
                            stream.read(len(b"TENSORACLE\x01"))
                            size = int.from_bytes(stream.read(8), "big")
                            header = json.loads(stream.read(size))
                            version = header["conditions"]["graph_version"]
                            header["conditions"]["graph_version"] = version[:-1] + "x"
                            encoded = json.dumps(
                                header, sort_keys=True, separators=(",", ":")
                            ).encode()
                            assert len(encoded) == size
                            stream.seek(len(b"TENSORACLE\x01") + 8)
                            stream.write(encoded)
                    elif not _damage_saved_plugin(target, damage):
                        continue
                    damaged = _engine_hash(target)
                    unavailable = damage if damage in ("missing-plugin", "plugin-version") else None
                    assert recognize(precision, unavailable) == expected
                    assert _engine_hash(target) != damaged, (precision, damage, target.name)
                    assert {path: path.stat().st_mtime_ns for path in unchanged} == unchanged
                    rebuilt = {path: path.stat().st_mtime_ns for path in files}
                    assert recognize(precision) == expected
                    assert {path: path.stat().st_mtime_ns for path in files} == rebuilt
                    print(f"{precision}: {damage}: {target.name}: rebuilt and reused", flush=True)
    assert identities[0].isdisjoint(identities[1])


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
