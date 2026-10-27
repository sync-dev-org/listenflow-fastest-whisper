"""The listenflow engine contract with only the native GPU boundary replaced."""

import asyncio

import numpy as np
import pytest
from tensoracle import CorruptEngineError, EngineConditionError, EngineIOError

from listenflow import (
    ConfigurationError,
    EngineClosedError,
    EngineInputFormat,
    EngineState,
    EngineStateError,
    EnvironmentUnavailableError,
    Final,
    InferenceError,
    OfflineModelDirectorySource,
    Partial,
    STTEngineBase,
    UtteranceBoundarySettings,
    UtteranceEnd,
    UtteranceStart,
)


def _engine(tmp_path, **overrides):
    from listenflow.fastest_whisper import FastestWhisperEngine

    arguments = {
        "model_source": OfflineModelDirectorySource(str(tmp_path)),
        "precision": "fp32",
        "device": 0,
        "language": "en",
        "boundary_settings": UtteranceBoundarySettings(preroll_s=0),
    }
    arguments.update(overrides)
    return FastestWhisperEngine(**arguments)


async def _chunks(*chunks):
    for chunk in chunks:
        yield chunk


def _collect(native, engine, *chunks):
    async def run():
        await engine.prepare()
        return [event async for event in engine.stream(_chunks(*chunks))]

    return native.run(run())


class _Model:
    """A native runtime boundary; the engine and core pipeline remain real."""

    def __init__(self, segments=None, language="en"):
        from listenflow.fastest_whisper._longform import TimedSegment

        self.segments = segments if segments is not None else [TimedSegment(0.25, 0.5, "hello")]
        self.language = language
        self.detect_calls = 0
        self.audio = []
        self.closed = 0
        self.warmed = 0

    def warmup(self):
        self.warmed += 1

    def close(self):
        self.closed += 1

    def transcribe(self, audio, **kwargs):
        self.audio.append(audio.copy())
        return self.segments

    def detect_language(self, audio):
        self.detect_calls += 1
        return self.language


@pytest.fixture
def native(monkeypatch):
    from listenflow.fastest_whisper import _cached_model
    from listenflow.fastest_whisper import _engine as engine_module

    # Device enumeration is a native CUDA boundary; no CUDA calls run in CPU tests.
    monkeypatch.setattr(engine_module, "_cuda_device_info", lambda device: (2, (12, 0)))

    def install(model):
        monkeypatch.setattr(_cached_model, "CachedWhisper", lambda *args, **kwargs: model)
        return model

    with asyncio.Runner() as runner:
        install.run = runner.run
        yield install


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_engine_requires_and_validates_its_public_configuration(tmp_path):
    """Invalid engine settings are rejected as configuration errors before recognition."""
    from listenflow.fastest_whisper import FastestWhisperEngine

    assert isinstance(_engine(tmp_path), STTEngineBase)
    for missing in ("model_source", "precision", "device", "language"):
        arguments = {
            "model_source": OfflineModelDirectorySource(str(tmp_path)),
            "precision": "fp32",
            "device": 0,
            "language": "en",
        }
        del arguments[missing]
        with pytest.raises(TypeError):
            FastestWhisperEngine(**arguments)
    for changed in (
        {"precision": "int8"},
        {"precision": 1},
        {"device": -1},
        {"device": True},
        {"device": "0"},
        {"model_source": str(tmp_path)},
        {"boundary_settings": None},
        {"input_format": "pcm_s16le"},
    ):
        with pytest.raises(ConfigurationError):
            _engine(tmp_path, **changed)
    with pytest.raises(ConfigurationError, match="pcm_s16le.*pcm_f32"):
        _engine(tmp_path, input_format=EngineInputFormat.WAV_PCM_S16LE)


@pytest.mark.req("SP-LFFASTESTWHISPER-008")
@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize("value", [False, True, 1])
def test_engine_rejects_caller_selected_cache_rebuild(tmp_path, value):
    """Engine construction has no setting for rebuilding saved TensorRT engines."""
    with pytest.raises(TypeError, match="force_rebuild"):
        _engine(tmp_path, force_rebuild=value)


@pytest.mark.req("SP-LFFASTESTWHISPER-009")
@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_initial_prompt_configuration_accepts_long_and_whitespace_hints(tmp_path):
    """The engine rejects empty and nontext hints while accepting text of any token length."""
    assert _engine(tmp_path, initial_prompt="  ").initial_prompt == "  "
    long_prompt = "comet " * 1_000
    assert _engine(tmp_path, initial_prompt=long_prompt).initial_prompt == long_prompt
    for value in ("", 7, b"comet", ["comet"]):
        with pytest.raises(ConfigurationError, match="initial_prompt"):
            _engine(tmp_path, initial_prompt=value)


@pytest.mark.req("SP-LFFASTESTWHISPER-009")
@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize("prompt", [None, "Quetzalcoatl"])
def test_health_reports_hint_only_after_a_successful_decode(tmp_path, native, prompt):
    """Each stream reports its configured hint and whether a decode received it."""
    observed = []
    engine = _engine(tmp_path, initial_prompt=prompt)

    class PromptModel(_Model):
        def transcribe(self, audio, initial_prompt=None, **kwargs):
            observed.append((initial_prompt, engine.health().bias_application))
            return super().transcribe(audio)

    native(PromptModel())
    assert engine.health().bias_application is None
    speech = np.full(160, 1000, dtype="<i2").tobytes()
    _collect(native, engine, speech)
    first = engine.health().bias_application
    _collect(native, engine, speech)
    second = engine.health().bias_application
    assert [seen_prompt for seen_prompt, _snapshot in observed] == [prompt, prompt]
    assert [snapshot.applied for _prompt, snapshot in observed] == [False, False]
    assert [
        (snapshot.kind, snapshot.prompt, snapshot.applied_value) for _, snapshot in observed
    ] == [("prompt" if prompt is not None else "none", prompt, prompt)] * 2
    assert first.applied and second.applied
    assert second.session_id > first.session_id


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_capabilities_match_accepted_languages_and_are_static(tmp_path):
    """The engine accepts exactly its declared language codes plus automatic detection."""
    english = _engine(tmp_path)
    automatic = _engine(tmp_path, language=None, input_format=EngineInputFormat.PCM_F32)
    assert english.capabilities.languages == automatic.capabilities.languages == frozenset({"en"})
    assert english.capabilities.auto_language is automatic.capabilities.auto_language is True
    for language in english.capabilities.languages:
        assert _engine(tmp_path, language=language).language == language
    for language in ("ja", "English", "en-US", "EN", "en_us", "eng", "", 7, b"en"):
        with pytest.raises(ConfigurationError, match="language"):
            _engine(tmp_path, language=language)
    assert automatic.capabilities.input_format is EngineInputFormat.PCM_F32
    assert english.capabilities.partial_mode.value == "none"
    assert {mode.value for mode in english.capabilities.finalization} == {
        "local_vad",
        "client_flush",
    }


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_stream_requires_prepare_and_health_records_progress(tmp_path, native):
    """Recognition requires preparation and reports PCM and completed-decode progress."""
    model = native(_Model())
    engine = _engine(tmp_path)
    quiet = np.zeros(160, dtype="<i2").tobytes()
    speech = np.full(160, 1000, dtype="<i2").tobytes()
    assert engine.health().state is EngineState.UNPREPARED
    assert engine.health().last_progress_monotonic_s is None

    async def run():
        with pytest.raises(EngineStateError, match="prepare"):
            await anext(engine.stream(_chunks(speech)))
        await engine.prepare()
        await engine.prepare()
        assert model.warmed == 1
        assert engine.health().state is EngineState.READY
        stream = engine.stream(_chunks(quiet, speech))
        start = await anext(stream)
        health = engine.health()
        assert health.busy
        assert health.last_progress_monotonic_s is not None
        before = health.last_progress_monotonic_s
        events = [start, *[event async for event in stream]]
        assert engine.health().last_progress_monotonic_s >= before
        assert not engine.health().busy
        await engine.aclose()
        assert engine.health().state is EngineState.CLOSED
        await engine.aclose()
        assert model.closed == 1
        with pytest.raises(EngineClosedError):
            await engine.prepare()
        return events

    events = native.run(run())
    assert [type(event) for event in events] == [UtteranceStart, Partial, Final, UtteranceEnd]
    assert events[2].text == "hello"
    assert events[2].ts_offset_s == pytest.approx(0.26)
    assert engine.health().processed_pcm_chunks == 2
    assert engine.health().emitted_events == 4
    assert len(model.audio) == 1


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize("text", [None, "", "   "])
def test_silent_decode_closes_utterance_without_text_events(tmp_path, native, text):
    """A spoken interval with no recognized text still has a start and end."""
    from listenflow.fastest_whisper._longform import TimedSegment

    segments = [] if text is None else [TimedSegment(0, 0.01, text)]
    native(_Model(segments=segments))
    events = _collect(native, _engine(tmp_path), np.full(160, 1000, dtype="<i2").tobytes())
    assert [type(event) for event in events] == [UtteranceStart, UtteranceEnd]
    assert events[0].utterance_id == events[1].utterance_id
    assert (events[0].ts_offset_s, events[1].ts_offset_s) == (0, 0.01)


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_automatic_language_rejects_non_english_and_recovers(tmp_path, native):
    """An unsupported detected language is a configuration error and permits reuse."""
    from listenflow.fastest_whisper import UnsupportedLanguageError

    model = native(_Model(language="ja"))
    engine = _engine(tmp_path, language=None)
    speech = np.full(160, 1000, dtype="<i2").tobytes()
    with pytest.raises(ConfigurationError, match="ja") as failure:
        _collect(native, engine, speech)
    assert isinstance(failure.value, UnsupportedLanguageError)
    assert failure.value.language == "ja"
    model.language = "en"
    assert [type(event) for event in _collect(native, engine, speech)] == [
        UtteranceStart,
        Partial,
        Final,
        UtteranceEnd,
    ]
    assert model.detect_calls == 2
    _collect(native, _engine(tmp_path, language="en"), speech)
    assert model.detect_calls == 2


@pytest.mark.req("SP-LFFASTESTWHISPER-008")
@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize("error_type", [CorruptEngineError, EngineConditionError, EngineIOError])
def test_prepare_preserves_cache_failure_without_caller_rebuild_guidance(
    tmp_path, native, monkeypatch, error_type
):
    """An unrecoverable cache failure reports its path and cause without a rebuild setting."""
    from listenflow.fastest_whisper import _cached_model

    native(_Model())
    cause = error_type(f"invalid cached engine: {tmp_path / 'broken.engine'}")

    def fail(*args, **kwargs):
        raise cause

    monkeypatch.setattr(_cached_model, "CachedWhisper", fail)
    with pytest.raises(EnvironmentUnavailableError, match="broken.engine") as failure:
        native.run(_engine(tmp_path).prepare())
    assert failure.value.__cause__ is cause
    assert "force_rebuild" not in str(failure.value)


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize(
    "cause", [OSError("checkpoint unavailable"), RuntimeError("native build failed")]
)
def test_prepare_maps_native_loading_errors_and_can_retry(tmp_path, native, monkeypatch, cause):
    """Native load errors keep their cause and leave the same engine retryable."""
    from listenflow.fastest_whisper import _cached_model

    model = native(_Model())
    calls = 0

    def load(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise cause
        return model

    monkeypatch.setattr(_cached_model, "CachedWhisper", load)
    engine = _engine(tmp_path)
    with pytest.raises(EnvironmentUnavailableError) as failure:
        native.run(engine.prepare())
    assert failure.value.__cause__ is cause
    assert engine.health().state is EngineState.UNPREPARED
    native.run(engine.prepare())
    assert calls == 2
    assert engine.health().state is EngineState.READY


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_failed_warmup_returns_resources_before_retry(tmp_path, native):
    """A failed trial inference returns its model before preparation is retried."""
    cause = RuntimeError("native warmup failed")

    class FailingModel(_Model):
        def warmup(self):
            if self.warmed == 0:
                self.warmed += 1
                raise cause
            super().warmup()

    model = native(FailingModel())
    engine = _engine(tmp_path)
    with pytest.raises(EnvironmentUnavailableError) as failure:
        native.run(engine.prepare())
    assert failure.value.__cause__ is cause
    assert model.closed == 1
    assert engine.health().state is EngineState.UNPREPARED
    native.run(engine.prepare())
    assert engine.health().state is EngineState.READY


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_second_stream_is_rejected_while_first_continues(tmp_path, native):
    """Concurrent streams are refused without closing the earlier stream."""
    native(_Model())
    engine = _engine(tmp_path)
    speech = np.full(160, 1000, dtype="<i2").tobytes()

    async def run():
        await engine.prepare()
        started = asyncio.Event()
        release = asyncio.Event()

        async def first_chunks():
            yield speech
            started.set()
            await release.wait()

        first = engine.stream(first_chunks())
        assert isinstance(await anext(first), UtteranceStart)
        await started.wait()
        with pytest.raises(EngineStateError, match="already has an active stream"):
            await anext(engine.stream(_chunks(speech)))
        release.set()
        return [event async for event in first]

    assert [type(event) for event in native.run(run())] == [Partial, Final, UtteranceEnd]
    assert isinstance(_collect(native, engine, speech)[-1], UtteranceEnd)


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize("detected", [False, True])
def test_decode_failure_preserves_native_cause_and_permits_reuse(tmp_path, native, detected):
    """A native transcription or language-detector failure maps to an inference error."""
    model = native(_Model())
    cause = RuntimeError("native decode failed")
    operation = "detect_language" if detected else "transcribe"
    original = getattr(model, operation)
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise cause
        return original(*args, **kwargs)

    setattr(model, operation, fail_once)
    engine = _engine(tmp_path, language=None if detected else "en")
    speech = np.full(160, 1000, dtype="<i2").tobytes()
    with pytest.raises(InferenceError) as failure:
        _collect(native, engine, speech)
    assert failure.value.__cause__ is cause
    assert isinstance(_collect(native, engine, speech)[-1], UtteranceEnd)


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_f32_pcm_reaches_decoder_as_little_endian_samples(tmp_path, native):
    """The selected f32 PCM input reaches the model without integer scaling."""
    model = native(_Model())
    engine = _engine(tmp_path, input_format=EngineInputFormat.PCM_F32)
    samples = np.full(160, 0.5, dtype="<f4")
    assert isinstance(_collect(native, engine, samples.tobytes())[-1], UtteranceEnd)
    np.testing.assert_array_equal(model.audio[0], samples)


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize(
    "problem", ["cuda", "device", "capability", "tensorrt_import", "tensorrt_os"]
)
def test_prepare_reports_runtime_unavailability_before_loading(
    tmp_path, native, monkeypatch, problem
):
    """CUDA, device, GPU generation, and TensorRT problems fail preparation with their cause."""
    from listenflow.fastest_whisper import _engine as engine_module

    model = native(_Model())
    cause = None
    device = 0
    if problem == "cuda":
        cause = RuntimeError("CUDA unavailable")

        def fail_cuda(_device):
            raise cause

        monkeypatch.setattr(engine_module, "_cuda_device_info", fail_cuda)
    elif problem == "device":
        device = 2
        monkeypatch.setattr(engine_module, "_cuda_device_info", lambda device: (1, None))
    elif problem == "capability":
        monkeypatch.setattr(engine_module, "_cuda_device_info", lambda device: (1, (9, 0)))
    else:
        error_type = ImportError if problem == "tensorrt_import" else OSError
        cause = error_type("TensorRT unavailable")
        original_import = engine_module.importlib.import_module

        def no_tensorrt(name):
            if name == "tensorrt":
                raise cause
            return original_import(name)

        monkeypatch.setattr(engine_module.importlib, "import_module", no_tensorrt)
    engine = _engine(tmp_path, device=device)
    with pytest.raises(EnvironmentUnavailableError, match="unavailable|unsupported") as failure:
        native.run(engine.prepare())
    if cause is not None:
        assert failure.value.__cause__ is cause
    else:
        assert isinstance(failure.value.__cause__, RuntimeError)
    assert model.warmed == 0
    assert engine.health().state is EngineState.UNPREPARED


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_failed_preparation_maps_cleanup_failure_and_keeps_both_causes(tmp_path, native):
    """A native cleanup failure after failed preparation is a core error with both native causes."""
    load_error = RuntimeError("trial inference failed")
    release_error = RuntimeError("native stream destruction failed")

    class FailingModel(_Model):
        def warmup(self):
            raise load_error

        def close(self):
            super().close()
            raise release_error

    model = native(FailingModel())
    engine = _engine(tmp_path)
    with pytest.raises(
        EnvironmentUnavailableError, match="trial inference.*stream destruction"
    ) as failure:
        native.run(engine.prepare())
    assert failure.value.__cause__ is release_error
    assert release_error.__context__ is load_error
    assert model.closed == 1
    assert engine.health().state is EngineState.UNPREPARED
    assert not engine.health().busy


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize("error_type", [ImportError, OSError])
def test_prepare_maps_native_binding_import_failure_before_allocating(
    tmp_path, native, monkeypatch, error_type
):
    """An unavailable native binding reports its original cause without allocating a model."""
    import builtins

    model = native(_Model())
    engine = _engine(tmp_path)
    cause = error_type("TensorRT native binding unavailable")
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "tensoracle":
            raise cause
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(EnvironmentUnavailableError, match="native binding unavailable") as failure:
        native.run(engine.prepare())
    assert failure.value.__cause__ is cause
    assert model.warmed == model.closed == 0
    assert engine.health().state is EngineState.UNPREPARED
    assert not engine.health().busy
