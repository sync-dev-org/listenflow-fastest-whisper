"""listenflow's stream contract for the TensorRT Whisper runtime."""

from __future__ import annotations

import asyncio
import importlib
import threading
import weakref
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
from listenflow.engine import validate_input_format

from listenflow import (
    BiasApplicationSnapshot,
    ClosedUtterance,
    ConfigurationError,
    DiarizationMode,
    EngineCapabilities,
    EngineInputFormat,
    EngineStateError,
    EngineTransport,
    EnvironmentUnavailableError,
    FinalizationMode,
    FinalObservation,
    InferenceError,
    OfflineModelDirectorySource,
    PartialMode,
    STTEngineBase,
    TimestampDetail,
    TranscriptEvent,
    UtteranceBoundarySettings,
    UtteranceDecodePipeline,
)

from ._decoding import DecodingSettings, decoding_settings

_DEFAULT_BOUNDARY_SETTINGS = UtteranceBoundarySettings()


class UnsupportedLanguageError(ConfigurationError):
    """The detector selected a language that this engine cannot transcribe."""

    def __init__(self, language: str):
        self.language = language
        super().__init__(f"detected unsupported language: {language}")


def _cuda_device_info(device: int) -> tuple[int, tuple[int, int] | None]:
    """Return the visible device count and the selected device's compute capability."""
    from cuda.bindings import runtime as cudart

    status, count = cudart.cudaGetDeviceCount()
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA device enumeration failed: {status}")
    if device >= count:
        return count, None
    attributes = cudart.cudaDeviceAttr
    major_status, major = cudart.cudaDeviceGetAttribute(
        attributes.cudaDevAttrComputeCapabilityMajor, device
    )
    minor_status, minor = cudart.cudaDeviceGetAttribute(
        attributes.cudaDevAttrComputeCapabilityMinor, device
    )
    if (
        major_status != cudart.cudaError_t.cudaSuccess
        or minor_status != cudart.cudaError_t.cudaSuccess
    ):
        raise RuntimeError("CUDA compute capability lookup failed")
    return count, (int(major), int(minor))


def _check_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise asyncio.CancelledError


async def _run_native[T](operation: Callable[[threading.Event], T]) -> T:
    """Keep the native thread owned until it has finished, including on cancellation."""
    cancelled = threading.Event()
    task = asyncio.create_task(asyncio.to_thread(operation, cancelled))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled.set()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:  # noqa: BLE001 - observe the native completion below
                break
        if not task.cancelled():
            task.exception()
        raise


class FastestWhisperEngine(STTEngineBase):
    """Decode one PCM stream at a time with a cached large-v3 TensorRT model."""

    def __init__(
        self,
        *,
        model_source: OfflineModelDirectorySource,
        precision: Literal["fp16", "fp32"],
        device: int,
        language: str | None,
        initial_prompt: str | None = None,
        beam_size: int = DecodingSettings.beam_size,
        temperature: float | Sequence[float] = DecodingSettings.temperatures,
        best_of: int = DecodingSettings.best_of,
        compression_ratio_threshold: float | None = DecodingSettings.compression_ratio_threshold,
        logprob_threshold: float | None = DecodingSettings.logprob_threshold,
        no_speech_threshold: float | None = DecodingSettings.no_speech_threshold,
        boundary_settings: UtteranceBoundarySettings = _DEFAULT_BOUNDARY_SETTINGS,
        input_format: EngineInputFormat = EngineInputFormat.PCM_S16LE,
        force_rebuild: bool = False,
    ) -> None:
        if not isinstance(model_source, OfflineModelDirectorySource):
            raise ConfigurationError("model_source must be OfflineModelDirectorySource")
        if not isinstance(precision, str):
            raise ConfigurationError("precision must be 'fp16' or 'fp32'")
        if precision not in ("fp16", "fp32"):
            raise ConfigurationError("precision must be 'fp16' or 'fp32'")
        if type(device) is bool:
            raise ConfigurationError("device must be a non-negative integer")
        if not isinstance(device, int):
            raise ConfigurationError("device must be a non-negative integer")
        if device < 0:
            raise ConfigurationError("device must be a non-negative integer")
        if language is not None:
            if not isinstance(language, str):
                raise ConfigurationError("language must be 'en' or None")
            if language != "en":
                if language in ("English", "en-US", "EN", "en_us", "eng"):
                    raise ConfigurationError(f"language {language!r} is invalid; use 'en'")
                raise ConfigurationError(
                    f"language {language!r} is not supported; only 'en' is supported"
                )
        if initial_prompt is not None:
            if not isinstance(initial_prompt, str):
                raise ConfigurationError("initial_prompt must be None or a non-empty string")
            if not initial_prompt:
                raise ConfigurationError("initial_prompt must be None or a non-empty string")
        if not isinstance(boundary_settings, UtteranceBoundarySettings):
            raise ConfigurationError("boundary_settings must be UtteranceBoundarySettings")
        if not isinstance(input_format, EngineInputFormat):
            raise ConfigurationError("input_format must be EngineInputFormat")
        try:
            validate_input_format(
                input_format,
                target="FastestWhisperEngine",
                accepted=(EngineInputFormat.PCM_S16LE, EngineInputFormat.PCM_F32),
            )
        except ValueError as error:
            raise ConfigurationError(str(error)) from error
        if type(force_rebuild) is not bool:
            raise ConfigurationError("force_rebuild must be bool")

        try:
            settings = decoding_settings(
                beam_size=beam_size,
                temperature=temperature,
                best_of=best_of,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
            )
        except (TypeError, ValueError) as error:
            raise ConfigurationError(str(error)) from error

        super().__init__()
        self.model_source = model_source
        self.precision = precision
        self.device = device
        self.language = language
        self.initial_prompt = initial_prompt
        self.boundary_settings = boundary_settings
        self.input_format = input_format
        self.force_rebuild = force_rebuild
        self._decoding_settings = settings
        self._model = None
        self._session_counter = 0
        self._bias_application: BiasApplicationSnapshot | None = None
        engine = weakref.proxy(self)
        self._pipeline = UtteranceDecodePipeline(
            boundary_settings=boundary_settings,
            decode_utterance=lambda utterance: engine._decode_closed_utterance(utterance),
            on_progress=lambda: engine._mark_progress(),
            input_format=input_format,
        )

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            transport=EngineTransport.IN_PROCESS,
            input_format=self.input_format,
            partial_mode=PartialMode.NONE,
            finalization=frozenset((FinalizationMode.LOCAL_VAD, FinalizationMode.CLIENT_FLUSH)),
            timestamp_detail=TimestampDetail.SEGMENT,
            diarization=DiarizationMode.NONE,
            languages=frozenset({"en"}),
            auto_language=True,
        )

    async def _load(self) -> None:
        try:
            from tensoracle import EngineCacheError
        except Exception as error:
            raise EnvironmentUnavailableError(str(error)) from error

        try:
            await _run_native(self._load_and_warm)
        except BaseException as error:
            try:
                await self._release()
            except Exception as cleanup_error:
                raise EnvironmentUnavailableError(
                    f"{error}; resource release failed: {cleanup_error}"
                ) from cleanup_error
            if isinstance(error, EngineCacheError):
                raise EnvironmentUnavailableError(
                    f"{error}; rebuild with force_rebuild=True"
                ) from error
            if isinstance(error, Exception):
                raise EnvironmentUnavailableError(str(error)) from error
            raise

    async def _release(self) -> None:
        model, self._model = self._model, None
        if model is not None:
            await _run_native(lambda cancelled: model.close())

    def _load_and_warm(self, cancelled: threading.Event) -> None:
        count, capability = _cuda_device_info(self.device)
        if self.device >= count:
            raise RuntimeError(
                f"CUDA device {self.device} is unavailable; {count} visible device(s)"
            )
        if capability not in ((8, 6), (8, 9), (12, 0)):
            raise RuntimeError(
                f"CUDA device {self.device} compute capability {capability} is unsupported"
            )
        importlib.import_module("tensorrt")
        _check_cancelled(cancelled)
        self._model = self._load_model()
        self._model._cancel_requested = cancelled
        try:
            _check_cancelled(cancelled)
            self._model.warmup()
            _check_cancelled(cancelled)
        finally:
            self._model._cancel_requested = None

    def _load_model(self):
        from ._cached_model import CachedWhisper

        return CachedWhisper(
            Path(self.model_source.directory_path),
            precision=self.precision,
            device=self.device,
            force_rebuild=self.force_rebuild,
        )

    async def _recognize(self, pcm_chunks: AsyncIterator[bytes]) -> AsyncIterator[TranscriptEvent]:
        self._session_counter += 1
        self._bias_application = BiasApplicationSnapshot(
            session_id=self._session_counter,
            kind="prompt" if self.initial_prompt is not None else "none",
            prompt=self.initial_prompt,
            applied_value=self.initial_prompt,
            applied=False,
            evidence_method="verified_apply_path",
            evidence_detail="CachedWhisper.transcribe(..., initial_prompt=...) supplies decoder context",
        )
        pipeline_stream = self._pipeline.stream(pcm_chunks)
        try:
            async for event in pipeline_stream:
                yield event
        finally:
            await pipeline_stream.aclose()

    def _health_details(self) -> dict[str, object]:
        return {
            "processed_pcm_chunks": self._pipeline.processed_pcm_chunks,
            "emitted_events": self._pipeline.emitted_events,
            "bias_application": self._bias_application,
            "decoding_utterance_id": self._pipeline.decoding_utterance_id,
            "pending_decode_utterances": self._pipeline.pending_decode_utterances,
        }

    async def _decode_closed_utterance(self, utterance: ClosedUtterance) -> list[FinalObservation]:
        if self._model is None:
            raise EngineStateError("FastestWhisperEngine is not prepared")
        if self.input_format is EngineInputFormat.PCM_S16LE:
            audio = np.frombuffer(utterance.pcm, dtype="<i2").astype(np.float32) / 32768.0
        else:
            audio = np.frombuffer(utterance.pcm, dtype="<f4")

        def decode(cancelled: threading.Event) -> list[FinalObservation]:
            model = self._model
            model._cancel_requested = cancelled
            try:
                _check_cancelled(cancelled)
                if self.language is None:
                    detected = model.detect_language(audio)
                    if detected != "en":
                        raise UnsupportedLanguageError(detected)
                settings = self._decoding_settings
                segments = model.transcribe(
                    audio,
                    initial_prompt=self.initial_prompt,
                    beam_size=settings.beam_size,
                    temperature=settings.temperatures,
                    best_of=settings.best_of,
                    compression_ratio_threshold=settings.compression_ratio_threshold,
                    logprob_threshold=settings.logprob_threshold,
                    no_speech_threshold=settings.no_speech_threshold,
                )
                return [
                    FinalObservation(
                        text=segment.text,
                        source_offset_s=utterance.start_offset_s + segment.start,
                    )
                    for segment in segments
                    if segment.text.strip()
                ]

            finally:
                model._cancel_requested = None

        try:
            result = await _run_native(decode)
        except ConfigurationError:
            raise
        except Exception as error:
            raise InferenceError(str(error)) from error
        if self._bias_application is not None:
            self._bias_application = replace(self._bias_application, applied=True)
        return result
