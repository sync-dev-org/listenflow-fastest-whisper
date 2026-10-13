"""Whisper-family inference using TensorRT."""

from ._engine import FastestWhisperEngine, UnsupportedLanguageError
from ._longform import LongTranscription, TimedSegment

__all__ = [
    "FastestWhisperEngine",
    "LongTranscription",
    "TimedSegment",
    "Transcription",
    "UnsupportedLanguageError",
    "transcribe_fp16",
    "transcribe_fp32",
    "transcribe_long_fp32",
]


def __getattr__(name: str):
    if name in {"Transcription", "transcribe_fp16", "transcribe_fp32", "transcribe_long_fp32"}:
        from . import _inference

        value = getattr(_inference, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
