"""CPU-side window control for a complete Whisper transcription."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._audio import HOP_SIZE, N_SAMPLES
from ._decoding import DecodingSettings

SAMPLE_RATE = 16_000
N_FRAMES = N_SAMPLES // HOP_SIZE


@dataclass(frozen=True)
class DecodedWindow:
    tokens: list[int]
    avg_logprob: float
    no_speech_prob: float
    temperature: float = 0.0


@dataclass(frozen=True)
class TimedSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SegmentTrace:
    start: float
    end: float
    text: str
    tokens: list[int]
    seek: int


@dataclass(frozen=True)
class WindowTrace:
    seek: int
    next_seek: int
    tokens: list[int]


@dataclass(frozen=True)
class LongTranscription:
    text: str
    segments: list[TimedSegment]


def decoder_prefix(
    prompt: list[int],
    *,
    initial_tokens: list[int],
    previous_token: int,
    text_positions: int,
) -> list[int]:
    """Keep the newest previous tokens before the next window's initial tokens."""
    if not prompt:
        return list(initial_tokens)
    return [previous_token] + prompt[-(text_positions // 2 - 1) :] + list(initial_tokens)


def apply_timestamp_rules(
    logits: np.ndarray, sampled: list[int], *, timestamp_begin: int, eot: int
) -> np.ndarray:
    """Mask logits using the official timestamp pair and ordering constraints."""
    last_is_timestamp = bool(sampled and sampled[-1] >= timestamp_begin)
    previous_is_timestamp = len(sampled) < 2 or sampled[-2] >= timestamp_begin
    if last_is_timestamp:
        if previous_is_timestamp:
            logits[timestamp_begin:] = -np.inf
        else:
            logits[:eot] = -np.inf

    timestamps = [token for token in sampled if token >= timestamp_begin]
    if timestamps:
        minimum = timestamps[-1] + (0 if last_is_timestamp and not previous_is_timestamp else 1)
        logits[timestamp_begin:minimum] = -np.inf

    if not sampled:
        logits[:timestamp_begin] = -np.inf
        logits[timestamp_begin + 51 :] = -np.inf

    timestamp_mass = np.logaddexp.reduce(logits[timestamp_begin:])
    if timestamp_mass > np.max(logits[:timestamp_begin]):
        logits[:timestamp_begin] = -np.inf
    return logits


def transcribe_mel(
    mel: np.ndarray,
    *,
    content_frames: int,
    decode: Callable[[np.ndarray, list[int]], DecodedWindow],
    trace: Callable[[WindowTrace], None] | None = None,
    segment_trace: Callable[[SegmentTrace], None] | None = None,
    text: Callable[[list[int]], str],
    timestamp_begin: int,
    eot: int,
    audio_positions: int,
    initial_prompt_tokens: list[int] | None = None,
    no_speech_threshold: float | None = DecodingSettings.no_speech_threshold,
    logprob_threshold: float | None = DecodingSettings.logprob_threshold,
) -> LongTranscription:
    """Advance across the original full-audio mel using decoded timestamp tokens."""
    if mel.ndim != 2 or mel.shape[-1] < content_frames + N_FRAMES:
        raise ValueError("mel must include 30 seconds of padding after the audio")
    stride = N_FRAMES // audio_positions
    if stride * audio_positions != N_FRAMES:
        raise ValueError("audio positions must divide a 30-second mel window")
    precision = stride * HOP_SIZE / SAMPLE_RATE
    seek = 0
    all_tokens = list(initial_prompt_tokens or [])
    initial_prompt_length = len(all_tokens)
    prompt_reset_since = 0
    segments: list[TimedSegment] = []
    while seek < content_frames:
        segment_size = min(N_FRAMES, content_frames - seek)
        window = mel[:, seek : seek + segment_size]
        if segment_size < N_FRAMES:
            window = np.pad(window, ((0, 0), (0, N_FRAMES - segment_size)))
        result = decode(window, all_tokens[prompt_reset_since:].copy())
        tokens = result.tokens
        next_seek = seek + segment_size
        skip_silence = (
            no_speech_threshold is not None and result.no_speech_prob > no_speech_threshold
        )
        if (
            skip_silence
            and logprob_threshold is not None
            and result.avg_logprob > logprob_threshold
        ):
            skip_silence = False
        if not skip_silence:
            offset = seek * HOP_SIZE / SAMPLE_RATE
            timestamp_flags = [token >= timestamp_begin for token in tokens]
            single_timestamp_ending = timestamp_flags[-2:] == [False, True]
            cuts = [
                index + 1
                for index in range(len(tokens) - 1)
                if timestamp_flags[index] and timestamp_flags[index + 1]
            ]
            pieces: list[tuple[float, float, list[int]]] = []
            if cuts:
                if single_timestamp_ending:
                    cuts.append(len(tokens))
                previous = 0
                for cut in cuts:
                    piece = tokens[previous:cut]
                    pieces.append(
                        (
                            offset + (piece[0] - timestamp_begin) * precision,
                            offset + (piece[-1] - timestamp_begin) * precision,
                            piece,
                        )
                    )
                    previous = cut
                if not single_timestamp_ending:
                    next_seek = seek + (tokens[previous - 1] - timestamp_begin) * stride
            else:
                duration = segment_size * HOP_SIZE / SAMPLE_RATE
                timestamps = [token for token in tokens if token >= timestamp_begin]
                if timestamps and timestamps[-1] != timestamp_begin:
                    duration = (timestamps[-1] - timestamp_begin) * precision
                pieces.append((offset, offset + duration, tokens))

            for start, end, piece in pieces:
                segment_text = text([token for token in piece if token < eot])
                kept_tokens = list(piece)
                if start == end or not segment_text.strip():
                    segment_text = ""
                    kept_tokens = []
                segments.append(TimedSegment(start, end, segment_text))
                if segment_trace is not None:
                    segment_trace(SegmentTrace(start, end, segment_text, kept_tokens, seek))
                all_tokens.extend(kept_tokens)

            if result.temperature > 0.5:
                prompt_reset_since = len(all_tokens)

        if trace is not None:
            trace(WindowTrace(seek, next_seek, list(tokens)))
        if next_seek <= seek:
            raise RuntimeError("decoded timestamp did not advance the audio window")
        seek = next_seek
    return LongTranscription(
        text([token for token in all_tokens[initial_prompt_length:] if token < eot]), segments
    )
