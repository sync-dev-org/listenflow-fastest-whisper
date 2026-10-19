"""CPU-side sequence selection and window retry rules for Whisper decoding."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Candidate:
    tokens: list[int]
    sum_logprob: float


@dataclass(frozen=True)
class DecodeAttempt:
    tokens: list[int]
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    temperature: float


@dataclass(frozen=True)
class DecodingSettings:
    beam_size: int = 1
    temperatures: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    best_of: int = 5
    compression_ratio_threshold: float | None = 2.4
    logprob_threshold: float | None = -1.0
    no_speech_threshold: float | None = 0.6

    @property
    def greedy_only(self) -> bool:
        return self.beam_size == 1 and self.temperatures == (0.0,)


def decoding_settings(
    *,
    beam_size: int = DecodingSettings.beam_size,
    temperature: float | Sequence[float] = DecodingSettings.temperatures,
    best_of: int = DecodingSettings.best_of,
    compression_ratio_threshold: float | None = DecodingSettings.compression_ratio_threshold,
    logprob_threshold: float | None = DecodingSettings.logprob_threshold,
    no_speech_threshold: float | None = DecodingSettings.no_speech_threshold,
) -> DecodingSettings:
    if type(beam_size) is not int or beam_size < 1:
        raise ValueError("beam_size must be a positive integer")
    if type(best_of) is not int or best_of < 1:
        raise ValueError("best_of must be a positive integer")
    if isinstance(temperature, (int, float)):
        values = (temperature,)
    elif isinstance(temperature, (str, bytes, bytearray)):
        # Preserve the numerical entries' existing ValueError for textual temperatures.
        raise ValueError("temperature must contain numbers, not text or bytes")  # noqa: TRY004
    elif isinstance(temperature, Sequence):
        values = tuple(temperature)
    else:
        raise TypeError("temperature must be a number or a sequence of numbers")
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValueError("temperature must be a nonempty sequence of finite nonnegative values")
    for name, threshold in (
        ("compression_ratio_threshold", compression_ratio_threshold),
        ("logprob_threshold", logprob_threshold),
        ("no_speech_threshold", no_speech_threshold),
    ):
        if threshold is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
        ):
            raise ValueError(f"{name} must be finite or None")
    return DecodingSettings(
        beam_size,
        tuple(float(value) for value in values),
        best_of,
        compression_ratio_threshold,
        logprob_threshold,
        no_speech_threshold,
    )


def _logprobs(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    largest = np.max(values)
    if not np.isfinite(largest):
        raise ValueError("all decoder tokens were suppressed")
    return values - np.logaddexp.reduce(values)


def _rank(candidates: Sequence[Candidate]) -> Candidate:
    return max(
        candidates, key=lambda candidate: candidate.sum_logprob / max(1, len(candidate.tokens))
    )


def search_candidates(
    initial_tokens: list[int],
    logits: Callable[[list[int]], np.ndarray],
    *,
    eot: int,
    max_tokens: int,
    context_length: int,
    beam_size: int | None = None,
    temperature: float = 0.0,
    best_of: int | None = None,
    sample_token: Callable[[np.ndarray], int] | None = None,
) -> Candidate:
    """Choose a completed sentence from beam paths or independent samples."""
    if beam_size is not None:
        live = [Candidate([], 0.0)]
        finished: dict[tuple[int, ...], Candidate] = {}
        for _ in range(max_tokens):
            scores: dict[tuple[int, ...], Candidate] = {}
            for path in live:
                prefix = initial_tokens + path.tokens
                if len(prefix) > context_length:
                    break
                distribution = _logprobs(logits(prefix))
                for token in np.argsort(distribution)[::-1][: beam_size + 1]:
                    if not np.isfinite(distribution[token]):
                        continue
                    sequence = (*path.tokens, int(token))
                    scores[sequence] = Candidate(
                        list(sequence), float(np.float32(path.sum_logprob + distribution[token]))
                    )
            if not scores:
                break
            live = []
            for sequence, candidate in sorted(
                scores.items(), key=lambda item: item[1].sum_logprob, reverse=True
            ):
                if sequence[-1] == eot:
                    if len(finished) < beam_size:
                        finished[sequence] = Candidate(candidate.tokens[:-1], candidate.sum_logprob)
                elif len(live) < beam_size:
                    live.append(candidate)
                    if len(live) == beam_size:
                        break
            if len(finished) >= beam_size or not live:
                break
            if len(initial_tokens) + len(live[0].tokens) > context_length:
                break
        if len(finished) < beam_size:
            for candidate in sorted(live, key=lambda item: item.sum_logprob, reverse=True):
                finished.setdefault(tuple(candidate.tokens + [eot]), candidate)
                if len(finished) >= beam_size:
                    break
        if not finished:
            raise RuntimeError("beam search produced no candidate")
        return _rank(list(finished.values()))

    if sample_token is None:
        generator = np.random.default_rng()
        sample_token = lambda probabilities: int(
            generator.choice(len(probabilities), p=probabilities)
        )
    candidates = []
    for _ in range(best_of or 1):
        tokens: list[int] = []
        score = np.float32(0.0)
        for _ in range(max_tokens):
            prefix = initial_tokens + tokens
            if len(prefix) > context_length:
                break
            values = np.asarray(logits(prefix), dtype=np.float32)
            raw_logprobs = _logprobs(values)
            if temperature == 0:
                token = int(np.argmax(values))
            else:
                sampled_values = values.astype(np.float64)
                probabilities = np.exp((sampled_values - np.max(sampled_values)) / temperature)
                probabilities /= probabilities.sum()
                token = sample_token(probabilities)
            score = np.float32(score + raw_logprobs[token])
            if token == eot:
                break
            tokens.append(token)
            if len(prefix) + 1 > context_length:
                break
        candidates.append(Candidate(tokens, float(score)))
    return _rank(candidates)


def decode_with_fallback(
    decode: Callable[[float, int | None, int | None], DecodeAttempt],
    *,
    temperatures: Sequence[float],
    beam_size: int,
    best_of: int,
    compression_ratio_threshold: float | None,
    logprob_threshold: float | None,
    no_speech_threshold: float | None,
) -> DecodeAttempt:
    """Retry a window only while the official failure checks request it."""
    if not temperatures:
        raise ValueError("temperature sequence must contain at least one value")
    for temperature in temperatures:
        attempt = decode(
            temperature,
            beam_size if temperature == 0 and beam_size > 1 else None,
            best_of if temperature > 0 else None,
        )
        failed = (
            compression_ratio_threshold is not None
            and attempt.compression_ratio > compression_ratio_threshold
        ) or (logprob_threshold is not None and attempt.avg_logprob < logprob_threshold)
        silent = (
            no_speech_threshold is not None
            and attempt.no_speech_prob > no_speech_threshold
            and logprob_threshold is not None
            and attempt.avg_logprob < logprob_threshold
        )
        if not failed or silent:
            break
    return attempt
