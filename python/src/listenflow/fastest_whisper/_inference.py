"""Run independently built TensorRT encoder and decoder engines."""

from __future__ import annotations

import asyncio
import threading
import zlib
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import numpy as np
from tokenizers import Tokenizer

from listenflow import MetadataRecord

from . import _native
from ._assets import FP8_SCALES, read_fp8_scales
from ._audio import _FILTERS, FFT_SIZE, HOP_SIZE, N_SAMPLES
from ._checkpoint import Checkpoint, ModelConfig
from ._decoding import (
    Candidate,
    DecodeAttempt,
    DecodingSettings,
    _rank,
    decode_with_fallback,
    decoding_settings,
    search_candidates,
    should_skip_window,
)
from ._languages import LANGUAGES
from ._longform import (
    DecodedWindow,
    LongTranscription,
    SegmentTrace,
    WindowTrace,
    apply_timestamp_constraints,
    apply_timestamp_rules,
    decoder_prefix,
    transcribe_mel,
)
from ._plugins import _fc1_library
from ._precision import PRECISIONS
from ._rejection import NO_REJECTION, RejectionSettings, decode_candidate_text

_DEFAULT_DECODING_SETTINGS = DecodingSettings()


@dataclass
class _PrefixState:
    prefix: object
    logits: np.ndarray

    def close(self) -> None:
        self.prefix.close()


@dataclass(frozen=True)
class Transcription:
    """The decoded text and token IDs, including the final end-of-text token."""

    text: str
    tokens: list[int]


def _suppressed_tokens(tokenizer: Tokenizer) -> set[int]:
    # Match the official multilingual tokenizer's -1 non-speech policy.
    symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
    symbols += [
        "<<",
        ">>",
        "<<<",
        ">>>",
        "--",
        "---",
        "-(",
        "-[",
        "('",
        '("',
        "((",
        "))",
        "(((",
        ")))",
        "[[",
        "]]",
        "{{",
        "}}",
        "♪♪",
        "♪♪♪",
    ]
    miscellaneous = set("♩♪♫♬♭♮♯")
    result = {
        tokenizer.encode(" -", add_special_tokens=False).ids[0],
        tokenizer.encode(" '", add_special_tokens=False).ids[0],
    }
    for symbol in symbols + sorted(miscellaneous):
        for text in (symbol, " " + symbol):
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if len(ids) == 1 or symbol in miscellaneous:
                result.add(ids[0])
    for symbol in (
        "<|transcribe|>",
        "<|translate|>",
        "<|startoftranscript|>",
        "<|startofprev|>",
        "<|startoflm|>",
        "<|nospeech|>",
    ):
        token_id = tokenizer.token_to_id(symbol)
        if token_id is not None:
            result.add(token_id)
    return result


class TensorRTWhisper:
    """Own the native model while Python controls transcription and decoding."""

    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        precision: str = "fp32",
        device: int = 0,
        settings: DecodingSettings | None = None,
    ):
        if precision not in PRECISIONS:
            raise ValueError(f"unsupported precision: {precision}")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.precision, self.device = precision, device
        self._settings = settings or _DEFAULT_DECODING_SETTINGS
        self._native_model = None
        self._mel_tensor = self._encoded_audio = None
        self._prefix_decoder = self._greedy_decoder = None
        self._prefix_states: dict[tuple[int, ...], _PrefixState] = {}
        self._root_prefix: tuple[int, ...] = ()
        self._greedy_length = 0
        self._cancel_requested: threading.Event | None = None
        self._sampled_batch = (
            settings.best_of
            if settings is not None
            and settings.best_of > 1
            and any(value > 0 for value in settings.temperatures)
            else None
        )
        self._sampled_decoder = None
        try:
            self.tokenizer = Tokenizer.from_file(str(self.checkpoint_dir / "tokenizer.json"))
            with Checkpoint(self.checkpoint_dir) as checkpoint:
                self.config: ModelConfig = checkpoint.config
                weights = (
                    [checkpoint.directory / "model.safetensors"]
                    if checkpoint.weight_map is None
                    else [
                        checkpoint.directory / name
                        for name in sorted(set(checkpoint.weight_map.values()))
                    ]
                )
            required = (
                "<|startoftranscript|>",
                "<|en|>",
                "<|transcribe|>",
                "<|notimestamps|>",
                "<|endoftext|>",
            )
            ids = [self.tokenizer.token_to_id(token) for token in required]
            if any(value is None for value in ids):
                raise ValueError("checkpoint tokenizer lacks a required transcription token")
            self.initial_tokens, self.eot = ids[:4], ids[4]
            self.suppressed = _suppressed_tokens(self.tokenizer)
            self.begin_suppressed = set(self.tokenizer.encode(" ", add_special_tokens=False).ids)
            self.begin_suppressed.add(self.eot)
            scales = FP8_SCALES
            source = {
                "config": asdict(self.config),
                "weights": [str(path) for path in weights],
                "config_file": str(self.checkpoint_dir / "config.json"),
                "tokenizer_file": str(self.checkpoint_dir / "tokenizer.json"),
                "scale_file": str(scales),
                "mel_filter_file": str(_FILTERS),
                "mel_weights": str(_FILTERS.with_suffix(".safetensors")),
                "fc1_library": str(_fc1_library()) if precision == "fp16" else "",
                "scales": read_fp8_scales(scales) if precision == "fp8" else {},
                "no_speech_token": self.tokenizer.token_to_id("<|nospeech|>"),
                "timestamp_begin_token": self.tokenizer.token_to_id("<|0.00|>"),
            }
            self._native_model = _native.Model(
                source,
                precision,
                device,
                self._sampled_batch or 0,
                sorted(self.suppressed),
                sorted(self.begin_suppressed),
            )
            self.build_seconds = self._native_model.info["open_seconds"]
            self._sampled_decoder = self._native_model if self._sampled_batch is not None else None
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Disable inference and release resources, retaining failed handles for another close."""
        handles = getattr(self, "_pending_close", [])
        self._pending_close = []
        handles.extend(getattr(self, "_prefix_states", {}).values())
        self._prefix_states, self._root_prefix = {}, ()
        for name in (
            "_prefix_decoder",
            "_greedy_decoder",
            "_encoded_audio",
            "_mel_tensor",
            "_native_model",
        ):
            value = getattr(self, name, None)
            setattr(self, name, None)
            if value is not None:
                handles.append(value)
        self._sampled_decoder = None
        self._greedy_length = 0
        first = None
        for handle in handles:
            try:
                handle.close()
            except Exception as error:  # noqa: BLE001 -- release every handle before reporting the first failure
                self._pending_close.append(handle)
                if first is None:
                    first = error
        if first is not None:
            raise first

    def _mel_spectrogram(self, audio: np.ndarray):
        self._check_cancelled()
        if self._native_model is None:
            raise RuntimeError("Whisper model is closed")
        samples = np.ascontiguousarray(audio, dtype=np.float32)
        previous, self._mel_tensor = getattr(self, "_mel_tensor", None), None
        if previous is not None:
            previous.close()
        self._mel_tensor = self._native_model.mel(samples)
        return self._mel_tensor

    def _mel_spectrogram_long(self, audio: np.ndarray) -> np.ndarray:
        self._check_cancelled()
        if self._native_model is None:
            raise RuntimeError("Whisper model is closed")
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("audio must be a finite mono waveform")
        padded = np.pad(samples, (0, N_SAMPLES))
        centered = np.pad(padded, (FFT_SIZE // 2, FFT_SIZE // 2), mode="reflect")
        frames, chunks = padded.size // HOP_SIZE, []
        for first in range(0, frames, 3000):
            self._check_cancelled()
            block = centered[first * HOP_SIZE : first * HOP_SIZE + N_SAMPLES + FFT_SIZE]
            block = np.pad(block, (0, N_SAMPLES + FFT_SIZE - block.size))
            output = self._native_model.log_mel_block(np.ascontiguousarray(block))
            chunks.append(output[:, : min(3000, frames - first)])
        log_mel = np.concatenate(chunks, axis=1)
        log_mel = np.maximum(log_mel, log_mel.max() - np.float32(8))
        return (log_mel + np.float32(4)) * np.float32(0.25)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def _check_cancelled(self) -> None:
        cancelled = getattr(self, "_cancel_requested", None)
        if cancelled is not None and cancelled.is_set():
            raise asyncio.CancelledError

    def _clear_prefix_states(self) -> None:
        states, self._prefix_states = getattr(self, "_prefix_states", {}), {}
        self._root_prefix = ()
        for state in states.values():
            state.close()
        decoder, self._prefix_decoder = getattr(self, "_prefix_decoder", None), None
        if decoder is not None:
            decoder.close()

    def encode(self, mel) -> _native.EncodedAudio:
        self._check_cancelled()
        if self._native_model is None:
            raise RuntimeError("Whisper model is closed")
        self._clear_prefix_states()
        self._clear_greedy_cache()
        previous, self._encoded_audio = self._encoded_audio, None
        if previous is not None:
            previous.close()
        self._encoded_audio = (
            self._native_model.encode(mel)
            if isinstance(mel, _native.Mel)
            else self._native_model.encode_features(np.ascontiguousarray(mel, dtype=np.float32))
        )
        return self._encoded_audio

    def logits(self, prefix: list[int], encoder: _native.EncodedAudio) -> np.ndarray:
        """Execute missing positions with immutable native KV states."""
        self._check_cancelled()
        if getattr(self, "_native_model", None) is None:
            raise RuntimeError("Whisper model is closed")
        length = len(prefix)
        if not 1 <= length <= self.config.text_positions:
            raise ValueError("decoder prefix exceeds checkpoint context length")
        if encoder is not self._encoded_audio:
            raise ValueError("decoder requires the current encoded audio window")
        path, root = tuple(prefix), self._root_prefix
        if root and path[: len(root)] != root:
            self._clear_prefix_states()
        states = self._prefix_states
        parent_path = next(
            (path[:offset] for offset in range(length - 1, 0, -1) if path[:offset] in states), None
        )
        for previous in list(states):
            if previous not in (self._root_prefix, parent_path) and len(previous) not in (
                length - 1,
                length,
            ):
                states.pop(previous).close()
        if path in states:
            return states[path].logits
        if self._prefix_decoder is None:
            self._prefix_decoder = encoder.decoder(_native.PREFIX)
        parent = None if parent_path is None else states[parent_path]
        offset = 0 if parent is None else len(parent.logits)
        rows = [] if parent is None else list(parent.logits)
        state = None if parent is None else parent.prefix
        owns = False
        try:
            for position in range(offset, length):
                self._check_cancelled()
                result, row = self._prefix_decoder.prefix_step(state, prefix[position])
                if owns:
                    state.close()
                state, owns = result, True
                rows.append(row)
            states[path] = _PrefixState(state, np.stack(rows))
            owns = False
            if not self._root_prefix:
                self._root_prefix = path
            return states[path].logits
        finally:
            if owns:
                state.close()

    def prompt_tokens(self, initial_prompt: str | None) -> list[int]:
        if initial_prompt is None:
            return []
        return self.tokenizer.encode(" " + initial_prompt.strip(), add_special_tokens=False).ids

    def _set_language(self, language: str) -> None:
        """Select the language token for subsequent short and timestamped decodes."""
        token = self.tokenizer.token_to_id(f"<|{language}|>")
        if language not in LANGUAGES or token is None:
            raise ValueError(f"checkpoint tokenizer has no language token for {language!r}")
        self.initial_tokens = [self.initial_tokens[0], token, *self.initial_tokens[2:]]

    def _detect_language(
        self, encoder: _native.EncodedAudio, languages: list[str] | None = None
    ) -> str:
        """Choose among the supplied languages, or all large-v3 codes, using start-token logits."""
        logits = self.logits([self.initial_tokens[0]], encoder)[-1]
        candidates = {
            token_id: language
            for language in sorted(LANGUAGES if languages is None else languages)
            if (token_id := self.tokenizer.token_to_id(f"<|{language}|>")) is not None
        }
        if not candidates:
            raise ValueError("checkpoint tokenizer has no language tokens")
        return candidates[max(candidates, key=lambda token_id: float(logits[token_id]))]

    def detect_language(self, audio: np.ndarray) -> str:
        """Detect the language of at most one audio window."""
        mel = self._mel_spectrogram(audio[:N_SAMPLES])
        return self._detect_language(self.encode(mel))

    def _clear_greedy_cache(self):
        decoder, self._greedy_decoder = getattr(self, "_greedy_decoder", None), None
        self._greedy_length = 0
        if decoder is not None:
            decoder.close()

    def _greedy_step(self, prefix, encoder, *, first, score=False, speech_position=0):
        if encoder is not self._encoded_audio:
            raise ValueError("decoder requires the current encoded audio window")
        if not 1 <= len(prefix) <= self.config.text_positions:
            raise ValueError("decoder prefix exceeds checkpoint context length")
        if self._greedy_decoder is None:
            self._greedy_decoder = encoder.decoder(_native.GREEDY)
        no_speech_prob = float("nan")
        for position in range(self._greedy_length, len(prefix)):
            self._check_cancelled()
            token, logprob, probability = self._greedy_decoder.greedy_step(
                prefix[position], first, score
            )
            self._greedy_length = position + 1
            if score and position == speech_position:
                no_speech_prob = probability
        return int(token), float(logprob), no_speech_prob

    def _decode_attempt(
        self,
        encoder: _native.EncodedAudio,
        prompt: list[int],
        *,
        with_timestamps: bool,
        temperature: float,
        beam_size: int | None,
        best_of: int | None,
        sample_token: Callable[[np.ndarray], int] | None = None,
        rejection: RejectionSettings = NO_REJECTION,
    ) -> DecodeAttempt:
        """Read one window with the chosen candidate selection rule."""
        if not with_timestamps and temperature == 0 and beam_size is None:
            return self._greedy_attempt(encoder, prompt, rejection=rejection)
        previous = self.tokenizer.token_to_id("<|startofprev|>")
        no_speech = self.tokenizer.token_to_id("<|nospeech|>")
        timestamp_begin = self.tokenizer.token_to_id("<|0.00|>") if with_timestamps else None
        no_timestamps = self.tokenizer.token_to_id("<|notimestamps|>") if with_timestamps else None
        if prompt and previous is None:
            raise ValueError("checkpoint tokenizer lacks the previous-text token")
        if with_timestamps and (timestamp_begin is None or no_timestamps is None):
            raise ValueError("checkpoint tokenizer lacks long-form tokens")
        initial_tokens = self.initial_tokens[:3] if with_timestamps else self.initial_tokens
        prefix = decoder_prefix(
            prompt,
            initial_tokens=initial_tokens,
            previous_token=previous,
            text_positions=self.config.text_positions,
        )
        if (
            temperature > 0
            and sample_token is None
            and getattr(self, "_sampled_decoder", None) is not None
            and best_of == self._sampled_batch
        ):
            chosen, no_speech_prob = self._sampled_candidates(
                encoder,
                prefix,
                temperature=temperature,
                with_timestamps=with_timestamps,
                rejection=rejection,
            )
            return self._candidate_attempt(chosen, no_speech_prob, temperature, rejection)
        first_logits = self.logits(prefix, encoder)
        if no_speech is None:
            no_speech_prob = float("nan")
        else:
            speech_logits = first_logits[len(prefix) - len(initial_tokens)].astype(np.float32)
            no_speech_prob = float(
                np.exp(speech_logits[no_speech] - np.logaddexp.reduce(speech_logits))
            )

        def filtered(current: list[int]) -> np.ndarray:
            self._check_cancelled()
            sampled = current[len(prefix) :]
            raw = first_logits[-1] if not sampled else self.logits(current, encoder)[-1]
            next_logits = raw.astype(np.float32, copy=True)
            next_logits[list(self.suppressed)] = -np.inf
            if not sampled:
                next_logits[list(self.begin_suppressed)] = -np.inf
            if with_timestamps:
                next_logits[no_timestamps] = -np.inf
                apply_timestamp_rules(
                    next_logits, sampled, timestamp_begin=timestamp_begin, eot=self.eot
                )
            return next_logits

        chosen = search_candidates(
            prefix,
            filtered,
            eot=self.eot,
            max_tokens=self.config.text_positions // 2,
            context_length=self.config.text_positions,
            beam_size=beam_size,
            temperature=temperature,
            best_of=best_of,
            sample_token=sample_token,
            stop_candidate=self._candidate_stop(rejection),
        )
        return self._candidate_attempt(chosen, no_speech_prob, temperature, rejection)

    def _candidate_stop(self, rejection):
        if not rejection.enabled:
            return None

        def stop(tokens, final):
            text_tokens = [token for token in tokens if token < self.eot]
            text = (
                decode_candidate_text(self.tokenizer, text_tokens, final=final)
                if rejection.patterns
                else ""
            )
            return rejection.reason(text_tokens, text, final=final) is not None

        return stop

    def _candidate_attempt(self, chosen, no_speech_prob, temperature, rejection=NO_REJECTION):
        text_tokens = [token for token in chosen.tokens if token < self.eot]
        text = self.tokenizer.decode(text_tokens, skip_special_tokens=False).strip()
        text_bytes = text.encode("utf-8")
        compression_ratio = len(text_bytes) / len(zlib.compress(text_bytes))
        return DecodeAttempt(
            chosen.tokens,
            text,
            chosen.sum_logprob / (len(chosen.tokens) + 1),
            no_speech_prob,
            compression_ratio,
            temperature,
            rejection.reason(text_tokens, text, final=True) if rejection.enabled else None,
        )

    def _sampled_candidates(
        self, encoder, prefix, *, temperature, with_timestamps, rejection=NO_REJECTION
    ):
        """Advance all candidates together, retaining only K/V and the sampled result."""
        c = self.config
        batch = self._sampled_batch
        paths = [[] for _ in range(batch)]
        scores = np.zeros(batch, dtype=np.float32)
        finished = np.zeros(batch, dtype=bool)
        decoder = encoder.decoder(_native.SAMPLED, batch)
        generator = np.random.default_rng()
        timestamp_begin = self.tokenizer.token_to_id("<|0.00|>") if with_timestamps else None
        no_timestamps = self.tokenizer.token_to_id("<|notimestamps|>") if with_timestamps else None
        speech_position = len(prefix) - (3 if with_timestamps else len(self.initial_tokens))
        no_speech_prob = float("nan")
        stop_candidate = self._candidate_stop(rejection)

        def step(tokens, position, *, first, select):
            nonlocal no_speech_prob
            self._check_cancelled()
            suppression = np.zeros((batch, 1, c.vocab_size), dtype=np.float32)
            suppression[..., list(self.suppressed)] = -np.inf
            if first:
                suppression[..., list(self.begin_suppressed)] = -np.inf
            if with_timestamps:
                suppression[..., no_timestamps] = -np.inf
                for row, path in zip(suppression[:, 0], paths, strict=True):
                    apply_timestamp_constraints(
                        row, path, timestamp_begin=timestamp_begin, eot=self.eot
                    )
            selected, logprobs, probabilities = decoder.sampled_step(
                np.asarray(tokens, dtype=np.int32),
                np.ascontiguousarray(suppression[:, 0]),
                temperature,
                generator.random(batch, dtype=np.float32)
                if select
                else np.zeros(batch, np.float32),
                with_timestamps,
            )
            sampled = np.stack((selected.astype(np.float32), logprobs, probabilities), axis=1)
            if position == speech_position:
                no_speech_prob = float(sampled[0, 2])
            return sampled

        try:
            for position, token in enumerate(prefix):
                sampled = step(
                    [token] * batch, position, first=True, select=position == len(prefix) - 1
                )
            for index in range(c.text_positions // 2):
                active = ~finished
                if not np.isfinite(sampled[active, 1]).all():
                    raise ValueError("all decoder tokens were suppressed")
                scores[active] += sampled[active, 1]
                tokens = sampled[:, 0].astype(np.int32)
                for row in np.flatnonzero(active):
                    if tokens[row] == self.eot:
                        finished[row] = True
                    else:
                        paths[row].append(int(tokens[row]))
                        if stop_candidate is not None and stop_candidate(paths[row], False):
                            finished[row] = True
                if finished.all() or len(prefix) + index + 1 > c.text_positions:
                    break
                if index + 1 == c.text_positions // 2:
                    break
                tokens[finished] = self.eot
                sampled = step(tokens, len(prefix) + index, first=False, select=True)
            return _rank(
                [Candidate(path, float(score)) for path, score in zip(paths, scores, strict=True)]
            ), no_speech_prob
        finally:
            decoder.close()

    def decode_short(
        self,
        encoder: _native.EncodedAudio,
        prompt: list[int],
        settings: DecodingSettings,
        *,
        on_metadata: Callable[[tuple[MetadataRecord, ...]], None] | None = None,
    ) -> Transcription:
        if (
            settings.greedy_only
            and settings.no_speech_threshold is None
            and not settings.rejection.enabled
        ):
            return self.greedy_decode(encoder, prompt)
        result = decode_with_fallback(
            lambda temperature, beam_size, best_of: self._decode_attempt(
                encoder,
                prompt,
                with_timestamps=False,
                temperature=temperature,
                beam_size=beam_size,
                best_of=best_of,
                rejection=settings.rejection,
            ),
            temperatures=settings.temperatures,
            beam_size=settings.beam_size,
            best_of=settings.best_of,
            compression_ratio_threshold=settings.compression_ratio_threshold,
            logprob_threshold=settings.logprob_threshold,
            no_speech_threshold=settings.no_speech_threshold,
            rejection=settings.rejection,
        )
        skip_silence = should_skip_window(
            result.no_speech_prob,
            result.avg_logprob,
            no_speech_threshold=settings.no_speech_threshold,
            logprob_threshold=settings.logprob_threshold,
        )
        if on_metadata is not None:
            on_metadata(result.metadata)
        return Transcription("" if skip_silence else result.text, result.tokens + [self.eot])

    def decode_long(
        self, encoder: _native.EncodedAudio, prompt: list[int], settings: DecodingSettings
    ) -> DecodedWindow:
        if settings.greedy_only and not settings.rejection.enabled:
            return self.timestamp_decode(encoder, prompt)
        result = decode_with_fallback(
            lambda temperature, beam_size, best_of: self._decode_attempt(
                encoder,
                prompt,
                with_timestamps=True,
                temperature=temperature,
                beam_size=beam_size,
                best_of=best_of,
                rejection=settings.rejection,
            ),
            temperatures=settings.temperatures,
            beam_size=settings.beam_size,
            best_of=settings.best_of,
            compression_ratio_threshold=settings.compression_ratio_threshold,
            logprob_threshold=settings.logprob_threshold,
            no_speech_threshold=settings.no_speech_threshold,
            rejection=settings.rejection,
        )
        return DecodedWindow(
            result.tokens,
            result.avg_logprob,
            result.no_speech_prob,
            result.temperature,
            rejected=result.rejection is not None,
            metadata=result.metadata,
        )

    def greedy_decode(
        self, encoder: _native.EncodedAudio, prompt: list[int] | None = None
    ) -> Transcription:
        result = self._greedy_attempt(encoder, prompt or [], score=False)
        return Transcription(result.text, result.tokens + [self.eot])

    def _greedy_attempt(
        self, encoder, prompt, *, score=True, rejection=NO_REJECTION
    ) -> DecodeAttempt:
        previous = self.tokenizer.token_to_id("<|startofprev|>") if prompt else None
        if prompt and previous is None:
            raise ValueError("checkpoint tokenizer lacks the previous-text token")
        prefix = decoder_prefix(
            prompt or [],
            initial_tokens=self.initial_tokens,
            previous_token=previous,
            text_positions=self.config.text_positions,
        )
        result: list[int] = []
        sum_logprob = np.float32(0.0)
        no_speech_prob = float("nan")
        stop_candidate = self._candidate_stop(rejection)
        speech_position = len(prefix) - len(self.initial_tokens)
        device_selection = getattr(self, "_native_model", None) is not None
        if device_selection:
            self._clear_prefix_states()
            self._clear_greedy_cache()
        try:
            for step in range(self.config.text_positions // 2):
                self._check_cancelled()
                if device_selection:
                    token, logprob, probability = self._greedy_step(
                        prefix,
                        encoder,
                        first=step == 0,
                        score=score,
                        speech_position=speech_position,
                    )
                    if step == 0:
                        no_speech_prob = probability
                else:
                    raw = self.logits(prefix, encoder)
                    if step == 0 and score:
                        no_speech = self.tokenizer.token_to_id("<|nospeech|>")
                        if no_speech is not None:
                            speech_logits = raw[speech_position].astype(np.float32)
                            no_speech_prob = float(
                                np.exp(
                                    speech_logits[no_speech] - np.logaddexp.reduce(speech_logits)
                                )
                            )
                    next_logits = raw[-1].astype(np.float32, copy=True)
                    next_logits[list(self.suppressed)] = -np.inf
                    if step == 0:
                        next_logits[list(self.begin_suppressed)] = -np.inf
                    token = int(np.argmax(next_logits))
                    logprob = (
                        float(next_logits[token] - np.logaddexp.reduce(next_logits))
                        if score
                        else float("nan")
                    )
                sum_logprob = np.float32(sum_logprob + logprob)
                result.append(token)
                if token == self.eot:
                    break
                if stop_candidate is not None and stop_candidate(result, False):
                    result.append(self.eot)
                    break
                prefix.append(token)
                if len(prefix) > self.config.text_positions:
                    result.append(self.eot)
                    break
            else:
                result.append(self.eot)
        finally:
            if device_selection:
                self._clear_greedy_cache()
        text = self.tokenizer.decode(result[:-1], skip_special_tokens=False).strip()
        text_bytes = text.encode("utf-8")
        return DecodeAttempt(
            result[:-1],
            text,
            float(sum_logprob) / len(result),
            no_speech_prob,
            len(text_bytes) / len(zlib.compress(text_bytes)),
            0.0,
            rejection.reason(result[:-1], text, final=True) if rejection.enabled else None,
        )

    def timestamp_decode(self, encoder: _native.EncodedAudio, prompt: list[int]) -> DecodedWindow:
        """Decode one window at temperature zero with timestamp tokens enabled."""
        timestamp_begin = self.tokenizer.token_to_id("<|0.00|>")
        previous = self.tokenizer.token_to_id("<|startofprev|>")
        no_speech = self.tokenizer.token_to_id("<|nospeech|>")
        no_timestamps = self.tokenizer.token_to_id("<|notimestamps|>")
        if any(value is None for value in (timestamp_begin, previous, no_speech, no_timestamps)):
            raise ValueError("checkpoint tokenizer lacks long-form tokens")
        prefix = decoder_prefix(
            prompt,
            initial_tokens=self.initial_tokens[:3],
            previous_token=previous,
            text_positions=self.config.text_positions,
        )
        sot_index = len(prefix) - len(self.initial_tokens[:3])
        first_logits = self.logits(prefix, encoder)
        speech_logits = first_logits[sot_index].astype(np.float32)
        no_speech_prob = float(
            np.exp(speech_logits[no_speech] - np.logaddexp.reduce(speech_logits))
        )
        result: list[int] = []
        sum_logprobs = 0.0
        for step in range(self.config.text_positions // 2):
            self._check_cancelled()
            if len(prefix) > self.config.text_positions:
                break
            raw = first_logits[-1] if step == 0 else self.logits(prefix, encoder)[-1]
            next_logits = raw.astype(np.float32, copy=True)
            next_logits[list(self.suppressed)] = -np.inf
            next_logits[no_timestamps] = -np.inf
            if step == 0:
                next_logits[list(self.begin_suppressed)] = -np.inf
            apply_timestamp_rules(
                next_logits, result, timestamp_begin=timestamp_begin, eot=self.eot
            )
            token = int(np.argmax(next_logits))
            sum_logprobs += float(next_logits[token] - np.logaddexp.reduce(next_logits))
            if token == self.eot:
                break
            result.append(token)
            prefix.append(token)
        return DecodedWindow(result, sum_logprobs / (len(result) + 1), no_speech_prob)


def transcribe_fp32(
    checkpoint_dir: Path,
    audio: np.ndarray,
    *,
    initial_prompt: str | None = None,
    beam_size: int = DecodingSettings.beam_size,
    temperature: float | Sequence[float] = DecodingSettings.temperatures,
    best_of: int = DecodingSettings.best_of,
    compression_ratio_threshold: float | None = DecodingSettings.compression_ratio_threshold,
    logprob_threshold: float | None = DecodingSettings.logprob_threshold,
    no_speech_threshold: float | None = DecodingSettings.no_speech_threshold,
) -> Transcription:
    """Transcribe one <=30 s 16 kHz mono waveform for fp32 correctness checks.

    Engine lifetime and reuse are left to the later listenflow engine-contract issue.
    """
    settings = decoding_settings(
        beam_size=beam_size,
        temperature=temperature,
        best_of=best_of,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
    )
    return _transcribe(
        checkpoint_dir, audio, "fp32", initial_prompt=initial_prompt, settings=settings
    )


def transcribe_fp16(
    checkpoint_dir: Path,
    audio: np.ndarray,
    *,
    initial_prompt: str | None = None,
    beam_size: int = DecodingSettings.beam_size,
    temperature: float | Sequence[float] = DecodingSettings.temperatures,
    best_of: int = DecodingSettings.best_of,
    compression_ratio_threshold: float | None = DecodingSettings.compression_ratio_threshold,
    logprob_threshold: float | None = DecodingSettings.logprob_threshold,
    no_speech_threshold: float | None = DecodingSettings.no_speech_threshold,
) -> Transcription:
    """Transcribe one <=30 s 16 kHz mono waveform for fp16 correctness checks."""
    settings = decoding_settings(
        beam_size=beam_size,
        temperature=temperature,
        best_of=best_of,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
    )
    return _transcribe(
        checkpoint_dir, audio, "fp16", initial_prompt=initial_prompt, settings=settings
    )


def transcribe_long_fp32(
    checkpoint_dir: Path,
    audio: np.ndarray,
    *,
    initial_prompt: str | None = None,
    beam_size: int = DecodingSettings.beam_size,
    temperature: float | Sequence[float] = DecodingSettings.temperatures,
    best_of: int = DecodingSettings.best_of,
    compression_ratio_threshold: float | None = DecodingSettings.compression_ratio_threshold,
    logprob_threshold: float | None = DecodingSettings.logprob_threshold,
    no_speech_threshold: float | None = DecodingSettings.no_speech_threshold,
) -> LongTranscription:
    """Transcribe a complete 16 kHz mono recording with timed segments."""
    settings = decoding_settings(
        beam_size=beam_size,
        temperature=temperature,
        best_of=best_of,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
    )
    return _transcribe_long_fp32(
        checkpoint_dir, audio, initial_prompt=initial_prompt, settings=settings
    )


def _transcribe_long_fp32(
    checkpoint_dir: Path,
    audio: np.ndarray,
    *,
    initial_prompt: str | None = None,
    settings: DecodingSettings = _DEFAULT_DECODING_SETTINGS,
    trace: Callable[[WindowTrace], None] | None = None,
    segment_trace: Callable[[SegmentTrace], None] | None = None,
) -> LongTranscription:
    """Run long-form decoding, optionally recording its windows for evaluation."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("audio must be a finite mono waveform")
    with TensorRTWhisper(checkpoint_dir, precision="fp32", settings=settings) as model:
        mel = model._mel_spectrogram_long(samples)
        timestamp_begin = model.tokenizer.token_to_id("<|0.00|>")
        if timestamp_begin is None:
            raise ValueError("checkpoint tokenizer lacks timestamp tokens")
        return transcribe_mel(
            mel,
            content_frames=mel.shape[-1] - 3_000,
            decode=lambda window, prompt: model.decode_long(model.encode(window), prompt, settings),
            trace=trace,
            segment_trace=segment_trace,
            text=lambda tokens: model.tokenizer.decode(tokens, skip_special_tokens=False),
            timestamp_begin=timestamp_begin,
            eot=model.eot,
            audio_positions=model.config.audio_positions,
            initial_prompt_tokens=model.prompt_tokens(initial_prompt),
            no_speech_threshold=settings.no_speech_threshold,
            logprob_threshold=settings.logprob_threshold,
        )


def _transcribe(
    checkpoint_dir: Path,
    audio: np.ndarray,
    precision: str,
    *,
    initial_prompt: str | None = None,
    settings: DecodingSettings = _DEFAULT_DECODING_SETTINGS,
) -> Transcription:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or samples.size > 480_000 or not np.isfinite(samples).all():
        raise ValueError("audio must be a finite mono waveform of at most 30 seconds")
    with TensorRTWhisper(checkpoint_dir, precision=precision, settings=settings) as model:
        mel = model._mel_spectrogram(samples)
        return model.decode_short(model.encode(mel), model.prompt_tokens(initial_prompt), settings)
