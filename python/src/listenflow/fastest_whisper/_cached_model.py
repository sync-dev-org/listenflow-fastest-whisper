"""A reusable Whisper runtime backed by tensoracle's serialized engines."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from ._audio import N_SAMPLES
from ._checkpoint import Checkpoint
from ._decoding import DecodingSettings, decoding_settings
from ._engines import _open_mel_engine, open_whisper_engine
from ._inference import TensorRTWhisper
from ._longform import TimedSegment, transcribe_mel

_LANGUAGE_TOKEN = re.compile(r"<\|([a-z]{2,3})\|>")


class CachedWhisper(TensorRTWhisper):
    def _open_engine(self, checkpoint: Checkpoint, *, decoder: bool):
        return open_whisper_engine(
            checkpoint,
            decoder=decoder,
            precision=self.precision,
            logger=self.logger,
        )

    def warmup(self) -> None:
        mel = self._mel_spectrogram(np.zeros(16_000, dtype=np.float32))
        self.greedy_decode(self.encode(mel))

    def _open_mel(self, checkpoint: Checkpoint):
        return _open_mel_engine(
            checkpoint,
            precision=self.precision,
            logger=self.logger,
        )

    def detect_language(self, audio: np.ndarray) -> str:
        mel = self._mel_spectrogram(audio[:N_SAMPLES])
        encoder = self.encode(mel)
        logits = self.logits([self.initial_tokens[0]], encoder)[-1]
        candidates = {
            token_id: match.group(1)
            for token, token_id in self.tokenizer.get_vocab().items()
            if (match := _LANGUAGE_TOKEN.fullmatch(token)) is not None
        }
        if not candidates:
            raise ValueError("checkpoint tokenizer has no language tokens")
        return candidates[max(candidates, key=lambda token_id: float(logits[token_id]))]

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        initial_prompt: str | None = None,
        beam_size: int = DecodingSettings.beam_size,
        temperature: float | Sequence[float] = DecodingSettings.temperatures,
        best_of: int = DecodingSettings.best_of,
        compression_ratio_threshold: float | None = DecodingSettings.compression_ratio_threshold,
        logprob_threshold: float | None = DecodingSettings.logprob_threshold,
        no_speech_threshold: float | None = DecodingSettings.no_speech_threshold,
    ) -> list[TimedSegment]:
        settings = decoding_settings(
            beam_size=beam_size,
            temperature=temperature,
            best_of=best_of,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
        )
        prompt_tokens = self.prompt_tokens(initial_prompt)
        if audio.size <= N_SAMPLES:
            mel = self._mel_spectrogram(audio)
            result = self.decode_short(self.encode(mel), prompt_tokens, settings)
            return [TimedSegment(0.0, audio.size / 16_000, result.text)] if result.text else []

        mel = self._mel_spectrogram_long(audio)
        timestamp_begin = self.tokenizer.token_to_id("<|0.00|>")
        if timestamp_begin is None:
            raise ValueError("checkpoint tokenizer lacks timestamp tokens")
        result = transcribe_mel(
            mel,
            content_frames=mel.shape[-1] - 3_000,
            decode=lambda window, prompt: self.decode_long(self.encode(window), prompt, settings),
            text=lambda tokens: self.tokenizer.decode(tokens, skip_special_tokens=False),
            timestamp_begin=timestamp_begin,
            eot=self.eot,
            audio_positions=self.config.audio_positions,
            initial_prompt_tokens=prompt_tokens,
            no_speech_threshold=settings.no_speech_threshold,
            logprob_threshold=settings.logprob_threshold,
        )
        return [segment for segment in result.segments if segment.text.strip()]
