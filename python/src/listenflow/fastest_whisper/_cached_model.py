"""A reusable Whisper runtime backed by tensoracle's serialized engines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._audio import N_SAMPLES
from ._checkpoint import Checkpoint
from ._decoding import DecodingSettings, decoding_settings
from ._engines import _open_mel_engine, open_whisper_engine
from ._inference import TensorRTWhisper
from ._longform import TimedSegment, transcribe_mel


class CachedWhisper(TensorRTWhisper):
    def _open_sampled_engine(self, checkpoint: Checkpoint, *, batch_size: int):
        return open_whisper_engine(
            checkpoint,
            decoder=True,
            precision=self.precision,
            logger=self.logger,
            sample_batch=batch_size,
        )

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

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | list[str] | None = "en",
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
        if isinstance(language, list) and len(language) == 1:
            language = language[0]
        if audio.size <= N_SAMPLES:
            mel = self._mel_spectrogram(audio)
            encoder = self.encode(mel)
            self._set_language(
                language if isinstance(language, str) else self._detect_language(encoder, language)
            )
            result = self.decode_short(encoder, prompt_tokens, settings)
            return [TimedSegment(0.0, audio.size / 16_000, result.text)] if result.text else []

        mel = self._mel_spectrogram_long(audio)
        timestamp_begin = self.tokenizer.token_to_id("<|0.00|>")
        if timestamp_begin is None:
            raise ValueError("checkpoint tokenizer lacks timestamp tokens")

        def decode(window, prompt):
            encoder = self.encode(window)
            self._set_language(
                language if isinstance(language, str) else self._detect_language(encoder, language)
            )
            return self.decode_long(encoder, prompt, settings)

        result = transcribe_mel(
            mel,
            content_frames=mel.shape[-1] - 3_000,
            decode=decode,
            text=lambda tokens: self.tokenizer.decode(tokens, skip_special_tokens=False),
            timestamp_begin=timestamp_begin,
            eot=self.eot,
            audio_positions=self.config.audio_positions,
            initial_prompt_tokens=prompt_tokens,
            no_speech_threshold=settings.no_speech_threshold,
            logprob_threshold=settings.logprob_threshold,
        )
        return [segment for segment in result.segments if segment.text.strip()]
