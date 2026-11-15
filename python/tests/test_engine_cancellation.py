"""Cancellation crosses the async/native boundary and stops the real token loops."""

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models

from listenflow import (
    EngineClosedError,
    EngineState,
    OfflineModelDirectorySource,
    UtteranceBoundarySettings,
    UtteranceEnd,
    UtteranceStart,
)
from listenflow.fastest_whisper import FastestWhisperEngine
from listenflow.fastest_whisper._cached_model import CachedWhisper


@pytest.fixture
def token_runtime(monkeypatch):
    """Block the native token call while keeping real CPU decode and cancellation loops."""
    from listenflow.fastest_whisper import _cached_model

    started, release, blocking = threading.Event(), threading.Event(), threading.Event()
    decoder_calls = []

    def execute(token):
        if blocking.is_set():
            decoder_calls.append([token])
            if not started.is_set():
                started.set()
                assert release.wait(5), "test did not release native inference"
        logits = np.full(11, -10.0, np.float32)
        logits[0] = 10
        return logits

    class Prefix:
        def close(self):
            pass

    class Decoder:
        def prefix_step(self, parent, token):
            return Prefix(), execute(token)

        def greedy_step(self, token, first, score):
            logits = execute(token)
            normalizer = np.logaddexp.reduce(logits)
            return 0, float(logits[0] - normalizer), float(np.exp(logits[7] - normalizer))

        def close(self):
            pass

    class Encoder:
        def decoder(self, kind, batch=1):
            return Decoder()

        def close(self):
            pass

    class Model(CachedWhisper):
        def __init__(self, *args, **kwargs):
            self.device = 0
            self.config = SimpleNamespace(
                n_mels=128,
                text_positions=16,
                audio_positions=1500,
                audio_state=4,
                text_layers=1,
                text_heads=1,
                vocab_size=11,
            )
            self.precision = "fp32"
            self.initial_tokens = [1, 2, 4, 5]
            self.eot = 3
            self.suppressed, self.begin_suppressed = set(), set()
            self.tokenizer = Tokenizer(
                models.WordLevel(
                    vocab={
                        "word": 0,
                        "<|startoftranscript|>": 1,
                        "<|en|>": 2,
                        "<|endoftext|>": 3,
                        "<|transcribe|>": 4,
                        "<|notimestamps|>": 5,
                        "<|startofprev|>": 6,
                        "<|nospeech|>": 7,
                        "<|0.00|>": 8,
                        "<|0.02|>": 9,
                        "<|0.04|>": 10,
                    },
                    unk_token="word",
                )
            )
            self._native_model = SimpleNamespace(close=lambda: None)
            self._mel_tensor = self._encoded_audio = None
            self._prefix_decoder = self._greedy_decoder = None
            self._prefix_states, self._root_prefix, self._greedy_length = {}, (), 0
            self._sampled_decoder = None

        def _mel_spectrogram(self, audio):
            return np.zeros((128, 3000), np.float32)

        def _mel_spectrogram_long(self, audio):
            return np.zeros((128, audio.size // 160 + 3000), np.float32)

        def encode(self, mel):
            self._clear_prefix_states()
            self._clear_greedy_cache()
            self._encoded_audio = Encoder()
            return self._encoded_audio

    monkeypatch.setattr(_cached_model, "CachedWhisper", Model)
    try:
        yield started, release, blocking, decoder_calls
    finally:
        release.set()


def _engine(tmp_path):
    return FastestWhisperEngine(
        model_source=OfflineModelDirectorySource(str(tmp_path)),
        precision="fp32",
        device=0,
        language="en",
        boundary_settings=UtteranceBoundarySettings(preroll_s=0),
    )


async def _chunks(samples=160):
    yield np.full(samples, 1000, dtype="<i2").tobytes()


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize("stop", ["caller", "iterator", "engine"])
@pytest.mark.parametrize("samples", [160, 480_160])
def test_cancellation_waits_for_native_token_and_starts_no_next_token(
    tmp_path, token_runtime, stop, samples
):
    """Caller cancellation, iterator close, and engine close wait and stop before another token."""
    started, release, blocking, calls = token_runtime
    engine = _engine(tmp_path)

    async def run():
        await engine.prepare()
        blocking.set()
        stream = engine.stream(_chunks(samples))
        assert isinstance(await anext(stream), UtteranceStart)
        assert await asyncio.to_thread(started.wait, 5)
        if stop == "caller":
            stopping = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            stopping.cancel()
        elif stop == "iterator":
            stopping = asyncio.create_task(stream.aclose())
        else:
            stopping = asyncio.create_task(engine.aclose())
        # Let the cancellation pass through the actual core/pipeline task cleanup.
        for _ in range(20):
            await asyncio.sleep(0)
        premature = stopping.done()
        busy = engine.health().busy
        release.set()
        if stop == "caller":
            with pytest.raises(asyncio.CancelledError):
                await stopping
        else:
            await stopping
        assert not premature, "cancellation returned while native inference was still running"
        assert busy, "the engine released its model slot before native work finished"
        assert len(calls) == 1, "a second token was started after cancellation"
        assert not engine.health().busy
        if stop == "engine":
            assert engine.health().state is EngineState.CLOSED
            with pytest.raises(EngineClosedError):
                await engine.prepare()
        else:
            blocking.clear()
            events = [event async for event in engine.stream(_chunks())]
            assert isinstance(events[-1], UtteranceEnd)
            await engine.aclose()

    asyncio.run(run())


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize("stop", ["caller", "engine"])
def test_cancelled_prepare_waits_and_returns_warmup_resources(tmp_path, token_runtime, stop):
    """Cancellation during trial inference waits for native completion and discards its model."""
    started, release, blocking, calls = token_runtime
    blocking.set()
    engine = _engine(tmp_path)

    async def run():
        preparing = asyncio.create_task(engine.prepare())
        assert await asyncio.to_thread(started.wait, 5)
        if stop == "caller":
            preparing.cancel()
            stopping = preparing
        else:
            stopping = asyncio.create_task(engine.aclose())
        for _ in range(20):
            await asyncio.sleep(0)
        premature = stopping.done()
        release.set()
        if stop == "caller":
            with pytest.raises(asyncio.CancelledError):
                await stopping
        else:
            await stopping
            with pytest.raises(EngineClosedError):
                await preparing
        assert not premature, "preparation returned while its warmup was still running"
        assert len(calls) == 1
        assert engine._model is None
        assert not engine.health().busy
        if stop == "caller":
            assert engine.health().state is EngineState.UNPREPARED
            blocking.clear()
            await engine.prepare()
            await engine.aclose()

    asyncio.run(run())
