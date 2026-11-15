"""Observe the real product decoder through warmup, cancellation and reuse."""

import asyncio
import inspect
import json
import os
import sys
import threading
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
from cuda.bindings import runtime as cudart

from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._inference import TensorRTWhisper


def cancel_prefix_after_steps(model, encoded, prefix, limit=4):
    """Cancel at a Python token boundary after the requested native steps finish."""
    cancelled = threading.Event()
    previous = model._cancel_requested
    original = TensorRTWhisper._check_cancelled
    steps = 0

    def check(self):
        nonlocal steps
        original(self)
        caller = inspect.currentframe().f_back
        token_step = (
            caller.f_code is TensorRTWhisper.logits.__code__ and "position" in caller.f_locals
        )
        del caller
        if self is model and token_step:
            if steps == limit:
                cancelled.set()
                original(self)
            steps += 1

    model._cancel_requested = cancelled
    try:
        with patch.object(TensorRTWhisper, "_check_cancelled", check):
            try:
                model.logits(prefix, encoded)
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("decoder ignored cancellation")
    finally:
        model._cancel_requested = previous
    assert steps == limit
    return steps


def main():
    log = Path(os.environ["LFFW_QKV_DRIVER_LOG"])
    status, properties = cudart.cudaGetDeviceProperties(0)
    assert status == cudart.cudaError_t.cudaSuccess
    sm = properties.major * 10 + properties.minor
    with CachedWhisper(Path(sys.argv[1]), precision="fp16") as model:
        model.warmup()
        warmup_launches = sum(
            line.startswith("fc1_launch|") for line in log.read_text().splitlines()
        )
        waveform = Path(__file__).parent / "fixtures/small_boat_16khz_mono_s16le.wav"
        with wave.open(str(waveform)) as source:
            samples = (
                np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(
                    np.float32
                )
                / 32768
            )
        encoded = model.encode(model._mel_spectrogram(samples))
        steps = cancel_prefix_after_steps(model, encoded, model.initial_tokens + [198] * 32)
        text = model.transcribe(samples, temperature=0.0)[0].text
    assert model._native_model is model._encoded_audio is model._mel_tensor is None
    assert not any(name in sys.modules for name in ("torch", "triton", "tensoracle.producer"))
    print(
        json.dumps(
            {
                "warmup_fc1_launches": warmup_launches,
                "cancelled_steps": steps,
                "text": text,
                "sm": sm,
            }
        )
    )


if __name__ == "__main__":
    main()
