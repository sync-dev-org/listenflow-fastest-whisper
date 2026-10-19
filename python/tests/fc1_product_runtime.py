"""Observe the real product decoder through warmup, cancellation and reuse."""

import asyncio
import json
import os
import sys
import threading
import wave
from pathlib import Path

import numpy as np
from cuda.bindings import runtime as cudart

from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._inference import _cuda, _CudaGraphs


def main():
    log = Path(os.environ["LFFW_QKV_DRIVER_LOG"])
    properties = _cuda(cudart.cudaGetDeviceProperties(0), "cudaGetDeviceProperties")
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
        cancelled = threading.Event()
        model._cancel_requested = cancelled
        steps = 0

        def observe(frame, event, _value):
            nonlocal steps
            if event == "return" and frame.f_code is _CudaGraphs.enqueue.__code__:
                steps += 1
                if steps == 4:
                    cancelled.set()

        previous = sys.getprofile()
        sys.setprofile(observe)
        try:
            try:
                model.logits(model.initial_tokens + [198] * 32, encoded)
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("decoder ignored cancellation")
        finally:
            sys.setprofile(previous)
            model._cancel_requested = None
        assert steps == 4
        text = model.transcribe(samples, temperature=0.0)[0].text
        graph_contexts = len(model._decoder._graphs.entries)
    assert model._decoder is model._encoder is model._mel is None
    assert not any(name in sys.modules for name in ("torch", "triton", "tensoracle.producer"))
    print(
        json.dumps(
            {
                "warmup_fc1_launches": warmup_launches,
                "cancelled_steps": steps,
                "graph_contexts": graph_contexts,
                "text": text,
                "sm": sm,
            }
        )
    )


if __name__ == "__main__":
    main()
