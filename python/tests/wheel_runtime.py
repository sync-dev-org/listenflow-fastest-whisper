"""Probe an installed wheel using only its declared runtime dependencies.

SP-LFFASTESTWHISPER-003, -006, -007, -008, -010: prepare, recognize,
compare token sequences, and inspect saved-engine reuse in a fresh process.
Run this file with the independent environment's Python and isolated mode.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import wave
from pathlib import Path

import numpy as np

from listenflow import Final, OfflineModelDirectorySource
from listenflow.fastest_whisper import FastestWhisperEngine
from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._decoding import DecodingSettings
from listenflow.fastest_whisper._plugins import _fc1_library


def saved_engines():
    directory = Path(os.environ["XDG_CACHE_HOME"]) / "tensoracle"
    result = {}
    for path in sorted(directory.glob("*.engine")):
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        result[path.name] = {
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": digest,
        }
    return result


def pcm(path):
    with wave.open(str(path)) as source:
        assert (source.getframerate(), source.getnchannels(), source.getsampwidth()) == (
            16000,
            1,
            2,
        )
        return source.readframes(source.getnframes())


async def stream(checkpoint, audio):
    async def chunks():
        yield pcm(audio)

    async with FastestWhisperEngine(
        model_source=OfflineModelDirectorySource(str(checkpoint)),
        precision="fp16",
        device=0,
        language="en",
        temperature=0.0,
    ) as engine:
        return [event.text async for event in engine.stream(chunks()) if isinstance(event, Final)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, action="append", required=True)
    parser.add_argument("--expect", type=Path)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    before = saved_engines()
    transcripts = asyncio.run(stream(args.checkpoint, args.audio[0]))
    assert transcripts and all(transcripts)
    tokens = []
    with CachedWhisper(args.checkpoint, precision="fp16") as model:
        for audio in args.audio:
            samples = np.frombuffer(pcm(audio), dtype="<i2").astype(np.float32) / 32768
            encoded = model.encode(model._mel_spectrogram(samples))
            result = model.decode_short(encoded, [], DecodingSettings(temperatures=(0.0,)))
            tokens.append({"audio": audio.name, "text": result.text, "tokens": list(result.tokens)})
    after = saved_engines()
    assert len(after) == 3
    if args.reuse:
        assert before == after, "a saved engine was rebuilt instead of reused"
    observed = {"transcripts": transcripts, "recognitions": tokens}
    if args.expect:
        assert observed == json.loads(args.expect.read_text())["observed"]
    assert not {"torch", "triton", "tensoracle.producer"}.intersection(sys.modules)
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "versions": {
                    dist.metadata["Name"]: dist.version
                    for dist in importlib.metadata.distributions()
                },
                "library": str(_fc1_library()),
                "observed": observed,
                "engines": after,
            }
        )
    )


if __name__ == "__main__":
    main()
