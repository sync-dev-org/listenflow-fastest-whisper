"""Prepare the fixed LibriSpeech chapter recordings used for window-boundary checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scripts.evaluate_wer import validate_mono_waveform

CHAPTER = ("test-clean", "8463", "294825")
SOURCE_SHA256 = (
    "22f91479f9e3a69d334e8c861bf5e42258bff2aadf7fd60473c8f53ce72e4cc6",
    "41969012dfe30603a0cd94c3638273e7a99de6281b4b1104e21ae0d8703ab9e2",
    "556e61572acda3390e90ca9e077859533707e9b9bfb0063304220de97fba5a31",
    "fbc296993dccac596e6b674e3884df3892f1aaf4871721aa4a30c966bb2f8c48",
    "a0ad0166180e16e30fdd3c4b28584ba569c329f7c9f2fd124ccb88c92e5cc386",
    "afa38fd15e34de4f5a62407a0c830168eeb2454a54cd71c4ec2e4043fe0e91bb",
    "27a534645f01ce5844ed1dc986511c8d7dbe8abf62dc5725b4eb981d8f6eefe4",
    "ae7287dd69015aa06f15d282e2ba917b28a7cdcd01ff069ff3c7ef59d5738be9",
)
BOUNDARIES = {
    "boundary_30_01": (
        4,
        480_160,
        "57ffeb313a5a649e7868ddb950a97b4d01ad0b5a80e57f8df7b3cca3ce93b08a",
    ),
    "boundary_34_675": (
        5,
        None,
        "1fb01f97f9b3f59beded51acc6145ab0583281305e9519b84a6ec7f91ffe7ff7",
    ),
    "boundary_74_56": (8, None, "593e975b5d41463bd1bf887e7b9653f8cfad19bdbef2474e34ed2ba43626e4f1"),
}


def assemble_chapter(
    directory: Path, utterance_ids: list[str], *, limit_samples: int | None = None
) -> np.ndarray:
    """Concatenate PCM16 utterances in the chosen order without adding a gap."""
    import soundfile as sf

    pieces = []
    for utterance_id in utterance_ids:
        path = directory / f"{utterance_id}.flac"
        info = sf.info(path)
        if (info.samplerate, info.channels, info.subtype) != (16_000, 1, "PCM_16"):
            raise ValueError(f"unexpected LibriSpeech FLAC format: {path}")
        pcm, rate = sf.read(path, dtype="int16")
        if rate != 16_000 or pcm.ndim != 1 or len(pcm) != info.frames:
            raise ValueError(f"LibriSpeech FLAC changed while reading: {path}")
        pieces.append(pcm.astype(np.float32) / np.float32(32768.0))
    audio = validate_mono_waveform(np.concatenate(pieces), sample_rate=16_000)
    if limit_samples is not None:
        if len(audio) < limit_samples:
            raise ValueError("LibriSpeech chapter is shorter than the requested boundary")
        audio = audio[:limit_samples]
    return audio


def prepare(librispeech_dir: Path, output_dir: Path) -> None:
    chapter = librispeech_dir.joinpath(*CHAPTER)
    source_ids = [f"8463-294825-{index:04}" for index in range(8)]
    for utterance_id, expected_hash in zip(source_ids, SOURCE_SHA256):
        source = chapter / f"{utterance_id}.flac"
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"LibriSpeech source does not match the fixed input: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (count, limit_samples, expected_hash) in BOUNDARIES.items():
        audio = assemble_chapter(chapter, source_ids[:count], limit_samples=limit_samples)
        actual_hash = hashlib.sha256(audio.tobytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"LibriSpeech boundary waveform does not match: {name}")
        np.save(output_dir / f"{name}.npy", audio)
        manifest[name] = {
            "source_ids": source_ids[:count],
            "gap_samples": 0,
            "samples": len(audio),
            "seconds": len(audio) / 16_000,
            "waveform_sha256": actual_hash,
        }
    (output_dir / "libri_boundaries.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.librispeech_dir, args.output_dir)


if __name__ == "__main__":
    main()
