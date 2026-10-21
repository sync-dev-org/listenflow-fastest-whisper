"""Collect static E4M3 input scales with the official fp32 CPU model on dev speech.

Run from python/. Only dev-clean/dev-other are accepted; evaluation speech is
never eligible. Each split contributes equally to all four duration bands. The
decoder is teacher-forced with the development reference, including the normal
English transcription prefix; this is calibration, not a WER measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def select_audio(root, per_band):
    import numpy as np
    import soundfile as sf

    selected, counts = [], {}
    for split in ("dev-clean", "dev-other"):
        directory = root / split
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        references = {}
        for path in sorted(directory.rglob("*.trans.txt")):
            for line in path.read_text().splitlines():
                key, text = line.split(" ", 1)
                references[key] = text
        bands = {band: [] for band in ("0-5", "5-10", "10-20", "20-30")}
        for path in sorted(directory.rglob("*.flac")):
            info = sf.info(path)
            if info.samplerate != 16_000 or info.channels != 1:
                raise ValueError(f"expected 16 kHz mono: {path}")
            seconds = info.frames / info.samplerate
            if not 0 < seconds <= 30:
                continue
            band = (
                "0-5"
                if seconds < 5
                else "5-10"
                if seconds < 10
                else "10-20"
                if seconds < 20
                else "20-30"
            )
            bands[band].append(
                {
                    "split": split,
                    "id": path.stem,
                    "audio": str(path.resolve()),
                    "frames": info.frames,
                    "seconds": seconds,
                    "band": band,
                    "reference": references[path.stem],
                    "sha256": digest(path),
                }
            )
        counts[split] = {band: len(rows) for band, rows in bands.items()}
        for band, rows in bands.items():
            rows.sort(key=lambda row: (row["frames"], row["id"]))
            if len(rows) < per_band:
                raise ValueError(f"not enough calibration speech in {split}/{band}")
            # Include the extremes and evenly spaced ranks, rather than the first speaker.
            selected.extend(
                rows[index] for index in np.linspace(0, len(rows) - 1, per_band, dtype=int)
            )
    return selected, counts


def graph_key(name):
    """Map official Python linear module names onto this repository's checkpoint keys."""
    if ".blocks." not in name:
        return None
    result = "model." + name.replace(".blocks.", ".layers.")
    result = result.replace(".cross_attn.", ".encoder_attn.").replace(".attn.", ".self_attn.")
    for official, local in (
        ("query", "q_proj"),
        ("key", "k_proj"),
        ("value", "v_proj"),
        ("out", "out_proj"),
    ):
        if result.endswith("." + official):
            return result[: -len(official)] + local
    for official, local in (("mlp.0", "fc1"), ("mlp.2", "fc2")):
        if result.endswith("." + official):
            return result[: -len(official)] + local
    return None


def run(checkpoint, librispeech_dir, output, *, per_band=8, threads=8):
    import numpy as np
    import soundfile as sf
    import torch
    import whisper
    from scripts.generate_oracle import MODEL_SHA256
    from whisper.tokenizer import get_tokenizer

    if per_band < 2 or threads < 1:
        raise ValueError("use at least two samples per band and one CPU thread")
    if output.exists() or output.with_suffix(".partial.json").exists():
        raise FileExistsError(output)
    selected, counts = select_audio(librispeech_dir, per_band)
    if digest(checkpoint) != MODEL_SHA256:
        raise ValueError("official large-v3 checkpoint hash mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    model = whisper.load_model(str(checkpoint), device="cpu").float().eval()
    tokenizer = get_tokenizer(
        multilingual=True, num_languages=model.num_languages, language="en", task="transcribe"
    )
    maxima, handles = {}, []
    for name, module in model.named_modules():
        key = graph_key(name)
        if key is None:
            continue
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"calibration target is not Linear: {name}")
        maxima[key] = 0.0

        def observe(_module, inputs, *, target=key):
            maximum = float(inputs[0].detach().abs().amax())
            if not np.isfinite(maximum):
                raise ValueError(f"non-finite activation: {target}")
            maxima[target] = max(maxima[target], maximum)

        handles.append(module.register_forward_pre_hook(observe))
    if len(maxima) != 512:
        raise ValueError(f"expected 512 large-v3 linear inputs, found {len(maxima)}")
    metadata = {
        "format": "lffw-fp8-trial-v1",
        "started_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": MODEL_SHA256,
        "model": "large-v3",
        "compute": "official fp32 model on CPU",
        "torch": torch.__version__,
        "cpu_threads": threads,
        "selection": "equal counts per split and duration band; evenly spaced duration ranks including extremes",
        "available_by_band": counts,
        "per_band_per_split": per_band,
        "decoder": "teacher forcing on the dev reference after SOT/en/transcribe/no-timestamps; no test data",
        "selected": selected,
        "completed": 0,
        "activation_scale": "one scalar per linear input; observed absolute maximum / 448; zero maximum uses 1",
        "weight_scale": "one scale per output column; absolute maximum of checkpoint weight / 448; zero column uses 1",
    }
    start = time.monotonic()
    partial = output.with_suffix(".partial.json")
    try:
        with torch.inference_mode():
            for index, row in enumerate(selected, 1):
                samples, rate = sf.read(row["audio"], dtype="int16")
                if rate != 16_000 or samples.ndim != 1 or len(samples) != row["frames"]:
                    raise ValueError(f"calibration audio changed: {row['audio']}")
                waveform = samples.astype(np.float32) / np.float32(32768)
                mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(waveform), n_mels=128)
                encoded = model.encoder(mel.unsqueeze(0))
                prefix = [
                    *tokenizer.sot_sequence_including_notimestamps,
                    *tokenizer.encode(" " + row["reference"].strip()),
                    tokenizer.eot,
                ]
                tokens = torch.tensor([prefix[: model.dims.n_text_ctx]], dtype=torch.long)
                model.decoder(tokens, encoded)
                metadata.update(
                    completed=index,
                    elapsed_seconds=time.monotonic() - start,
                    input_amax=dict(sorted(maxima.items())),
                    scales={
                        key: value / 448 if value else 1.0 for key, value in sorted(maxima.items())
                    },
                )
                partial.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
                print(
                    f"calibration {index}/{len(selected)} {row['split']}/{row['id']} elapsed={time.monotonic() - start:.1f}s",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()
    metadata["finished_at"] = datetime.now(UTC).isoformat()
    partial.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    partial.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--librispeech-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-band", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    run(
        args.checkpoint,
        args.librispeech_dir,
        args.output,
        per_band=args.per_band,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
