"""Collect static E4M3 input scales with the official fp32 CPU model.

Run from python/. Keep LibriSpeech dev-clean/dev-other and optionally add the
nine representative languages from FLEURS validation shards. Evaluation splits
are never eligible. The decoder is teacher-forced with each recording's language
and reference. Record FLEURS-only per-language maxima against the existing scales
alongside the combined new scales; this is calibration, not an error-rate measure.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

BANDS = ("0-5", "5-10", "10-20", "20-30")
DEFAULT_SCALES = (
    Path(__file__).resolve().parents[1] / "src/listenflow/fastest_whisper/assets/fp8-scales.json"
)


def duration_band(seconds):
    if not 0 < seconds <= 30:
        return None
    return (
        "0-5" if seconds < 5 else "5-10" if seconds < 10 else "10-20" if seconds < 20 else "20-30"
    )


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
        bands = {band: [] for band in BANDS}
        for path in sorted(directory.rglob("*.flac")):
            info = sf.info(path)
            if info.samplerate != 16_000 or info.channels != 1:
                raise ValueError(f"expected 16 kHz mono: {path}")
            seconds = info.frames / info.samplerate
            band = duration_band(seconds)
            if band is None:
                continue
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


def _validation_shard(path):
    if any(
        not name.startswith("validation-") or not name.endswith(".parquet")
        for name in (path.name, path.resolve().name)
    ):
        raise ValueError(f"only validation parquet is eligible; test speech is forbidden: {path}")


def select_fleurs_audio(root, per_band):
    """Balance validation duration bands across all nine languages, without replacement.

    Each band contributes min(per_band, availability in every language) recordings
    per language. An empty band in any language is therefore omitted for all nine.
    WAV headers determine durations; declared num_samples remains separate evidence.
    """
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from scripts.evaluate_fleurs import CONFIGS

    root = Path(root)
    if not root.is_dir():
        raise ValueError(
            f"expected a dataset directory containing validation shards, not test: {root}"
        )
    if per_band < 2:
        raise ValueError("use at least two samples per band")
    shards = {}
    for language, config in CONFIGS.items():
        shards[language] = sorted((root / config).glob("validation-*.parquet"))
        if not shards[language]:
            raise FileNotFoundError(f"no validation parquet for {language}: {root / config}")
        for path in shards[language]:
            _validation_shard(path)
    bands = {language: {band: [] for band in BANDS} for language in CONFIGS}
    columns = ["id", "path", "transcription", "num_samples", "audio"]
    for language, config in CONFIGS.items():
        seen = set()
        for path in shards[language]:
            for index, batch in enumerate(
                pq.ParquetFile(path).iter_batches(batch_size=1, columns=columns, use_threads=False)
            ):
                source = batch.to_pylist()[0]
                filename = Path(source["path"]).name
                key = f"{config}/{filename}"
                if key in seen:
                    raise ValueError(f"duplicate FLEURS recording: {key}")
                seen.add(key)
                audio = source["audio"]
                if audio["path"] and Path(audio["path"]).name != filename:
                    raise ValueError(f"audio identity differs: {key}")
                info = sf.info(io.BytesIO(audio["bytes"]))
                if info.samplerate != 16000 or info.channels != 1:
                    raise ValueError(f"expected 16 kHz mono: {key}")
                seconds = info.frames / info.samplerate
                band = duration_band(seconds)
                if band is None:
                    continue
                bands[language][band].append(
                    {
                        "dataset": "fleurs",
                        "split": "validation",
                        "language": language,
                        "config": config,
                        "id": key,
                        "fleurs_id": int(source["id"]),
                        "parquet": str(path.resolve()),
                        "row_index": index,
                        "frames": info.frames,
                        "declared_samples": int(source["num_samples"]),
                        "seconds": seconds,
                        "band": band,
                        "reference": source["transcription"],
                        "sha256": hashlib.sha256(audio["bytes"]).hexdigest(),
                    }
                )
    counts = {
        language: {band: len(rows) for band, rows in groups.items()}
        for language, groups in bands.items()
    }
    quotas = {band: min(per_band, *(counts[lang][band] for lang in CONFIGS)) for band in BANDS}
    selected = []
    for language, groups in bands.items():
        language_rows = []
        for band, rows in groups.items():
            rows.sort(key=lambda row: (row["frames"], row["id"]))
            language_rows.extend(
                rows[index] for index in np.linspace(0, len(rows) - 1, quotas[band], dtype=int)
            )
        # Read each shard once during inference, in the same order as this manifest.
        selected.extend(sorted(language_rows, key=lambda row: (row["parquet"], row["row_index"])))
    if not selected:
        raise ValueError("no duration band contains validation speech in every language")
    return selected, counts


def read_audio(selected):
    """Read selected files and stream embedded WAVs without retaining the full corpus."""
    import numpy as np
    import soundfile as sf

    shards = {}
    for row in selected:
        if row.get("dataset") == "fleurs":
            shards.setdefault(row["parquet"], {})[row["row_index"]] = row
            continue
        path = Path(row["audio"])
        if digest(path) != row["sha256"]:
            raise ValueError(f"calibration audio changed: {path}")
        samples, rate = sf.read(path, dtype="int16")
        if rate != 16000 or samples.ndim != 1 or len(samples) != row["frames"]:
            raise ValueError(f"calibration audio changed: {path}")
        yield row, samples.astype(np.float32) / np.float32(32768)
    if shards:
        import pyarrow.parquet as pq

        for filename, pending in shards.items():
            path = Path(filename)
            _validation_shard(path)
            for index, batch in enumerate(
                pq.ParquetFile(path).iter_batches(
                    batch_size=1, columns=["audio"], use_threads=False
                )
            ):
                row = pending.pop(index, None)
                if row is None:
                    continue
                payload = batch.to_pylist()[0]["audio"]["bytes"]
                if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                    raise ValueError(f"calibration audio changed: {row['id']}")
                samples, rate = sf.read(io.BytesIO(payload), dtype="float32")
                if (
                    rate != 16000
                    or samples.ndim != 1
                    or len(samples) != row["frames"]
                    or not np.isfinite(samples).all()
                ):
                    raise ValueError(f"expected unchanged finite mono 16 kHz audio: {row['id']}")
                yield row, samples
                if not pending:
                    break
            if pending:
                raise ValueError(f"selected calibration recordings are missing: {path}")


def teacher_forcing_tokens(reference, language, *, num_languages, text_ctx):
    from whisper.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(
        multilingual=True, num_languages=num_languages, language=language, task="transcribe"
    )
    return [
        *tokenizer.sot_sequence_including_notimestamps,
        *tokenizer.encode(" " + reference.strip()),
        tokenizer.eot,
    ][:text_ctx]


def compare_scales(maxima_by_language, scales):
    """Retain every ratio and rank the ten largest ratios per language."""
    result = {}
    for language, maxima in maxima_by_language.items():
        if maxima.keys() != scales.keys():
            raise ValueError(f"baseline scale layer keys differ: {language}")
        layers = {}
        for key, maximum in sorted(maxima.items()):
            scale = scales[key]
            if not math.isfinite(scale) or scale <= 0 or not math.isfinite(scale * 448):
                raise ValueError(f"expected positive finite baseline scale: {key}")
            if not math.isfinite(maximum) or maximum < 0:
                raise ValueError(f"expected nonnegative finite activation: {key}")
            layers[key] = {
                "input_amax": maximum,
                "scale": scale,
                "limit": scale * 448,
                "ratio": maximum / (scale * 448),
            }
        result[language] = {
            "layers": layers,
            "clipped_layers": sum(row["ratio"] > 1 for row in layers.values()),
            "top_layers": sorted(
                ({"layer": key, **row} for key, row in layers.items()),
                key=lambda row: (-row["ratio"], row["layer"]),
            )[:10],
        }
    return result


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


def run(
    checkpoint,
    librispeech_dir,
    output,
    *,
    fleurs_dir=None,
    baseline_scales=DEFAULT_SCALES,
    per_band=8,
    threads=8,
):
    import numpy as np
    import torch
    import whisper
    from scripts.generate_oracle import MODEL_SHA256

    if per_band < 2 or threads < 1:
        raise ValueError("use at least two samples per band and one CPU thread")
    if output.exists() or output.with_suffix(".partial.json").exists():
        raise FileExistsError(output)
    selected, counts = select_audio(librispeech_dir, per_band)
    fleurs_counts = {}
    if fleurs_dir is not None:
        fleurs_selected, fleurs_counts = select_fleurs_audio(fleurs_dir, per_band)
        selected.extend(fleurs_selected)
    baseline = json.loads(baseline_scales.read_text())
    if not isinstance(baseline, dict):
        raise TypeError("expected baseline scales to be a layer-to-scale mapping")
    if digest(checkpoint) != MODEL_SHA256:
        raise ValueError("official large-v3 checkpoint hash mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    model = whisper.load_model(str(checkpoint), device="cpu").float().eval()
    maxima, by_language, handles = {}, {}, []
    active_language = None
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
            if active_language is not None:
                current = by_language[active_language]
                current[target] = max(current[target], maximum)

        handles.append(module.register_forward_pre_hook(observe))
    if len(maxima) != 512:
        for handle in handles:
            handle.remove()
        raise ValueError(f"expected 512 large-v3 linear inputs, found {len(maxima)}")
    by_language = {language: dict(maxima) for language in fleurs_counts}
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
        "available_fleurs_by_band": fleurs_counts,
        "selected_fleurs_by_band": {
            language: {
                band: sum(
                    row.get("language") == language and row["band"] == band for row in selected
                )
                for band in BANDS
            }
            for language in fleurs_counts
        },
        "fleurs_selection": "equal counts per language and band, capped by the sparsest language; empty bands omitted; evenly spaced ranks including extremes",
        "per_band_per_split": per_band,
        "decoder": "teacher forcing on the reference after SOT/language/transcribe/no-timestamps; LibriSpeech uses en; no test data",
        "language_maxima": "FLEURS validation only; combined maxima and scales also include LibriSpeech dev speech",
        "baseline_scales": str(baseline_scales.resolve()),
        "baseline_scales_sha256": digest(baseline_scales),
        "selected": selected,
        "completed": 0,
        "activation_scale": "one scalar per linear input; observed absolute maximum / 448; zero maximum uses 1",
        "weight_scale": "one scale per output column; absolute maximum of checkpoint weight / 448; zero column uses 1",
    }
    start = time.monotonic()
    partial = output.with_suffix(".partial.json")
    try:
        compare_scales(by_language, baseline)
        with torch.inference_mode():
            for index, (row, waveform) in enumerate(read_audio(selected), 1):
                active_language = row.get("language")
                mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(waveform), n_mels=128)
                encoded = model.encoder(mel.unsqueeze(0))
                prefix = teacher_forcing_tokens(
                    row["reference"],
                    row.get("language", "en"),
                    num_languages=model.num_languages,
                    text_ctx=model.dims.n_text_ctx,
                )
                tokens = torch.tensor([prefix], dtype=torch.long)
                model.decoder(tokens, encoded)
                metadata.update(
                    completed=index,
                    elapsed_seconds=time.monotonic() - start,
                    input_amax=dict(sorted(maxima.items())),
                    input_amax_by_language=by_language,
                    scale_comparison_by_language=compare_scales(by_language, baseline),
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
    parser.add_argument(
        "--fleurs-dir",
        type=Path,
        help="dataset revision directory with <config>/validation-*.parquet",
    )
    parser.add_argument("--baseline-scales", type=Path, default=DEFAULT_SCALES)
    parser.add_argument("--per-band", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    run(
        args.checkpoint,
        args.librispeech_dir,
        args.output,
        fleurs_dir=args.fleurs_dir,
        baseline_scales=args.baseline_scales,
        per_band=args.per_band,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
