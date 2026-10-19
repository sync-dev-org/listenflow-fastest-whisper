"""Generate the fixed large-v3 fp32 oracle with openai-whisper==20250625 on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

import numpy as np

MODEL_SHA256 = "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb"
MODEL_URL = f"https://openaipublic.azureedge.net/main/whisper/models/{MODEL_SHA256}/large-v3.pt"
HF_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
FIXTURE_NAMES = ("small_boat_16khz_mono_s16le", "silence_16khz_mono_s16le")
FIXTURE_SOURCES = {
    "small_boat_16khz_mono_s16le": {
        "source": "Qwen3-TTS-12Hz-1.7B-CustomVoice, built-in Aiden voice, no reference audio",
        "source_url": "https://github.com/sync-dev-org/listenflow-faster-whisper/tree/d7c6087edd7b871bebda8d245a71a5f22b01cdc3/tests/fixtures",
        "record": "tests/fixtures/small_boat_speech_record.txt",
        "license": "MIT",
        "change": "24 kHz synthetic speech resampled to mono 16 kHz PCM16 with FFmpeg",
    },
    "silence_16khz_mono_s16le": {
        "source": "generated for this repository",
        "license": "no third-party source",
        "change": "one second of zero-valued PCM16 samples",
    },
}


def read_pcm16_mono(path: Path) -> np.ndarray:
    """Read the exact samples stored in a 16 kHz mono PCM16 WAV file."""
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getframerate() != 16_000
            or audio.getcomptype() != "NONE"
        ):
            raise ValueError(f"{path} must be 16 kHz mono PCM16 WAV")
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / np.float32(32768.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(
    checkpoint: Path,
    audio_dir: Path,
    output_dir: Path,
    *,
    initial_prompt: str | None = None,
    beam_size: int | None = None,
) -> None:
    import torch
    import whisper
    from whisper.decoding import DecodingOptions, DecodingTask
    from whisper.tokenizer import get_tokenizer

    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("the official large-v3.pt SHA-256 does not match the fixed oracle")

    torch.set_num_threads(min(torch.get_num_threads(), 8))
    model = whisper.load_model(str(checkpoint), device="cpu").eval()
    prompt_tokens = (
        get_tokenizer(multilingual=model.is_multilingual, language="en", task="transcribe").encode(
            " " + initial_prompt.strip()
        )
        if initial_prompt is not None
        else None
    )
    options = DecodingOptions(
        language="en",
        task="transcribe",
        temperature=0.0,
        beam_size=beam_size,
        best_of=None,
        without_timestamps=True,
        suppress_tokens="-1",
        suppress_blank=True,
        fp16=False,
        prompt=prompt_tokens,
    )
    task = DecodingTask(model, options)
    initial_tokens = task.initial_tokens
    eot = task.tokenizer.eot
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "implementation": "openai-whisper==20250625",
        "model": {
            "source": MODEL_URL,
            "revision": "content-addressed URL; no separate revision",
            "file": "large-v3.pt",
            "sha256": MODEL_SHA256,
            "equivalent_hf_checkpoint": {
                "source": "openai/whisper-large-v3",
                "revision": HF_REVISION,
                "file": "model.safetensors",
                "sha256": "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95",
            },
        },
        "audio": "16 kHz mono PCM16; samples divided by 32768; right-pad or trim to 480000 samples",
        "mel": "whisper.log_mel_spectrogram(padded_audio, n_mels=model.dims.n_mels)",
        "decode": {
            "call": "whisper.decode(model, mel, options)",
            "language": "en",
            "task": "transcribe",
            "temperature": 0.0,
            "beam_size": beam_size,
            "best_of": None,
            "without_timestamps": True,
            "suppress_tokens": "-1",
            "suppress_blank": True,
            "fp16": False,
            "sample_len": None,
            "prompt": prompt_tokens,
            "prefix": None,
        },
        "logits": "raw model.decoder outputs before token suppression; same initial prefix and oracle tokens",
        "fixtures": {},
    }
    if initial_prompt is not None:
        metadata["decode"]["initial_prompt"] = initial_prompt

    for name in FIXTURE_NAMES:
        wav_path = audio_dir / f"{name}.wav"
        samples = read_pcm16_mono(wav_path)
        padded = whisper.pad_or_trim(samples, length=whisper.audio.N_SAMPLES)
        mel = whisper.log_mel_spectrogram(padded, n_mels=model.dims.n_mels)
        with torch.inference_mode():
            result = whisper.decode(model, mel, options)
            prefix = initial_tokens + tuple(result.tokens)
            tokens = torch.tensor(prefix, dtype=torch.long).unsqueeze(0)
            raw_logits = model.decoder(tokens, result.audio_features.unsqueeze(0))
            step_logits = raw_logits[0, len(initial_tokens) - 1 :].numpy()
        oracle_path = output_dir / f"{name}.npz"
        data = {
            "mel": mel.numpy(),
            "encoder": result.audio_features.numpy(),
            "logits": step_logits,
            "initial_tokens": np.asarray(initial_tokens, dtype=np.int32),
            "beam_tokens" if beam_size is not None else "greedy_tokens": np.asarray(
                [*result.tokens, eot], dtype=np.int32
            ),
        }
        np.savez_compressed(oracle_path, **data)
        metadata["fixtures"][name] = {
            **FIXTURE_SOURCES[name],
            "audio_sha256": sha256(wav_path),
            "audio_samples": int(samples.size),
            "oracle_sha256": sha256(oracle_path),
            "text": result.text,
            "tokens_without_eot": result.tokens,
            "initial_tokens": list(initial_tokens),
            "eot": eot,
            "shapes": {
                "mel": list(mel.shape),
                "encoder": list(result.audio_features.shape),
                "logits": list(step_logits.shape),
            },
        }
        print(f"{name}: {result.text!r}, {len(result.tokens)} tokens")

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if initial_prompt is None and beam_size is None:
        _generate_text_references(model, audio_dir, output_dir, metadata)


def _generate_text_references(model, audio_dir: Path, output_dir: Path, metadata: dict) -> None:
    """Regenerate every prompted and beam reference with the same official fp32 model."""
    import torch
    import whisper
    from whisper.decoding import DecodingOptions, DecodingTask
    from whisper.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(multilingual=model.is_multilingual, language="en", task="transcribe")
    # Preserve the original tests' long unrelated hint and short neutral hint.
    prompts = {
        "long_hint": ", ".join(["Quetzalcoatl"] * 91) + ".",
        "one_word_hint": "the",
    }
    waveforms = {name: read_pcm16_mono(audio_dir / f"{name}.wav") for name in FIXTURE_NAMES}
    mels = {
        name: whisper.log_mel_spectrogram(whisper.pad_or_trim(samples), n_mels=model.dims.n_mels)
        for name, samples in waveforms.items()
    }

    def write_reference(name, data):
        (output_dir / name).write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def short_reference(name, prompt, beam_size):
        prompt_tokens = tokenizer.encode(" " + prompt.strip()) if prompt is not None else None
        options = DecodingOptions(
            language="en",
            task="transcribe",
            temperature=0.0,
            beam_size=beam_size,
            best_of=None,
            without_timestamps=True,
            suppress_tokens="-1",
            suppress_blank=True,
            fp16=False,
            prompt=prompt_tokens,
        )
        with torch.inference_mode():
            result = whisper.decode(model, mels[name], options)
        return {
            "text": result.text,
            "tokens_without_eot": result.tokens,
            "eot": tokenizer.eot,
            "audio_sha256": sha256(audio_dir / f"{name}.wav"),
            "initial_tokens": list(DecodingTask(model, options).initial_tokens),
        }

    prompt_reference = {
        "implementation": metadata["implementation"],
        "checkpoint_sha256": MODEL_SHA256,
        "decode": "whisper.decode, en/transcribe, temperature=0, beam=None, "
        "without_timestamps=True, fp16=False, prompt=transcribe-style initial_prompt tokens",
        "cases": {},
    }
    for label, prompt in prompts.items():
        for name in FIXTURE_NAMES:
            result = short_reference(name, prompt, None)
            prompt_reference["cases"][f"{label}/{name}"] = {
                "prompt": prompt,
                "text": result["text"],
                "tokens": [*result["tokens_without_eot"], result["eot"]],
                "audio_sha256": result["audio_sha256"],
            }
    write_reference("initial_prompt.json", prompt_reference)

    for filename, prompt in (("beam_size_2.json", None), ("beam_with_prompt.json", "the")):
        decode = dict(metadata["decode"], beam_size=2)
        if prompt is not None:
            decode.update(initial_prompt=prompt, prompt=tokenizer.encode(" " + prompt))
        write_reference(
            filename,
            {
                "implementation": metadata["implementation"],
                "checkpoint_sha256": MODEL_SHA256,
                "decode": decode,
                "fixtures": {name: short_reference(name, prompt, 2) for name in FIXTURE_NAMES},
            },
        )

    name = "small_boat_16khz_mono_s16le"
    samples = np.tile(waveforms[name], 3)
    for filename, prompt, beam_size in (
        ("initial_prompt_long_small_boat_repeat.json", prompts["long_hint"], None),
        ("beam_long_with_prompt.json", "the", 2),
    ):
        calls = []
        original_decode = model.decode

        def record_decode(mel, options, *, _decode=original_decode, _calls=calls):
            result = _decode(mel, options)
            _calls.append({"tokens": result.tokens, "prompt": list(options.prompt or [])})
            return result

        model.decode = record_decode
        try:
            with torch.inference_mode():
                result = model.transcribe(
                    samples,
                    language="en",
                    task="transcribe",
                    temperature=0.0,
                    beam_size=beam_size,
                    best_of=None,
                    initial_prompt=prompt,
                    fp16=False,
                    verbose=None,
                )
        finally:
            del model.decode
        write_reference(
            filename,
            {
                "implementation": metadata["implementation"],
                "checkpoint_sha256": MODEL_SHA256,
                "audio": f"three consecutive copies of tests/fixtures/{name}.wav",
                "source_audio_sha256": sha256(audio_dir / f"{name}.wav"),
                "audio_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
                "audio_hash_format": "normalized float32 waveform, three copies, little-endian",
                "audio_samples": int(samples.size),
                "prompt": prompt,
                "beam_size": beam_size,
                "temperature": 0.0,
                "text": result["text"],
                "segments": [
                    {key: segment[key] for key in ("start", "end", "text")}
                    for segment in result["segments"]
                ],
                "decode_calls": calls,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--output-dir", type=Path, default=Path("tests/fixtures/oracle"))
    parser.add_argument("--initial-prompt")
    parser.add_argument("--beam-size", type=int)
    args = parser.parse_args()
    if args.beam_size is not None and args.beam_size < 1:
        parser.error("--beam-size must be positive")
    kwargs = {"initial_prompt": args.initial_prompt}
    if args.beam_size is not None:
        kwargs["beam_size"] = args.beam_size
    generate(args.checkpoint, args.audio_dir, args.output_dir, **kwargs)


if __name__ == "__main__":
    main()
