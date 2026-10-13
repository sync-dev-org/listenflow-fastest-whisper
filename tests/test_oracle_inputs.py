import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from scripts import generate_oracle
from scripts.generate_oracle import read_pcm16_mono


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_checked_in_references_identify_the_exact_synthetic_and_silent_inputs():
    """Reference hashes and tokens identify the actual audio and arrays distributed with the tests."""
    fixtures = Path(__file__).parent / "fixtures"
    metadata = json.loads((fixtures / "oracle" / "metadata.json").read_text())
    assert set(metadata["fixtures"]) == {
        "small_boat_16khz_mono_s16le",
        "silence_16khz_mono_s16le",
    }
    for name, expected in metadata["fixtures"].items():
        audio = fixtures / f"{name}.wav"
        reference = fixtures / "oracle" / f"{name}.npz"
        assert hashlib.sha256(audio.read_bytes()).hexdigest() == expected["audio_sha256"]
        assert read_pcm16_mono(audio).size == expected["audio_samples"]
        assert hashlib.sha256(reference.read_bytes()).hexdigest() == expected["oracle_sha256"]
        with np.load(reference) as arrays:
            assert arrays["initial_tokens"].tolist() == expected["initial_tokens"]
            assert arrays["greedy_tokens"].tolist() == [
                *expected["tokens_without_eot"],
                expected["eot"],
            ]
            for field, shape in expected["shapes"].items():
                assert list(arrays[field].shape) == shape


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_oracle_reads_16khz_mono_pcm16_as_normalized_fp32(tmp_path):
    """The reference calculation receives the recorded waveform without resampling."""
    fixture = tmp_path / "three_samples.wav"
    with wave.open(str(fixture), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(np.array([-32768, 0, 16384], dtype="<i2").tobytes())

    audio = read_pcm16_mono(fixture)

    assert audio.dtype == np.float32
    np.testing.assert_array_equal(audio, np.array([-1.0, 0.0, 0.5], dtype=np.float32))


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_oracle_rejects_audio_with_the_wrong_sample_rate(tmp_path):
    """The oracle does not silently reinterpret audio recorded at another rate."""
    fixture = tmp_path / "wrong_rate.wav"
    with wave.open(str(fixture), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8_000)
        wav.writeframes(np.array([0], dtype="<i2").tobytes())

    with pytest.raises(ValueError, match="16 kHz mono PCM16"):
        read_pcm16_mono(fixture)


@pytest.mark.req("SP-LFFASTESTWHISPER-009")
def test_short_oracle_cli_accepts_an_initial_prompt(tmp_path, monkeypatch):
    """The short official reference can be regenerated with the same hint as the GPU route."""
    seen = []

    def generate(checkpoint, audio_dir, output_dir, *, initial_prompt=None):
        seen.append((checkpoint, audio_dir, output_dir, initial_prompt))

    monkeypatch.setattr(generate_oracle, "generate", generate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_oracle",
            "--checkpoint",
            str(tmp_path / "large-v3.pt"),
            "--audio-dir",
            str(tmp_path / "audio"),
            "--output-dir",
            str(tmp_path / "output"),
            "--initial-prompt",
            "the",
        ],
    )

    generate_oracle.main()

    assert seen == [(tmp_path / "large-v3.pt", tmp_path / "audio", tmp_path / "output", "the")]


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_short_oracle_cli_accepts_a_beam_width(tmp_path, monkeypatch):
    """The official short reference can be regenerated with the selected beam width."""
    seen = []

    def generate(checkpoint, audio_dir, output_dir, *, initial_prompt=None, beam_size=None):
        seen.append((checkpoint, audio_dir, output_dir, beam_size))

    monkeypatch.setattr(generate_oracle, "generate", generate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_oracle",
            "--checkpoint",
            str(tmp_path / "large-v3.pt"),
            "--audio-dir",
            str(tmp_path / "audio"),
            "--output-dir",
            str(tmp_path / "output"),
            "--beam-size",
            "2",
        ],
    )

    generate_oracle.main()

    assert seen == [(tmp_path / "large-v3.pt", tmp_path / "audio", tmp_path / "output", 2)]
