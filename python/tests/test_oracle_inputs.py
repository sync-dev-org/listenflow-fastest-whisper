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


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_multilingual_oracle_cli_keeps_existing_references_separate(tmp_path, monkeypatch):
    """Multilingual generation selects its own path without regenerating the English references."""
    seen = []

    def generate(checkpoint, audio_dir, output_dir):
        seen.append((checkpoint, audio_dir, output_dir))

    monkeypatch.setattr(generate_oracle, "generate_multilingual", generate, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_oracle", "--checkpoint", str(tmp_path / "large-v3.pt"), "--multilingual"],
    )

    generate_oracle.main()

    assert seen == [
        (tmp_path / "large-v3.pt", Path("tests/fixtures"), Path("tests/fixtures/oracle"))
    ]


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_multilingual_references_identify_audio_languages_and_official_results():
    """Each representative language has exact fp32 arrays, token results, and detected-language data."""
    from whisper.tokenizer import LANGUAGES, get_tokenizer

    fixtures = Path(__file__).parent / "fixtures"
    root = fixtures / "oracle/multilingual"
    metadata = json.loads((root / "metadata.json").read_text())
    assert not (root / "en.npz").exists()
    assert sum(path.stat().st_size for path in root.iterdir() if path.is_file()) <= 20_000_000
    assert set(metadata["fixtures"]) == {"en", "ja", "zh", "yue", "ko", "ru", "ar", "hi", "th"}
    assert metadata["implementation"] == "openai-whisper==20250625"
    assert metadata["checkpoint_sha256"] == generate_oracle.MODEL_SHA256
    assert metadata["decode"]["fp16"] is False
    assert metadata["decode"]["task"] == "transcribe"
    assert metadata["decode"]["without_timestamps"] is True
    for language, expected in metadata["fixtures"].items():
        audio = fixtures / f"{expected['name']}.wav"
        reference = root / expected["oracle_file"]
        samples = read_pcm16_mono(audio)
        tokenizer = get_tokenizer(multilingual=True, num_languages=100, language=language)
        assert 0 < samples.size < 480_000
        assert hashlib.sha256(audio.read_bytes()).hexdigest() == expected["audio_sha256"]
        assert samples.size == expected["audio_samples"]
        assert hashlib.sha256(reference.read_bytes()).hexdigest() == expected["oracle_sha256"]
        assert expected["detected_language"] in LANGUAGES
        assert expected["text"].strip()
        assert expected["auto_text"].strip()
        assert tokenizer.decode(expected["tokens_without_eot"]).strip() == expected["text"]
        with np.load(reference) as arrays:
            assert arrays["initial_tokens"].tolist() == list(
                tokenizer.sot_sequence_including_notimestamps
            )
            assert arrays["greedy_tokens"].tolist() == [
                *expected["tokens_without_eot"],
                tokenizer.eot,
            ]
            for field, shape in expected["shapes"].items():
                assert list(arrays[field].shape) == shape
            if language == "en":
                assert reference.resolve() == (root.parent / f"{expected['name']}.npz").resolve()
            else:
                assert set(arrays.files) == {
                    "initial_tokens",
                    "greedy_tokens",
                    "logit_token_ids",
                    "logit_values",
                }
                assert arrays["logit_values"].dtype == np.float32
                assert arrays["logit_token_ids"].shape == arrays["logit_values"].shape
                assert arrays["logit_values"].shape == (len(expected["tokens_without_eot"]) + 1, 32)
        if language not in ("en", "ja"):
            source = generate_oracle.FIXTURE_SOURCES[expected["name"]]
            for field in (
                "config",
                "split",
                "utterance_id",
                "original_file",
                "dataset_revision",
                "parquet_file",
                "reference_field",
                "author",
                "license",
                "license_url",
                "source_url",
                "change",
                "record",
            ):
                assert source[field] == expected[field]
            assert source["license"] == "CC-BY-4.0"
            assert source["reference_field"] == "transcription"
            assert (fixtures.parents[1] / source["record"]).is_file()


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_multilingual_conversion_keeps_top_logits_tokens_and_detection_without_a_model(tmp_path):
    """Array-only conversion preserves exact tokens and strong fp32 logits while reusing English."""
    root = tmp_path / "multilingual"
    root.mkdir()
    initial = np.array([1, 2, 3, 4], dtype=np.int32)
    greedy = np.array([8, 5], dtype=np.int32)
    logits = np.arange(80, dtype=np.float32).reshape(2, 40)
    data = {
        "mel": np.zeros((2, 3), dtype=np.float32),
        "encoder": np.ones((3, 4), dtype=np.float32),
        "logits": logits,
        "initial_tokens": initial,
        "greedy_tokens": greedy,
    }
    legacy = tmp_path / "small_boat_16khz_mono_s16le.npz"
    np.savez_compressed(legacy, **data)
    original = legacy.read_bytes()
    cases = {}
    for language, name in (("en", "small_boat_16khz_mono_s16le"), ("ja", "japanese")):
        np.savez_compressed(root / f"{language}.npz", **data)
        cases[language] = {
            "name": name,
            "initial_tokens": initial.tolist(),
            "tokens_without_eot": [8],
            "eot": 5,
            "text": "example",
            "detected_language": language,
            "auto_text": "automatic",
            "auto_tokens_without_eot": [9],
        }
    (root / "metadata.json").write_text(json.dumps({"fixtures": cases}))

    generate_oracle.reduce_multilingual_references(tmp_path)

    metadata = json.loads((root / "metadata.json").read_text())
    assert legacy.read_bytes() == original
    assert not (root / "en.npz").exists()
    assert metadata["fixtures"]["en"]["oracle_file"] == "../small_boat_16khz_mono_s16le.npz"
    case = metadata["fixtures"]["ja"]
    for field in (
        "initial_tokens",
        "tokens_without_eot",
        "eot",
        "text",
        "detected_language",
        "auto_text",
        "auto_tokens_without_eot",
    ):
        assert case[field] == cases["ja"][field]
    with np.load(root / "ja.npz") as arrays:
        assert set(arrays.files) == {
            "initial_tokens",
            "greedy_tokens",
            "logit_token_ids",
            "logit_values",
        }
        np.testing.assert_array_equal(arrays["initial_tokens"], initial)
        np.testing.assert_array_equal(arrays["greedy_tokens"], greedy)
        expected_ids = np.tile(np.arange(39, 7, -1, dtype=np.int32), (2, 1))
        np.testing.assert_array_equal(arrays["logit_token_ids"], expected_ids)
        np.testing.assert_array_equal(
            arrays["logit_values"], np.take_along_axis(logits, expected_ids, axis=1)
        )
        assert arrays["logit_values"].dtype == np.float32
