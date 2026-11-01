"""CPU checks for validation-only multilingual FP8 calibration and clipping evidence."""

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from scripts import calibrate_fp8 as calibration
from scripts.evaluate_fleurs import CONFIGS


def _wav(seconds, *, rate=16000, amplitude=0.125):
    stream = io.BytesIO()
    sf.write(
        stream,
        np.full(round(seconds * rate), amplitude, dtype=np.float32),
        rate,
        format="WAV",
        subtype="PCM_16",
    )
    return stream.getvalue()


def _validation(root, *, durations=(0.002, 0.004), overrides=None):
    payloads = {seconds: _wav(seconds) for seconds in durations}
    for language, config in CONFIGS.items():
        rows = [
            {
                "id": 17,  # Sentence ids may repeat; recording filenames are distinct.
                "path": f"/original/{index:02}.wav",
                "num_samples": 1,  # Selection must use the embedded WAV's real duration.
                "transcription": f"reference {language} {index}",
                "audio": {"path": f"{index:02}.wav", "bytes": payloads[seconds]},
            }
            for index, seconds in enumerate(durations)
        ]
        if overrides is not None:
            rows = overrides(language, rows)
        directory = root / config
        directory.mkdir(parents=True)
        # More than one validation shard must be eligible.
        for index, subset in enumerate((rows[:1], rows[1:])):
            if subset:
                pq.write_table(
                    pa.Table.from_pylist(subset),
                    directory / f"validation-{index:05}-of-00002.parquet",
                )
        # Reading this sentinel as parquet would fail immediately.
        (directory / "test-00000-of-00001.parquet").write_bytes(b"never read evaluation speech")


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_validation_selection_balances_languages_and_keeps_even_duration_ranks(tmp_path):
    """Nine languages contribute equal counts per band, capped by the sparsest language."""
    durations = (1, 2, 3, 4, 6, 8, 10, 19, 20, 30, 31)
    _validation(
        tmp_path,
        durations=durations,
        overrides=lambda language, rows: rows[:-2] if language == "th" else rows,
    )
    selected, available = calibration.select_fleurs_audio(tmp_path, per_band=3)
    assert set(available) == set(CONFIGS)
    for language, config in CONFIGS.items():
        rows = [row for row in selected if row["language"] == language]
        assert [row["seconds"] for row in rows] == [1, 2, 4, 6, 8, 10, 19, 20]
        assert available[language] == {
            "0-5": 4,
            "5-10": 2,
            "10-20": 2,
            "20-30": 1 if language == "th" else 2,
        }
        assert all(row["dataset"] == "fleurs" and row["split"] == "validation" for row in rows)
        assert len({row["id"] for row in rows}) == 8
        for row in rows:
            assert row["id"].startswith(config + "/")
            assert row["frames"] == row["seconds"] * 16000
            assert row["declared_samples"] == 1
            assert row["sha256"] == hashlib.sha256(_wav(row["seconds"])).hexdigest()
    # Selection and metadata are reproducible across calls.
    assert calibration.select_fleurs_audio(tmp_path, per_band=3) == (selected, available)


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_empty_duration_band_is_omitted_for_every_language_and_is_counted(tmp_path):
    """A missing long-speech band cannot duplicate shorter recordings to fill a quota."""
    _validation(tmp_path)
    rows, available = calibration.select_fleurs_audio(tmp_path, per_band=8)
    assert len(rows) == 18
    assert all(
        counts == {"0-5": 2, "5-10": 0, "10-20": 0, "20-30": 0} for counts in available.values()
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
@pytest.mark.parametrize("symlink", [False, True])
def test_test_split_is_rejected_when_passed_directly_or_as_a_validation_symlink(tmp_path, symlink):
    """Evaluation speech cannot become calibration speech through a path argument or alias."""
    test = tmp_path / "test-00000-of-00001.parquet"
    test.write_bytes(b"not opened")
    if symlink:
        directory = tmp_path / "en_us"
        directory.mkdir()
        (directory / "validation-00000-of-00001.parquet").symlink_to(test)
    with pytest.raises(ValueError, match="validation|test"):
        calibration.select_fleurs_audio(tmp_path if symlink else test, per_band=2)


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
@pytest.mark.parametrize("defect", ["missing_language", "duplicate", "identity", "sample_rate"])
def test_invalid_validation_inputs_stop_before_model_loading(tmp_path, defect):
    """Missing languages, repeated recordings and invalid audio never silently enter calibration."""

    def override(language, rows):
        if language == "en":
            if defect == "duplicate":
                rows[1] = rows[0].copy()
            elif defect == "identity":
                rows[0]["audio"]["path"] = "different.wav"
            elif defect == "sample_rate":
                rows[0]["audio"]["bytes"] = _wav(0.002, rate=8000)
        return rows

    _validation(tmp_path, overrides=override)
    if defect == "missing_language":
        for path in (tmp_path / CONFIGS["th"]).glob("validation-*.parquet"):
            path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        calibration.select_fleurs_audio(tmp_path, per_band=2)


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.req("SP-LFFASTESTWHISPER-040")
@pytest.mark.parametrize("language", CONFIGS)
def test_teacher_forcing_uses_each_recordings_language_and_reference(language):
    """Decoder input is the official language prefix, reference tokens and EOT."""
    from whisper.tokenizer import get_tokenizer

    reference = "  日本語 हिंदी ภาษาไทย  "
    tokenizer = get_tokenizer(
        multilingual=True, num_languages=100, language=language, task="transcribe"
    )
    actual = calibration.teacher_forcing_tokens(
        reference, language, num_languages=100, text_ctx=448
    )
    assert actual == [
        *tokenizer.sot_sequence_including_notimestamps,
        *tokenizer.encode(" " + reference.strip()),
        tokenizer.eot,
    ]


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_scale_comparison_reports_all_layers_and_ranks_clipping_per_language():
    """Ratios use the existing scale times 448, with strict clipping above one."""
    report = calibration.compare_scales(
        {"hi": {"a": 896, "b": 224, "c": 0}, "th": {"a": 448, "b": 112, "c": 0}},
        {"a": 1, "b": 0.25, "c": 1},
    )
    assert report["hi"]["clipped_layers"] == 2
    assert report["th"]["clipped_layers"] == 0
    assert report["hi"]["layers"]["a"] == {
        "input_amax": 896,
        "scale": 1,
        "limit": 448,
        "ratio": 2,
    }
    assert [row["layer"] for row in report["hi"]["top_layers"]] == ["a", "b", "c"]
    assert report["hi"]["layers"]["c"]["ratio"] == 0
    for invalid in ({"a": 0}, {"a": float("nan")}, {"b": 1}):
        with pytest.raises(ValueError):
            calibration.compare_scales({"hi": {"a": 1}}, invalid)


def _librispeech(root):
    for split in ("dev-clean", "dev-other"):
        directory = root / split
        directory.mkdir(parents=True)
        references = []
        for index, seconds in enumerate((1, 2, 6, 7, 11, 12, 21, 22)):
            key = f"{split}-{index}"
            references.append(f"{key} reference en {index}")
            sf.write(directory / f"{key}.flac", np.zeros(seconds * 16000), 16000, subtype="PCM_16")
        (directory / "references.trans.txt").write_text("\n".join(references))


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.req("SP-LFFASTESTWHISPER-003")
@pytest.mark.req("SP-LFFASTESTWHISPER-016")
@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_cpu_run_combines_dev_and_validation_maxima_and_records_512_scales(tmp_path, monkeypatch):
    """The real collector aggregates all inputs while retaining FLEURS-only per-language evidence."""
    import torch
    import whisper
    from whisper.tokenizer import get_tokenizer

    assert Path(calibration.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])
    fleurs, libri = tmp_path / "fleurs", tmp_path / "libri"
    _validation(fleurs)
    _librispeech(libri)
    libri_selection, _ = calibration.select_audio(libri, per_band=2)
    modules = {}
    for part in ("encoder", "decoder"):
        for index in range(32):
            names = [f"attn.{name}" for name in ("query", "key", "value", "out")]
            if part == "decoder":
                names += [f"cross_attn.{name}" for name in ("query", "key", "value", "out")]
            names += ["mlp.0", "mlp.2"]
            for name in names:
                modules[f"{part}.blocks.{index}.{name}"] = torch.nn.Linear(1, 1, bias=False)

    calls = []

    class OfficialModel:
        num_languages = 100
        dims = SimpleNamespace(n_text_ctx=448)

        def float(self):
            return self

        def eval(self):
            return self

        def named_modules(self):
            return modules.items()

        def encoder(self, mel):
            # Dev speech is deliberately larger than FLEURS, proving global aggregation.
            value = 4.0 if not mel.any() else 2.0
            tensor = torch.tensor([[value]])
            for name, module in modules.items():
                if name.startswith("encoder."):
                    module(tensor)
            return tensor

        def decoder(self, tokens, encoded):
            calls.append(tokens.tolist()[0])
            for name, module in modules.items():
                if name.startswith("decoder."):
                    module(encoded * 2)

    monkeypatch.setattr(whisper, "load_model", lambda path, *, device: OfficialModel())
    monkeypatch.setattr(whisper, "pad_or_trim", lambda waveform: waveform)
    monkeypatch.setattr(
        whisper, "log_mel_spectrogram", lambda waveform, **kw: torch.from_numpy(waveform)
    )
    checkpoint = tmp_path / "large-v3.pt"
    checkpoint.write_bytes(b"unread: test model is supplied at the external boundary")
    monkeypatch.setattr("scripts.generate_oracle.MODEL_SHA256", calibration.digest(checkpoint))
    baseline = tmp_path / "old-scales.json"
    baseline.write_text(json.dumps({calibration.graph_key(name): 1 / 448 for name in modules}))
    before = baseline.read_bytes()
    output = tmp_path / "calibration.json"
    calibration.run(
        checkpoint,
        libri,
        output,
        fleurs_dir=fleurs,
        baseline_scales=baseline,
        per_band=2,
        threads=1,
    )
    record = json.loads(output.read_text())
    assert record["completed"] == 34
    assert len(record["scales"]) == 512
    assert record["selected"][:16] == libri_selection
    assert set(record["input_amax_by_language"]) == set(CONFIGS)
    for language in CONFIGS:
        observed = record["input_amax_by_language"][language]
        assert len(observed) == 512
        assert set(observed.values()) == {2, 4}
        assert record["scale_comparison_by_language"][language]["clipped_layers"] == 512
        assert record["selected_fleurs_by_band"][language] == {
            "0-5": 2,
            "5-10": 0,
            "10-20": 0,
            "20-30": 0,
        }
    assert set(record["input_amax"].values()) == {4, 8}
    assert set(record["scales"].values()) == {4 / 448, 8 / 448}
    assert len(calls) == record["completed"]
    for row, tokens in zip(record["selected"], calls, strict=True):
        language = row.get("language", "en")
        tokenizer = get_tokenizer(
            multilingual=True, num_languages=100, language=language, task="transcribe"
        )
        assert tokens == [
            *tokenizer.sot_sequence_including_notimestamps,
            *tokenizer.encode(" " + row["reference"].strip()),
            tokenizer.eot,
        ]
        assert row["split"] in ("dev-clean", "dev-other", "validation")
    assert baseline.read_bytes() == before
    assert not output.with_suffix(".partial.json").exists()
    assert not any(module._forward_pre_hooks for module in modules.values())
