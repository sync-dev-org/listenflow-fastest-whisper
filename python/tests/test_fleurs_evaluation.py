"""CPU checks for reproducible FLEURS inputs, scoring, and interrupted measurements."""

import csv
import io
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from scripts import evaluate_fleurs as evaluation


def _shard(tmp_path, *, config="en_us", rate=16000, declared_samples=None, samples=None):
    records = []
    for index, filename in enumerate(("first.wav", "second.wav")):
        audio = io.BytesIO()
        waveform = (
            np.array([index / 4, -0.5, 0.5], dtype=np.float32)
            if samples is None
            else np.asarray(samples, dtype=np.float32)
        )
        sf.write(
            audio,
            waveform,
            rate,
            format="WAV",
            subtype="FLOAT",
        )
        records.append(
            {
                "id": 17,
                "path": f"/original/{filename}",
                "num_samples": len(waveform) if declared_samples is None else declared_samples,
                "transcription": "cat dog" if index == 0 else "bird",
                "audio": {"path": filename, "bytes": audio.getvalue()},
            }
        )
    target = tmp_path / config / "test-00000-of-00001.parquet"
    target.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(records), target)
    return records


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
@pytest.mark.parametrize("language", ["zh", "yue", "ja", "ko", "th"])
def test_character_languages_remove_normalized_whitespace_before_scoring(language):
    """Character languages count missing characters without counting spaces as errors."""
    assert evaluation.error_unit(language) == "CER"
    measured = evaluation.score_text("甲 乙 丙 丁", "甲乙丙", language)
    assert measured["reference_units"] == 4
    assert measured["errors"] == 1
    assert measured["normalized_reference"] == "甲乙丙丁"


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
@pytest.mark.parametrize("language", ["en", "ru", "ar", "hi"])
def test_word_languages_aggregate_edits_on_normalized_words(language):
    """Word languages count one missing normalized word rather than its characters."""
    assert evaluation.error_unit(language) == "WER"
    measured = evaluation.score_text("Cat, DOG!", "cat", language)
    assert measured["reference_units"] == 2
    assert measured["errors"] == measured["deletions"] == 1


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_normalization_uses_the_official_english_and_multilingual_rules():
    """English number normalization and the multilingual basic cleaner match their official oracles."""
    from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

    text = "I paid $20 [noise]"
    assert evaluation.normalize_text(text, "en") == EnglishTextNormalizer()(text)
    assert evaluation.normalize_text(text, "ru") == BasicTextNormalizer()(text)
    assert evaluation.normalize_text(text, "en") != evaluation.normalize_text(text, "ru")
    with pytest.raises(ValueError, match="language"):
        evaluation.error_unit("xx")


def _result(utterance_id, reference, output, temperatures=(0.0,)):
    return {
        "id": utterance_id,
        "reference": reference,
        "output": output,
        "temperatures": json.dumps(temperatures),
    }


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_scoring_weights_reference_units_and_applies_only_the_fp16_gate():
    """The fp16 gate accepts exactly 0.05 point on total reference units; fp8 is measured only."""
    rows = {
        "fp32": [_result("short", "cat dog", "cat"), _result("long", "cat " * 1998, "cat " * 1998)],
        "fp16": [_result("short", "cat dog", ""), _result("long", "cat " * 1998, "cat " * 1998)],
        "fp8": [_result("short", "cat dog", "cat dog"), _result("long", "cat " * 1998, "")],
    }
    actual = evaluation.summarize_language("en", rows)
    assert actual["reference_units"] == 2000
    assert actual["precisions"]["fp32"]["errors"] == 1
    assert actual["precisions"]["fp32"]["rate_percent"] == pytest.approx(0.05)
    assert actual["precisions"]["fp16"]["difference_points"] == pytest.approx(0.05)
    assert actual["fp16_gate"] is True
    assert actual["precisions"]["fp8"]["errors"] == 1998
    assert "passed" not in actual["precisions"]["fp8"]
    failed = {
        precision: [
            _result("one", "word " * 1000, "word " * (999 if precision == "fp16" else 1000))
        ]
        for precision in rows
    }
    assert evaluation.summarize_language("ru", failed)["fp16_gate"] is False


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_top_regressions_retain_reference_outputs_and_skipped_fallbacks():
    """The five largest positive error increases retain both outputs and the actual retry evidence."""
    rows = {precision: [] for precision in ("fp32", "fp16", "fp8")}
    for index in range(7):
        reference = "cat " * 10
        for precision, records in rows.items():
            records.append(
                _result(
                    str(index),
                    reference,
                    "cat " * (10 if precision == "fp32" else 10 - index),
                    (0.0, 0.2) if precision == "fp32" else (0.0,),
                )
            )
    actual = evaluation.summarize_language("en", rows)
    regressions = actual["top_regressions"]["fp8"]
    assert [row["id"] for row in regressions] == ["6", "5", "4", "3", "2"]
    assert regressions[0]["error_increase"] == 6
    assert regressions[0]["reference"] == rows["fp32"][6]["reference"]
    assert regressions[0]["fp32_output"] == rows["fp32"][6]["output"]
    assert regressions[0]["output"] == rows["fp8"][6]["output"]
    assert regressions[0]["skipped_fp32_fallback"] is True


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_regression_identifies_a_later_fallback_omitted_by_fp8():
    """A shortened fp8 retry sequence identifies the specific later temperature taken by fp32."""
    rows = {
        "fp32": [_result("one", "cat dog", "cat dog", (0.0, 0.2, 0.4))],
        "fp16": [_result("one", "cat dog", "cat dog", (0.0, 0.2, 0.4))],
        "fp8": [_result("one", "cat dog", "cat", (0.0, 0.2))],
    }
    row = evaluation.summarize_language("en", rows)["top_regressions"]["fp8"][0]
    assert row["skipped_fp32_fallback"] is True
    assert row["skipped_fp32_temperatures"] == [0.4]


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_report_keeps_completed_baseline_rates_and_lists_unfinished_pairs(tmp_path):
    """A partial report retains a completed fp32 rate without claiming a missing precision's gate."""
    rows = [_result("one", "cat dog", "cat")]
    path = tmp_path / "en-fp32.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "status": "complete",
        "expected_utterances": 1,
        "completed_utterances": 1,
        "csv_sha256": evaluation._sha256(path),
        "provenance": {"language": "en", "precision": "fp32"},
    }
    (tmp_path / "en-fp32.json").write_text(json.dumps(metadata))
    result = evaluation.build_report(tmp_path)
    assert result["languages"]["en"]["reference_units"] == 2
    assert result["languages"]["en"]["precisions"]["fp32"]["rate_percent"] == 50
    assert result["languages"]["en"]["precisions"]["fp16"] is None
    assert result["languages"]["en"]["fp16_gate"] is None
    assert len(result["remaining"]) == 26
    assert {
        "language": "en",
        "precision": "fp16",
        "reason": "not run",
        "completed_utterances": 0,
    } in result["remaining"]


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
@pytest.mark.parametrize("defect", ["missing", "duplicate", "reference", "count"])
def test_scoring_rejects_unpaired_or_inconsistent_records(defect):
    """A comparison cannot hide missing recordings, duplicate ids, changed references, or false counts."""
    rows = {
        precision: [_result("a", "cat", "cat"), _result("b", "dog", "dog")]
        for precision in ("fp32", "fp16", "fp8")
    }
    if defect == "missing":
        rows["fp16"].pop()
    elif defect == "duplicate":
        rows["fp16"].append(rows["fp16"][0])
    elif defect == "reference":
        rows["fp16"][0]["reference"] = "different"
    else:
        rows["fp16"][0]["errors"] = 1
    with pytest.raises(ValueError):
        evaluation.summarize_language("en", rows)


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_reader_preserves_float_audio_references_and_recording_identity(tmp_path):
    """Each original recording remains distinct even when FLEURS sentence ids repeat."""
    _shard(tmp_path)
    utterances = list(evaluation.read_fleurs(tmp_path, "en"))
    assert [row["id"] for row in utterances] == ["en_us/first.wav", "en_us/second.wav"]
    assert [row["fleurs_id"] for row in utterances] == [17, 17]
    assert [row["reference"] for row in utterances] == ["cat dog", "bird"]
    assert utterances[0]["duration"] == 3 / 16000
    assert utterances[0]["audio"].dtype == np.float32
    np.testing.assert_array_equal(utterances[0]["audio"], [0, -0.5, 0.5])
    remaining = list(evaluation.read_fleurs(tmp_path, "en", skip_ids={utterances[0]["id"]}))
    assert [row["id"] for row in remaining] == ["en_us/second.wav"]


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_reader_uses_decoded_audio_and_records_a_declared_length_discrepancy(tmp_path):
    """A valid WAV remains in the split with both lengths when its metadata declares more samples."""
    _shard(tmp_path, declared_samples=5)
    utterances = list(evaluation.read_fleurs(tmp_path, "en"))
    assert [row["id"] for row in utterances] == ["en_us/first.wav", "en_us/second.wav"]
    assert [(row["declared_samples"], row["decoded_samples"]) for row in utterances] == [
        (5, 3),
        (5, 3),
    ]
    assert [row["duration"] for row in utterances] == [3 / 16000, 3 / 16000]
    np.testing.assert_array_equal(utterances[0]["audio"], [0, -0.5, 0.5])


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_reader_rejects_an_incorrect_audio_sample_rate(tmp_path):
    """The reader refuses to reinterpret a recording at the wrong sample rate."""
    _shard(tmp_path, rate=8000)
    with pytest.raises(ValueError, match="16 kHz"):
        list(evaluation.read_fleurs(tmp_path, "en"))


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
@pytest.mark.parametrize(
    "samples",
    [np.zeros((3, 2), dtype=np.float32), np.array([0, np.nan, 1], dtype=np.float32)],
    ids=["stereo", "non-finite"],
)
def test_reader_rejects_multichannel_or_nonfinite_audio(tmp_path, samples):
    """Relaxing declared lengths still rejects stereo audio and non-finite samples."""
    _shard(tmp_path, samples=samples)
    with pytest.raises(ValueError, match="finite mono 16 kHz"):
        list(evaluation.read_fleurs(tmp_path, "en"))


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_interrupted_run_resumes_without_decoding_completed_utterances(tmp_path, monkeypatch):
    """A resumed run retains declared and decoded audio lengths and skips completed recordings."""
    _shard(tmp_path / "data", declared_samples=5)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"synthetic checkpoint identity")
    output = tmp_path / "results"
    stop = tmp_path / "stop"
    calls = []

    class Runtime:
        def __init__(self):
            self.build_seconds = {}

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

        def _decode_attempt(self, *args, **kwargs):
            return SimpleNamespace(
                temperature=kwargs["temperature"],
                avg_logprob=-0.1,
                no_speech_prob=0.0,
                compression_ratio=1.0,
            )

        def transcribe(self, audio, **kwargs):
            assert kwargs == {"language": "en"}
            calls.append(float(audio[0]))
            self._decode_attempt(temperature=0.0)
            if len(calls) == 1:
                stop.touch()
            return [SimpleNamespace(text="cat dog" if audio[0] == 0 else "bird")]

    monkeypatch.setattr(evaluation, "_open_model", lambda *args, **kwargs: Runtime())
    arguments = (tmp_path / "data", checkpoint, output, "en", "fp32")
    assert evaluation.run_evaluation(*arguments, stop_file=stop) == 75
    first = json.loads((output / "en-fp32.json").read_text())
    assert first["status"] == "stopped" and first["completed_utterances"] == 1
    stop.unlink()
    assert evaluation.run_evaluation(*arguments, stop_file=stop) == 0
    assert calls == [0, 0.25]
    complete = json.loads((output / "en-fp32.json").read_text())
    assert complete["status"] == "complete" and complete["completed_utterances"] == 2
    assert len(complete["sessions"]) == 2
    with (output / "en-fp32.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and all(int(row["errors"]) == 0 for row in rows)
    assert [(int(row["declared_samples"]), int(row["decoded_samples"])) for row in rows] == [
        (5, 3),
        (5, 3),
    ]
    assert evaluation.run_evaluation(*arguments, stop_file=stop) == 0
    assert calls == [0, 0.25]
