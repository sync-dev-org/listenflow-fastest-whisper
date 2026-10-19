"""CPU-only acceptance checks for the paired LibriSpeech evaluation."""

import json

import pytest
from scripts.evaluate_wer import (
    PairedResult,
    included_utterances,
    read_csv,
    report,
    score_subsets,
    write_csv,
)


@pytest.fixture
def report_inputs(tmp_path):
    manifest = [
        {"subset": "test-clean", "id": "clean", "reference": "HELLO WORLD", "frames": 1},
        {"subset": "test-other", "id": "other", "reference": "GOOD MORNING", "frames": 1},
    ]
    write_csv(tmp_path / "manifest.csv", manifest, list(manifest[0]))
    for precision in ("fp16", "fp32"):
        rows = [
            {
                "subset": row["subset"],
                "id": row["id"],
                "reference": row["reference"],
                "output": "hello there" if precision == "fp16" and index == 0 else row["reference"],
                "reference_words": 2,
                "errors": int(precision == "fp16" and index == 0),
            }
            for index, row in enumerate(manifest)
        ]
        write_csv(tmp_path / f"{precision}.csv", rows, list(rows[0]))
        (tmp_path / f"{precision}.json").write_text("{}", encoding="utf-8")
    return tmp_path


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_gate_uses_word_weighted_wer_and_requires_both_subsets():
    """Each subset passes only when fp16 WER is at most fp32 WER plus 0.05 point."""
    rows = [
        PairedResult("test-clean", "clean-1", 2, 0, 1, "same", "same"),
        PairedResult("test-clean", "clean-2", 1998, 101, 100, "sixteen", "thirtytwo"),
        PairedResult("test-other", "other-1", 2000, 102, 100, "sixteen", "thirtytwo"),
    ]

    result = score_subsets(rows)

    assert result["test-clean"].fp16_wer == pytest.approx(101 / 2000)
    assert result["test-clean"].fp32_wer == pytest.approx(101 / 2000)
    assert result["test-clean"].changed_outputs == 1
    assert result["test-clean"].passed
    assert result["test-other"].difference_points == pytest.approx(0.1)
    assert not result["test-other"].passed
    assert not all(subset.passed for subset in result.values())


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_gate_accepts_exact_tolerance_and_excludes_overlong_utterances():
    """A 0.05-point gap is allowed, and audio beyond 30 seconds is listed by ID."""
    rows = [
        ("test-clean", "within", 480_000),
        ("test-clean", "over", 480_001),
        ("test-other", "other", 16_000),
    ]

    included, excluded = included_utterances(rows)
    result = score_subsets(
        [
            PairedResult("test-clean", "within", 2000, 101, 100, "a", "b"),
            PairedResult("test-other", "other", 2000, 100, 100, "a", "a"),
        ]
    )

    assert [row[1] for row in included] == ["within", "other"]
    assert excluded == {"test-clean": ["over"], "test-other": []}
    assert result["test-clean"].difference_points == pytest.approx(0.05)
    assert all(subset.passed for subset in result.values())


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_report_rejects_outputs_with_a_different_reference_from_manifest(tmp_path):
    """The evaluation rejects paired runs against a changed transcript."""
    manifest = [
        {"subset": subset, "id": subset, "audio": "unused", "reference": "CORRECT", "frames": 1}
        for subset in ("test-clean", "test-other")
    ]
    write_csv(tmp_path / "manifest.csv", manifest, list(manifest[0]))
    rows = [
        {
            "subset": subset,
            "id": subset,
            "reference": "WRONG",
            "reference_words": 1,
            "errors": 0,
            "output": "CORRECT",
        }
        for subset in ("test-clean", "test-other")
    ]
    for precision in ("fp16", "fp32"):
        write_csv(tmp_path / f"{precision}.csv", rows, list(rows[0]))
        (tmp_path / f"{precision}.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="reference mismatch"):
        report(tmp_path)


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
@pytest.mark.parametrize(
    "precisions,field,value",
    [
        (("fp16",), "errors", 0),
        (("fp32",), "errors", 1),
        (("fp16", "fp32"), "reference_words", 2000),
    ],
)
def test_report_rejects_changed_counts_instead_of_accepting_inaccurate_outputs(
    report_inputs, precisions, field, value
):
    """Falsified error counts or reference lengths cannot turn worse fp16 output into a pass."""
    for precision in precisions:
        path = report_inputs / f"{precision}.csv"
        rows = read_csv(path)
        rows[0][field] = value
        path.unlink()
        write_csv(path, rows, list(rows[0]))

    with pytest.raises(ValueError, match=rf"{field} mismatch"):
        report(report_inputs)

    assert not (report_inputs / "report.json").exists()


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_report_rejects_missing_utterances(report_inputs):
    """A greedy report rejects a completed CSV that omits an included manifest utterance."""
    path = report_inputs / "fp16.csv"
    rows = read_csv(path)
    path.unlink()
    write_csv(path, rows[:-1], list(rows[0]))

    with pytest.raises(ValueError, match="results do not match the manifest"):
        report(report_inputs)

    assert not (report_inputs / "report.json").exists()


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_report_preserves_counts_and_gate_for_valid_greedy_results(report_inputs):
    """Valid CSVs retain their word counts, error counts, and per-subset acceptance results."""
    report(report_inputs)

    result = json.loads((report_inputs / "report.json").read_text(encoding="utf-8"))
    assert result["passed"] is False
    for subset, errors in (("test-clean", 1), ("test-other", 0)):
        score = result["subsets"][subset]
        assert score["utterances"] == 1
        assert score["reference_words"] == 2
        assert score["fp16_errors"] == errors
        assert score["fp32_errors"] == 0
        assert score["fp16_wer"] == errors / 2
        assert score["fp32_wer"] == 0
        assert score["passed"] == (errors == 0)
