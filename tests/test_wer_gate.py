"""CPU-only acceptance checks for the paired LibriSpeech evaluation."""

import pytest
from scripts.evaluate_wer import PairedResult, included_utterances, report, score_subsets, write_csv


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
