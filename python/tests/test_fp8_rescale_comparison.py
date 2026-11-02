"""FP8 calibration comparisons use paired deterministic speech and retain retries."""

import csv
import hashlib
import json

import pytest


def _rows():
    def row(key, reference, output, temperatures=(0.0,)):
        return {
            "id": key,
            "reference": reference,
            "output": output,
            "seed": key,
            "temperatures": json.dumps(temperatures),
        }

    return (
        [row("a", "cat dog", "cat dog"), row("b", "bird", "bird", (0.0, 0.2))],
        [row("a", "cat dog", "cat"), row("b", "bird", "", (0.0, 0.2))],
        [row("a", "cat dog", "cat dog"), row("b", "bird", "bird")],
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_comparison_pairs_temperature_zero_and_keeps_other_speech_separate():
    """A retry never enters the deterministic score, even when its output is correct."""
    from scripts.compare_fp8_scales import compare_language

    fp32, old, new = _rows()
    result = compare_language("en", fp32, old, new)
    assert result["old"]["temperature_zero"]["utterances"] == 1
    assert result["old"]["temperature_zero"]["reference_units"] == 2
    assert result["old"]["temperature_zero"]["difference_points"] == 50
    assert result["new"]["temperature_zero"]["difference_points"] == 0
    assert result["new"]["other"]["utterances"] == 1
    assert result["old"]["other"]["difference_points"] == 100
    assert result["common_temperature_zero"]["old"]["difference_points"] == 50
    assert result["common_temperature_zero"]["new"]["difference_points"] == 0


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_old_and_new_cohorts_are_counted_and_shared_cohort_allows_a_fair_comparison():
    """Different retry decisions change pair counts without contaminating the shared cohort."""
    from scripts.compare_fp8_scales import compare_language

    fp32, old, new = _rows()
    fp32[1]["temperatures"] = "[0.0]"
    result = compare_language("en", fp32, old, new)
    assert result["old"]["temperature_zero"]["utterances"] == 1
    assert result["new"]["temperature_zero"]["utterances"] == 2
    assert result["common_temperature_zero"]["old"]["utterances"] == 1
    old[0]["temperatures"] = "[0.0, 0.2]"
    result = compare_language("en", fp32, old, new)
    assert result["old"]["temperature_zero"]["difference_points"] is None


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_multiple_windows_decoded_only_at_temperature_zero_remain_in_the_comparison():
    """Long speech stays eligible when every window uses temperature zero without fallback."""
    from scripts.compare_fp8_scales import compare_language

    fp32, old, new = _rows()
    for rows in (fp32, old, new):
        rows[0]["temperatures"] = "[0.0, 0.0]"
    result = compare_language("en", fp32, old, new)
    assert result["old"]["temperature_zero"]["utterances"] == 1
    assert result["old"]["temperature_zero"]["difference_points"] == 50
    assert result["common_temperature_zero"]["new"]["difference_points"] == 0


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "reference", "score", "seed", "temperature"]
)
def test_comparison_rejects_unpaired_or_invalid_measurements(defect):
    """Incomplete IDs, changed references or seeds, false scores and absent trials stop comparison."""
    from scripts.compare_fp8_scales import compare_language

    fp32, old, new = _rows()
    if defect == "missing":
        new.pop()
    elif defect == "duplicate":
        new.append(new[0])
    elif defect == "reference":
        new[0]["reference"] = "different"
    elif defect == "score":
        new[0]["errors"] = 9
    elif defect == "seed":
        new[0]["seed"] = "changed"
    else:
        new[0]["temperatures"] = "[]"
    with pytest.raises(ValueError):
        compare_language("en", fp32, old, new)


@pytest.mark.req("SP-LFFASTESTWHISPER-040")
def test_saved_run_requires_completion_counts_and_unchanged_csv(tmp_path):
    """An interrupted or altered saved run cannot be presented as a completed evaluation."""
    from scripts.compare_fp8_scales import load_run

    path = tmp_path / "en-fp8.csv"
    rows = _rows()[2]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "status": "complete",
        "expected_utterances": 2,
        "completed_utterances": 2,
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "provenance": {"language": "en", "precision": "fp8"},
    }
    target = path.with_suffix(".json")
    target.write_text(json.dumps(metadata))
    assert load_run(tmp_path, "en", "fp8")[0] == rows
    for field, value in (
        ("status", "stopped"),
        ("completed_utterances", 1),
        ("csv_sha256", "wrong"),
    ):
        target.write_text(json.dumps({**metadata, field: value}))
        with pytest.raises(ValueError):
            load_run(tmp_path, "en", "fp8")
