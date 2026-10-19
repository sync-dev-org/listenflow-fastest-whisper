"""Exact comparison of full-length transcript and timed segments."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scripts import evaluate_longform
from scripts.evaluate_longform import compare_results
from scripts.evaluate_wer import validate_mono_waveform, word_errors


def _oracle():
    return {
        "text": " A B",
        "segments": [
            {"text": " A", "start": 0.0, "end": 1.0, "tokens": [100, 1, 150], "seek": 0},
            {"text": "", "start": 1.0, "end": 1.2, "tokens": [], "seek": 0},
            {"text": " B", "start": 30.0, "end": 31.0, "tokens": [100, 2, 150], "seek": 3000},
        ],
        "decode_calls": [
            {"tokens": [100, 1, 150], "prompt": []},
            {"tokens": [100, 2, 150], "prompt": [100, 1, 150]},
        ],
    }


def _engine():
    return {
        "text": " A B",
        "segments": [
            {"text": " A", "start": 0.0, "end": 1.0, "tokens": [100, 1, 150], "seek": 0},
            {"text": "", "start": 1.0, "end": 1.2, "tokens": [], "seek": 0},
            {"text": " B", "start": 30.0, "end": 31.0, "tokens": [100, 2, 150], "seek": 3000},
        ],
        "windows": [
            {"seek": 0, "next_seek": 3000, "tokens": [100, 1, 150]},
            {"seek": 3000, "next_seek": 3100, "tokens": [100, 2, 150]},
        ],
    }


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_full_length_acceptance_requires_exact_text_positions_and_empty_segments():
    """Matching text alone cannot hide a changed boundary or a removed empty segment."""
    oracle, engine = _oracle(), _engine()
    assert compare_results(oracle, engine)["passed"]

    engine["segments"][1]["end"] = 1.22
    report = compare_results(oracle, engine)
    assert not report["passed"]
    assert report["first_difference"]["segment_index"] == 1
    assert report["first_difference"]["field"] == "end"

    engine = _engine()
    engine["segments"].pop(1)
    assert not compare_results(oracle, engine)["passed"]


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_token_difference_identifies_its_window_and_text_keeps_edge_spaces():
    """The first changed token and the original whitespace are part of the result."""
    oracle, engine = _oracle(), _engine()
    engine["windows"][1]["tokens"][1] = 3
    engine["text"] = "A B"

    report = compare_results(oracle, engine)

    assert not report["passed"]
    assert report["first_token_difference"] == {
        "window_index": 1,
        "seek_frame": 3000,
        "token_index": 1,
        "oracle": 2,
        "fp32": 3,
    }
    assert report["first_difference"]["field"] == "text"


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_report_keeps_the_first_segment_difference_and_later_spread_when_text_differs():
    """A changed transcript still identifies the first segment and downstream differences."""
    oracle, engine = _oracle(), _engine()
    engine["text"] = " A C"
    engine["segments"][1]["end"] = 1.4
    engine["segments"][2]["text"] = " C"

    report = compare_results(oracle, engine)

    assert report["first_difference"]["field"] == "text"
    assert report["first_segment_difference"]["segment_index"] == 1
    assert report["segment_differences_after_first"] == 1


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_evaluations_share_finite_mono_16_khz_waveform_validation():
    """The prepared waveform must have the same shape, rate, and finite values in both evaluations."""
    assert validate_mono_waveform(np.zeros(4, dtype=np.float32), sample_rate=16_000).shape == (4,)
    for waveform, rate in [
        (np.zeros((2, 2)), 16_000),
        (np.array([np.nan]), 16_000),
        (np.zeros(4), 24_000),
    ]:
        with pytest.raises(ValueError, match="16 kHz mono"):
            validate_mono_waveform(waveform, sample_rate=rate)


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_diagnostic_word_errors_use_the_same_normalizer_as_short_audio():
    """A text diagnostic shares the short-audio WER normalization and error count."""
    pytest.importorskip("jiwer")
    pytest.importorskip("whisper")
    measured = word_errors("Hello, world!", "hello world")
    assert measured["normalized_reference"] == "hello world"
    assert measured["normalized_output"] == "hello world"
    assert measured["errors"] == 0


@pytest.mark.req("SP-LFFASTESTWHISPER-009")
@pytest.mark.parametrize("phase", ["oracle", "fp32"])
def test_longform_evaluation_cli_passes_the_hint_to_each_decoder(phase, tmp_path, monkeypatch):
    """The recorded long-form comparison supplies the same hint to both decode routes."""
    seen = []

    def run(checkpoint, output_dir, *, initial_prompt=None):
        seen.append((checkpoint, output_dir, initial_prompt))

    monkeypatch.setattr(evaluate_longform, phase, run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_longform",
            phase,
            "--checkpoint",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path),
            "--initial-prompt",
            "the",
        ],
    )

    evaluate_longform.main()

    assert seen == [(tmp_path / "model", Path(tmp_path), "the")]


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("phase", ["oracle", "fp32"])
def test_longform_evaluation_cli_passes_the_beam_width_to_both_routes(phase, tmp_path, monkeypatch):
    """The full-length comparison uses the same beam width in both decode routes."""
    seen = []

    def run(checkpoint, output_dir, *, initial_prompt=None, beam_size=1):
        seen.append((checkpoint, output_dir, beam_size))

    monkeypatch.setattr(evaluate_longform, phase, run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_longform",
            phase,
            "--checkpoint",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path),
            "--beam-size",
            "2",
        ],
    )

    evaluate_longform.main()

    assert seen == [(tmp_path / "model", Path(tmp_path), 2)]
