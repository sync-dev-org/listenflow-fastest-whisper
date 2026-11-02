"""Rejected-window records delivered through the real listenflow pipeline."""

import asyncio
import math

import numpy as np
import pytest
from scripts.evaluate_wer import DecodeRetryRecorder

from listenflow import Final, FinalObservation, transcribe, transcribe_utterance
from test_engine_decoding import _engine
from test_engine_decoding import runtime as runtime  # noqa: PLC0414 -- pytest fixture


def _finals(engine, *, seconds=1, leading_samples=0, collector=None):
    pcm = np.full(seconds * 16000, 1000, dtype="<i2").tobytes()

    async def chunks():
        if leading_samples:
            yield np.zeros(leading_samples, dtype="<i2").tobytes()
        yield pcm

    async def collect():
        async with engine:
            if collector is not None:
                return await collector(engine, pcm)
            return [event async for event in engine.stream(chunks()) if isinstance(event, Final)]

    with asyncio.Runner() as runner:
        return runner.run(collect())


def _record(final):
    assert len(final.metadata) == 1
    return dict(final.metadata[0])


@pytest.mark.req("SP-LFFASTESTWHISPER-051")
@pytest.mark.req("SP-LFFASTESTWHISPER-052")
@pytest.mark.parametrize("seconds", [1, 31])
def test_exhausted_phrase_pass_keeps_final_count_and_original_phrase(tmp_path, runtime, seconds):
    """Passing rejected text adds one record per window without adding Final events."""
    runtime(starts=(0,))
    phrase = "a C!"
    phrases = [phrase]
    engine = _engine(tmp_path, temperature=(0.0, 0.4), best_of=1, reject_phrases=phrases)
    phrases[:] = ["changed"]

    finals = _finals(engine, seconds=seconds)

    assert [final.text for final in finals] == ["A C"] * (2 if seconds > 30 else 1)
    for index, final in enumerate(finals):
        assert _record(final) == {
            "type": "rejection",
            "outcome": "passed",
            "reason": "phrase",
            "phrase": phrase,
            "text": final.text,
            "temperature": 0.4,
            "window": index,
            "window_start_s": index * 30.0,
            "window_end_s": min((index + 1) * 30.0, float(seconds)),
        }


@pytest.mark.req("SP-LFFASTESTWHISPER-051")
@pytest.mark.req("SP-LFFASTESTWHISPER-052")
@pytest.mark.parametrize("seconds", [1, 31])
@pytest.mark.parametrize("collector", [None, transcribe, transcribe_utterance])
def test_dropped_phrase_remains_a_metadata_only_final(tmp_path, runtime, seconds, collector):
    """Streams and both complete-input helpers retain rejected text after dropping its output."""
    runtime(starts=(0,))
    engine = _engine(tmp_path, temperature=0.0, reject_phrases=["A"], on_rejected_exhausted="drop")

    finals = _finals(engine, seconds=seconds, collector=collector)

    assert [final.text for final in finals] == [""] * (2 if seconds > 30 else 1)
    for index, final in enumerate(finals):
        record = _record(final)
        assert record["outcome"] == "dropped"
        assert record["reason"] == "phrase"
        assert record["phrase"] == "A"
        assert record["text"] == "A C"
        assert final.ts_offset_s == index * 30.0


@pytest.mark.req("SP-LFFASTESTWHISPER-051")
@pytest.mark.parametrize("seconds", [1, 31])
@pytest.mark.parametrize("silence_rule", ["early_phrase", "official", "repetition"])
def test_silent_rejected_windows_record_only_phrase_hits(tmp_path, runtime, seconds, silence_rule):
    """Both silence rules report phrase hits, while silent repetition has no rejection record."""
    repetition = silence_rule == "repetition"
    sequence = ([11] if seconds > 30 else []) + ([0, 0, 0] if repetition else [0, 2])
    runtime(sequence=sequence, no_speech=0.8)
    engine = _engine(
        tmp_path,
        temperature=(0.0, 0.4),
        reject_phrases=[] if repetition else ["A"],
        reject_repetition=repetition,
        phrase_hit_no_speech=silence_rule == "early_phrase",
        logprob_threshold=-1.0 if silence_rule == "early_phrase" else None,
    )

    finals = _finals(engine, seconds=seconds)

    if repetition:
        assert finals == []
    else:
        assert [final.text for final in finals] == [""] * (2 if seconds > 30 else 1)
        for final in finals:
            record = _record(final)
            assert record["outcome"] == "no_speech"
            assert record["reason"] == "phrase"
            assert record["text"] == "A C"
            assert record["temperature"] == 0.0


@pytest.mark.req("SP-LFFASTESTWHISPER-051")
@pytest.mark.parametrize("seconds", [1, 31])
@pytest.mark.parametrize("mode", ["pass", "drop"])
def test_exhausted_repetition_has_no_matched_phrase(tmp_path, runtime, seconds, mode):
    """An exhausted repeated token sequence reports repetition and a null phrase."""
    runtime(sequence=([11] if seconds > 30 else []) + [0, 0, 0])
    engine = _engine(tmp_path, temperature=0.0, reject_repetition=True, on_rejected_exhausted=mode)

    finals = _finals(engine, seconds=seconds)

    assert [final.text for final in finals] == ["A A A" if mode == "pass" else ""] * (
        2 if seconds > 30 else 1
    )
    for final in finals:
        record = _record(final)
        assert record["outcome"] == ("passed" if mode == "pass" else "dropped")
        assert record["reason"] == "repetition"
        assert record["phrase"] is None
        assert record["text"] == "A A A"


@pytest.mark.req("SP-LFFASTESTWHISPER-051")
@pytest.mark.req("SP-LFFASTESTWHISPER-052")
def test_records_follow_window_order_and_utterance_relative_offsets(tmp_path, runtime):
    """Rejected first and last windows surround an unrecorded clean window in one utterance."""
    runtime(starts=(0, 1, 0))
    engine = _engine(
        tmp_path, temperature=0.4, best_of=1, reject_phrases=["A"], on_rejected_exhausted="drop"
    )

    finals = _finals(engine, seconds=61, leading_samples=512)

    assert [final.text for final in finals] == ["", "B", ""]
    assert len({final.utterance_id for final in finals}) == 1
    assert finals[1].metadata == ()
    records = [_record(finals[index]) for index in (0, 2)]
    assert [record["window"] for record in records] == [0, 2]
    assert [record["window_start_s"] for record in records] == [0.0, 60.0]
    assert [record["window_end_s"] for record in records] == [30.0, 61.0]
    assert [finals[index].ts_offset_s for index in (0, 2)] == [0.032, 60.032]


@pytest.mark.req("SP-LFFASTESTWHISPER-051")
@pytest.mark.parametrize("seconds", [1, 31])
def test_successful_retry_has_no_rejection_record(tmp_path, runtime, seconds):
    """A rejected earlier temperature leaves no record when the later temperature succeeds."""
    runtime(starts=(0, 1))
    engine = _engine(tmp_path, temperature=(0.2, 0.4), best_of=1, reject_phrases=["A"])

    finals = _finals(engine, seconds=seconds)

    assert [final.text for final in finals] == ["B"] * (2 if seconds > 30 else 1)
    assert all(final.metadata == () for final in finals)


@pytest.mark.req("SP-LFFASTESTWHISPER-052")
@pytest.mark.parametrize("seconds", [1, 31])
def test_disabled_rejection_preserves_text_count_and_empty_metadata(tmp_path, runtime, seconds):
    """Explicitly empty rejection settings preserve the original stream output."""
    runtime(starts=(0,))
    original = _finals(_engine(tmp_path, temperature=0.0), seconds=seconds)
    runtime(starts=(0,))
    empty = _finals(
        _engine(tmp_path, temperature=0.0, reject_phrases=[], reject_repetition=False),
        seconds=seconds,
    )

    assert [final.text for final in original] == ["A C"] * (2 if seconds > 30 else 1)
    assert empty == original
    assert all(final.metadata == () for final in (*original, *empty))


@pytest.mark.req("SP-LFFASTESTWHISPER-052")
@pytest.mark.parametrize("seconds", [1, 31])
def test_rejection_records_use_only_core_metadata_values(tmp_path, runtime, seconds):
    """Every record passes the core constructor and contains typed finite scalar values."""
    runtime(starts=(0,))
    finals = _finals(_engine(tmp_path, temperature=0.0, reject_phrases=["A"]), seconds=seconds)

    assert finals
    for final in finals:
        record = _record(final)
        assert type(final.metadata) is tuple and type(final.metadata[0]) is tuple
        assert all(type(pair) is tuple and type(pair[0]) is str for pair in final.metadata[0])
        assert set(record) == {
            "type",
            "outcome",
            "reason",
            "phrase",
            "text",
            "temperature",
            "window",
            "window_start_s",
            "window_end_s",
        }
        assert type(record["window"]) is int
        for key in ("temperature", "window_start_s", "window_end_s"):
            assert type(record[key]) is float and math.isfinite(record[key])
        observation = FinalObservation(final.text, final.ts_offset_s, metadata=final.metadata)
        assert observation.metadata == final.metadata


@pytest.mark.req("SP-LFFASTESTWHISPER-052")
def test_window_record_is_attached_only_to_its_first_text_segment(tmp_path, runtime):
    """A window with two text segments carries its full rejected attempt on the first Final."""
    runtime(sequence=[11, 0, 61, 61, 1, 2])
    finals = _finals(_engine(tmp_path, temperature=0.0, reject_phrases=["B C"]), seconds=60)

    assert [final.text for final in finals] == ["A", "B C", "A", "B C"]
    assert [len(final.metadata) for final in finals] == [1, 0, 1, 0]
    assert [_record(finals[index])["window"] for index in (0, 2)] == [0, 1]
    assert [_record(finals[index])["text"] for index in (0, 2)] == ["A B C", "A B C"]


@pytest.mark.req("SP-LFFASTESTWHISPER-052")
@pytest.mark.parametrize("seconds", [1, 31])
def test_whitespace_only_rejected_text_still_delivers_its_record(tmp_path, runtime, seconds):
    """A passed repetition without visible text produces one metadata-only Final per window."""
    runtime(sequence=([11] if seconds > 30 else []) + [0, 0, 0], word=" ")
    finals = _finals(_engine(tmp_path, temperature=0.0, reject_repetition=True), seconds=seconds)

    assert [final.text for final in finals] == [""] * (2 if seconds > 30 else 1)
    for index, final in enumerate(finals):
        record = _record(final)
        assert (record["outcome"], record["reason"], record["text"]) == ("passed", "repetition", "")
        assert final.ts_offset_s == index * 30.0


@pytest.mark.req("SP-LFFASTESTWHISPER-051")
def test_retry_recording_preserves_the_rejected_window_metadata(tmp_path, runtime):
    """Retry measurement preserves the rejected text's metadata and its attempt temperatures."""
    runtime(starts=(0,))
    engine = _engine(
        tmp_path,
        precision="fp8",
        temperature=(0.0, 0.4),
        best_of=1,
        reject_phrases=["A"],
    )

    with DecodeRetryRecorder("decode_short") as recorder:
        finals = _finals(engine)

    assert [final.text for final in finals] == ["A C"]
    assert _record(finals[0])["outcome"] == "passed"
    assert recorder.records[0]["temperatures"] == [0.0, 0.4]
    assert recorder.records[0]["attempts"] == 2
