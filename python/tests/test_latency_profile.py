"""Behavior checks for the fp16 latency breakdown without a CUDA GPU."""

import csv
import json

import pytest
from scripts import profile_latency

pytest_plugins = ("test_native_failures",)


def _row(name, seconds, subset="test-clean"):
    return {
        "subset": subset,
        "id": name,
        "audio": f"/{name}.flac",
        "reference": "HELLO",
        "frames": str(int(seconds * 16_000)),
    }


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_selects_first_utterances_in_each_band_and_preserves_manifest(tmp_path):
    """The profile selects each band's first inputs in manifest order without changing it."""
    rows = [
        _row("other", 3, "test-other"),
        _row("long", 20),
        _row("short", 4.99),
        _row("five", 5),
        _row("ten", 10),
        _row("thirty", 30),
        _row("too-long", 30.01),
        _row("short-extra", 1),
    ]
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    original = manifest.read_bytes()

    selected = profile_latency.select_utterances(manifest, per_band=1)

    assert [row["id"] for row in selected] == ["long", "short", "five", "ten"]
    assert manifest.read_bytes() == original


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_rejects_insufficient_inputs_and_duplicate_selected_ids(tmp_path):
    """A run fails before using the GPU if a band is incomplete or IDs are ambiguous."""
    manifest = tmp_path / "manifest.csv"
    rows = [_row("same", seconds) for seconds in (1, 5, 10, 20)]
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="duplicate"):
        profile_latency.select_utterances(manifest, per_band=1)
    rows = [_row(str(seconds), seconds) for seconds in (1, 5, 10)]
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="20-30"):
        profile_latency.select_utterances(manifest, per_band=1)


def _measurement(name, frames, mel, encoder, steps, other=0):
    gpu = sum(step["gpu_seconds"] for step in steps)
    host = sum(step["host_seconds"] for step in steps)
    return {
        "id": name,
        "frames": frames,
        "total_seconds": mel + encoder + gpu + host + other,
        "mel_seconds": mel,
        "encoder_seconds": encoder,
        "decoder_gpu_seconds": gpu,
        "decoder_host_seconds": host,
        "other_seconds": other,
        "decoder_steps": steps,
        "generated_tokens": len(steps),
    }


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_reports_component_medians_total_shares_and_pooled_step_medians():
    """Bands report real component totals and pool decoder steps across their utterances."""
    rows = [
        _measurement("one", 16_000, 1, 2, [{"gpu_seconds": 2, "host_seconds": 1}], 1),
        _measurement(
            "two",
            32_000,
            3,
            4,
            [{"gpu_seconds": 4, "host_seconds": 2}, {"gpu_seconds": 8, "host_seconds": 3}],
            1,
        ),
        _measurement("boundary", 5 * 16_000, 1, 1, [{"gpu_seconds": 1, "host_seconds": 1}]),
    ]

    report = profile_latency.summarize(rows)
    short = report["0-5"]

    assert short["count"] == 2
    assert short["total_median_seconds"] == 16
    assert short["components"]["mel"]["median_seconds"] == 2
    assert short["components"]["decoder_gpu"]["median_seconds"] == 7
    assert short["components"]["decoder_gpu"]["share_of_total"] == pytest.approx(14 / 32)
    assert sum(part["share_of_total"] for part in short["components"].values()) == 1
    assert short["decoder_step_gpu_median_seconds"] == 4
    assert short["decoder_step_host_median_seconds"] == 2
    assert short["decoder_steps_median"] == 1.5
    assert report["5-10"]["count"] == 1
    assert report["10-20"]["total_median_seconds"] is None
    markdown = profile_latency.markdown_table(report)
    assert "0-5" in markdown and "43.8%" in markdown


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_rejects_missing_gpu_steps_and_invalid_durations():
    """The profile refuses measurements whose GPU timing or input duration is missing."""
    row = _measurement("one", 16_000, 1, 1, [])
    with pytest.raises(ValueError, match="decoder"):
        profile_latency.summarize([row])
    row = _measurement("one", 480_001, 1, 1, [{"gpu_seconds": 1, "host_seconds": 1}])
    with pytest.raises(ValueError, match="frames"):
        profile_latency.summarize([row])


@pytest.fixture
def cpu_native(fault_observer, source_checkpoint):
    """Use real native execution with deterministic external device operations."""
    from diagnostic_runtime_fixture import DiagnosticRuntime

    return DiagnosticRuntime(fault_observer, source_checkpoint)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_times_real_greedy_steps_and_brackets_enqueue_on_its_stream(cpu_native):
    """Timing preserves decoded text and counts EOT while separating GPU from host work."""
    cpu_native.run("""
with profile_latency.DecoderProfiler(model) as profiler:
    row = profiler.measure({'id':'one', 'frames':'16000'}, waveform)
    second = profiler.measure({'id':'two', 'frames':'16000'}, waveform)
assert row['tokens'] == second['tokens'] == [1, 3]
assert row['generated_tokens'] == len(row['decoder_steps']) == 2
assert row['decoder_enqueues'] == 5
assert row['decoder_gpu_seconds'] > 0
assert [item['bucket'] for item in row['decoder_buckets']] == [1, 8, 8, 8, 8]
assert row['decoder_gpu_seconds'] == sum(item['gpu_seconds'] for item in row['decoder_buckets'])
profile_latency.summarize([row, second])
""")


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_restores_inference_and_destroys_events_after_failed_enqueue(cpu_native):
    """An enqueue error leaves no profiler hooks or CUDA events behind."""
    cpu_native.run("""
original = model._greedy_step
observer.lffw_test_fail_next(3)
try:
    with profile_latency.DecoderProfiler(model) as profiler:
        profiler.measure({'id':'bad', 'frames':'16000'}, waveform)
except RuntimeError:
    pass
else:
    raise AssertionError('enqueue failure ignored')
assert model._greedy_step == original
with diagnostic.Session(1) as next_session:
    assert _native.abi_version() == (1, 0, 0)
assert len(next_session.records) == 1
next_session.close()
""")


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_measures_the_gpu_audio_features_used_by_inference(cpu_native):
    """The profile includes native mel execution when the model supplies its GPU audio path."""
    cpu_native.run("""
with profile_latency.DecoderProfiler(model) as profiler:
    row = profiler.measure({'id':'mel', 'frames':'16000'}, waveform)
assert row['tokens'] == [1, 3]
assert any(item['name'] == 'lffw_mel_compute' for item in row['native_operations'])
assert any(item['name'] == 'lffw_encode' for item in row['native_operations'])
assert row['mel_seconds'] > 0 and row['encoder_seconds'] > 0
""")


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_greedy_reads_only_the_selected_token_from_each_decoder_execution(cpu_native):
    """Greedy decoding preserves tokens while each GPU-to-host transfer contains one token ID."""
    cpu_native.run("""
audio = model.encode(features)
expected = model.greedy_decode(audio)
with diagnostic.Session() as session:
    actual = model.greedy_decode(audio)
steps = [item for item in session.records if item['name'] == 'lffw_greedy_step']
assert actual.tokens == expected.tokens == [1, 3]
assert [span['bytes'] for item in steps for span in item['spans'] if span['phase']=='d2h'] == [4]*5
session.close()
""")


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_default_first_attempt_keeps_greedy_tokens_and_only_copies_scalar_results(cpu_native):
    """An accepted first attempt matches greedy and transfers no vocabulary-sized logits."""
    cpu_native.run("""
audio = model.encode(features)
expected = model.greedy_decode(audio)
with diagnostic.Session() as session:
    actual = model.decode_short(audio, [], DecodingSettings())
steps = [item for item in session.records if item['name'] == 'lffw_greedy_step']
assert actual == expected
assert [span['bytes'] for item in steps for span in item['spans'] if span['phase']=='d2h'] == [12]*5
session.close()
""")


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("failure", ["logprob", "compression", "silence"])
def test_default_retries_failed_quality_checks_and_stops_on_silence(cpu_native, failure):
    """Poor scores or repeated text trigger hotter attempts with five candidates; silence ends retries."""
    cpu_native.run(
        "failure="
        + repr(failure)
        + "\n"
        + """
import sys
from tokenizers import Tokenizer, models
from listenflow.fastest_whisper._inference import TensorRTWhisper
if failure == 'compression':
    model.tokenizer = Tokenizer(models.WordLevel({'[UNK]':0, 'hello'*100:1, '<|endoftext|>':3}, unk_token='[UNK]'))
else:
    observer.lffw_test_quality(-3.0, 0.7 if failure == 'silence' else 0.1)
audio = model.encode(features)
expected = model.decode_short(audio, [], DecodingSettings())
attempts=[]
def observe(frame,event,arg):
    if event=='call' and frame.f_code is TensorRTWhisper._decode_attempt.__code__:
        attempts.append(frame.f_locals['temperature'])
previous=sys.getprofile()
sys.setprofile(observe)
try:
    with diagnostic.Session() as session:
        actual=model.decode_short(audio, [], DecodingSettings())
finally:
    sys.setprofile(previous)
assert actual == expected
assert attempts == ([0.0] if failure=='silence' else [0.0,0.2] if failure=='logprob' else [0.0,0.2,0.4,0.6,0.8,1.0]), attempts
session.close()
"""
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_repeated_audio_reuses_captured_decoder_work_and_close_returns_graphs(cpu_native):
    """Repeated decoding replays captured work without recapture and closure releases its graphs."""
    cpu_native.run("""
audio = model.encode(features)
first = model.greedy_decode(audio)
captures = observer.lffw_test_graph_captures()
with diagnostic.Session() as session:
    second = model.greedy_decode(audio)
assert first.tokens == second.tokens == [1, 3]
assert observer.lffw_test_graph_captures() == captures
assert not any(span['graph_capture'] for item in session.records for span in item['spans'])
session.close()
model.close()
assert observer.lffw_test_live_resources() == 0
""")


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_writes_machine_readable_report_and_rejects_incomplete_run(tmp_path):
    """A report retains selection and measurement conditions and requires every selected ID."""
    selected = [_row(str(seconds), seconds) for seconds in (1, 5, 10, 20)]
    rows = [
        _measurement(row["id"], int(row["frames"]), 1, 2, [{"gpu_seconds": 2, "host_seconds": 1}])
        for row in selected
    ]
    metadata = {"gpu_exclusive": False, "gpu_access": "排他なし", "precision": "fp16"}
    with pytest.raises(ValueError, match="selected"):
        profile_latency.write_report(tmp_path, selected, rows[:-1], metadata)
    assert not (tmp_path / "report.json").exists()

    profile_latency.write_report(tmp_path, selected, rows, metadata)

    result = json.loads((tmp_path / "report.json").read_text())
    assert result["metadata"] == metadata
    assert result["selected_utterance_ids"] == [row["id"] for row in selected]
    assert result["bands"]["20-30"]["count"] == 1
    markdown = (tmp_path / "report.md").read_text()
    assert "排他なし" in markdown and "20-30" in markdown
    with pytest.raises(FileExistsError):
        profile_latency.write_report(tmp_path, selected, rows, metadata)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_reports_interpolated_p95_for_components_and_decoder_steps():
    """The slow tail is reported separately for utterances and pooled decoder steps."""
    rows = [
        _measurement("one", 16_000, 1, 2, [{"gpu_seconds": 2, "host_seconds": 1}], 1),
        _measurement(
            "two",
            32_000,
            3,
            4,
            [{"gpu_seconds": 4, "host_seconds": 2}, {"gpu_seconds": 8, "host_seconds": 3}],
            1,
        ),
    ]
    short = profile_latency.summarize(rows)["0-5"]
    assert short["total_p95_seconds"] == pytest.approx(24.1)
    assert short["components"]["mel"]["p95_seconds"] == pytest.approx(2.9)
    assert short["decoder_step_gpu_p95_seconds"] == pytest.approx(7.6)
    assert short["decoder_step_host_p95_seconds"] == pytest.approx(2.9)
    assert profile_latency.summarize(rows)["5-10"]["total_p95_seconds"] is None


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_uses_product_transcription_and_records_executed_cache_buckets(cpu_native):
    """Product transcription retains its text, tokens and actual padded cache work."""
    cpu_native.run("""
expected = model.transcribe(waveform, beam_size=1, temperature=0, best_of=1)
with profile_latency.DecoderProfiler(model) as profiler:
    row = profiler.measure({'id':'product', 'frames':'16000'}, waveform)
assert row['output'] == ' '.join(segment.text for segment in expected)
assert row['tokens'] == [1, 3]
assert row['entrypoint'] == 'CachedWhisper.transcribe'
assert [item['bucket'] for item in row['decoder_buckets']] == [1, 8, 8, 8, 8]
""")


@pytest.mark.req("SP-LFFASTESTWHISPER-005")
@pytest.mark.parametrize("script", ["profile", "benchmark", "trial", "timeline"])
def test_profile_queries_gpu_only_before_and_after_all_utterances(cpu_native, script):
    """Each runner brackets real native transcription with two spaced GPU snapshots."""
    cpu_native.run(
        f"script = {script!r}\n"
        + r"""
import csv
import os
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
import soundfile as sf
from scripts import benchmark_latency
from gpu_query_fixture import gpu_queries
from cuda.bindings import runtime as cudart

measured = []
def observe(frame, event, argument):
    if event == 'return' and frame.f_code is model._greedy_attempt.__code__:
        measured.append(argument.text)
        snapshots.completed = len(measured)

with TemporaryDirectory() as temporary:
    root = Path(temporary)
    selected = []
    for band, seconds in enumerate((1, 5, 10, 20)):
        audio = root / f'{band}.wav'
        sf.write(audio, np.zeros(seconds * 16000, np.float32), 16000)
        selected.append(dict(subset='test-clean', id=f'{band}', audio=str(audio),
                             frames=seconds * 16000, reference='a'))
    manifest = root / 'manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        with (
            gpu_queries() as snapshots,
            patch.dict(os.environ, {'XDG_CACHE_HOME': str(root / 'cache')}),
            patch.object(cudart, 'cudaDeviceSynchronize', lambda: (cudart.cudaError_t.cudaSuccess,)),
            patch.object(cudart, 'cudaProfilerStart', lambda: (cudart.cudaError_t.cudaSuccess,)),
            patch.object(cudart, 'cudaProfilerStop', lambda: (cudart.cudaError_t.cudaSuccess,)),
        ):
            if script == 'profile':
                profile_latency.run(model.checkpoint_dir, manifest, root / 'output', per_band=1)
            elif script == 'benchmark':
                data = root / 'data'
                from test_latency_comparison import _fleurs_shard
                _fleurs_shard(data, 'en', ['a'] * 4)
                benchmark_latency.run('fp16', model.checkpoint_dir, None, root / 'output',
                                      None, data_dir=data, limit=4)
            elif script == 'trial':
                from scripts import fp8_trial
                with patch.object(sys, 'argv', [
                    'fp8_trial', 'smoke', '--precision', 'fp32',
                    '--checkpoint-dir', str(model.checkpoint_dir), '--manifest', str(manifest),
                    '--output-dir', str(root / 'output'), '--per-band', '1',
                ]):
                    fp8_trial.main()
            else:
                from scripts import profile_timeline
                profile_timeline.run(model.checkpoint_dir, manifest, root / 'output')
    finally:
        sys.setprofile(previous)
    assert len(snapshots.calls) == 2, snapshots.calls
    assert [count for _, count, _ in snapshots.calls] == [0, len(measured)]
    assert measured
    assert all(args == ['nvidia-smi', '-q', '-x'] for args, _, _ in snapshots.calls)
    assert snapshots.calls[1][2] - snapshots.calls[0][2] >= 60
    assert all(count in (0, len(measured)) for count, seconds in snapshots.sleeps)
    output = root / 'output'
    if script == 'profile':
        report = json.loads((output / 'report.json').read_text())
        assert report['metadata']['gpu_before_load']['completed_utterances'] == 0
        assert [row['completed_utterances'] for row in report['metadata']['gpu_observations']] == [0, 4]
        assert report['selected_utterance_ids'] == [row['id'] for row in selected]
        assert all(band['count'] == 1 for band in report['bands'].values())
        assert len((output / 'utterances.jsonl').read_text().splitlines()) == 4
        assert not (output / 'utterances.jsonl.partial').exists()
    elif script == 'benchmark':
        report = json.loads((output / 'fp16.json').read_text())
        assert report['tensor_rt_path']['engine_reused'] == dict.fromkeys(('mel','encoder','decoder'),False)
        assert len(list(csv.DictReader((output / 'fp16.csv').open()))) == 4
    elif script == 'trial':
        report = json.loads((output / 'report.json').read_text())
        assert len(report['utterances']) == 4
    else:
        report = json.loads((output / 'timeline.json').read_text())
        assert len(report['utterances']) == len(report['warmup']) == 4
"""
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_timeline_preserves_product_output_and_closes_capture_after_failure(cpu_native):
    """CUDA tracing keeps real transcription and closes its capture if native execution fails."""
    cpu_native.run("""
from scripts.decoder_host_capture import capture_transcriptions
samples = [({'id':'trace', 'frames':'16000'}, waveform)]
rows = capture_transcriptions(model, samples)
assert rows[0]['output'] == 'a'
assert rows[0]['native_operations']
observer.lffw_test_fail_next(4)
try:
    capture_transcriptions(model, samples)
except RuntimeError:
    pass
else:
    raise AssertionError('copy failure ignored')
with diagnostic.Session(1) as session:
    assert _native.abi_version() == (1,0,0)
assert len(session.records)==1
session.close()
""")


@pytest.mark.req("SP-LFFASTESTWHISPER-012")
def test_capture_stops_and_drains_gpu_after_a_failed_product_call():
    """A failed observed call still drains before stopping the external CUDA collector."""
    from scripts import profile_layers

    calls = []

    class Driver:
        def cuProfilerStart(self):
            calls.append("start")
            return (0,)

        def cuCtxSynchronize(self):
            calls.append("drain")
            return (0,)

        def cuProfilerStop(self):
            calls.append("stop")
            return (0,)

    with (
        pytest.raises(RuntimeError, match="product failed"),
        profile_layers.capture_range(Driver()),
    ):
        calls.append("product")
        raise RuntimeError("product failed")
    assert calls == ["start", "product", "drain", "stop"]
