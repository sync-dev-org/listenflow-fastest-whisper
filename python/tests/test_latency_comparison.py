"""CPU-only behavior checks for the single-utterance speed comparison."""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scripts import benchmark_latency
from scripts.benchmark_latency import (
    Latency,
    _transcribe_tensor_rt,
    mode_name,
    report,
    summarize_latencies,
)
from scripts.evaluate_wer import word_errors, write_csv
from whisper.normalizers import EnglishTextNormalizer

from listenflow.fastest_whisper._decoding import DecodingSettings
from listenflow.fastest_whisper._inference import Transcription


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_transcription_cpu_and_wall_clocks_include_final_gpu_completion(tmp_path, monkeypatch):
    """Both clocks cover transcription and its last GPU wait, excluding the initial drain."""
    from cuda.bindings import runtime as cu

    clocks = {"wall": 0, "cpu": 0}
    waits = []

    def synchronize():
        waits.append(True)
        clocks["wall"] += 2
        clocks["cpu"] += 1
        return (cu.cudaError_t.cudaSuccess,)

    def transcribe():
        (tmp_path / "text").write_text("recognized")
        clocks["wall"] += 100
        clocks["cpu"] += 30
        return "recognized"

    monkeypatch.setattr(cu, "cudaDeviceSynchronize", synchronize)
    monkeypatch.setattr(benchmark_latency.time, "perf_counter_ns", lambda: clocks["wall"])
    monkeypatch.setattr(benchmark_latency.time, "process_time_ns", lambda: clocks["cpu"])
    text, wall, cpu = benchmark_latency._measure(transcribe)
    assert text == (tmp_path / "text").read_text() == "recognized"
    assert len(waits) == 2
    assert wall == pytest.approx(102e-9)
    assert cpu == pytest.approx(31e-9)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.req("SP-LFFASTESTWHISPER-005")
def test_latency_report_groups_every_utterance_by_duration_and_uses_median_and_p95():
    """Each duration band reports its count, median, and 95th percentile in seconds."""
    rows = [
        Latency("short", 4 * 16_000, 0.1),
        Latency("five", 5 * 16_000, 0.2),
        Latency("ten", 10 * 16_000, 0.3),
        Latency("twenty", 20 * 16_000, 0.4),
        Latency("thirty", 30 * 16_000, 0.6),
    ]

    result = summarize_latencies(rows, {row.utterance_id for row in rows})

    assert result == {
        "0-5": {"count": 1, "median_seconds": 0.1, "p95_seconds": 0.1},
        "5-10": {"count": 1, "median_seconds": 0.2, "p95_seconds": 0.2},
        "10-20": {"count": 1, "median_seconds": 0.3, "p95_seconds": 0.3},
        "20-30": {"count": 2, "median_seconds": 0.5, "p95_seconds": 0.59},
    }


@pytest.mark.req("SP-LFFASTESTWHISPER-005")
def test_latency_report_rejects_a_missing_or_duplicate_utterance():
    """A run is unusable when it does not measure each selected utterance once."""
    with pytest.raises(ValueError, match="utterance IDs"):
        summarize_latencies([Latency("one", 16_000, 0.1)], {"one", "two"})
    with pytest.raises(ValueError, match="utterance IDs"):
        summarize_latencies([Latency("one", 16_000, 0.1), Latency("one", 16_000, 0.2)], {"one"})


@pytest.mark.req("SP-LFFASTESTWHISPER-005")
def test_faster_whisper_receives_the_same_suppression_list_for_every_utterance():
    """Repeated transcriptions preserve the comparison's suppression settings and full text."""
    from faster_whisper.transcribe import get_suppressed_tokens

    tokenizer = SimpleNamespace(
        transcribe=10, translate=11, sot=12, sot_prev=13, sot_lm=14, no_speech=15
    )
    received = []
    effective = []

    class Model:
        def transcribe(self, _samples, **options):
            received.append(list(options["suppress_tokens"]))
            effective.append(get_suppressed_tokens(tokenizer, options["suppress_tokens"]))
            return iter(SimpleNamespace(text=text) for text in (" Hello ", " ", "world ")), None

    options = {"language": "en", "suppress_tokens": [1, 2, 3]}
    samples = np.zeros(16_000, dtype=np.float32)
    texts = [
        benchmark_latency._transcribe_faster_whisper(Model(), samples, options) for _ in range(3)
    ]

    assert texts == ["Hello world"] * 3
    assert received == [[1, 2, 3]] * 3
    assert effective == [(1, 2, 3, 10, 11, 12, 13, 14, 15)] * 3
    assert options == {"language": "en", "suppress_tokens": [1, 2, 3]}


@pytest.mark.req("SP-LFFASTESTWHISPER-005")
def test_latency_runner_includes_the_models_gpu_features_in_the_timed_transcription():
    """The comparison transcribes the GPU features used by the product's model."""
    features = object()

    class Model:
        config = SimpleNamespace(n_mels=128)

        def _mel_spectrogram(self, samples):
            assert samples.shape == (16_000,)
            return features

        def encode(self, mel):
            assert mel is features
            return "encoded"

        def greedy_decode(self, encoder):
            assert encoder == "encoded"
            return Transcription("GPU features", [1])

    assert _transcribe_tensor_rt(Model(), np.zeros(16_000, dtype=np.float32)) == "GPU features"


@pytest.mark.req("SP-LFFASTESTWHISPER-005")
def test_gpu_observation_rejects_another_compute_process(monkeypatch):
    """The comparison records its own process and rejects a competing GPU process."""
    pids = [str(os.getpid())]

    def query(args, **_kwargs):
        if "--query-compute-apps=pid" in args:
            return "\n".join(pids) + "\n"
        return "2500, 50, 4000\n"

    monkeypatch.setattr(benchmark_latency.subprocess, "check_output", query)
    assert benchmark_latency._gpu_sample(100)["compute_process_pids"] == [os.getpid()]
    pids.append(str(os.getpid() + 1))
    with pytest.raises(RuntimeError, match="other GPU compute processes"):
        benchmark_latency._gpu_sample(200)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_latency_runner_uses_selected_beam_and_retry_settings():
    """Each timed utterance uses the selected decoder rather than the greedy-only route."""
    seen = []

    class Model:
        config = SimpleNamespace(n_mels=128)

        def _mel_spectrogram(self, samples):
            from listenflow.fastest_whisper._audio import log_mel_spectrogram

            return log_mel_spectrogram(samples, n_mels=self.config.n_mels)

        def encode(self, _mel):
            return np.empty((0,), dtype=np.float32)

        def decode_short(self, _encoder, prompt, settings):
            seen.append((prompt, settings))
            return Transcription("selected", [1, 3])

    settings = DecodingSettings(beam_size=2, temperatures=(0.0, 0.4), best_of=3)
    text = _transcribe_tensor_rt(Model(), np.zeros(16_000, dtype=np.float32), settings)

    assert text == "selected"
    assert seen == [([], settings)]


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_latency_modes_keep_separate_outputs_for_beam_and_retry_settings():
    """The benchmark does not replace its greedy record with another decode mode."""
    assert mode_name(DecodingSettings(temperatures=(0.0,), best_of=1)) == "fp16"
    assert mode_name(DecodingSettings(beam_size=2, temperatures=(0.0,), best_of=1)) == "fp16-b2-t0"
    assert (
        mode_name(DecodingSettings(beam_size=2, temperatures=(0.0, 0.4), best_of=1))
        == "fp16-b2-t0_0p4"
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_latency_cli_passes_beam_and_retry_options_to_the_existing_measurement(
    tmp_path, monkeypatch
):
    """The option run shares the existing 2,611-utterance manifest and timer."""
    seen = []

    def run(route, checkpoint_dir, wer_output_dir, output_dir, model_dir, *, settings=None):
        seen.append((route, checkpoint_dir, wer_output_dir, output_dir, settings))

    monkeypatch.setattr(benchmark_latency, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_latency",
            "fp16",
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--wer-output-dir",
            str(tmp_path / "wer"),
            "--output-dir",
            str(tmp_path / "output"),
            "--beam-size",
            "2",
            "--temperature",
            "0,0.4",
            "--best-of",
            "3",
        ],
    )

    benchmark_latency.main()

    assert seen == [
        (
            "fp16",
            tmp_path / "checkpoint",
            tmp_path / "wer",
            tmp_path / "output",
            DecodingSettings(beam_size=2, temperatures=(0.0, 0.4), best_of=3),
        )
    ]


@pytest.mark.req("SP-LFFASTESTWHISPER-005")
def test_comparison_uses_existing_wer_normalization_and_word_errors():
    """The faster-whisper reference score uses the WER evaluator's normalization rule."""
    score = word_errors("HELLO WORLD", "Hello, there!", normalizer=EnglishTextNormalizer())

    assert score["normalized_reference"] == "hello world"
    assert score["normalized_output"] == "hello there"
    assert score["reference_words"] == 2
    assert score["substitutions"] == 1
    assert score["errors"] == 1


@pytest.mark.req("SP-LFFASTESTWHISPER-005")
def test_report_requires_the_prior_fp16_reference_and_records_text_diagnostics(tmp_path):
    """The comparison rejects a changed reference and reports output differences and WER."""
    wer_dir = tmp_path / "wer"
    benchmark_dir = tmp_path / "benchmark"
    wer_dir.mkdir()
    benchmark_dir.mkdir()
    manifest = [
        {
            "subset": "test-clean",
            "id": f"utterance-{index}",
            "audio": "unused",
            "reference": "HELLO WORLD",
            "frames": (16_000, 6 * 16_000, 11 * 16_000, 21 * 16_000)[min(index, 3)],
        }
        for index in range(2611)
    ]
    write_csv(wer_dir / "manifest.csv", manifest, list(manifest[0]))
    prior = [
        {
            "subset": "test-clean",
            "id": row["id"],
            "reference": "WRONG REFERENCE" if index == 0 else row["reference"],
            "output": "OLDER WORDS" if index == 0 else "HELLO WORLD",
        }
        for index, row in enumerate(manifest)
    ]
    write_csv(wer_dir / "fp16.csv", prior, list(prior[0]))
    for route in ("fp16", "faster-whisper"):
        rows = [
            {
                "subset": "test-clean",
                "id": row["id"],
                "frames": row["frames"],
                "seconds": 0.2 if route == "fp16" else 0.3,
                "output": (
                    "HELLO THERE" if route == "faster-whisper" and index == 0 else "HELLO WORLD"
                ),
            }
            for index, row in enumerate(manifest)
        ]
        write_csv(benchmark_dir / f"{route}.csv", rows, list(rows[0]))
        (benchmark_dir / f"{route}.json").write_text(
            json.dumps({"checkpoint_sha256": {"model.safetensors": "same"}}), encoding="utf-8"
        )

    with pytest.raises(ValueError, match="prior fp16 reference"):
        report(wer_dir, benchmark_dir)

    prior[0]["reference"] = "HELLO WORLD"
    (wer_dir / "fp16.csv").unlink()
    write_csv(wer_dir / "fp16.csv", prior, list(prior[0]))
    report(wer_dir, benchmark_dir)
    result = json.loads((benchmark_dir / "report.json").read_text(encoding="utf-8"))
    assert result["utterances"] == 2611
    assert result["duration_bands"]["fp16"]["20-30"]["count"] == 2608
    assert result["duration_bands"]["faster-whisper"]["0-5"]["median_seconds"] == 0.3
    assert result["prior_fp16_matching"] == 2610
    assert result["prior_fp16_differences"][0]["id"] == "utterance-0"
    assert result["prior_fp16_differences"][0]["prior_errors"] == 2
    assert result["prior_fp16_differences"][0]["new_errors"] == 0
    assert result["faster_whisper_wer"] == pytest.approx(1 / (2611 * 2))
