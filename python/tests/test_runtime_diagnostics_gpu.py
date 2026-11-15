"""Opt-in script integration using real native inference and disposable test storage."""

import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.req("SP-LFFASTESTWHISPER-017")
@pytest.mark.req("SP-LFFASTESTWHISPER-043")
def test_diagnostic_scripts_keep_product_tokens_and_cover_every_bucket(
    tmp_path, monkeypatch, record_property
):
    """All four diagnostic scripts run on sm120 and retain product results and native buckets."""
    checkpoint = os.environ.get("LFFW_CHECKPOINT_DIR")
    if os.environ.get("LFFW_USE_GPU") != "1" or not checkpoint:
        pytest.skip("reserve sm120 and set LFFW_USE_GPU=1 and LFFW_CHECKPOINT_DIR")
    import numpy as np
    import soundfile as sf
    from scripts import decoder_host_capture, fp8_trial, profile_latency
    from scripts.profile_layers import decoder_buckets

    original, rate = sf.read(
        Path(__file__).parent / "fixtures/small_boat_16khz_mono_s16le.wav", dtype="int16"
    )
    assert rate == 16000
    rows = []
    for index, seconds in enumerate((1, 6, 11, 21)):
        waveform = np.resize(original, seconds * rate)
        target = tmp_path / f"input-{index}.wav"
        sf.write(target, waveform, rate, subtype="PCM_16")
        rows.append(
            {
                "subset": "test-clean",
                "id": f"diagnostic-{index}",
                "audio": str(target),
                "reference": "diagnostic audio",
                "frames": str(len(waveform)),
            }
        )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    profile_output = tmp_path / "latency"
    profile_latency.run(Path(checkpoint), manifest, profile_output, per_band=1)
    observations = [
        json.loads(line) for line in (profile_output / "utterances.jsonl").read_text().splitlines()
    ]
    assert all(row["native_operations"] and row["tokens"] for row in observations)
    record_property(
        "profile_latency", json.dumps({"utterances": len(observations), "native": True})
    )
    print("diagnostic: profile_latency completed", flush=True)
    trial_output = tmp_path / "trial"
    with monkeypatch.context() as patch:
        patch.setattr(
            sys,
            "argv",
            [
                "fp8_trial",
                "latency",
                "--precision",
                "fp16",
                "--checkpoint-dir",
                checkpoint,
                "--manifest",
                str(manifest),
                "--output-dir",
                str(trial_output),
                "--per-band",
                "1",
                "--iterations",
                "1",
                "--rounds",
                "1",
                "--quiet-interval",
                "reserved-unit-5",
            ],
        )
        fp8_trial.main()
    trial = json.loads((trial_output / "report.json").read_text())
    buckets = trial["rounds"][0]["isolated"]["decoder"]
    assert list(map(int, buckets)) == list(decoder_buckets())
    assert all(
        row["input_kind"] == "product_prefix" and row["samples_ms"] for row in buckets.values()
    )
    record_property(
        "fp8_trial", json.dumps({"precision": "fp16", "buckets": list(map(int, buckets))})
    )
    print("diagnostic: fp8_trial completed all cache buckets", flush=True)
    for mode in ("build", "sections", "tail", "wall", "wait", "trace"):
        output = tmp_path / f"host-{mode}"
        args = SimpleNamespace(
            mode=mode,
            condition="L0",
            run_id=f"unit-5-{mode}",
            checkpoint_dir=Path(checkpoint),
            manifest=manifest,
            output_dir=output,
            schedule=None,
            nvtx_library=None,
        )
        decoder_host_capture.run(args, per_band=1, iterations=1)
        report = json.loads((output / "run.json").read_text())
        if mode != "build":
            assert report["step_count"] > 0
            assert all(
                report["tokens"][key] == tokens for key, tokens in report["baseline_tokens"].items()
            )
            assert (output / "steps.csv").is_file()
        if mode == "trace":
            assert report["native_trace"]
        record_property(
            f"decoder_host_{mode}",
            json.dumps({"steps": report.get("step_count", 0), "condition": "L0"}),
        )
        print(f"diagnostic: decoder_host_capture {mode} completed", flush=True)
