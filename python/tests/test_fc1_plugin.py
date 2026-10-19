"""The single-token MLP expansion preserves the graph's fp16 GELU expression."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from test_cubin_plugin import bundle, gpu_plugin, native_plugin  # noqa: F401 -- pytest fixtures

ROOT = Path(__file__).resolve().parents[1]
WIDTH, OUTPUT = 1280, 5120


def reference(inputs, weight, bias):
    """Independent NumPy/math calculation with a half rounding at every graph edge."""
    half = lambda x: x.astype(np.float16).astype(np.float64)
    dot = inputs.astype(np.float64) @ weight.astype(np.float64).T
    x = half(half(dot) + bias.astype(np.float64))
    scaled = half(x * float(np.float16(2**-0.5)))
    erf = half(np.array([math.erf(float(v)) for v in scaled.flat]).reshape(scaled.shape))
    return (half(x * 0.5) * half(1 + erf)).astype(np.float16)


@pytest.fixture(scope="module")
def fc1_artifacts(request):
    artifacts = request.getfixturevalue("gpu_plugin")
    artifacts = artifacts[0], artifacts[1].with_name("liblffw_fc1_plugin.so")
    checkpoint = os.environ.get("LFFW_CHECKPOINT_DIR")
    if not checkpoint:
        pytest.skip("set LFFW_CHECKPOINT_DIR to the large-v3 checkpoint")
    from listenflow.fastest_whisper._checkpoint import Checkpoint

    with Checkpoint(Path(checkpoint)) as source:
        weight = source.tensor("model.decoder.layers.0.fc1.weight").astype(np.float16)
        bias = source.tensor("model.decoder.layers.0.fc1.bias").astype(np.float16)
    assert weight.shape == (OUTPUT, WIDTH)
    rng = np.random.default_rng(3318)
    inputs = rng.normal(size=(3, WIDTH)).astype(np.float16)
    # Isolate bias, then arrange a real row near a dot/bias cancellation and
    # the GELU minimum (-0.75) and origin. Other rows still use real weights.
    row = weight[17].astype(np.float64)
    directed = np.stack(
        [row * (target - float(bias[17])) / (row @ row) for target in (0.0, -0.75, 1e-3)]
    ).astype(np.float16)
    inputs = np.concatenate((inputs, np.zeros((1, WIDTH), np.float16), directed))
    expected = reference(inputs, weight, bias)
    # Bound the fp32 tree's reduction error using exact fp16 products. Eleven
    # addition levels suffice for the padded 2048-wide reduction. The following
    # perturbed reference also includes every half quantization discontinuity.
    unit = 2**-24
    gamma = 11 * unit / (1 - 11 * unit)
    bounds = gamma * (np.abs(inputs.astype(np.float64)) @ np.abs(weight.astype(np.float64)).T)
    path = artifacts[1].parent / "fc1.safetensors"
    save_file({"weight": weight, "bias": bias}, path)
    np.savez(path.with_suffix(".cases.npz"), inputs=inputs, expected=expected, dot_error=bounds)
    return *artifacts, path


def _runtime(artifacts, mode, cache, *, observe=False):
    environment = {}
    if observe:
        environment = {
            "LD_PRELOAD": str(artifacts[1].parent / "liblffw_qkv_driver_observer.so"),
            "LFFW_QKV_DRIVER_LOG": str(cache.parent / "driver.log"),
        }
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests/fc1_plugin_runtime.py"), mode, *map(str, artifacts)],
        cwd=cache.parent,
        env={
            **os.environ,
            "XDG_CACHE_HOME": str(cache),
            "CUDA_HOME": str(cache / "absent-cuda"),
            "TENSORRT_ROOT": str(cache / "absent-trt"),
            **environment,
        },
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-012")
def test_fc1_real_weights_preserve_fp16_rounding_and_erf_gelu(fc1_artifacts, tmp_path):
    """Real checkpoint weights match an independent rounded erf GELU, including cancellation."""
    result = _runtime(fc1_artifacts, "build", tmp_path / "cache")
    assert result["cases"] == 7
    assert result["shape"] == [1, 1, OUTPUT]
    assert not result["producer_loaded"]


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-012")
def test_fc1_rounds_dot_before_bias_and_each_gelu_edge(fc1_artifacts, tmp_path):
    """Cancellation sentinels distinguish graph rounding from one final fp16 conversion."""
    weight = np.zeros((OUTPUT, WIDTH), np.float16)
    weight[:, 0] = 1
    weight[:, 1] = np.resize(
        np.array([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.001, 0.0015], np.float16),
        OUTPUT,
    )
    bias = np.full(OUTPUT, -1, np.float16)
    inputs = np.zeros((1, WIDTH), np.float16)
    inputs[0, :2] = 1
    weights = tmp_path / "rounding.safetensors"
    save_file({"weight": weight, "bias": bias}, weights)
    expected = reference(inputs, weight, bias)
    np.savez(weights.with_suffix(".cases.npz"), inputs=inputs, expected=expected, exact=True)
    result = _runtime((*fc1_artifacts[:2], weights), "build", tmp_path / "cache")
    assert result["cases"] == 1


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-008")
def test_fc1_saved_engine_reuses_constants_without_rebuilding(fc1_artifacts, tmp_path):
    """A new process reuses serialized constants without calling the graph or reading weights."""
    cache = tmp_path / "cache"
    built = _runtime(fc1_artifacts, "build", cache)
    reused = _runtime(fc1_artifacts, "reuse", cache)
    assert (built["sha256"], built["output_sha256"]) == (reused["sha256"], reused["output_sha256"])
    assert not reused["producer_loaded"]


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-012")
def test_fc1_constants_are_shared_by_execution_contexts(fc1_artifacts, tmp_path):
    """Four contexts and a replacement reuse identical constants, without plugin allocations."""
    cache = tmp_path / "cache"
    built = _runtime(fc1_artifacts, "build", cache)
    observed = _runtime(fc1_artifacts, "contexts", cache, observe=True)
    assert observed["output_hashes"] == [built["output_sha256"]] * 4
    assert observed["survivor_hashes"] == [built["output_sha256"]] * 3
    assert observed["replacement_hash"] == built["output_sha256"]
    for phase in ("one_context", "four_contexts", "one_closed", "replacement", "last_owner"):
        snapshot = observed["resources"][phase]
        assert snapshot["alloc"] == snapshot["free"] == snapshot["upload"] == 0
        assert snapshot["upload_bytes"] == 0
        for name in ("fc1_weight", "fc1_bias"):
            assert len(snapshot["constant_pointers"][name]) == 1
    assert observed["resources"]["closed"]["load"] == observed["resources"]["closed"]["unload"] == 5
    assert observed["balanced_pointers"]


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-012")
def test_fc1_rejects_incompatible_fields_shapes_and_dtypes(fc1_artifacts, tmp_path):
    """The fixed single-token expansion rejects incompatible fields and input contracts."""
    result = _runtime(fc1_artifacts, "reject", tmp_path / "cache")
    assert result["fields_rejected"] == 5
    assert result["inputs_rejected"] == 9
