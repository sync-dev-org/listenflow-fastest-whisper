"""A single-token fp16 projection preserves Q/K/V values and serialized weights."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from test_cubin_plugin import bundle, gpu_plugin, native_plugin  # noqa: F401 -- pytest fixtures

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 1280


@pytest.fixture(scope="module")
def qkv_artifacts(request):
    artifacts = request.getfixturevalue("gpu_plugin")
    _, library = artifacts
    rng = np.random.default_rng(33)
    weight = (rng.normal(size=(3, WIDTH, WIDTH)) / np.sqrt(WIDTH)).astype(np.float16)
    q_bias = rng.uniform(-0.5, 0.5, WIDTH).astype(np.float16)
    v_bias = rng.uniform(-0.5, 0.5, WIDTH).astype(np.float16)
    weights = library.parent / "qkv.safetensors"
    save_file({"qkv_weight": weight, "q_bias": q_bias, "v_bias": v_bias}, weights)
    inputs = rng.normal(size=(3, WIDTH)).astype(np.float16)
    # Zero isolates the two biases; basis inputs isolate layout and projection order.
    inputs = np.concatenate(
        (inputs, np.zeros((1, WIDTH), np.float16), np.eye(1, WIDTH, 731, dtype=np.float16))
    )
    references = np.stack(
        [inputs.astype(np.float32) @ projection.astype(np.float32).T for projection in weight],
        axis=1,
    )
    references[:, 0] += q_bias.astype(np.float32)
    references[:, 2] += v_bias.astype(np.float32)
    np.savez(
        weights.with_suffix(".cases.npz"), inputs=inputs, expected=references.astype(np.float16)
    )
    return *artifacts, weights


def _runtime(artifacts, mode, cache, *, observe=False, plan=None):
    environment = {}
    if observe:
        observer = artifacts[1].parent / "liblffw_qkv_driver_observer.so"
        environment = {
            "LD_PRELOAD": str(observer),
            "LFFW_QKV_DRIVER_LOG": str(cache.parent / "driver.log"),
        }
    command = [
        sys.executable,
        str(ROOT / "tests" / "qkv_plugin_runtime.py"),
        mode,
        *map(str, artifacts),
    ]
    if plan is not None:
        command += ["--plan", str(plan)]
    result = subprocess.run(
        command,
        cwd=cache.parent,
        env={
            **os.environ,
            "XDG_CACHE_HOME": str(cache),
            "CUDA_HOME": str(cache / "absent-cuda-sdk"),
            "TENSORRT_ROOT": str(cache / "absent-tensorrt-sdk"),
            **environment,
        },
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # TensorRT reports expected shape/type rejections on stdout before the result.
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-012")
def test_single_token_qkv_matches_independent_fp32_projection(qkv_artifacts, tmp_path):
    """All three 1280-wide projections match NumPy, including Q/V biases and unbiased K."""
    result = _runtime(qkv_artifacts, "build", tmp_path / "cache")
    assert result["cases"] == 5
    assert result["shape"] == [1, 1, 3 * WIDTH]
    assert result["producer_loaded"] is False


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-008")
def test_qkv_reopens_without_materializing_projection_weights(qkv_artifacts, tmp_path):
    """A fresh process reuses serialized QKV weights without graph construction or weight loading."""
    cache = tmp_path / "cache"
    built = _runtime(qkv_artifacts, "build", cache)
    reused = _runtime(qkv_artifacts, "reuse", cache)
    assert (built["path"], built["sha256"], built["output_sha256"]) == (
        reused["path"],
        reused["sha256"],
        reused["output_sha256"],
    )
    assert reused["producer_loaded"] is False


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-008")
def test_qkv_rejects_engine_saved_with_previous_plugin_version(qkv_artifacts, tmp_path):
    """Native loading reports an unusable legacy plugin plan so the engine can rebuild it."""
    directory, library, weights = qkv_artifacts
    legacy = library.parent / "liblffw_qkv_v1_fixture.so"
    plan = tmp_path / "version-1.plan"
    _runtime((directory, legacy, weights), "legacy-build", tmp_path / "legacy-cache", plan=plan)
    rejected = _runtime(qkv_artifacts, "legacy-reject", tmp_path / "new-cache", plan=plan)
    assert rejected["rejected"]
    assert rejected["legacy_creator_present"] is False
    assert "could not deserialize" in rejected["error"] or "execution context" in rejected["error"]


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-012")
@pytest.mark.req("SP-LFFASTESTWHISPER-008")
def test_qkv_contexts_reuse_constant_weights_and_keep_survivors_alive(
    qkv_artifacts, tmp_path, record_property
):
    """Four contexts reuse engine-owned constants without plugin uploads and keep survivors working."""
    cache = tmp_path / "cache"
    built = _runtime(qkv_artifacts, "build", cache)
    observed = _runtime(qkv_artifacts, "contexts", cache, observe=True)
    assert observed["sha256"] == built["sha256"]
    assert observed["output_hashes"] == [built["output_sha256"]] * 4
    assert observed["survivor_hashes"] == [built["output_sha256"]] * 3
    assert observed["replacement_hash"] == built["output_sha256"]
    snapshots = observed["resources"]
    # Device-wide usage includes other processes, so retain it only as a record.
    record_property("qkv_resources", json.dumps(snapshots, sort_keys=True))
    for phase in ("one_context", "four_contexts", "one_closed", "replacement", "last_owner"):
        assert snapshots[phase]["upload"] == 0, observed
        assert snapshots[phase]["upload_bytes"] == 0
        assert snapshots[phase]["alloc"] == 0
        assert snapshots[phase]["free"] == 0
        for name in ("weight", "q_bias", "v_bias"):
            assert len(snapshots[phase]["constant_pointers"][name]) == 1, observed
    assert snapshots["closed"]["alloc"] == snapshots["closed"]["free"] == 0
    assert snapshots["closed"]["load"] == snapshots["closed"]["unload"] == 5
    assert snapshots["closed"]["live_bytes"] == 0
    assert observed["balanced_pointers"]
    assert observed["cuda_contexts"] == 1
    assert observed["producer_loaded"] is False


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-012")
def test_qkv_rejects_incompatible_fields_shapes_and_dtypes(qkv_artifacts, tmp_path):
    """The projection rejects incompatible bundle fields and feature/constant shapes or types."""
    result = _runtime(qkv_artifacts, "reject", tmp_path / "reject")
    assert result["fields_rejected"] == 6
    assert result["inputs_rejected"] == 12
