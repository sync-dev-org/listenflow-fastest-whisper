"""The caller's fp16 kernel can be reproduced and embedded without a GPU."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from tensoracle.bundle import inspect_bundle

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent


def _run(script, *arguments, cwd=ROOT):
    result = subprocess.run(
        [sys.executable, str(ROOT / "cpp" / script), *map(str, arguments)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    if importlib.util.find_spec("triton") is None:
        pytest.skip("install the cubin dependency group to produce cubins")
    directory = tmp_path_factory.mktemp("cubin-bundle")
    _run("generate_cubin_bundle.py", "--output-dir", directory, cwd=directory.parent)
    return directory


@pytest.mark.req("SP-LFFASTESTWHISPER-006")
@pytest.mark.req("SP-LFFASTESTWHISPER-011")
@pytest.mark.req("SP-LFFASTESTWHISPER-012")
def test_fp16_bundle_contains_reproducible_cubins_for_each_supported_gpu(bundle, tmp_path):
    """Two fresh productions contain identical fp16 cubins for sm86, sm89 and sm120."""
    outputs = [bundle, tmp_path / "second"]
    _run("generate_cubin_bundle.py", "--output-dir", outputs[1], cwd=tmp_path)
    for output in outputs:
        assert inspect_bundle(output)

    manifest = json.loads((outputs[0] / "bundle.json").read_text())
    assert {entry["id"] for entry in manifest["entries"]} == {
        f"sm{sm}.{kernel}"
        for sm in (86, 89, 120)
        for kernel in ("bias_fp16", "qkv_fp16", "fc1_fp16")
    }
    assert all(
        entry["signature"]
        == [
            {"name": "x_ptr", "type": "*fp16"},
            {"name": "bias_ptr", "type": "*fp16"},
            {"name": "out_ptr", "type": "*fp16"},
            {"name": "count", "type": "i32"},
        ]
        for entry in manifest["entries"]
        if entry["id"].endswith(".bias_fp16")
    )
    assert all(
        entry["signature"]
        == [
            {"name": "x_ptr", "type": "*fp16"},
            {"name": "weight_ptr", "type": "*fp16"},
            {"name": "q_bias_ptr", "type": "*fp16"},
            {"name": "v_bias_ptr", "type": "*fp16"},
            {"name": "out_ptr", "type": "*fp16"},
        ]
        for entry in manifest["entries"]
        if entry["id"].endswith(".qkv_fp16")
    )
    contents = [
        {
            path.relative_to(output): path.read_bytes()
            for path in output.rglob("*")
            if path.is_file()
        }
        for output in outputs
    ]
    assert contents[0] == contents[1]


@pytest.fixture(scope="module")
def native_plugin(request, tmp_path_factory):
    sdk = os.environ.get("TENSORRT_ROOT")
    if not sdk:
        pytest.skip("set TENSORRT_ROOT to a verified TensorRT SDK for the native build")
    bundle = request.getfixturevalue("bundle")
    directory = tmp_path_factory.mktemp("cubin-plugin")
    _run(
        "build_native.py",
        "--bundle-dir",
        bundle,
        "--build-dir",
        directory,
        "--tensorrt-root",
        sdk,
        cwd=directory.parent,
    )
    import numpy as np
    from safetensors.numpy import save_file

    save_file({"bias": np.array([-0.375], dtype=np.float16)}, directory / "bias.safetensors")
    return bundle, directory / "liblffw_cubin_plugin.so"


@pytest.mark.req("SP-LFFASTESTWHISPER-007")
@pytest.mark.req("SP-LFFASTESTWHISPER-011")
def test_native_plugin_embeds_the_bundle_and_passes_registration_checks(native_plugin):
    """A standalone build exports registration symbols and only the permitted ELF dependencies."""
    bundle, library = native_plugin
    assert inspect_bundle(bundle, plugin=library) == inspect_bundle(bundle)
    checked = subprocess.run(
        ["ctest", "--test-dir", str(library.parent), "--output-on-failure", "--no-tests=error"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


@pytest.mark.req("SP-LFFASTESTWHISPER-007")
@pytest.mark.req("SP-LFFASTESTWHISPER-033")
def test_native_build_and_ctest_without_the_system_cuda_toolkit(bundle, tmp_path):
    """Wheel headers and the driver suffice when /usr/local and toolkit variables are absent."""
    sdk = os.environ.get("TENSORRT_ROOT")
    if not sdk:
        pytest.skip("set TENSORRT_ROOT to a verified TensorRT SDK for the native build")
    bwrap = shutil.which("bwrap")
    if not bwrap:
        pytest.skip("install bubblewrap to verify the build with /usr/local hidden")
    isolated = [
        bwrap,
        "--bind",
        "/",
        "/",
        "--proc",
        "/proc",
        "--dev-bind",
        "/dev",
        "/dev",
        "--tmpfs",
        "/usr/local",
        "--unsetenv",
        "CUDA_HOME",
        "--unsetenv",
        "CUDA_PATH",
        "--unsetenv",
        "CUDAToolkit_ROOT",
        "--unsetenv",
        "CUDACXX",
        "--unsetenv",
        "LD_LIBRARY_PATH",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
    ]
    build = tmp_path / "native"
    commands = [
        [
            sys.executable,
            str(ROOT / "cpp/build_native.py"),
            "--bundle-dir",
            str(bundle),
            "--build-dir",
            str(build),
            "--tensorrt-root",
            sdk,
        ],
        ["ctest", "--test-dir", str(build), "--output-on-failure", "--no-tests=error"],
    ]
    for command in commands:
        result = subprocess.run(
            [*isolated, *command], capture_output=True, text=True, timeout=180, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert inspect_bundle(bundle, plugin=build / "liblffw_fc1_plugin.so") == inspect_bundle(bundle)


@pytest.fixture(scope="module")
def gpu_plugin(request):
    if os.environ.get("LFFW_USE_GPU") != "1":
        pytest.skip("set LFFW_USE_GPU=1 for the plugin's GPU checks")
    return request.getfixturevalue("native_plugin")


def _runtime(artifacts, mode, cache):
    bundle, library = artifacts
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "tests" / "cubin_plugin_runtime.py"),
            mode,
            str(bundle),
            str(library),
            str(library.parent / "bias.safetensors"),
        ],
        cwd=cache.parent,
        env={
            **os.environ,
            "XDG_CACHE_HOME": str(cache),
            "CUDA_HOME": str(cache / "absent-cuda-sdk"),
            "TENSORRT_ROOT": str(cache / "absent-tensorrt-sdk"),
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-006")
@pytest.mark.req("SP-LFFASTESTWHISPER-011")
def test_plugin_builds_through_both_tensoracle_entry_points(gpu_plugin, tmp_path):
    """Cached and direct builds run fp16 bias addition for full and partial blocks."""
    for mode in ("build", "serialized"):
        observed = _runtime(gpu_plugin, mode, tmp_path / mode)
        assert observed["counts"] == [1, 127, 128, 129, 4097]
        assert observed["producer_loaded"] is False


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-008")
def test_plugin_engine_reopens_in_a_fresh_process_without_calling_graph(gpu_plugin, tmp_path):
    """A saved plugin engine reuses its bytes and serialized bias in a later process."""
    cache = tmp_path / "cache"
    built = _runtime(gpu_plugin, "build", cache)
    reused = _runtime(gpu_plugin, "reuse", cache)
    assert reused["path"] == built["path"]
    assert reused["sha256"] == built["sha256"]
    assert reused["counts"] == built["counts"]
    assert reused["producer_loaded"] is False


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-011")
def test_plugin_rejects_fields_outside_its_bundle_and_fp16_bias_schema(gpu_plugin, tmp_path):
    """Missing fields, another bundle, wrong bias types and wrong lengths cannot create a plugin."""
    result = _runtime(gpu_plugin, "fields", tmp_path / "fields")
    assert result["rejected"] == 6
