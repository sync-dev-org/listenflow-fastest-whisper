"""The caller's fp16 kernel can be reproduced and embedded without a GPU."""

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
    code = (
        "import runpy, sys; from pathlib import Path; sys.argv = sys.argv[1:]; "
        "sys.path.insert(0, str(Path(sys.argv[0]).parent)); "
        "runpy.run_path(sys.argv[0], run_name='__main__'); "
        "assert not any(name in sys.modules for name in ('torch', 'triton'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "cpp" / script), *map(str, arguments)],
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
    cuda_root = os.environ.get("CUDAToolkit_ROOT")
    if not cuda_root:
        pytest.skip("prepare Conan dependencies and source build/conan-prepare/conanbuild.sh")
    directory = tmp_path_factory.mktemp("cubin-bundle")
    _run(
        "generate_cubin_bundle.py",
        "--cuda-root",
        cuda_root,
        "--output-dir",
        directory,
        cwd=directory.parent,
    )
    return directory


@pytest.mark.req("SP-LFFASTESTWHISPER-006")
@pytest.mark.req("SP-LFFASTESTWHISPER-011")
@pytest.mark.req("SP-LFFASTESTWHISPER-012")
@pytest.mark.req("SP-LFFASTESTWHISPER-055")
def test_fp16_bundle_contains_reproducible_cubins_for_each_supported_gpu(bundle, tmp_path):
    """Conan's CUDA tools reproduce fp16 cubins for sm86, sm89 and sm120."""
    outputs = [bundle, tmp_path / "second"]
    _run(
        "generate_cubin_bundle.py",
        "--cuda-root",
        os.environ["CUDAToolkit_ROOT"],
        "--output-dir",
        outputs[1],
        cwd=tmp_path,
    )
    for output in outputs:
        assert inspect_bundle(output)

    manifest = json.loads((outputs[0] / "bundle.json").read_text())
    assert manifest["schema_version"] == 2
    assert "nvrtc" in manifest["producer"]
    assert manifest["producer"]["nvrtc"] == "13.2"
    assert manifest["producer"]["ptxas"] == "13.2.86"
    assert "triton" not in manifest["producer"]
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
    assert all(
        entry["signature"]
        == [
            {"name": "x_ptr", "type": "*fp16"},
            {"name": "weight_ptr", "type": "*fp16"},
            {"name": "bias_ptr", "type": "*fp16"},
            {"name": "out_ptr", "type": "*fp16"},
        ]
        for entry in manifest["entries"]
        if entry["id"].endswith(".fc1_fp16")
    )
    for entry in manifest["entries"]:
        assert entry["entry_symbol"] == entry["id"].split(".", 1)[1]
        assert "hidden_trailing_parameters" not in entry
        assert "global_scratch_bytes" not in entry["resources"]
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
    bundle = request.getfixturevalue("bundle")
    directory = tmp_path_factory.mktemp("cubin-plugin")
    _run(
        "build_native.py",
        "--bundle-dir",
        bundle,
        "--build-dir",
        directory,
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
        [
            "ctest",
            "--test-dir",
            str(library.parent),
            "--output-on-failure",
            "--no-tests=error",
            "-LE",
            "gpu",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


@pytest.mark.req("SP-LFFASTESTWHISPER-007")
@pytest.mark.req("SP-LFFASTESTWHISPER-033")
def test_native_build_and_ctest_without_the_system_cuda_toolkit(bundle, tmp_path):
    """Conan headers and the driver suffice when /usr/local and toolkit variables are absent."""
    bwrap = shutil.which("bwrap")
    if not bwrap:
        pytest.skip("install bubblewrap to verify the build with /usr/local hidden")
    conan = shutil.which("conan")
    assert conan, "install the pinned Conan CLI before the native build tests"
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
        f"{Path(conan).parent}:/usr/bin:/bin",
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
        ],
        [
            "ctest",
            "--test-dir",
            str(build),
            "--output-on-failure",
            "--no-tests=error",
            "-LE",
            "gpu",
        ],
    ]
    for command in commands:
        result = subprocess.run(
            [*isolated, *command], capture_output=True, text=True, timeout=180, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert inspect_bundle(bundle, plugin=build / "liblffw_fc1_plugin.so") == inspect_bundle(bundle)
