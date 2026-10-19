"""Locate the mandatory fc1 library in installed packages and source checkouts."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def _locate(tmp_path, *, packaged=False, development=False):
    package = tmp_path / "python/src/plugin_fixture/inner"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package.parent / "__init__.py").touch()
    shutil.copyfile(PROJECT / "src/listenflow/fastest_whisper/_plugins.py", package / "_plugins.py")
    expected = package / "liblffw_fc1_plugin.so"
    if packaged:
        expected.write_bytes(b"packaged library")
    if development:
        (tmp_path / "cpp").mkdir()
        (tmp_path / "cpp/CMakeLists.txt").touch()
        library = tmp_path / "build/cubin-plugin/liblffw_fc1_plugin.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"development library")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from plugin_fixture.inner._plugins import _fc1_library; print(_fc1_library())",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(package.parent.parent)},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return result, expected


@pytest.mark.req("SP-LFFASTESTWHISPER-007")
@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_packaged_fc1_library_takes_precedence_over_development_build(tmp_path):
    """An installed fc1 library is selected even when a development build also exists."""
    result, expected = _locate(tmp_path, packaged=True, development=True)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == expected


@pytest.mark.req("SP-LFFASTESTWHISPER-007")
def test_source_checkout_uses_its_standalone_cpp_build(tmp_path):
    """A source checkout can use the fc1 target built independently by CMake."""
    result, _ = _locate(tmp_path, development=True)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == tmp_path / "build/cubin-plugin/liblffw_fc1_plugin.so"


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_absent_fc1_library_reports_the_required_native_file(tmp_path):
    """An incomplete fp16 installation fails with the required library's name."""
    result, _ = _locate(tmp_path)
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert "liblffw_fc1_plugin.so" in result.stderr
