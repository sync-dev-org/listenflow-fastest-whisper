"""Inspect every native file and the declared platform in a release wheel."""

import runpy
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def wheel(tmp_path):
    source = tmp_path / "source.c"
    source.write_text("int upstream(void) { return 1; }\n")
    upstream = tmp_path / "libtensoracle.so.2"
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            str(source),
            "-Wl,-soname,libtensoracle.so.2",
            "-o",
            str(upstream),
        ],
        check=True,
        capture_output=True,
    )
    source.write_text("extern int upstream(void); int entry(void) { return upstream(); }\n")
    runtime = tmp_path / "liblffw_runtime.so"
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            str(source),
            str(upstream),
            "-Wl,-rpath,$ORIGIN/../../tensoracle/cpp/lib",
            "-o",
            str(runtime),
        ],
        check=True,
        capture_output=True,
    )
    source.write_text("int entry(void) { return 1; }\n")
    native = tmp_path / "_native.abi3.so"
    plugin = tmp_path / "liblffw_fc1_plugin.so"
    for path in (native, plugin):
        command = ["cc", "-shared", "-fPIC", str(source), "-o", str(path)]
        if path == native:
            command.append("-Wl,-rpath,$ORIGIN")
        subprocess.run(command, check=True, capture_output=True)
    name = "listenflow_fastest_whisper-1.2.3-cp312-abi3-manylinux_2_28_x86_64.whl"
    archive = tmp_path / name
    with zipfile.ZipFile(archive, "w") as output:
        for path in (runtime, native, plugin):
            output.write(path, f"listenflow/fastest_whisper/{path.name}")
        output.writestr("listenflow/fastest_whisper/__init__.py", "")
        output.writestr(
            "listenflow_fastest_whisper-1.2.3.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp312-abi3-manylinux_2_28_x86_64\n",
        )
    return archive


def inspector():
    return runpy.run_path(str(ROOT / "cpp/inspect_release_wheel.py"))


@pytest.mark.req("SP-LFFASTESTWHISPER-024")
def test_symbol_version_maxima_use_numeric_components():
    """glibc 2.28 and libstdc++ 3.4.25 sort after 2.9 and 3.4.9 respectively."""
    assert inspector()["symbol_versions"](
        "Name: GLIBC_2.9 Name: GLIBC_2.28 Name: GLIBCXX_3.4.9 "
        "Name: GLIBCXX_3.4.25 Name: CXXABI_1.3.9 Name: CXXABI_1.3.11"
    ) == {"GLIBC": "2.28", "GLIBCXX": "3.4.25", "CXXABI": "1.3.11"}


@pytest.mark.req("SP-LFFASTESTWHISPER-023")
@pytest.mark.req("SP-LFFASTESTWHISPER-024")
def test_release_inspection_reports_all_three_native_files(wheel, tmp_path):
    """Release inspection records dependencies, search paths and symbol maxima for each ELF."""
    report = inspector()["inspect_wheel"](wheel, tmp_path / "report")
    assert set(report["elf"]) == {
        "_native.abi3.so",
        "liblffw_runtime.so",
        "liblffw_fc1_plugin.so",
    }
    runtime = report["elf"]["liblffw_runtime.so"]
    assert "libtensoracle.so.2" in runtime["needed"]
    assert runtime["runpath"] == ["$ORIGIN/../../tensoracle/cpp/lib"]
    assert runtime["rpath"] == []
    assert set(runtime["symbol_versions"]) == {"GLIBC", "GLIBCXX", "CXXABI"}


@pytest.mark.req("SP-LFFASTESTWHISPER-023")
@pytest.mark.parametrize(
    "extra", ["libtensoracle.so.2", "libnvinfer.so.11", "libcuda.so.1", "libcudart.so.13"]
)
def test_release_rejects_bundled_external_libraries(wheel, tmp_path, extra):
    """External tensoracle, TensorRT, driver and CUDA runtime libraries cannot enter the wheel."""
    with zipfile.ZipFile(wheel, "a") as output:
        output.writestr(f"listenflow/fastest_whisper/{extra}", b"\x7fELFextra")
    with pytest.raises(ValueError, match="native"):
        inspector()["inspect_wheel"](wheel, tmp_path / "report")


@pytest.mark.req("SP-LFFASTESTWHISPER-024")
def test_release_rejects_a_newer_symbol_requirement():
    """Symbols newer than the manylinux 2.28 baselines require a rebuild."""
    with pytest.raises(ValueError, match="GLIBCXX"):
        inspector()["validate_versions"]({"GLIBC": "2.28", "GLIBCXX": "3.4.26", "CXXABI": None})
