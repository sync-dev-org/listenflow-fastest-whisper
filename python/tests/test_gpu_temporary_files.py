"""GPU test storage and cleanup can be checked without loading a GPU."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def test_checkout(tmp_path):
    checkout = tmp_path / "checkout" / "python"
    tests = checkout / "tests"
    tests.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name("conftest.py"), tests / "conftest.py")
    (checkout / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    gpu: GPU integration check\n    req(id): parent requirement\n"
    )
    memory_tmp = tmp_path / "memory-tmp"
    memory_tmp.mkdir()
    caller_cache = tmp_path / "caller-cache"
    caller_cache.mkdir()
    protected_engine = caller_cache / "existing.engine"
    protected_engine.write_bytes(b"keep the caller's cache")
    environment = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "LFFW_USE_GPU": "1",
        "XDG_CACHE_HOME": str(caller_cache),
        "TMPDIR": str(memory_tmp),
    }
    environment.pop("LFFW_TEST_GPU_CACHE_HOME", None)
    try:
        yield checkout, memory_tmp, protected_engine, environment
    finally:
        shutil.rmtree(tmp_path)


def _run(checkout, environment, *arguments):
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *arguments],
        cwd=checkout,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize(
    ("outcome", "exit_code"),
    [("pass", 0), ("failure", 1), ("setup_failure", 1), ("interrupt", 2)],
)
@pytest.mark.parametrize("explicit_basetemp", [False, True])
def test_gpu_files_use_disk_and_are_removed_after_the_run(
    test_checkout, outcome, exit_code, explicit_basetemp
):
    """GPU fixtures and caches use disk or the caller's base and leave no engines on exit."""
    checkout, memory_tmp, protected_engine, environment = test_checkout
    (checkout / "tests" / "test_sample.py").write_text(
        f"""
import json
import os
from pathlib import Path
import pytest

@pytest.fixture(scope="module")
def model(tmp_path_factory):
    cache = Path(os.environ["XDG_CACHE_HOME"])
    cache.mkdir(parents=True, exist_ok=True)
    engine = cache / "model.engine"
    with engine.open("wb") as stream:
        stream.truncate(2 * 1024**3)
    shared = tmp_path_factory.mktemp("shared-gpu-cache")
    (shared / "shared.engine").write_bytes(b"cached engine")
    Path("observed.json").write_text(json.dumps({{"cache": str(cache), "shared": str(shared)}}))
    if {outcome!r} == "setup_failure":
        raise RuntimeError("model setup failed after writing its engine")
    return cache

@pytest.mark.gpu
def test_gpu(model, tmp_path):
    (tmp_path / "local.engine").write_bytes(b"per-test engine")
    observed = json.loads(Path("observed.json").read_text())
    observed["temporary"] = str(tmp_path)
    Path("observed.json").write_text(json.dumps(observed))
    if {outcome!r} == "failure":
        pytest.fail("inference failed after writing its engine")
    if {outcome!r} == "interrupt":
        raise KeyboardInterrupt
"""
    )
    basetemp = checkout / "explicit-base"
    arguments = ["--basetemp", str(basetemp)] if explicit_basetemp else []

    result = _run(checkout, environment, *arguments)

    assert result.returncode == exit_code, result.stdout + result.stderr
    observed = json.loads((checkout / "observed.json").read_text())
    expected_parent = basetemp if explicit_basetemp else checkout.parent / ".nf/tmp/pytest-gpu"
    for value in observed.values():
        directory = Path(value)
        assert directory.is_relative_to(expected_parent), observed
        assert not directory.exists(), observed
    assert not list(expected_parent.rglob("*.engine"))
    assert not list(memory_tmp.rglob("*.engine"))
    assert protected_engine.read_bytes() == b"keep the caller's cache"


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("enable_gpu", [False, True])
def test_cpu_storage_and_environment_are_preserved_between_gpu_tests(test_checkout, enable_gpu):
    """CPU tests retain pytest's temporary root and the caller's cache even in a mixed run."""
    checkout, memory_tmp, protected_engine, environment = test_checkout
    environment["LFFW_USE_GPU"] = "1" if enable_gpu else "0"
    (checkout / "tests" / "test_sample.py").write_text(
        """
import json
import os
from pathlib import Path
import pytest

def record(name, tmp_path):
    Path(name + ".json").write_text(json.dumps({
        "temporary": str(tmp_path), "cache": os.environ["XDG_CACHE_HOME"]
    }))

def test_cpu_before(tmp_path):
    record("before", tmp_path)

@pytest.mark.gpu
def test_gpu(tmp_path):
    record("gpu", tmp_path)

def test_cpu_after(tmp_path):
    record("after", tmp_path)
"""
    )

    result = _run(checkout, environment)

    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("before", "after"):
        observed = json.loads((checkout / f"{name}.json").read_text())
        assert Path(observed["temporary"]).is_relative_to(memory_tmp)
        assert observed["cache"] == str(protected_engine.parent)
    if not enable_gpu:
        observed = json.loads((checkout / "gpu.json").read_text())
        assert Path(observed["temporary"]).is_relative_to(memory_tmp)
        assert observed["cache"] == str(protected_engine.parent)
        assert not (checkout.parent / ".nf/tmp/pytest-gpu").exists()
