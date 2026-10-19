"""Keep GPU test files on disk and release files and native allocations after use."""

import ctypes
import gc
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

_GPU_DIRECTORY = pytest.StashKey[Path]()


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    if not item.get_closest_marker("gpu") or os.environ.get("LFFW_USE_GPU") != "1":
        return (yield)

    config = item.config
    factory = config._tmp_path_factory
    if _GPU_DIRECTORY not in config.stash:
        parent = (
            factory.getbasetemp()
            if config.option.basetemp is not None
            else config.rootpath.parent / ".nf" / "tmp" / "pytest-gpu"
        )
        parent.mkdir(parents=True, exist_ok=True)
        temporary = TemporaryDirectory(prefix="run-", dir=parent)
        config.add_cleanup(temporary.cleanup)
        config.stash[_GPU_DIRECTORY] = Path(temporary.name)

    directory = config.stash[_GPU_DIRECTORY]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CACHE_HOME", str(directory / "cache"))
        # Both tmp_path and module-scoped tmp_path_factory fixtures share this
        # factory. Restore its original base before the next CPU test runs.
        patch.setattr(factory, "_basetemp", directory)
        return (yield)


def _trim_free_memory():
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except AttributeError:
        return
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    trim(0)


@pytest.fixture(autouse=True)
def release_memory_after_gpu_test(request):
    yield
    if request.node.get_closest_marker("gpu") and os.environ.get("LFFW_USE_GPU") == "1":
        gc.collect()
        _trim_free_memory()
