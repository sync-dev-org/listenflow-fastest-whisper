"""The selected fp16 decoder requires the production fc1 library."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tensoracle.bundle import inspect_bundle

from listenflow.fastest_whisper import _engines
from listenflow.fastest_whisper._checkpoint import ModelConfig
from test_cubin_plugin import bundle, native_plugin  # noqa: F401 -- pytest fixtures


def checkpoint(directory):
    return SimpleNamespace(
        directory=directory,
        config=ModelConfig(128, 1500, 1280, 20, 5120, 32, 20, 5120, 32, 448, 51866),
        weight_map=None,
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-004")
@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_only_fp16_decoder_declares_the_required_fc1_library(tmp_path, monkeypatch):
    """Only the fp16 decoder binds its cache and build to the required native library."""
    library = tmp_path / "liblffw_fc1_plugin.so"
    calls = []
    monkeypatch.setattr(_engines, "_fc1_library", lambda: library, raising=False)

    def opened(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(create_execution=lambda: SimpleNamespace(close=lambda: None))

    monkeypatch.setattr(_engines, "open_engine", opened)
    for precision in ("fp32", "fp16"):
        for decoder in (False, True):
            _engines.open_whisper_engine(
                checkpoint(tmp_path), decoder=decoder, precision=precision, logger=None
            )
    _engines._open_mel_engine(checkpoint(tmp_path), precision="fp16", logger=None)
    assert [tuple(call.get("plugins", ())) for call in calls] == [(), (), (), (library,), ()]
    assert calls[3]["graph_version"] == "5-fp16"


@pytest.mark.req("SP-LFFASTESTWHISPER-008")
def test_cached_plugin_engine_that_cannot_create_a_context_is_rebuilt_once(monkeypatch):
    """Deferred plugin incompatibility is detected and rebuilt before returning the engine."""
    calls = []
    closed = []
    failure = RuntimeError("TensorRT failed to create an execution context")
    reject_rebuilt = False

    def opened(*args, **kwargs):
        calls.append(kwargs)

        def create():
            if reject_rebuilt or not kwargs["force_rebuild"]:
                raise failure
            return SimpleNamespace(close=lambda: closed.append(True))

        return SimpleNamespace(create_execution=create)

    monkeypatch.setattr(_engines, "open_engine", opened)
    _engines._open_cached_engine(None, None, force_rebuild=False, plugins=(Path("fc1.so"),))
    assert [call["force_rebuild"] for call in calls] == [False, True]
    assert closed == [True]
    calls.clear()
    reject_rebuilt = True
    with pytest.raises(RuntimeError) as caught:
        _engines._open_cached_engine(None, None, force_rebuild=True, plugins=(Path("fc1.so"),))
    assert caught.value is failure
    assert len(calls) == 1


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_missing_fc1_library_stops_fp16_before_building_a_standard_decoder(tmp_path, monkeypatch):
    """A missing required library fails instead of silently building standard fp16 layers."""
    calls = []

    def absent():
        raise FileNotFoundError("required fc1 plugin library is missing")

    monkeypatch.setattr(_engines, "_fc1_library", absent, raising=False)
    monkeypatch.setattr(_engines, "open_engine", lambda *a, **kw: calls.append(kw))
    with pytest.raises(FileNotFoundError, match="fc1"):
        _engines.open_whisper_engine(
            checkpoint(tmp_path), decoder=True, precision="fp16", logger=None
        )
    assert calls == []


@pytest.mark.req("SP-LFFASTESTWHISPER-006")
@pytest.mark.req("SP-LFFASTESTWHISPER-007")
@pytest.mark.req("SP-LFFASTESTWHISPER-011")
def test_production_registers_only_fc1_and_development_retains_all_creators(request):
    """The product exports one fc1 creator; the development library retains all three."""
    bundle_dir, development = request.getfixturevalue("native_plugin")
    product = development.with_name("liblffw_fc1_plugin.so")
    assert inspect_bundle(bundle_dir, plugin=product) == inspect_bundle(bundle_dir)
    code = """
import ctypes, json, sys
import tensorrt as trt
library = ctypes.CDLL(sys.argv[1])
count = ctypes.c_int()
library.getCreators(ctypes.byref(count))
registry = trt.get_plugin_registry()
assert registry.load_library(sys.argv[1]) is not None
print(json.dumps({'count': count.value, 'creators': [
    name for name, version in [('LffwFp16Fc1', '1'), ('LffwFp16Qkv', '2'), ('LffwFp16Bias', '1')]
    if registry.get_creator(name, version, 'listenflow.fastest_whisper') is not None]}))
"""
    for library, expected in (
        (product, ["LffwFp16Fc1"]),
        (development, ["LffwFp16Fc1", "LffwFp16Qkv", "LffwFp16Bias"]),
    ):
        result = subprocess.run(
            [sys.executable, "-c", code, str(library)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        observed = json.loads(result.stdout.splitlines()[-1])
        assert observed == {"count": len(expected), "creators": expected}
