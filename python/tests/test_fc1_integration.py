"""The selected fp16 decoder requires the production fc1 library."""

import json
import subprocess
import sys

import pytest
from tensoracle.bundle import inspect_bundle

from test_cubin_plugin import bundle, native_plugin  # noqa: F401 -- pytest fixtures


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
