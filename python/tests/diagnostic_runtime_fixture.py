"""Run real product and diagnostic bindings with only external operations replaced."""

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path


def diagnostic_build_directory():
    root = Path(__file__).resolve().parents[2]
    specification = importlib.util.find_spec("_lffw_diagnostics")
    return (
        Path(specification.origin).parent
        if specification is not None
        else root / "build/python-diagnostics"
    )


class DiagnosticRuntime:
    def __init__(self, observer, checkpoint):
        self.observer, self.checkpoint = observer, checkpoint

    def run(self, body):
        root = Path(__file__).resolve().parents[2]
        module_root = diagnostic_build_directory()
        assert list(module_root.glob("_lffw_diagnostics*.so")), (
            "Build python/ with LFFW_RUNTIME_DIAGNOSTICS=ON in build/python-diagnostics "
            "or add its build directory to PYTHONPATH"
        )
        prefix = f"""
import ctypes
import json
import math
from pathlib import Path
import numpy as np
import _lffw_diagnostics as diagnostic
from listenflow.fastest_whisper import _native
from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._decoding import DecodingSettings
from scripts import profile_latency, decoder_host_profile as host_profile
observer = ctypes.CDLL({str(self.observer)!r})
observer.lffw_test_enable_product_tokens(1)
observer.lffw_test_quality.argtypes = (ctypes.c_float, ctypes.c_float)
model = CachedWhisper(Path({str(self.checkpoint)!r}), settings=DecodingSettings())
features = np.zeros((80, 3000), np.float32)
waveform = np.zeros(16000, np.float32)
"""
        source = prefix + "\ntry:\n" + textwrap.indent(textwrap.dedent(body), "    ")
        source += (
            "\nfinally:\n    model.close()\n    assert observer.lffw_test_live_resources() == 0\n"
        )
        environment = dict(os.environ)
        environment["LD_PRELOAD"] = str(self.observer)
        environment["PYTHONPATH"] = os.pathsep.join(
            map(str, (module_root, root / "python/src", root / "python", root / "python/tests"))
        )
        result = subprocess.run(
            [sys.executable, "-c", source],
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
