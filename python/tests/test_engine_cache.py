"""Saved native engines survive process boundaries and recover from corruption."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from listenflow.fastest_whisper._cached_model import CachedWhisper
from native_runtime_fixture import encoded, record_engines

pytestmark = [pytest.mark.gpu, pytest.mark.req("SP-LFFASTESTWHISPER-008")]


def test_encoder_and_decoder_keep_cache_identity_in_another_process(
    native_checkpoint, native_model
):
    """A fresh process reuses all three saved engines without rewriting their files."""
    paths = native_model._native_model.info["engine_paths"]
    before = {
        name: (Path(path).stat().st_ino, hashlib.sha256(Path(path).read_bytes()).hexdigest())
        for name, path in paths.items()
    }
    expected = np.asarray(encoded(native_model)).sum().item()
    code = """
import json, sys, numpy as np
from pathlib import Path
from listenflow.fastest_whisper._cached_model import CachedWhisper
with CachedWhisper(Path(sys.argv[1])) as model:
    audio = model.encode(np.zeros((80, 3000), np.float32))
    print(json.dumps({'paths': model._native_model.info['engine_paths'], 'sum': np.asarray(audio).sum().item()}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", code, str(native_checkpoint)],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    actual = json.loads(result.stdout.splitlines()[-1])
    assert actual["paths"] == paths
    assert actual["sum"] == expected
    after = {
        # tensoracle records cache hits in mtime; an actual write atomically
        # replaces the inode. Preserve both inode and serialized bytes here.
        name: (Path(path).stat().st_ino, hashlib.sha256(Path(path).read_bytes()).hexdigest())
        for name, path in paths.items()
    }
    assert after == before


def test_corrupt_saved_graph_recovers_without_a_caller_option(native_checkpoint, native_model):
    """A damaged saved encoder is rebuilt once before native inference is returned."""
    paths = native_model._native_model.info["engine_paths"]
    native_model.close()
    path = Path(paths["encoder"])
    path.write_bytes(b"invalid native cache fixture")
    with CachedWhisper(native_checkpoint) as recovered:
        record_engines(native_checkpoint, recovered)
        assert np.isfinite(np.asarray(encoded(recovered))).all()
        assert recovered._native_model.info["engine_paths"]["encoder"] == str(path)
        assert path.read_bytes() != b"invalid native cache fixture"
