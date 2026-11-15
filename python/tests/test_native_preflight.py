"""CUDA capability and import checks use the real C ABI without a GPU context."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def device_observer(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path_factory.mktemp("device-observer") / "observer.so"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            str(root / "cpp/tests/runtime_preflight_observer.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def run(device_observer, body, sm=86, failure=None):
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(device_observer)
    env["LFFW_TEST_SM"] = str(sm)
    if failure is not None:
        env["LFFW_TEST_FAILURE"] = failure
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", body, str(device_observer)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_import_and_version_never_initialize_cuda(device_observer):
    """Importing the native module and reading its ABI never enter CUDA initialization."""
    run(
        device_observer,
        """
import ctypes, sys
observer = ctypes.CDLL(sys.argv[1])
from listenflow.fastest_whisper import _native
assert _native.abi_version() == (1, 0, 0)
assert observer.lffw_test_init_calls() == 0
""",
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-013")
def test_fp8_on_sm86_reports_precision_without_creating_context(device_observer):
    """Unsupported FP8 is rejected before context retention, engine opening or GPU buffers."""
    run(
        device_observer,
        """
from listenflow.fastest_whisper import _native
config = dict(n_mels=80, audio_positions=1500, audio_state=16, audio_heads=1,
              audio_ffn=32, audio_layers=1, text_heads=1, text_ffn=32,
              text_layers=1, text_positions=64, vocab_size=16)
source = dict(config=config, weights=['not-opened.safetensors'], config_file='config.json',
              tokenizer_file='tokenizer.json', mel_filter_file='filters.npz', mel_weights='mel.safetensors')
try:
    _native.Model(source, 'fp8', 0, 0, [], [])
except _native.NativeError as error:
    assert error.status == 9
    assert 'fp8' in str(error) and 'sm86' in str(error)
else:
    raise AssertionError('FP8 ran on sm86')
""",
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.req("SP-LFFASTESTWHISPER-019")
@pytest.mark.parametrize(
    "failure", ["stream", "event", "open-encoder", "open-decoder", "execution"]
)
def test_failed_model_open_releases_prior_native_owners_while_error_is_retained(
    device_observer, source_checkpoint, failure
):
    """Partial model construction returns its context, queue, engines and executions before retry."""
    run(
        device_observer,
        f"""
import ctypes, sys
from pathlib import Path
from listenflow.fastest_whisper import _native
from listenflow.fastest_whisper._cached_model import CachedWhisper
observer = ctypes.CDLL(sys.argv[1])
retained = []
for attempt in range(2):
    try:
        CachedWhisper(Path({str(source_checkpoint)!r}))
    except _native.NativeError as error:
        retained.append(error)
        assert error.status == 3, (error.status, str(error))
    else:
        raise AssertionError('injected failure did not reach the caller')
    assert observer.lffw_test_live_resources() == 0
assert observer.lffw_test_init_calls() == 2
assert len(retained) == 2
""",
        failure=failure,
    )
