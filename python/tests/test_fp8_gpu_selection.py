"""FP8 integration tests run only on product-supported GPU generations."""

from pathlib import Path

import pytest

import test_fp8_gpu as fp8_gpu

pytest_plugins = ["pytester"]


@pytest.mark.req("SP-LFFASTESTWHISPER-013")
@pytest.mark.parametrize("capability", [(8, 6), (8, 9), (12, 0)])
def test_fp8_gpu_selection_skips_only_unsupported_devices(monkeypatch, capability):
    """SM86 skips with the supported generations stated; SM89 and SM120 proceed."""
    queried = []

    def query(device):
        queried.append(device)
        return capability

    monkeypatch.setattr(fp8_gpu, "_device_capability", query, raising=False)
    if capability == (8, 6):
        with pytest.raises(pytest.skip.Exception) as failure:
            fp8_gpu._require_fp8_device()
        assert all(sm in str(failure.value) for sm in ("sm86", "sm89", "sm120"))
    else:
        fp8_gpu._require_fp8_device()
    assert queried == [0]


@pytest.mark.req("SP-LFFASTESTWHISPER-013")
def test_each_fp8_gpu_case_skips_before_model_loading_on_sm86(pytester):
    """Every FP8 integration case uses the guard before its model or cache work."""
    pytester.makepyfile(test_fp8_gpu_probe=Path(fp8_gpu.__file__).read_text())
    pytester.makeini("[pytest]\nmarkers =\n    gpu: GPU integration\n    req(id): spec item\n")
    pytester.makeconftest(
        """
from pathlib import Path
import pytest

@pytest.fixture
def gpu(monkeypatch):
    import test_fp8_gpu_probe as module
    assert hasattr(module, '_device_capability'), 'FP8 capability guard is missing'
    monkeypatch.setattr(module, '_device_capability', lambda device: (8, 6))
    return Path('checkpoint-must-not-be-opened')
"""
    )
    result = pytester.runpytest("-q", "-rs")
    result.assert_outcomes(skipped=5)
    assert all(sm in result.stdout.str() for sm in ("sm86", "sm89", "sm120"))
