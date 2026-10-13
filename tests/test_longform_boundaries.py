"""Build reproducible LibriSpeech boundary recordings without an external dataset."""

import numpy as np
import pytest
from scripts.prepare_longform_boundaries import assemble_chapter


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_chapter_assembly_keeps_order_and_adds_no_silence(tmp_path):
    """Adjacent utterances retain their PCM16 samples with no inserted gap."""
    soundfile = pytest.importorskip("soundfile")
    first = np.array([100, -200, 300], dtype=np.int16)
    second = np.array([-400, 500], dtype=np.int16)
    soundfile.write(tmp_path / "a.flac", first, 16_000, subtype="PCM_16")
    soundfile.write(tmp_path / "b.flac", second, 16_000, subtype="PCM_16")

    full = assemble_chapter(tmp_path, ["a", "b"])
    cut = assemble_chapter(tmp_path, ["a", "b"], limit_samples=4)

    np.testing.assert_array_equal(full, np.array([100, -200, 300, -400, 500]) / 32768)
    np.testing.assert_array_equal(cut, full[:4])
    assert full.dtype == np.float32
