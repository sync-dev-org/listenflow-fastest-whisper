from pathlib import Path

import numpy as np
import pytest
from scripts.generate_oracle import read_pcm16_mono

from listenflow.fastest_whisper._audio import log_mel_spectrogram

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("name", ["small_boat_16khz_mono_s16le", "silence_16khz_mono_s16le"])
def test_log_mel_matches_the_official_fp32_reference(name):
    """The fixed audio reaches the encoder with the official 30-second log-mel values."""
    samples = read_pcm16_mono(FIXTURES / f"{name}.wav")
    with np.load(FIXTURES / "oracle" / f"{name}.npz") as oracle:
        expected = oracle["mel"]

    actual = log_mel_spectrogram(samples, n_mels=expected.shape[0])

    # CPU FFT implementations round differently, so allow 3e-5 absolute/relative error.
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
