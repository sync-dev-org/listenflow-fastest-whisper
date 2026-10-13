"""Public fp16 input validation without a GPU or checkpoint."""

import numpy as np
import pytest

from listenflow.fastest_whisper import transcribe_fp16


@pytest.mark.req("SP-LFFASTESTWHISPER-004")
@pytest.mark.parametrize("audio", [np.zeros((2, 2)), np.zeros(480_001), np.array([np.nan])])
def test_fp16_entry_rejects_audio_outside_one_thirty_second_mono_segment(audio):
    """The fp16 entry accepts only finite mono audio of at most 30 seconds."""
    with pytest.raises(ValueError, match="audio must be"):
        transcribe_fp16("missing-checkpoint", audio)
