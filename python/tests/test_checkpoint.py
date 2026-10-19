import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from listenflow.fastest_whisper._checkpoint import Checkpoint


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_checkpoint_reads_dimensions_and_prefers_indexed_fp32_weights(tmp_path):
    """The network shape and fp32 weights come from the supplied HF directory."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "num_mel_bins": 4,
                "max_source_positions": 7,
                "d_model": 8,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 16,
                "encoder_layers": 1,
                "decoder_attention_heads": 4,
                "decoder_ffn_dim": 24,
                "decoder_layers": 2,
                "max_target_positions": 11,
                "vocab_size": 23,
            }
        )
    )
    key = "model.encoder.conv1.weight"
    save_file({key: np.array([1], dtype=np.float16)}, tmp_path / "model.safetensors")
    save_file({key: np.array([2], dtype=np.float32)}, tmp_path / "fp32.safetensors")
    (tmp_path / "model.safetensors.index.fp32.json").write_text(
        json.dumps({"weight_map": {key: "fp32.safetensors"}})
    )

    with Checkpoint(tmp_path) as checkpoint:
        assert checkpoint.config.n_mels == 4
        assert checkpoint.config.audio_positions == 7
        assert checkpoint.config.audio_heads == 2
        assert checkpoint.config.text_heads == 4
        assert checkpoint.config.text_ffn == 24
        assert checkpoint.config.text_positions == 11
        assert checkpoint.config.vocab_size == 23
        weight = checkpoint.tensor(key)

    assert weight.dtype == np.float32
    np.testing.assert_array_equal(weight, np.array([2], dtype=np.float32))
