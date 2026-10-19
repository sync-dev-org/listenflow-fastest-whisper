"""Read Hugging Face Whisper configuration and safetensors without torch."""

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from safetensors import safe_open


@dataclass(frozen=True)
class ModelConfig:
    n_mels: int
    audio_positions: int
    audio_state: int
    audio_heads: int
    audio_ffn: int
    audio_layers: int
    text_heads: int
    text_ffn: int
    text_layers: int
    text_positions: int
    vocab_size: int


class Checkpoint:
    """Open the weight files for one engine build."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        raw = json.loads((self.directory / "config.json").read_text(encoding="utf-8"))
        self.config = ModelConfig(
            n_mels=raw["num_mel_bins"],
            audio_positions=raw["max_source_positions"],
            audio_state=raw["d_model"],
            audio_heads=raw["encoder_attention_heads"],
            audio_ffn=raw["encoder_ffn_dim"],
            audio_layers=raw["encoder_layers"],
            text_heads=raw["decoder_attention_heads"],
            text_ffn=raw["decoder_ffn_dim"],
            text_layers=raw["decoder_layers"],
            text_positions=raw["max_target_positions"],
            vocab_size=raw["vocab_size"],
        )
        indexed = self.directory / "model.safetensors.index.fp32.json"
        self.weight_map = (
            json.loads(indexed.read_text(encoding="utf-8"))["weight_map"]
            if indexed.exists()
            else None
        )
        self._stack = ExitStack()
        self._readers = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stack.close()

    def tensor(self, key: str) -> np.ndarray:
        filename = self.weight_map[key] if self.weight_map is not None else "model.safetensors"
        if filename not in self._readers:
            self._readers[filename] = self._stack.enter_context(
                safe_open(self.directory / filename, framework="np")
            )
        return np.array(self._readers[filename].get_tensor(key), dtype=np.float32, copy=True)
