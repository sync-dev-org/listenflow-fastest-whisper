"""Reproduce the native mel filters from the existing CPU reference asset."""

from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


def main() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/listenflow/fastest_whisper/assets/mel_filters.npz"
    )
    with np.load(source) as filters:
        values = {name: np.ascontiguousarray(filters[name]) for name in sorted(filters.files)}
    save_file(values, source.with_suffix(".safetensors"))


if __name__ == "__main__":
    main()
