"""Paths to immutable feature and precision inputs shipped with the package."""

import json
from pathlib import Path

_ROOT = Path(__file__).parent / "assets"
MEL_FILTERS = _ROOT / "mel_filters.npz"
FP8_SCALES = _ROOT / "fp8-scales.json"


def read_fp8_scales(path: Path) -> dict[str, float]:
    """Read the calibration mapping before handing model inputs to native code."""
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("FP8 scales must be a mapping of layer names to values")  # noqa: TRY004 -- invalid file content
    return values
