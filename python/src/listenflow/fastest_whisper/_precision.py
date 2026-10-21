"""Product precision choices and their surrounding operation types and GPU support."""

from dataclasses import dataclass
from typing import Literal

type Precision = Literal["fp32", "fp16", "fp8"]


@dataclass(frozen=True)
class PrecisionPolicy:
    dtype: str
    capabilities: tuple[tuple[int, int], ...]


PRECISIONS: dict[Precision, PrecisionPolicy] = {
    "fp32": PrecisionPolicy("float32", ((8, 6), (8, 9), (12, 0))),
    "fp16": PrecisionPolicy("float16", ((8, 6), (8, 9), (12, 0))),
    "fp8": PrecisionPolicy("float16", ((8, 9), (12, 0))),
}
