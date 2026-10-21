"""E4M3 linear operands with fp16 surrounding operations for product fp8."""

from pathlib import Path

import numpy as np
import tensorrt as trt

from ._network import _Graph

_SCALES = Path(__file__).parent / "assets" / "fp8-scales.json"
_GRAPH_VERSION = "fp8-1"


class _Fp8Graph(_Graph):
    """Quantize linear operands and keep all surrounding product operations in fp16."""

    def __init__(self, network, checkpoint, scales):
        super().__init__(network, checkpoint, "fp16")
        self.scales = scales

    def _qdq(self, value, scale, name, *, axis=0):
        # Keep the scale in fp32, including subnormal fp16 weight scales.
        array = np.array(scale, dtype=np.float32, copy=True)
        self._weights.append(array)
        coefficients = self.net.add_constant(array.shape, array).get_output(0)
        quantized = self.net.add_quantize(value, coefficients, trt.fp8)
        quantized.axis = axis
        quantized.name = f"{name}.Q"
        restored = self.net.add_dequantize(quantized.get_output(0), coefficients, trt.float16)
        restored.axis = axis
        restored.name = f"{name}.DQ"
        return restored.get_output(0)

    def linear(self, x, key, *, bias=True):
        scale = self.scales.get(key)
        if scale is None or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"missing or invalid activation scale for {key}")
        weights = self.weight(f"{key}.weight")
        # The product stores (1, input, output); each output column has one scale.
        weight_scale = np.max(np.abs(weights), axis=1) / np.float32(448)
        weight_scale = np.where(weight_scale > 0, weight_scale, np.float32(1))
        weight = self.constant(weights.T[None, :, :])
        weight = self._qdq(weight, weight_scale, f"{key}.weight", axis=2)
        x = self._qdq(x, scale, f"{key}.input")
        layer = self.net.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.NONE
        )
        layer.name = f"{key}.matmul"
        y = layer.get_output(0)
        if bias:
            y = self.add(y, self.constant(self.weight(f"{key}.bias")[None, None, :]))
        return y

    def decoder_fc1(self, x, key):
        # The product's fp16 fc1 plugin cannot consume an FP8 operand.
        return self.gelu(self.linear(x, key))
