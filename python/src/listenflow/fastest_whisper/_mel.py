"""Declare fp32 audio features using a factored 400-point DFT on TensorRT."""

import numpy as np
import tensorrt as trt

from ._audio import _FILTERS, FFT_SIZE, HOP_SIZE, N_SAMPLES


class _MelGraph:
    def __init__(self, network: trt.INetworkDefinition, *, n_mels: int):
        self.net, self.n_mels = network, n_mels
        self._weights: list[np.ndarray] = []

    def constant(self, value) -> trt.ITensor:
        array = np.ascontiguousarray(value)
        self._weights.append(array)
        return self.net.add_constant(array.shape, array).get_output(0)

    def binary(self, left, right, operation):
        return self.net.add_elementwise(left, right, operation).get_output(0)

    def multiply(self, left, right):
        return self.binary(left, right, trt.ElementWiseOperation.PROD)

    def matrix(self, left, right, *, transpose=False):
        return self.net.add_matrix_multiply(
            left,
            trt.MatrixOperation.NONE,
            right,
            trt.MatrixOperation.TRANSPOSE if transpose else trt.MatrixOperation.NONE,
        ).get_output(0)

    def define(self) -> None:
        frames = N_SAMPLES // HOP_SIZE
        audio = self.net.add_input("audio", trt.float32, (N_SAMPLES + FFT_SIZE,))
        indices = np.arange(frames)[:, None] * HOP_SIZE + np.arange(FFT_SIZE)[None, :]
        gathered = self.net.add_gather(
            audio, self.constant(indices.astype(np.int32)), 0
        ).get_output(0)
        window = self.constant(np.hanning(FFT_SIZE + 1)[:-1].astype(np.float32)[None])
        shuffle = self.net.add_shuffle(self.multiply(gathered, window))
        shuffle.reshape_dims = (frames, 20, 20)
        signal = shuffle.get_output(0)

        # n = 20*n1+n2, k = k1+20*k2: two 20-point transforms keep fp32 sums short.
        axis = np.arange(20, dtype=np.float64)
        angle = -2 * np.pi * axis[:, None] * axis[None, :] / 20
        cosine = self.constant(np.cos(angle).astype(np.float32)[None])
        sine = self.constant(np.sin(angle).astype(np.float32)[None])
        real, imag = self.matrix(cosine, signal), self.matrix(sine, signal)
        angle = -2 * np.pi * axis[:, None] * axis[None, :] / FFT_SIZE
        twiddle_cos = self.constant(np.cos(angle).astype(np.float32)[None])
        twiddle_sin = self.constant(np.sin(angle).astype(np.float32)[None])
        real, imag = (
            self.binary(
                self.multiply(real, twiddle_cos),
                self.multiply(imag, twiddle_sin),
                trt.ElementWiseOperation.SUB,
            ),
            self.binary(
                self.multiply(real, twiddle_sin),
                self.multiply(imag, twiddle_cos),
                trt.ElementWiseOperation.SUM,
            ),
        )
        real, imag = (
            self.binary(
                self.matrix(real, cosine, transpose=True),
                self.matrix(imag, sine, transpose=True),
                trt.ElementWiseOperation.SUB,
            ),
            self.binary(
                self.matrix(real, sine, transpose=True),
                self.matrix(imag, cosine, transpose=True),
                trt.ElementWiseOperation.SUM,
            ),
        )

        def positive_frequencies(tensor):
            shuffle = self.net.add_shuffle(tensor)
            shuffle.first_transpose = (0, 2, 1)
            shuffle.reshape_dims = (frames, FFT_SIZE)
            return self.net.add_slice(
                shuffle.get_output(0), (0, 0), (frames, FFT_SIZE // 2 + 1), (1, 1)
            ).get_output(0)

        real, imag = positive_frequencies(real), positive_frequencies(imag)
        power = self.binary(
            self.multiply(real, real), self.multiply(imag, imag), trt.ElementWiseOperation.SUM
        )
        with np.load(_FILTERS) as filters:
            key = f"mel_{self.n_mels}"
            if key not in filters:
                raise ValueError(f"unsupported mel filter count: {self.n_mels}")
            mel = self.matrix(self.constant(filters[key]), power, transpose=True)

        def scalar(value):
            return self.constant(np.full((1, 1), value, dtype=np.float32))

        mel = self.binary(mel, scalar(1e-10), trt.ElementWiseOperation.MAX)
        log_mel = self.net.add_unary(mel, trt.UnaryOperation.LOG).get_output(0)
        log_mel = self.multiply(log_mel, scalar(1 / np.log(10)))
        log_mel.name = "log_mel"
        self.net.mark_output(log_mel)
        peak = self.net.add_reduce(log_mel, trt.ReduceOperation.MAX, 3, True).get_output(0)
        floor = self.binary(peak, scalar(8), trt.ElementWiseOperation.SUB)
        log_mel = self.binary(log_mel, floor, trt.ElementWiseOperation.MAX)
        shifted = self.binary(log_mel, scalar(4), trt.ElementWiseOperation.SUM)
        output = self.multiply(shifted, scalar(0.25))
        output.name = "mel"
        self.net.mark_output(output)
