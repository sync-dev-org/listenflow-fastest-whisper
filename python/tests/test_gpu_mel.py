"""The GPU mel graph preserves the official fp32 audio features."""

import os
from pathlib import Path

import numpy as np
import pytest
import tensorrt as trt
from scripts.generate_oracle import read_pcm16_mono

from test_native_resources import native_boundary as native_boundary  # noqa: PLC0414

FIXTURES = Path(__file__).parent / "fixtures"


class Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)
        self.name = ""


class Layer:
    def __init__(self, array):
        self.tensor = Tensor(array)

    def get_output(self, index):
        return self.tensor


class Shuffle:
    def __init__(self, tensor):
        self.tensor = tensor
        self.first_transpose = None
        self.reshape_dims = None

    def get_output(self, index):
        array = self.tensor.array
        if self.first_transpose is not None:
            array = array.transpose(self.first_transpose)
        if self.reshape_dims is not None:
            array = array.reshape(self.reshape_dims)
        return Tensor(array)


class NativeNetwork:
    """Replace TensorRT native operators with fp32 NumPy operators, never the mel graph."""

    def __init__(self, samples):
        self.samples = samples
        self.output = None
        self.outputs = {}

    def add_input(self, name, dtype, shape):
        assert name == "audio" and dtype == trt.float32 and shape == (480_400,)
        return Tensor(self.samples)

    def add_constant(self, shape, array):
        return Layer(array.reshape(shape))

    def add_gather(self, data, indices, axis):
        return Layer(np.take(data.array, indices.array, axis=axis))

    def add_elementwise(self, left, right, operation):
        functions = {
            trt.ElementWiseOperation.SUM: np.add,
            trt.ElementWiseOperation.SUB: np.subtract,
            trt.ElementWiseOperation.PROD: np.multiply,
            trt.ElementWiseOperation.MAX: np.maximum,
        }
        return Layer(functions[operation](left.array, right.array))

    def add_matrix_multiply(self, left, left_op, right, right_op):
        left, right = left.array, right.array
        if left_op == trt.MatrixOperation.TRANSPOSE:
            left = left.swapaxes(-1, -2)
        if right_op == trt.MatrixOperation.TRANSPOSE:
            right = right.swapaxes(-1, -2)
        return Layer(left @ right)

    def add_shuffle(self, tensor):
        return Shuffle(tensor)

    def add_slice(self, tensor, start, size, stride):
        return Layer(
            tensor.array[tuple(slice(a, a + n * s, s) for a, n, s in zip(start, size, stride))]
        )

    def add_unary(self, tensor, operation):
        assert operation == trt.UnaryOperation.LOG
        return Layer(np.log(tensor.array))

    def add_reduce(self, tensor, operation, axes, keep_dims):
        assert operation == trt.ReduceOperation.MAX
        reduced = tuple(index for index in range(tensor.array.ndim) if axes & (1 << index))
        return Layer(np.max(tensor.array, axis=reduced, keepdims=keep_dims))

    def mark_output(self, tensor):
        self.outputs[tensor.name] = tensor.array
        if tensor.name == "mel":
            self.output = tensor.array


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("name", ["small_boat_16khz_mono_s16le", "silence_16khz_mono_s16le"])
def test_gpu_mel_graph_math_preserves_official_features(name):
    """The declared GPU operations reproduce the official window, FFT, mel and clipping."""
    from listenflow.fastest_whisper._mel import _MelGraph

    samples = read_pcm16_mono(FIXTURES / f"{name}.wav")
    network = NativeNetwork(
        np.pad(np.pad(samples, (0, 480_000 - samples.size)), (200, 200), mode="reflect")
    )
    with np.load(FIXTURES / "oracle" / f"{name}.npz") as oracle:
        expected = oracle["mel"]
    graph = _MelGraph(network, n_mels=128)
    graph.define()
    np.testing.assert_allclose(network.output, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.req("SP-LFFASTESTWHISPER-008")
def test_mel_native_input_is_separate_from_model_weights(source_checkpoint, native_boundary):
    """Native mel uses the checked filter asset instead of treating auxiliary JSON/NPZ as weights."""
    from listenflow.fastest_whisper._audio import _FILTERS
    from listenflow.fastest_whisper._cached_model import CachedWhisper

    calls, _ = native_boundary
    with CachedWhisper(source_checkpoint):
        pass
    source = calls[0][0]
    assert source["mel_filter_file"] == str(_FILTERS)
    assert source["mel_weights"] == str(_FILTERS.with_suffix(".safetensors"))
    assert source["mel_weights"] not in source["weights"]
    assert all(path.endswith(".safetensors") for path in source["weights"])


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("name", ["small_boat_16khz_mono_s16le", "silence_16khz_mono_s16le"])
def test_native_gpu_mel_matches_official_fp32_features(name):
    """The native model returns mel values matching the official FP32 feature oracle."""
    checkpoint = os.environ.get("LFFW_CHECKPOINT_DIR")
    if os.environ.get("LFFW_USE_GPU") != "1" or not checkpoint:
        pytest.skip("set LFFW_USE_GPU=1 and LFFW_CHECKPOINT_DIR for GPU comparisons")
    from listenflow.fastest_whisper._inference import TensorRTWhisper

    samples = read_pcm16_mono(FIXTURES / f"{name}.wav")
    with TensorRTWhisper(Path(checkpoint)) as model:
        actual = np.asarray(model._mel_spectrogram(samples))[0]
        with np.load(FIXTURES / "oracle" / f"{name}.npz") as oracle:
            np.testing.assert_allclose(actual, oracle["mel"], atol=3e-5, rtol=3e-5)


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_long_gpu_features_keep_cross_window_samples_and_global_clipping():
    """A long recording's transforms use adjacent samples and share one clipping threshold."""
    from listenflow.fastest_whisper._audio import log_mel_spectrogram_long
    from listenflow.fastest_whisper._inference import TensorRTWhisper
    from listenflow.fastest_whisper._mel import _MelGraph

    class NativeMelIO:
        def run(self, inputs, *, device_outputs):
            network = NativeNetwork(inputs["audio"])
            graph = _MelGraph(network, n_mels=128)
            graph.define()
            return network.outputs

        def log_mel_block(self, centered):
            return self.run({"audio": centered}, device_outputs=())["log_mel"]

        def close(self):
            pass

    samples = read_pcm16_mono(FIXTURES / "small_boat_16khz_mono_s16le.wav")
    waveform = np.concatenate((samples, samples * np.float32(0.001), samples))
    model = TensorRTWhisper.__new__(TensorRTWhisper)
    model._native_model = NativeMelIO()
    expected = log_mel_spectrogram_long(waveform, n_mels=128)
    actual = model._mel_spectrogram_long(waveform)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
