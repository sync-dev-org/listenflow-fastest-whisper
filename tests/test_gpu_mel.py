"""The GPU mel graph preserves the official fp32 audio features."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorrt as trt
from scripts.generate_oracle import read_pcm16_mono

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
def test_mel_engine_cache_identifies_configuration_precision_filters_and_rebuild(
    tmp_path, monkeypatch
):
    """The mel build forwards cache identity and rebuild intent while keeping fp32 arithmetic."""
    from listenflow.fastest_whisper import _engines

    calls = []

    def open_native(graph, declaration, **kwargs):
        network = NativeNetwork(np.zeros(480_400, dtype=np.float32))
        graph(network)
        calls.append((declaration, kwargs))
        return object()

    monkeypatch.setattr(_engines, "open_engine", open_native)
    checkpoint = SimpleNamespace(directory=tmp_path, config=SimpleNamespace(n_mels=128))
    for precision, force in (("fp16", True), ("fp32", False)):
        _engines._open_mel_engine(
            checkpoint,
            precision=precision,
            force_rebuild=force,
            logger=trt.Logger(trt.Logger.ERROR),
        )
    assert all(
        declaration.precision == "fp32" and not declaration.settings.tf32
        for declaration, _ in calls
    )
    assert [entry["force_rebuild"] for _, entry in calls] == [True, False]
    assert calls[0][1]["graph_version"] != calls[1][1]["graph_version"]
    assert all(entry["weights"][0] == tmp_path / "config.json" for _, entry in calls)
    assert all(entry["weights"][1].name == "mel_filters.npz" for _, entry in calls)
    assert all(entry["io"].outputs["mel"].dtype == trt.float32 for _, entry in calls)
    assert all(entry["io"].outputs["mel"].shape == (128, 3000) for _, entry in calls)


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.parametrize("name", ["small_boat_16khz_mono_s16le", "silence_16khz_mono_s16le"])
def test_native_gpu_mel_matches_official_fp32_features(name):
    """The native TensorRT mel engine keeps official fp32 features on the device."""
    checkpoint = os.environ.get("LFFW_CHECKPOINT_DIR")
    if os.environ.get("LFFW_USE_GPU") != "1" or not checkpoint:
        pytest.skip("set LFFW_USE_GPU=1 and LFFW_CHECKPOINT_DIR for GPU comparisons")
    from listenflow.fastest_whisper._checkpoint import Checkpoint
    from listenflow.fastest_whisper._engines import _open_mel_engine
    from listenflow.fastest_whisper._inference import _HostIO

    with Checkpoint(Path(checkpoint)) as weights:
        handle = _open_mel_engine(
            weights, precision="fp16", force_rebuild=True, logger=trt.Logger(trt.Logger.ERROR)
        )
    samples = read_pcm16_mono(FIXTURES / f"{name}.wav")
    io = _HostIO(handle)
    try:
        result = io.run(
            {
                "audio": np.pad(
                    np.pad(samples, (0, 480_000 - samples.size)), (200, 200), mode="reflect"
                )
            },
            device_outputs=("mel", "log_mel"),
        )
        with np.load(FIXTURES / "oracle" / f"{name}.npz") as oracle:
            np.testing.assert_allclose(result["mel"].numpy(), oracle["mel"], atol=3e-5, rtol=3e-5)
        result["mel"].close()
        result["log_mel"].close()
    finally:
        io.close()


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

        def close(self):
            pass

    samples = read_pcm16_mono(FIXTURES / "small_boat_16khz_mono_s16le.wav")
    waveform = np.concatenate((samples, samples * np.float32(0.001), samples))
    model = TensorRTWhisper.__new__(TensorRTWhisper)
    model._mel = NativeMelIO()
    expected = log_mel_spectrogram_long(waveform, n_mels=128)
    actual = model._mel_spectrogram_long(waveform)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
