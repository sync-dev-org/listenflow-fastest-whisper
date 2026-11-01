"""GPU sampling frequencies and resident candidate batches using the real runtime."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorrt as trt
from scripts.evaluate_fp8_product import seeded_utterance
from scripts.generate_oracle import read_pcm16_mono
from tensoracle import BuilderSettings, EngineBuild, EngineIO, InputShape, TensorSpec, open_engine

from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._decoding import DecodingSettings
from listenflow.fastest_whisper._inference import _HostIO
from listenflow.fastest_whisper._network import _Graph

pytestmark = [pytest.mark.gpu, pytest.mark.req("SP-LFFASTESTWHISPER-002")]
COUNT = 40_000


@pytest.fixture(scope="module")
def sampler():
    if os.environ.get("LFFW_USE_GPU") != "1":
        pytest.skip("set LFFW_USE_GPU=1 for GPU sampling checks")
    shapes = {
        "logits": (COUNT, 1, 6),
        "suppression": (COUNT, 1, 6),
        "temperature": (1, 1, 1),
        "uniform": (COUNT, 1, 1),
        "timestamps": (1, 1, 1),
    }
    owners = []

    def define(network):
        graph = _Graph(network, SimpleNamespace(config=SimpleNamespace(vocab_size=6)), "fp32")
        owners.append(graph)
        inputs = {
            name: network.add_input(name, trt.bool if name == "timestamps" else trt.float32, shape)
            for name, shape in shapes.items()
        }
        result = graph._sampling_result(**inputs, timestamp_begin=4, no_speech=5)
        result.name = "sampled"
        network.mark_output(result)

    handle = open_engine(
        define,
        EngineBuild(
            inputs={name: InputShape.fixed(shape) for name, shape in shapes.items()},
            settings=BuilderSettings(workspace_limit=256 * 1024**2, tf32=False),
        ),
        name="lffw-test-sampling",
        graph_version="1",
        weights=[Path(__file__)],
        io=EngineIO(
            inputs={
                name: TensorSpec(trt.bool if name == "timestamps" else trt.float32, shape)
                for name, shape in shapes.items()
            },
            outputs={"sampled": TensorSpec(trt.float32, (COUNT, 1, 3))},
        ),
        logger=trt.Logger(trt.Logger.WARNING),
    )
    io = _HostIO(handle)
    try:
        yield io
    finally:
        io.close()


@pytest.mark.parametrize("temperature", [0.2, 0.8, 2.0])
def test_gpu_sampling_matches_softmax_with_six_sigma_bound_and_same_seed(sampler, temperature):
    """GPU frequencies follow softmax, scores remain unscaled, and a repeated seed repeats tokens."""
    row = np.array([1_000_000, 999_999.9375, 999_999.875, -np.inf, -np.inf, -np.inf], np.float32)
    outputs = []
    for _ in range(2):
        inputs = {
            "logits": np.tile(row, (COUNT, 1, 1)),
            "suppression": np.zeros((COUNT, 1, 6), np.float32),
            "temperature": np.full((1, 1, 1), temperature, np.float32),
            "uniform": np.random.default_rng(421).random((COUNT, 1, 1), dtype=np.float32),
            "timestamps": np.zeros((1, 1, 1), bool),
        }
        outputs.append(sampler.run(inputs)["sampled"])
    np.testing.assert_array_equal(outputs[0], outputs[1])
    tokens = outputs[0][:, 0, 0].astype(int)
    weights = np.exp((row[:3].astype(np.float64) - row[0]) / temperature)
    expected = weights / weights.sum()
    counts = np.bincount(tokens, minlength=6)
    assert not counts[3:].any()
    assert np.all(
        abs(counts[:3] - COUNT * expected) < 6 * np.sqrt(COUNT * expected * (1 - expected))
    )
    raw = row[:3].astype(np.float64) - row[0]
    np.testing.assert_allclose(
        outputs[0][:, 0, 1], (raw - np.log(np.exp(raw).sum()))[tokens], atol=3e-6
    )


@pytest.fixture(scope="module")
def sampled_model():
    checkpoint = os.environ.get("LFFW_CHECKPOINT_DIR")
    if os.environ.get("LFFW_USE_GPU") != "1" or not checkpoint:
        pytest.skip("set LFFW_USE_GPU=1 and LFFW_CHECKPOINT_DIR for candidate batch checks")
    with CachedWhisper(Path(checkpoint), precision="fp16", settings=DecodingSettings()) as model:
        yield model


def test_prepared_gpu_batch_repeats_utterance_seed_without_logit_or_prefix_transfers(
    sampled_model, monkeypatch
):
    """Prepared retries are reproducible and transfer only three scalar results per candidate."""
    from cuda.bindings import runtime as cudart

    model = sampled_model
    samples = read_pcm16_mono(Path(__file__).parent / "fixtures/small_boat_16khz_mono_s16le.wav")
    encoder = model.encode(model._mel_spectrogram(samples))
    copies = []
    native_copy = cudart.cudaMemcpyAsync

    def copy(*args):
        if args[3] == cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost:
            copies.append(args[2])
        return native_copy(*args)

    def host_logits(*_args, **_kwargs):
        pytest.fail("a prepared sampled retry transferred or stacked vocabulary logits")

    monkeypatch.setattr(cudart, "cudaMemcpyAsync", copy)
    monkeypatch.setattr(model, "logits", host_logits)
    monkeypatch.setattr(np, "stack", host_logits)
    results = []
    for _ in range(2):
        with seeded_utterance("sampled-batch-fixture"):
            results.append(
                model._decode_attempt(
                    encoder, [], with_timestamps=False, temperature=0.4, beam_size=None, best_of=5
                )
            )
    assert results[0] == results[1]
    assert copies and set(copies) == {5 * 3 * np.dtype(np.float32).itemsize}
    assert len(model._sampled_decoder._graphs.entries) <= 16
