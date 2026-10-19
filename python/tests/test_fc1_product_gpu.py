"""Product graphs and execution keep the adopted fp16 fc1 implementation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from test_cubin_plugin import bundle, native_plugin  # noqa: F401 -- pytest fixtures


@pytest.fixture
def checkpoint():
    value = os.environ.get("LFFW_CHECKPOINT_DIR")
    if os.environ.get("LFFW_USE_GPU") != "1" or not value:
        pytest.skip("set LFFW_USE_GPU=1 and LFFW_CHECKPOINT_DIR for product fc1 checks")
    return Path(value)


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-002")
@pytest.mark.req("SP-LFFASTESTWHISPER-004")
@pytest.mark.parametrize(
    "precision,part", [(p, g) for p in ("fp32", "fp16") for g in ("encoder", "decoder")]
)
def test_assembled_whisper_graph_uses_fc1_only_for_the_fp16_decoder(checkpoint, precision, part):
    """All 32 fp16 decoder expansions are plugins; both encoders and the fp32 decoder use none."""
    import tensorrt as trt
    from tensoracle._plugins import register_libraries

    from listenflow.fastest_whisper._checkpoint import Checkpoint
    from listenflow.fastest_whisper._network import _Graph
    from listenflow.fastest_whisper._plugins import _fc1_library

    register_libraries((_fc1_library(),))
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    with Checkpoint(checkpoint) as source:
        graph = _Graph(network, source, precision)
        getattr(graph, part)()
        layers = [network.get_layer(index) for index in range(network.num_layers)]
        plugins = [layer for layer in layers if layer.type == trt.LayerType.PLUGIN_V3]
        expected = precision == "fp16" and part == "decoder"
        assert [layer.name for layer in plugins] == (
            [f"model.decoder.layers.{index}.fc1.gelu" for index in range(32)] if expected else []
        )
        for layer in plugins:
            assert tuple(layer.get_input(0).shape) == (1, 1, 1280)
            assert tuple(layer.get_input(1).shape) == (5120, 1280)
            assert tuple(layer.get_input(2).shape) == (5120,)
            assert tuple(layer.get_output(0).shape) == (1, 1, 5120)


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.req("SP-LFFASTESTWHISPER-006")
@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_product_warmup_graphs_cancellation_and_release_execute_fc1(checkpoint, request, tmp_path):
    """Product warmup and captured contexts use the selected GPU's fc1 and release it after cancellation."""
    _, library = request.getfixturevalue("native_plugin")
    log = tmp_path / "driver.log"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("fc1_product_runtime.py")), str(checkpoint)],
        env={
            **os.environ,
            "LD_PRELOAD": str(library.with_name("liblffw_qkv_driver_observer.so")),
            "LFFW_QKV_DRIVER_LOG": str(log),
        },
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout.splitlines()[-1])
    assert observed["warmup_fc1_launches"] > 0
    assert observed["cancelled_steps"] == 4
    assert observed["graph_contexts"] >= 2
    assert observed["text"]
    events = [line.split("|") for line in log.read_text().splitlines()]
    assert all(int(event[4]) == 0 for event in events)
    assert {int(size) for event, _, _, size, _ in events if event == "load"} == {observed["sm"]}
    assert not any(event in ("alloc", "upload") for event, *_ in events)
    assert len({pointer for event, _, pointer, _, _ in events if event == "fc1_weight"}) == 32
    loads = sorted(
        (context, pointer) for event, context, pointer, _, _ in events if event == "load"
    )
    unloads = sorted(
        (context, pointer) for event, context, pointer, _, _ in events if event == "unload"
    )
    assert loads and loads == unloads
