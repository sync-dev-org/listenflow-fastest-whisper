"""Declare the shared encoder and decoder builds for cached and comparison runs."""

import json

import tensorrt as trt
from tensoracle import (
    BuilderSettings,
    EngineBuild,
    EngineCacheError,
    EngineHandle,
    EngineIO,
    InputShape,
    TensorSpec,
    open_engine,
)

from ._audio import _FILTERS, FFT_SIZE, N_SAMPLES
from ._checkpoint import Checkpoint
from ._mel import _MelGraph
from ._network import _Graph
from ._plugins import _fc1_library
from ._precision import PRECISIONS

# Increase the relevant value when a graph change makes an older serialized engine incompatible.
_GRAPH_VERSION = {"mel": "2", "encoder": "3", "decoder": "5"}
_SETTINGS = BuilderSettings(workspace_limit=4 * 1024**3, optimization_level=5, tf32=False)


def _open_cached_engine(graph, declaration, *, force_rebuild: bool, **kwargs) -> EngineHandle:
    """Rebuild an unusable saved engine once, preserving the original declaration."""

    def validate(handle):
        if kwargs.get("plugins"):
            # TensorRT can defer plugin creation until the first execution context.
            # Detect an unusable plugin plan while cache recovery still owns it.
            handle.create_execution().close()
        return handle

    try:
        handle = open_engine(graph, declaration, force_rebuild=force_rebuild, **kwargs)
    except EngineCacheError:
        if force_rebuild:
            raise
    else:
        try:
            return validate(handle)
        except RuntimeError:
            if force_rebuild:
                raise
        handle = None
    return validate(open_engine(graph, declaration, force_rebuild=True, **kwargs))


def _open_mel_engine(
    checkpoint: Checkpoint, *, precision: str, force_rebuild: bool = False, logger: trt.ILogger
) -> EngineHandle:
    """Use the same cache recovery policy for fp32 features on both recognition paths."""
    owners: list[_MelGraph] = []

    def define(network):
        graph = _MelGraph(network, n_mels=checkpoint.config.n_mels)
        owners.append(graph)
        graph.define()

    return _open_cached_engine(
        define,
        EngineBuild(
            inputs={"audio": InputShape.fixed((N_SAMPLES + FFT_SIZE,))},
            settings=_SETTINGS,
        ),
        name="listenflow-fastest-whisper-mel",
        graph_version=f"{_GRAPH_VERSION['mel']}-{precision}",
        weights=[checkpoint.directory / "config.json", _FILTERS],
        io=EngineIO(
            inputs={"audio": TensorSpec(trt.float32, (N_SAMPLES + FFT_SIZE,))},
            outputs={
                name: TensorSpec(trt.float32, (checkpoint.config.n_mels, 3000))
                for name in ("mel", "log_mel")
            },
        ),
        force_rebuild=force_rebuild,
        logger=logger,
    )


def open_whisper_engine(
    checkpoint: Checkpoint,
    *,
    decoder: bool,
    precision: str,
    force_rebuild: bool = False,
    logger: trt.ILogger,
) -> EngineHandle:
    """Open one graph with the same identity and build settings on both inference paths."""
    c = checkpoint.config
    name = "decoder" if decoder else "encoder"
    output_dtype = getattr(trt, PRECISIONS[precision].dtype)
    head_dim = c.audio_state // c.text_heads
    cross_shape = (c.text_layers, 1, c.text_heads, c.audio_positions, head_dim)
    if decoder:
        positions = c.text_positions
        self_shape = (c.text_layers, 1, c.text_heads, -1, head_dim)
        self_range = InputShape.ranged(
            (c.text_layers, 1, c.text_heads, 1, head_dim),
            (c.text_layers, 1, c.text_heads, 64, head_dim),
            (c.text_layers, 1, c.text_heads, positions, head_dim),
        )
        inputs = {
            "tokens": InputShape.fixed((1, 1)),
            "positions": InputShape.fixed((1, 1)),
            "mask": InputShape.ranged((1, 1, 1, 2), (1, 1, 1, 65), (1, 1, 1, positions + 1)),
            "self_keys": self_range,
            "self_values": self_range,
            "cross_keys": InputShape.fixed(cross_shape),
            "cross_values": InputShape.fixed(cross_shape),
            "suppression": InputShape.fixed((1, 1, c.vocab_size)),
        }
        io = EngineIO(
            inputs={
                "tokens": TensorSpec(trt.int32, (1, 1)),
                "positions": TensorSpec(trt.int32, (1, 1)),
                "mask": TensorSpec(trt.float32, (1, 1, 1, -1)),
                "self_keys": TensorSpec(output_dtype, self_shape),
                "self_values": TensorSpec(output_dtype, self_shape),
                "cross_keys": TensorSpec(output_dtype, cross_shape),
                "cross_values": TensorSpec(output_dtype, cross_shape),
                "suppression": TensorSpec(trt.float32, (1, 1, c.vocab_size)),
            },
            outputs={
                "logits": TensorSpec(output_dtype, (1, 1, c.vocab_size)),
                "token": TensorSpec(trt.int32, (1, 1, 1)),
                "greedy": TensorSpec(trt.float32, (1, 1, 3)),
                "present_keys": TensorSpec(output_dtype, self_shape),
                "present_values": TensorSpec(output_dtype, self_shape),
            },
        )
    else:
        shape = (1, c.n_mels, c.audio_positions * 2)
        inputs = {"mel": InputShape.fixed(shape)}
        io = EngineIO(
            inputs={"mel": TensorSpec(trt.float32, shape)},
            outputs={
                "encoder": TensorSpec(output_dtype, (1, c.audio_positions, c.audio_state)),
                "cross_keys": TensorSpec(output_dtype, cross_shape),
                "cross_values": TensorSpec(output_dtype, cross_shape),
            },
        )

    if checkpoint.weight_map is None:
        weight_files = [checkpoint.directory / "model.safetensors"]
    else:
        weight_files = [
            checkpoint.directory / filename
            for filename in sorted(set(checkpoint.weight_map.values()))
        ]
    # The configuration changes graph dimensions and therefore belongs to its cache identity.
    weight_files.insert(0, checkpoint.directory / "config.json")
    if decoder:
        weight_files.append(checkpoint.directory / "tokenizer.json")
    graph_owners: list[_Graph] = []
    scales = None
    graph_version = f"{_GRAPH_VERSION[name]}-{precision}"
    if precision == "fp8":
        from . import _fp8

        scales = json.loads(_fp8._SCALES.read_text())
        if not isinstance(scales, dict):
            raise ValueError("FP8 scales must map linear input names to scalar values")
        weight_files.append(_fp8._SCALES)
        graph_version = f"{_GRAPH_VERSION[name]}-{_fp8._GRAPH_VERSION}"
    plugins = (_fc1_library(),) if decoder and precision == "fp16" else ()

    def define_graph(network: trt.INetworkDefinition) -> None:
        if scales is None:
            graph = _Graph(network, checkpoint, precision)
        else:
            from ._fp8 import _Fp8Graph

            graph = _Fp8Graph(network, checkpoint, scales)
        graph_owners.append(graph)
        if decoder:
            graph.decoder()
        else:
            graph.encoder()

    # Tensoracle's cache identity omits network dtypes, so distinguish precisions here.
    handle = _open_cached_engine(
        define_graph,
        EngineBuild(inputs=inputs, settings=_SETTINGS),
        name=f"listenflow-fastest-whisper-{name}",
        graph_version=graph_version,
        weights=weight_files,
        plugins=plugins,
        io=io,
        force_rebuild=force_rebuild,
        logger=logger,
    )
    return handle
