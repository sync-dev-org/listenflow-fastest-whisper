"""Native ownership survives failed CUDA/TensorRT initialization and repeated close."""

import itertools
import json
import weakref
from pathlib import Path

import numpy as np
import pytest
from tensoracle import EngineHandle
from tokenizers import Tokenizer, models

from listenflow.fastest_whisper import _engines
from listenflow.fastest_whisper._inference import TensorRTWhisper, _HostIO


@pytest.fixture
def native(monkeypatch):
    from cuda.bindings import runtime as cudart

    monkeypatch.setattr(_engines, "_fc1_library", lambda: Path("native-test-fc1.so"))

    streams = set()
    stream_ids = itertools.count(100)
    destroyed = []
    contexts = []
    engines = []
    runtimes = []

    class Context:
        def infer_shapes(self):
            return []

    class Engine:
        num_io_tensors = 0

        def create_execution_context(self):
            context = Context()
            contexts.append(weakref.ref(context))
            return context

    class Runtime:
        pass

    def create_stream():
        stream = next(stream_ids)
        streams.add(stream)
        return cudart.cudaError_t.cudaSuccess, stream

    def destroy_stream(stream):
        assert stream in streams, "a CUDA stream was destroyed twice"
        streams.remove(stream)
        destroyed.append(stream)
        return (cudart.cudaError_t.cudaSuccess,)

    def build(*args, **kwargs):
        engine, runtime = Engine(), Runtime()
        engines.append(weakref.ref(engine))
        runtimes.append(weakref.ref(runtime))
        return EngineHandle(engine, runtime, Path("native-test.engine"))

    monkeypatch.setattr(cudart, "cudaSetDevice", lambda device: (cudart.cudaError_t.cudaSuccess,))
    monkeypatch.setattr(cudart, "cudaStreamCreate", create_stream)
    monkeypatch.setattr(cudart, "cudaStreamDestroy", destroy_stream)
    monkeypatch.setattr(
        cudart, "cudaStreamSynchronize", lambda stream: (cudart.cudaError_t.cudaSuccess,)
    )
    monkeypatch.setattr(_engines, "open_engine", build)
    return cudart, streams, destroyed, contexts, engines, runtimes, Engine


def _checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "num_mel_bins": 128,
                "max_source_positions": 1500,
                "d_model": 4,
                "encoder_attention_heads": 1,
                "encoder_ffn_dim": 4,
                "encoder_layers": 1,
                "decoder_attention_heads": 1,
                "decoder_ffn_dim": 4,
                "decoder_layers": 1,
                "max_target_positions": 448,
                "vocab_size": 12,
            }
        )
    )
    tokenizer = Tokenizer(
        models.WordLevel(
            vocab={
                "[UNK]": 0,
                "<|startoftranscript|>": 1,
                "<|en|>": 2,
                "<|transcribe|>": 3,
                "<|notimestamps|>": 4,
                "<|endoftext|>": 5,
                "<|startofprev|>": 6,
                "<|startoflm|>": 7,
                "<|nospeech|>": 8,
                "<|translate|>": 9,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    return tmp_path


@pytest.mark.req("SP-LFFASTESTWHISPER-005")
def test_latency_model_keeps_saved_engines_and_times_opening_each_graph(
    tmp_path, native, monkeypatch
):
    """Repeated benchmark loads keep the same engine files and report all three opening times."""
    from scripts.benchmark_latency import _TimedCachedWhisper

    original = _engines.open_engine
    opened = []

    def cached_native(*args, name, force_rebuild, **kwargs):
        path = tmp_path / f"{name}.engine"
        if not path.exists() or force_rebuild:
            path.write_bytes(f"build-{len(opened)}".encode())
        opened.append((name, force_rebuild))
        handle = original(*args, name=name, force_rebuild=force_rebuild, **kwargs)
        return EngineHandle(handle.engine, handle.runtime, path)

    monkeypatch.setattr(_engines, "open_engine", cached_native)
    checkpoint = _checkpoint(tmp_path)
    snapshots = []
    for _ in range(2):
        with _TimedCachedWhisper(checkpoint, precision="fp16", device=0) as model:
            assert set(model.build_seconds) == {"encoder", "decoder", "mel"}
            assert all(value >= 0 for value in model.build_seconds.values())
            snapshots.append({path.name: path.read_bytes() for path in tmp_path.glob("*.engine")})

    assert len(snapshots[0]) == 3
    assert snapshots[0] == snapshots[1]
    assert len(opened) == 6
    assert all(force_rebuild is False for _, force_rebuild in opened)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_native_close_drops_contexts_engines_and_runtimes_once(tmp_path, native):
    """Closing an owned runtime returns all streams and releases all native owners once."""
    _, streams, destroyed, contexts, engines, runtimes, _ = native
    model = TensorRTWhisper(_checkpoint(tmp_path))
    try:
        assert len(streams) == 3
    finally:
        model.close()
    assert not streams
    assert len(destroyed) == 3
    assert all(reference() is None for reference in contexts + engines + runtimes)
    model.close()
    assert len(destroyed) == 3


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_closed_model_rejects_further_inference(tmp_path, native):
    """An explicitly closed model reports an error before another native operation."""
    model = TensorRTWhisper(_checkpoint(tmp_path))
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.encode(np.zeros((128, 3000), dtype=np.float32))
    with pytest.raises(RuntimeError, match="closed"):
        model.logits([1], np.zeros((1500, 4), dtype=np.float32))


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_partial_execution_creation_releases_context_even_while_error_is_retained(
    native, monkeypatch
):
    """A failed CUDA stream creation drops its TensorRT context before returning the error."""
    cudart, streams, _, contexts, _, _, Engine = native
    monkeypatch.setattr(
        cudart, "cudaStreamCreate", lambda: (cudart.cudaError_t.cudaErrorMemoryAllocation, None)
    )
    with pytest.raises(RuntimeError, match="cudaStreamCreate") as failure:
        _HostIO(EngineHandle(Engine(), object(), Path("native-test.engine")))
    assert failure.value.__traceback__ is not None
    assert not streams
    assert contexts[0]() is None


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_partial_model_creation_returns_prior_streams_and_engines(tmp_path, native, monkeypatch):
    """A decoder initialization failure releases the encoder even while its error is retained."""
    cudart, streams, destroyed, contexts, engines, runtimes, _ = native
    create = cudart.cudaStreamCreate
    calls = 0

    def fail_second_stream():
        nonlocal calls
        calls += 1
        if calls == 2:
            return cudart.cudaError_t.cudaErrorMemoryAllocation, None
        return create()

    monkeypatch.setattr(cudart, "cudaStreamCreate", fail_second_stream)
    with pytest.raises(RuntimeError, match="cudaStreamCreate") as failure:
        TensorRTWhisper(_checkpoint(tmp_path))
    assert failure.value.__traceback__ is not None
    assert not streams
    assert len(destroyed) == 1
    assert all(reference() is None for reference in contexts + engines + runtimes)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_second_engine_build_failure_drops_first_engine(tmp_path, native, monkeypatch):
    """A decoder build error releases the already opened encoder and its runtime."""
    _, streams, _, _, engines, runtimes, _ = native
    build = _engines.open_engine
    cause = RuntimeError("decoder unavailable")

    def fail_decoder(*args, name, **kwargs):
        if name.endswith("-decoder"):
            raise cause
        return build(*args, name=name, **kwargs)

    monkeypatch.setattr(_engines, "open_engine", fail_decoder)
    with pytest.raises(RuntimeError) as failure:
        TensorRTWhisper(_checkpoint(tmp_path))
    assert failure.value is cause
    assert not streams
    assert all(reference() is None for reference in engines + runtimes)
