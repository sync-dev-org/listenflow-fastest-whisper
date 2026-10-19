"""Saved TensorRT graphs are reused or rebuilt without caller configuration."""

import gc
import weakref
from types import SimpleNamespace

import pytest
from tensoracle import CorruptEngineError, EngineConditionError, EngineIOError

from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._checkpoint import ModelConfig


@pytest.mark.req("SP-LFFASTESTWHISPER-008")
def test_encoder_and_decoder_keep_cache_identity_without_caller_rebuild(tmp_path, monkeypatch):
    """Graph identities separate precisions and graph versions while retaining saved engines."""
    from listenflow.fastest_whisper import _engines

    monkeypatch.setattr(_engines, "_fc1_library", lambda: tmp_path / "fc1.so")

    calls = []
    graph_refs = []

    class FakeGraph:
        def __init__(self, network, checkpoint, precision):
            graph_refs.append(weakref.ref(self))

        def encoder(self):
            pass

        def decoder(self):
            pass

    monkeypatch.setattr(_engines, "_Graph", FakeGraph)

    def open_stub(graph, declaration, **kwargs):
        graph(None)
        gc.collect()
        assert graph_refs[-1]() is not None
        calls.append((declaration, kwargs))
        return SimpleNamespace(
            engine=object(),
            runtime=object(),
            create_execution=lambda: SimpleNamespace(close=lambda: None),
        )

    monkeypatch.setattr(_engines, "open_engine", open_stub)
    config = ModelConfig(128, 1500, 1280, 20, 5120, 32, 20, 5120, 32, 448, 51866)
    checkpoint = SimpleNamespace(directory=tmp_path, config=config, weight_map=None)
    model = object.__new__(CachedWhisper)
    model.precision = "fp32"
    model.logger = object()

    model._open_engine(checkpoint, decoder=False)
    model._open_engine(checkpoint, decoder=True)

    assert [call[1]["name"] for call in calls] == [
        "listenflow-fastest-whisper-encoder",
        "listenflow-fastest-whisper-decoder",
    ]
    original_versions = _engines._GRAPH_VERSION
    changed_versions = {"encoder": "encoder-next", "decoder": "decoder-next"}
    monkeypatch.setattr(_engines, "_GRAPH_VERSION", changed_versions)
    model._open_engine(checkpoint, decoder=False)
    model._open_engine(checkpoint, decoder=True)
    assert all(
        original[1]["graph_version"] != changed[1]["graph_version"]
        for original, changed in zip(calls[:2], calls[-2:], strict=True)
    )
    assert all(not call[1].get("force_rebuild", False) for call in calls)
    assert all(
        call[1]["weights"]
        == [tmp_path / "config.json", tmp_path / "model.safetensors"]
        + ([tmp_path / "tokenizer.json"] if index % 2 else [])
        for index, call in enumerate(calls)
    )
    assert calls[0][0].inputs["mel"].minimum == (1, 128, 3000)
    assert calls[1][0].inputs["tokens"].maximum == (1, 1)
    assert calls[1][0].inputs["self_keys"].minimum == (32, 1, 20, 1, 64)
    assert calls[1][0].inputs["self_keys"].maximum == (32, 1, 20, 448, 64)
    assert calls[1][0].inputs["cross_values"].maximum == (32, 1, 20, 1500, 64)
    assert set(calls[0][1]["io"].outputs) == {"encoder", "cross_keys", "cross_values"}
    assert calls[1][1]["io"].outputs["logits"].shape == (1, 1, 51866)
    assert calls[1][1]["io"].outputs["greedy"].shape == (1, 1, 3)

    monkeypatch.setattr(_engines, "_GRAPH_VERSION", original_versions)
    model.precision = "fp16"
    model._open_engine(checkpoint, decoder=False)
    model._open_engine(checkpoint, decoder=True)
    for fp32, fp16 in zip(calls[:2], calls[-2:], strict=True):
        assert (fp32[1]["name"], fp32[1]["graph_version"]) != (
            fp16[1]["name"],
            fp16[1]["graph_version"],
        )


@pytest.mark.req("SP-LFFASTESTWHISPER-008")
@pytest.mark.parametrize("graph", ["mel", "encoder", "decoder"])
@pytest.mark.parametrize("error_type", [CorruptEngineError, EngineConditionError, EngineIOError])
def test_unusable_saved_graph_is_rebuilt_once(tmp_path, monkeypatch, graph, error_type):
    """Each damaged or incompatible graph is rebuilt while keeping its cache identity."""
    from listenflow.fastest_whisper import _engines

    calls = []
    handle = SimpleNamespace(create_execution=lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(_engines, "_fc1_library", lambda: tmp_path / "fc1.so")

    def open_native(*args, **kwargs):
        calls.append((args, kwargs))
        if not kwargs.get("force_rebuild", False):
            raise error_type("saved graph cannot be opened")
        return handle

    monkeypatch.setattr(_engines, "open_engine", open_native)
    config = ModelConfig(128, 1500, 1280, 20, 5120, 32, 20, 5120, 32, 448, 51866)
    checkpoint = SimpleNamespace(directory=tmp_path, config=config, weight_map=None)
    model = object.__new__(CachedWhisper)
    model.precision = "fp16"
    model.logger = object()
    if graph == "mel":
        actual = model._open_mel(checkpoint)
    else:
        actual = model._open_engine(checkpoint, decoder=graph == "decoder")

    assert actual is handle
    assert len(calls) == 2
    assert not calls[0][1].get("force_rebuild", False)
    retry_args, retry_kwargs = calls[1]
    assert retry_kwargs.pop("force_rebuild") is True
    original_args, original_kwargs = calls[0]
    original_kwargs.pop("force_rebuild", False)
    assert (retry_args, retry_kwargs) == (original_args, original_kwargs)


@pytest.mark.req("SP-LFFASTESTWHISPER-008")
@pytest.mark.parametrize(
    "failure",
    [CorruptEngineError("rebuilt graph is invalid"), RuntimeError("building graph failed")],
)
def test_failed_rebuild_is_not_retried_again(tmp_path, monkeypatch, failure):
    """A rebuild failure escapes unchanged after one attempt rather than looping."""
    from listenflow.fastest_whisper import _engines

    calls = []

    def open_native(*args, **kwargs):
        calls.append(kwargs.get("force_rebuild", False))
        if calls[-1]:
            raise failure
        raise CorruptEngineError("saved graph is invalid")

    monkeypatch.setattr(_engines, "open_engine", open_native)
    checkpoint = SimpleNamespace(directory=tmp_path, config=SimpleNamespace(n_mels=128))
    model = object.__new__(CachedWhisper)
    model.precision = "fp16"
    model.logger = object()
    with pytest.raises(type(failure)) as caught:
        model._open_mel(checkpoint)
    assert caught.value is failure
    assert calls == [False, True]
