"""The engine declares both TensorRT graphs to the persistent cache."""

import gc
import weakref
from types import SimpleNamespace

import pytest

from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._checkpoint import ModelConfig


@pytest.mark.req("SP-LFFASTESTWHISPER-008")
def test_encoder_and_decoder_use_tensoracle_with_explicit_rebuild(tmp_path, monkeypatch):
    """Both graph identities include configuration and weights and forward force_rebuild."""
    from listenflow.fastest_whisper import _engines

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
        return SimpleNamespace(engine=object(), runtime=object())

    monkeypatch.setattr(_engines, "open_engine", open_stub)
    config = ModelConfig(128, 1500, 1280, 20, 5120, 32, 20, 5120, 32, 448, 51866)
    checkpoint = SimpleNamespace(directory=tmp_path, config=config, weight_map=None)
    model = object.__new__(CachedWhisper)
    model.precision = "fp32"
    model.force_rebuild = True
    model.logger = object()

    model._open_engine(checkpoint, decoder=False)
    model._open_engine(checkpoint, decoder=True)

    assert [call[1]["name"] for call in calls] == [
        "listenflow-fastest-whisper-encoder",
        "listenflow-fastest-whisper-decoder",
    ]
    original_versions = _engines._GRAPH_VERSION
    assert [call[1]["graph_version"] for call in calls] == [
        original_versions["encoder"],
        original_versions["decoder"],
    ]
    changed_versions = {"encoder": "encoder-next", "decoder": "decoder-next"}
    monkeypatch.setattr(_engines, "_GRAPH_VERSION", changed_versions)
    model._open_engine(checkpoint, decoder=False)
    model._open_engine(checkpoint, decoder=True)
    assert [call[1]["graph_version"] for call in calls[-2:]] == [
        changed_versions["encoder"],
        changed_versions["decoder"],
    ]
    assert all(call[1]["force_rebuild"] is True for call in calls)
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
