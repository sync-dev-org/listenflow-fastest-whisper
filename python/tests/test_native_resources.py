"""Python model ownership and selected sampling options cross the native boundary once."""

import numpy as np
import pytest

from listenflow import OfflineModelDirectorySource
from listenflow.fastest_whisper import FastestWhisperEngine, _inference
from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._decoding import decoding_settings
from native_runtime_fixture import encoded


@pytest.fixture
def native_boundary(monkeypatch):
    calls, closed = [], []

    class Model:
        def __init__(self, source, precision, device, batch, suppressed, begin):
            calls.append((source, precision, device, batch, suppressed, begin))
            self.info = {"open_seconds": {name: 0.0 for name in ("mel", "encoder", "decoder")}}
            if batch:
                self.info["open_seconds"]["sampled_decoder"] = 0.0

        def close(self):
            closed.append(True)

    monkeypatch.setattr(_inference._native, "Model", Model)
    return calls, closed


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
@pytest.mark.parametrize(
    "temperature,best_of,expected", [(0.0, 5, 0), ((0.0, 0.4), 2, 2), (0.4, 1, 0)]
)
def test_only_selected_sampled_batches_prepare_an_extra_engine(
    source_checkpoint, native_boundary, temperature, best_of, expected
):
    """Only positive-temperature multiple candidates request a native sampled engine."""
    calls, closed = native_boundary
    settings = decoding_settings(temperature=temperature, best_of=best_of)
    model = CachedWhisper(source_checkpoint, settings=settings)
    assert calls[0][3] == expected
    assert (model._sampled_decoder is not None) is bool(expected)
    assert set(model.build_seconds) == {"mel", "encoder", "decoder"} | (
        {"sampled_decoder"} if expected else set()
    )
    model.close()
    model.close()
    assert closed == [True]


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_product_preparation_supplies_selected_sampling_configuration(
    source_checkpoint, native_boundary
):
    """The public constructor passes the selected decoding batch into native model creation."""
    calls, _ = native_boundary
    engine = FastestWhisperEngine(
        model_source=OfflineModelDirectorySource(str(source_checkpoint)),
        precision="fp32",
        device=0,
        language="en",
        temperature=(0.0, 0.4),
        best_of=2,
    )
    model = engine._load_model()
    assert calls[0][3] == 2
    assert calls[0][1:3] == ("fp32", 0)
    model.close()


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_greedy_suppression_lists_are_provided_once_at_model_creation(
    source_checkpoint, native_boundary
):
    """Normal and additional beginning-token lists are native model inputs, not step arrays."""
    calls, _ = native_boundary
    with CachedWhisper(source_checkpoint) as model:
        assert calls[0][4] == sorted(model.suppressed)
        assert calls[0][5] == sorted(model.begin_suppressed)
        assert len(calls) == 1
        assert model.eot in calls[0][5]


@pytest.mark.req("SP-LFFASTESTWHISPER-019")
def test_closed_model_rejects_further_inference(source_checkpoint, native_boundary):
    """A closed Python wrapper never calls the released native model again."""
    model = CachedWhisper(source_checkpoint)
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.encode(np.zeros((80, 3000), np.float32))


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_missing_fc1_stops_fp16_before_native_creation(
    source_checkpoint, native_boundary, monkeypatch
):
    """A missing selected plugin fails without silently opening another precision."""
    calls, _ = native_boundary

    def missing():
        raise FileNotFoundError("required fc1 plugin is missing")

    monkeypatch.setattr(_inference, "_fc1_library", missing)
    with pytest.raises(FileNotFoundError, match="fc1"):
        CachedWhisper(source_checkpoint, precision="fp16")
    assert calls == []


@pytest.mark.gpu
@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_native_close_invalidates_children_and_is_idempotent(native_model):
    """Native close releases a model even while Python retains its decoder children."""
    audio = encoded(native_model)
    decoder = audio.decoder(_inference._native.GREEDY)
    native_model.close()
    native_model.close()
    with pytest.raises(_inference._native.NativeError) as failure:
        decoder.greedy_step(4, True, False)
    assert failure.value.status == 5
    decoder.close()
