"""Native host outputs retain validation and independent ownership."""

import numpy as np
import pytest

from listenflow.fastest_whisper import _native
from native_runtime_fixture import encoded

pytestmark = [pytest.mark.gpu]


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
@pytest.mark.parametrize("shape", [(80, 2999), (79, 3000), (1, 80, 3000)])
def test_exchange_rejects_incompatible_input(native_model, shape):
    """Feature shapes are rejected before entering encoder execution."""
    with pytest.raises((TypeError, ValueError)):
        native_model._native_model.encode_features(np.zeros(shape, np.float32))


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_exchange_copies_strided_host_inputs(native_model):
    """The Python adapter normalizes strided features without changing their values."""
    features = np.zeros((80, 6000), np.float32)[:, ::2]
    expected = np.asarray(native_model.encode(features.copy())).copy()
    actual = np.asarray(native_model.encode(features))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.req("SP-LFFASTESTWHISPER-002")
def test_host_results_keep_values_after_another_exchange_and_close(native_model):
    """Returned encoder arrays are independent of native staging and model lifetime."""
    first = np.asarray(encoded(native_model))
    expected = first.copy()
    native_model.encode(np.ones((80, 3000), np.float32))
    native_model.close()
    np.testing.assert_array_equal(first, expected)


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
def test_closed_model_rejects_retained_child_inference(native_model):
    """Model close invalidates a retained encoded handle and still permits its release."""
    child = encoded(native_model)
    native_model._encoded_audio = None
    native_model.close()
    with pytest.raises(_native.NativeError) as failure:
        child.numpy()
    assert failure.value.status == 5
    child.close()


@pytest.mark.req("SP-LFFASTESTWHISPER-010")
@pytest.mark.parametrize(
    "bad",
    [np.zeros((2, 2), np.float32), np.array([np.nan], np.float32), np.zeros(480001, np.float32)],
)
def test_mel_rejects_invalid_waveforms_and_can_retry(native_model, bad):
    """Malformed audio does not poison the model's next completed mel exchange."""
    with pytest.raises((TypeError, ValueError)):
        native_model._mel_spectrogram(bad)
    actual = np.asarray(native_model._mel_spectrogram(np.zeros(160, np.float32)))
    assert actual.shape == (1, 80, 3000)
    assert np.isfinite(actual).all()
