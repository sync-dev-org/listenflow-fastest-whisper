"""CPU preparation of 16 kHz mono audio for a 30-second Whisper encoder input."""

from pathlib import Path

import numpy as np

N_SAMPLES = 480_000
FFT_SIZE = 400
HOP_SIZE = 160
_FILTERS = Path(__file__).parent / "assets" / "mel_filters.npz"


def log_mel_spectrogram(audio: np.ndarray, *, n_mels: int) -> np.ndarray:
    """Compute the 30-second fp32 log-mel input used by the official model."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or samples.size > N_SAMPLES or not np.isfinite(samples).all():
        raise ValueError("audio must be a finite mono waveform of at most 30 seconds")
    samples = np.pad(samples, (0, N_SAMPLES - samples.size))
    centered = np.pad(samples, (FFT_SIZE // 2, FFT_SIZE // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(centered, FFT_SIZE)[::HOP_SIZE]
    window = np.hanning(FFT_SIZE + 1)[:-1].astype(np.float32)
    fft = np.fft.rfft(frames * window, axis=-1)[:-1].astype(np.complex64)
    power = (fft.real**2 + fft.imag**2).T
    with np.load(_FILTERS) as filters:
        key = f"mel_{n_mels}"
        if key not in filters:
            raise ValueError(f"unsupported mel filter count: {n_mels}")
        mel = filters[key] @ power
    log_mel = np.log10(np.maximum(mel, np.float32(1e-10)))
    log_mel = np.maximum(log_mel, log_mel.max() - np.float32(8.0))
    return ((log_mel + np.float32(4.0)) / np.float32(4.0)).astype(np.float32)


def log_mel_spectrogram_long(audio: np.ndarray, *, n_mels: int) -> np.ndarray:
    """Compute a whole recording's mel before slicing it into 30-second windows."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("audio must be a finite mono waveform")
    padded = np.pad(samples, (0, N_SAMPLES))
    centered = np.pad(padded, (FFT_SIZE // 2, FFT_SIZE // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(centered, FFT_SIZE)[::HOP_SIZE][:-1]
    window = np.hanning(FFT_SIZE + 1)[:-1].astype(np.float32)
    with np.load(_FILTERS) as filters:
        key = f"mel_{n_mels}"
        if key not in filters:
            raise ValueError(f"unsupported mel filter count: {n_mels}")
        filterbank = filters[key]
    log_mel = np.empty((n_mels, len(frames)), dtype=np.float32)
    for start in range(0, len(frames), 2_048):
        fft = np.fft.rfft(frames[start : start + 2_048] * window, axis=-1).astype(np.complex64)
        power = (fft.real**2 + fft.imag**2).T
        mel = filterbank @ power
        log_mel[:, start : start + mel.shape[1]] = np.log10(np.maximum(mel, np.float32(1e-10)))
    log_mel = np.maximum(log_mel, log_mel.max() - np.float32(8.0))
    return ((log_mel + np.float32(4.0)) / np.float32(4.0)).astype(np.float32)
