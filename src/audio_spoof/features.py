"""Dependency-light acoustic feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


FEATURE_NAMES = [
    "duration_s",
    "rms_mean",
    "rms_std",
    "zcr_mean",
    "zcr_std",
    "centroid_mean",
    "centroid_std",
    "rolloff_mean",
    "rolloff_std",
    "flatness_mean",
    "flatness_std",
    "high_low_ratio_mean",
    "high_low_ratio_std",
]


def load_audio(path: str | Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load mono WAV audio, normalize it, and resample when required."""
    sample_rate, signal = wavfile.read(path)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if np.issubdtype(signal.dtype, np.integer):
        limit = max(abs(np.iinfo(signal.dtype).min), np.iinfo(signal.dtype).max)
        signal = signal.astype(np.float32) / limit
    else:
        signal = signal.astype(np.float32)
    if sample_rate != target_sr:
        divisor = np.gcd(sample_rate, target_sr)
        signal = resample_poly(signal, target_sr // divisor, sample_rate // divisor)
        sample_rate = target_sr
    peak = np.max(np.abs(signal)) if signal.size else 0.0
    if peak > 1.0:
        signal = signal / peak
    return signal, sample_rate


def _frames(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size))
    count = 1 + (signal.size - frame_length) // hop_length
    shape = (count, frame_length)
    strides = (signal.strides[0] * hop_length, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides).copy()


def extract_features(
    signal: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_fft: int = 512,
) -> np.ndarray:
    """Return summary acoustic features for one waveform."""
    frame_length = int(sample_rate * frame_ms / 1000)
    hop_length = int(sample_rate * hop_ms / 1000)
    framed = _frames(np.asarray(signal, dtype=np.float32), frame_length, hop_length)
    windowed = framed * np.hanning(frame_length)

    rms = np.sqrt(np.mean(framed**2, axis=1) + 1e-12)
    zcr = np.mean(np.diff(np.signbit(framed), axis=1), axis=1)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft, axis=1)) + 1e-12
    power = spectrum**2
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    centroid = np.sum(power * frequencies, axis=1) / np.sum(power, axis=1)

    cumulative = np.cumsum(power, axis=1)
    thresholds = 0.85 * cumulative[:, -1]
    rolloff_bins = np.argmax(cumulative >= thresholds[:, None], axis=1)
    rolloff = frequencies[rolloff_bins]

    flatness = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
    split = np.searchsorted(frequencies, 4_000)
    low = np.sum(power[:, :split], axis=1) + 1e-12
    high_low_ratio = np.sum(power[:, split:], axis=1) / low

    values = [signal.size / sample_rate]
    for statistic in (rms, zcr, centroid, rolloff, flatness, high_low_ratio):
        values.extend((float(np.mean(statistic)), float(np.std(statistic))))
    return np.asarray(values, dtype=np.float64)


def features_from_file(path: str | Path, sample_rate: int = 16_000) -> np.ndarray:
    signal, actual_rate = load_audio(path, sample_rate)
    return extract_features(signal, actual_rate)

