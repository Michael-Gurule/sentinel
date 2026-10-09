"""Handcrafted temporal features for the classical baselines.

All features are computed on noise-normalized signals (units of noise σ), so
they are comparable across sensors and scenes. Each captures phenomenology
an analyst would look at: brightness, onset sharpness, persistence, decay,
and roughness.
"""

import numpy as np
from scipy.stats import kurtosis, skew

from sentinel.classification.preprocess import noise_normalize
from sentinel.core.linalg import FloatArray

FEATURE_NAMES: tuple[str, ...] = (
    "log_peak",
    "peak_position",
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "fraction_above_3sigma",
    "fraction_above_half_peak",
    "rise_frames_10_90",
    "decay_frames_to_half",
    "late_minus_early",
    "log_max_rise",
    "log_max_fall",
    "half_peak_crossings",
    "low_frequency_power_ratio",
    "lag1_autocorrelation",
    "final_over_peak",
    "log_area",
)


def _signed_log(x: FloatArray) -> FloatArray:
    return np.sign(x) * np.log1p(np.abs(x))


def extract_features(signals: FloatArray) -> np.ndarray:
    """Feature matrix (N, len(FEATURE_NAMES)) from raw irradiance windows."""
    x = noise_normalize(signals)
    n, t = x.shape
    idx = np.arange(n)
    peak_pos = np.argmax(x, axis=1)
    peak = x[idx, peak_pos]
    safe_peak = np.where(np.abs(peak) > 1e-9, peak, 1e-9)
    frames = np.arange(t)[None, :]

    above_10 = x >= 0.1 * peak[:, None]
    above_90 = x >= 0.9 * peak[:, None]
    before_peak = frames <= peak_pos[:, None]
    first_10 = np.argmax(above_10 & before_peak, axis=1)
    first_90 = np.argmax(above_90 & before_peak, axis=1)
    after_peak = frames >= peak_pos[:, None]
    below_half_after = (x < 0.5 * peak[:, None]) & after_peak
    decay = np.where(
        below_half_after.any(axis=1),
        np.argmax(below_half_after, axis=1) - peak_pos,
        t - peak_pos,
    )
    quarter = max(1, t // 4)
    diffs = np.diff(x, axis=1)
    half = x >= 0.5 * peak[:, None]
    centered = x - x.mean(axis=1, keepdims=True)
    lag1 = np.sum(centered[:, 1:] * centered[:, :-1], axis=1) / np.maximum(
        np.sum(centered**2, axis=1), 1e-12
    )
    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    cut = max(1, spectrum.shape[1] // 20)
    low_ratio = spectrum[:, 1 : cut + 1].sum(axis=1) / np.maximum(
        spectrum[:, 1:].sum(axis=1), 1e-12
    )

    features = np.column_stack(
        [
            _signed_log(peak),
            peak_pos / t,
            x.mean(axis=1),
            x.std(axis=1),
            np.nan_to_num(skew(x, axis=1)),
            np.nan_to_num(kurtosis(x, axis=1)),
            (x > 3.0).mean(axis=1),
            half.mean(axis=1),
            np.clip(first_90 - first_10, 0, None),
            decay,
            x[:, -quarter:].mean(axis=1) - x[:, :quarter].mean(axis=1),
            _signed_log(diffs.max(axis=1)),
            _signed_log(-diffs.min(axis=1)),
            np.sum(np.diff(half.astype(np.int8), axis=1) == 1, axis=1),
            low_ratio,
            lag1,
            x[:, -1] / safe_peak,
            _signed_log(np.clip(x, 0.0, None).sum(axis=1)),
        ]
    )
    return np.asarray(np.nan_to_num(features), dtype=np.float64)
