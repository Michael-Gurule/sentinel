"""Preprocessing shared by training, evaluation, and inference.

One function, parameterized by a :class:`PreprocessSpec` that is stored in every
model artifact, so the transform applied at inference is exactly the one used
in training.

Normalizations:

* ``noise_asinh`` (default): subtract a low-percentile baseline, divide by
  the white-noise σ estimated from first differences, then apply ``asinh``.
  Values are in units of noise σ (amplitude carries SNR information), and
  asinh compresses the 1000× dynamic range of explosions while staying
  linear near zero.
* ``zscore``: per-window mean/std, the v1 choice. It discards absolute
  amplitude, so a faint fire and a bright launch with the same shape look
  alike; kept for the E2 ablation.
"""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from locant.core.linalg import FloatArray
from locant.detection.glrt import noise_sigma_from_differences

BASELINE_PERCENTILE = 10.0
"""Percentile used as the background level: robust to positive transients
(events, glints) covering up to ~90% of the window."""


class PreprocessSpec(BaseModel):
    """How raw pixel windows become model inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    normalization: Literal["noise_asinh", "zscore"] = "noise_asinh"
    length: int | None = Field(None, ge=8)
    """Resample to this many frames; ``None`` keeps the native length."""


def noise_normalize(signals: FloatArray) -> FloatArray:
    """(x - baseline) / σ_noise per row, in units of white-noise σ."""
    x = np.atleast_2d(np.asarray(signals, dtype=np.float64))
    baseline = np.percentile(x, BASELINE_PERCENTILE, axis=1, keepdims=True)
    sigma = np.maximum(noise_sigma_from_differences(x), 1e-9)[:, None]
    return np.asarray((x - baseline) / sigma)


def _resample(x: FloatArray, length: int) -> FloatArray:
    if x.shape[1] == length:
        return x
    old = np.linspace(0.0, 1.0, x.shape[1])
    new = np.linspace(0.0, 1.0, length)
    return np.asarray(np.stack([np.interp(new, old, row) for row in x]))


def preprocess(signals: FloatArray, spec: PreprocessSpec) -> np.ndarray:
    """Raw (N, T) or (T,) irradiance → float32 model input (N, L)."""
    x = np.atleast_2d(np.asarray(signals, dtype=np.float64))
    if x.shape[1] < 2:
        raise ValueError("signals need at least 2 frames")
    if spec.normalization == "noise_asinh":
        x = np.arcsinh(noise_normalize(x))
    else:
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
    if spec.length is not None:
        x = _resample(x, spec.length)
    return x.astype(np.float32)
