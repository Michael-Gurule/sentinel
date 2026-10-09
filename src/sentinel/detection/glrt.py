"""Generalized likelihood ratio test for a step increase at an unknown frame.

Under H0 the window is a constant level plus white Gaussian noise; under H1
the level steps up by an unknown amount at an unknown frame k. Maximizing the
likelihood over the step size gives, for each candidate k,

    T(k) = (mean(x[k:]) - mean(x[:k])) / (σ · sqrt(1/k + 1/(T - k))),

and the score is max_k T(k). σ is estimated robustly from first differences
(insensitive to the step itself). The white-noise model ignores correlated
clutter, which E1 shows inflates false alarms unless the threshold is
calibrated.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from sentinel.core.linalg import FloatArray
from sentinel.detection.base import DetectionScores, as_batch

MAD_TO_SIGMA = 1.482602218505602


def noise_sigma_from_differences(x: FloatArray) -> FloatArray:
    """Robust white-noise σ per row: 1.4826 · MAD(Δx) / √2."""
    diffs = np.diff(as_batch(x), axis=1)
    mad = np.median(np.abs(diffs - np.median(diffs, axis=1, keepdims=True)), axis=1)
    return np.asarray(MAD_TO_SIGMA * mad / np.sqrt(2.0))


@dataclass(frozen=True)
class StepGLRTDetector:
    """Step GLRT; candidate onsets keep ``margin_s`` of data on each side."""

    margin_s: float = 2.0
    name: str = "glrt"

    def score(self, signals: FloatArray, fs: float) -> DetectionScores:
        x = as_batch(signals)
        n = x.shape[1]
        margin = max(1, round(self.margin_s * fs))
        k = np.arange(margin, n - margin)
        csum = np.cumsum(x, axis=1)
        before = csum[:, k - 1] / k
        after = (csum[:, -1:] - csum[:, k - 1]) / (n - k)
        sigma = np.maximum(noise_sigma_from_differences(x), 1e-12)[:, None]
        stat = (after - before) / (sigma * np.sqrt(1.0 / k + 1.0 / (n - k)))
        best = np.argmax(stat, axis=1)
        return DetectionScores(
            score=stat[np.arange(x.shape[0]), best], onset_index=k[best]
        )

    def analytic_threshold(self, pfa: float, num_frames: int, fs: float) -> float:
        """Bonferroni over candidate onsets: η = Φ⁻¹(1 - pfa / M) (conservative)."""
        margin = max(1, round(self.margin_s * fs))
        m = max(1, num_frames - 2 * margin)
        return float(ndtri(1.0 - pfa / m))
