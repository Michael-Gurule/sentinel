"""Causal cell-averaging CFAR on a pixel time series.

For each frame t the background mean and standard deviation are estimated
from a leading reference window [t - guard - L, t - guard). The guard gap
keeps the start of an event out of its own reference. The statistic is

    z(t) = (x(t) - mean_ref(t)) / std_ref(t),

and the window score is max_t z(t). Under white Gaussian noise with a known
level, z(t) ~ N(0, 1) and the analytic threshold follows from the maximum of
M independent normals. Real backgrounds violate both assumptions (correlated
clutter; σ estimated from L samples gives heavier-than-Gaussian tails), so
thresholds should be calibrated empirically; E1 measures the gap.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from sentinel.core.linalg import FloatArray
from sentinel.detection.base import DetectionScores, as_batch

CFAR_THRESHOLD_PFA_1E2 = 5.27
"""Calibrated CFAR threshold for a 1e-2 false-alarm rate per 64 s window at
10 Hz on glint-free background (E1, ``reports/phase3/e1_detection.json``; a
test keeps them in sync). Glint alarms are left to the classifier."""


def _window_sums(x: FloatArray, start: np.ndarray, stop: np.ndarray) -> FloatArray:
    """Sums of x[:, start[t]:stop[t]] for every t, via cumulative sums."""
    csum = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)
    return np.asarray(csum[:, stop] - csum[:, start])


@dataclass(frozen=True)
class CFARDetector:
    """Causal cell-averaging CFAR.

    Attributes:
        reference_s: Length of the leading reference window, s.
        guard_s: Gap between the reference window and the test frame, s.
        min_reference_s: Frames with less reference history than this are
            not tested.
    """

    reference_s: float = 10.0
    guard_s: float = 1.0
    min_reference_s: float = 2.0
    name: str = "cfar"

    def _frames(self, fs: float) -> tuple[int, int, int]:
        return (
            max(2, round(self.reference_s * fs)),
            round(self.guard_s * fs),
            max(2, round(self.min_reference_s * fs)),
        )

    def standardized(self, signals: FloatArray, fs: float) -> FloatArray:
        """z(t) for every frame; NaN where the reference history is too short."""
        x = as_batch(signals)
        length, guard, minimum = self._frames(fs)
        t = np.arange(x.shape[1])
        stop = np.clip(t - guard, 0, None)
        start = np.clip(stop - length, 0, None)
        count = (stop - start).astype(np.float64)
        valid = count >= minimum
        safe = np.where(valid, count, 1.0)
        mean = _window_sums(x, start, stop) / safe
        mean_sq = _window_sums(x**2, start, stop) / safe
        var = (
            np.clip(mean_sq - mean**2, 0.0, None)
            * safe
            / np.clip(safe - 1.0, 1.0, None)
        )
        z = (x - mean) / np.sqrt(np.maximum(var, 1e-24))
        z[:, ~valid] = np.nan
        return np.asarray(z)

    def tested_frames(self, num_frames: int, fs: float) -> int:
        _, guard, minimum = self._frames(fs)
        return max(1, num_frames - guard - minimum)

    def score(self, signals: FloatArray, fs: float) -> DetectionScores:
        z = self.standardized(signals, fs)
        filled = np.where(np.isnan(z), -np.inf, z)
        onset = np.argmax(filled, axis=1)
        return DetectionScores(
            score=filled[np.arange(z.shape[0]), onset], onset_index=onset
        )

    def analytic_threshold(self, pfa: float, num_frames: int, fs: float) -> float:
        """η with P(max of M iid N(0,1) > η) = pfa, M = tested frames."""
        m = self.tested_frames(num_frames, fs)
        return float(ndtri((1.0 - pfa) ** (1.0 / m)))
