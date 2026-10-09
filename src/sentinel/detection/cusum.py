"""One-sided CUSUM for an upward mean shift.

Runs Page's recursion on the CFAR-standardized residual,

    S(t) = max(0, S(t-1) + z(t) - k),

with drift k = δ/2 tuned for a shift of δ standard deviations. The window
score is max_t S(t). Unlike CFAR, CUSUM accumulates evidence over many frames,
so it detects slow, sustained rises (fires) that a single-frame test misses.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from sentinel.core.linalg import FloatArray
from sentinel.detection.base import DetectionScores
from sentinel.detection.cfar import CFARDetector


def siegmund_arl0(threshold: float, drift: float) -> float:
    """In-control average run length of a CUSUM on N(0, 1) increments.

    Siegmund's approximation with the 1.166 boundary correction:
    ARL0 ≈ (exp(2kb) - 2kb - 1) / (2k²), b = h + 1.166.
    """
    b = threshold + 1.166
    return float((np.exp(2.0 * drift * b) - 2.0 * drift * b - 1.0) / (2.0 * drift**2))


@dataclass(frozen=True)
class CUSUMDetector:
    """CUSUM on a long-reference CFAR residual.

    The long reference window (30 s) with a wide guard (5 s) keeps the
    reference from absorbing a slowly rising event before CUSUM accumulates.
    """

    drift: float = 0.5
    cfar: CFARDetector = field(
        default_factory=lambda: CFARDetector(
            reference_s=30.0, guard_s=5.0, min_reference_s=2.0
        )
    )
    name: str = "cusum"

    def statistic(self, signals: FloatArray, fs: float) -> FloatArray:
        z = np.nan_to_num(self.cfar.standardized(signals, fs), nan=0.0)
        s = np.zeros_like(z)
        running = np.zeros(z.shape[0])
        for t in range(z.shape[1]):
            running = np.maximum(0.0, running + z[:, t] - self.drift)
            s[:, t] = running
        return s

    def score(self, signals: FloatArray, fs: float) -> DetectionScores:
        s = self.statistic(signals, fs)
        peak = np.argmax(s, axis=1)
        # Onset estimate: last zero of the statistic before its peak.
        onset = np.array(
            [
                int(np.flatnonzero(row[: p + 1] == 0.0)[-1]) + 1
                if np.any(row[: p + 1] == 0.0)
                else 0
                for row, p in zip(s, peak, strict=True)
            ]
        )
        return DetectionScores(score=s[np.arange(s.shape[0]), peak], onset_index=onset)

    def analytic_threshold(self, pfa: float, num_frames: int, fs: float) -> float:
        """h with P(alarm within M frames) ≈ 1 - exp(-M / ARL0(h)) = pfa."""
        m = self.cfar.tested_frames(num_frames, fs)
        target_arl = -m / np.log1p(-pfa)
        return float(
            brentq(lambda h: siegmund_arl0(h, self.drift) - target_arl, 0.0, 200.0)
        )
