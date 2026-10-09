"""Common detector interface, scores, and threshold calibration."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from locant.core.linalg import FloatArray


@dataclass(frozen=True, eq=False)
class DetectionScores:
    """Per-window detection statistics.

    Attributes:
        score: Scalar test statistic per window (N,); larger means more
            evidence of an event.
        onset_index: Frame at which the statistic peaked (N,), an onset
            estimate. Meaningful only for windows declared detections.
    """

    score: FloatArray
    onset_index: np.ndarray


class Detector(Protocol):
    """A window-level event detector."""

    @property
    def name(self) -> str:
        """Short identifier used in reports."""
        ...

    def score(self, signals: FloatArray, fs: float) -> DetectionScores:
        """Score a batch of pixel time series of shape (N, T) sampled at ``fs`` Hz."""
        ...

    def analytic_threshold(self, pfa: float, num_frames: int, fs: float) -> float:
        """Threshold giving window false-alarm probability ``pfa`` under the
        detector's noise model (white Gaussian; see each detector)."""
        ...


def as_batch(signals: FloatArray) -> FloatArray:
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("signals must have shape (T,) or (N, T)")
    return x


def calibrate_threshold(background_scores: FloatArray, pfa: float) -> float:
    """Empirical threshold with window false-alarm rate ``pfa`` on background data.

    Returns the smallest score exceeded by at most ``pfa`` of the background
    windows. Needs on the order of 10 / ``pfa`` windows for a stable estimate.

    Raises:
        ValueError: if there are too few windows to resolve ``pfa``.
    """
    scores = np.sort(np.asarray(background_scores, dtype=np.float64))
    if not 0.0 < pfa < 1.0:
        raise ValueError("pfa must be in (0, 1)")
    if scores.size * pfa < 1.0:
        raise ValueError(
            f"{scores.size} background windows cannot resolve pfa={pfa}; "
            f"need at least {int(np.ceil(1.0 / pfa))}"
        )
    allowed = int(np.floor(pfa * scores.size))  # windows allowed above threshold
    return float(np.nextafter(scores[scores.size - allowed - 1], np.inf))
