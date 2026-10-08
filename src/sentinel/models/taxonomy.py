"""Single source of truth for the OPIR event taxonomy and model input contract.

Training, evaluation, inference, and the pipeline all import these names, so
class order and input length cannot drift apart (audit defect C4).
"""

from typing import Final

import numpy as np

from sentinel.core.linalg import FloatArray

EVENT_CLASSES: Final[tuple[str, ...]] = (
    "launch",
    "explosion",
    "fire",
    "aircraft",
    "background",
)
"""Class names in model output order."""

INPUT_LENGTH: Final[int] = 100
"""Number of samples the classifier consumes."""


def preprocess_signal(signal: FloatArray, length: int = INPUT_LENGTH) -> FloatArray:
    """Linearly resample a 1-D series to ``length`` and z-score normalize it.

    Resampling happens first so the model input is exactly zero-mean and
    unit-variance. The same function is used for training and inference so
    preprocessing cannot differ between them.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("signal must be a 1-D array with at least 2 samples")
    if x.size != length:
        x = np.interp(np.linspace(0.0, 1.0, length), np.linspace(0.0, 1.0, x.size), x)
    return np.asarray((x - x.mean()) / (x.std() + 1e-8), dtype=np.float64)
