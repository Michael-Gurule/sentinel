"""Shared numerical primitives, statistics, errors, and constants."""

from sentinel.core.errors import (
    GeometryError,
    InsufficientMeasurementsError,
    NotPositiveDefiniteError,
    SentinelError,
)
from sentinel.core.linalg import (
    FloatArray,
    is_psd,
    mahalanobis_sq,
    nees,
    symmetrize,
)
from sentinel.core.stats import chi2_gate, mean_nees_bounds

__all__ = [
    "FloatArray",
    "GeometryError",
    "InsufficientMeasurementsError",
    "NotPositiveDefiniteError",
    "SentinelError",
    "chi2_gate",
    "is_psd",
    "mahalanobis_sq",
    "mean_nees_bounds",
    "nees",
    "symmetrize",
]
