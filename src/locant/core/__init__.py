"""Shared numerical primitives, statistics, errors, and constants."""

from locant.core.errors import (
    GeometryError,
    InsufficientMeasurementsError,
    LocantError,
    NotPositiveDefiniteError,
)
from locant.core.linalg import (
    FloatArray,
    is_psd,
    mahalanobis_sq,
    nees,
    symmetrize,
)
from locant.core.stats import chi2_gate, mean_nees_bounds

__all__ = [
    "FloatArray",
    "GeometryError",
    "InsufficientMeasurementsError",
    "LocantError",
    "NotPositiveDefiniteError",
    "chi2_gate",
    "is_psd",
    "mahalanobis_sq",
    "mean_nees_bounds",
    "nees",
    "symmetrize",
]
