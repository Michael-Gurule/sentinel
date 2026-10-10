"""Measurement and result types for emitter geolocation.

Conventions (local Cartesian frame, SI units):

* Positions in meters, velocities in m/s.
* TDOA and FDOA use the *reference-sensor* form. For a reference receiver 0 and
  others k = 1..m, ``tdoa[k] = (r_k - r_0) / c`` in seconds and
  ``fdoa[k] = f_k - f_0`` in Hz, where r is emitter-receiver range and f the
  received frequency.
* Measurement covariances are full matrices. Reference-sensor differences
  share the reference receiver's error, so they are correlated even when the
  per-receiver errors are independent.
"""

from dataclasses import dataclass, field

import numpy as np

from locant.core.linalg import FloatArray, is_psd
from locant.core.stats import chi2_gate


def _as_vector(value: FloatArray, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return array


@dataclass(frozen=True, eq=False)
class Receiver:
    """A receiving sensor with known position and velocity."""

    id: int
    position: FloatArray
    velocity: FloatArray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _as_vector(self.position, 3, "position"))
        object.__setattr__(self, "velocity", _as_vector(self.velocity, 3, "velocity"))


@dataclass(frozen=True, eq=False)
class _DifferenceMeasurement:
    reference: int
    others: tuple[int, ...]
    values: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        m = len(self.others)
        if m == 0:
            raise ValueError("at least one non-reference receiver is required")
        if self.reference in self.others:
            raise ValueError("reference receiver must not appear in `others`")
        if len(set(self.others)) != m:
            raise ValueError("`others` contains duplicate receiver ids")
        object.__setattr__(self, "others", tuple(self.others))
        object.__setattr__(self, "values", _as_vector(self.values, m, "values"))
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (m, m):
            raise ValueError(f"covariance must have shape ({m}, {m})")
        if not is_psd(covariance):
            raise ValueError("covariance must be symmetric positive semidefinite")
        object.__setattr__(self, "covariance", covariance)

    def __len__(self) -> int:
        return len(self.others)


@dataclass(frozen=True, eq=False)
class TDOAMeasurement(_DifferenceMeasurement):
    """Reference-sensor time differences of arrival (seconds, covariance s²)."""


@dataclass(frozen=True, eq=False)
class FDOAMeasurement(_DifferenceMeasurement):
    """Reference-sensor frequency differences of arrival (Hz, covariance Hz²)."""

    carrier_frequency: float = 1e9

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.carrier_frequency <= 0:
            raise ValueError("carrier_frequency must be positive")


@dataclass(frozen=True, eq=False)
class GeolocationResult:
    """Emitter state estimate with its covariance.

    Attributes:
        position: Estimated emitter position, shape (3,).
        position_covariance: Position covariance, shape (3, 3).
        velocity: Estimated emitter velocity, or ``None`` if unobservable from
            the measurements used (e.g. TDOA only).
        velocity_covariance: Velocity covariance, or ``None`` with ``velocity``.
        state_covariance: Full covariance of the estimated state ``[p]`` (3×3)
            or ``[p, v]`` (6×6), including position-velocity cross terms.
        chi2: Whitened sum of squared residuals at the solution.
        dof: Degrees of freedom of ``chi2`` (measurements minus unknowns).
        num_measurements: Number of scalar measurements used.
        converged: Whether the solver met its convergence criteria.
        method: Name of the estimator that produced the result.
        systematic_covariance: The part of ``state_covariance`` due to
            receiver errors that are fixed over a scenario (clock bias,
            survey error, LO offsets), when the solver was given their
            levels; ``None`` otherwise. Fixes from one network share this
            error, so averaging them does not reduce it.
    """

    position: FloatArray
    position_covariance: FloatArray
    velocity: FloatArray | None
    velocity_covariance: FloatArray | None
    state_covariance: FloatArray
    chi2: float
    dof: int
    num_measurements: int
    converged: bool
    method: str
    systematic_covariance: FloatArray | None = None

    def fits(self, probability: float = 0.999) -> bool:
        """χ² goodness-of-fit test of the residuals at the solution.

        A solver can stop at a stationary point that does not explain the
        measurements (e.g. a far-field solution along a hyperboloid's
        asymptote); its residual χ² is then far above its ``dof``. With zero
        degrees of freedom the test is vacuous and passes.
        """
        if self.dof < 1:
            return True
        return bool(self.chi2 <= chi2_gate(self.dof, probability))
