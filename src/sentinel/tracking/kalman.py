"""Linear Kalman filter operations on Gaussian states."""

from dataclasses import dataclass

import numpy as np

from sentinel.core.linalg import FloatArray, solve_psd, symmetrize
from sentinel.tracking.models import ConstantVelocity


@dataclass(frozen=True, eq=False)
class Gaussian:
    """Gaussian state estimate N(mean, covariance)."""

    mean: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if mean.ndim != 1 or covariance.shape != (mean.size, mean.size):
            raise ValueError("covariance must be square and match the mean")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True, eq=False)
class Innovation:
    """Measurement residual ν = z - Hx̂, its covariance S = HPHᵀ + R, and NIS νᵀS⁻¹ν."""

    residual: FloatArray
    covariance: FloatArray
    nis: float


def predict(state: Gaussian, model: ConstantVelocity, dt: float) -> Gaussian:
    """Propagate a state estimate ``dt`` seconds forward."""
    if dt == 0.0:
        return state
    f = model.transition(dt)
    return Gaussian(
        mean=f @ state.mean,
        covariance=symmetrize(f @ state.covariance @ f.T + model.process_noise(dt)),
    )


def innovation(
    state: Gaussian,
    measurement: FloatArray,
    matrix: FloatArray,
    noise: FloatArray,
) -> Innovation:
    """Innovation of a linear measurement z = Hx + v, v ~ N(0, R)."""
    residual = np.asarray(measurement, dtype=np.float64) - matrix @ state.mean
    covariance = symmetrize(matrix @ state.covariance @ matrix.T + noise)
    nis = float(residual @ solve_psd(covariance, residual))
    return Innovation(residual=residual, covariance=covariance, nis=nis)


def update(
    state: Gaussian,
    measurement: FloatArray,
    matrix: FloatArray,
    noise: FloatArray,
) -> tuple[Gaussian, Innovation]:
    """Kalman measurement update in Joseph form.

    P⁺ = (I - KH) P (I - KH)ᵀ + K R Kᵀ stays symmetric positive semidefinite
    under rounding, unlike the short form (I - KH) P.
    """
    innov = innovation(state, measurement, matrix, noise)
    gain = solve_psd(innov.covariance, matrix @ state.covariance).T  # P Hᵀ S⁻¹
    identity_minus_kh = np.eye(state.mean.size) - gain @ matrix
    covariance = (
        identity_minus_kh @ state.covariance @ identity_minus_kh.T
        + gain @ noise @ gain.T
    )
    return (
        Gaussian(
            mean=state.mean + gain @ innov.residual, covariance=symmetrize(covariance)
        ),
        innov,
    )
