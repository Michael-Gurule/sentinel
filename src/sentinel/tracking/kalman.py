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
    predicted: FloatArray | None = None,
) -> Innovation:
    """Innovation of z = h(x) + v, v ~ N(0, R), linearized as H.

    ``predicted`` is h(x̂) for a nonlinear measurement (EKF); by default the
    measurement is linear and h(x̂) = H x̂.
    """
    expected = matrix @ state.mean if predicted is None else np.asarray(predicted)
    residual = np.asarray(measurement, dtype=np.float64) - expected
    covariance = symmetrize(matrix @ state.covariance @ matrix.T + noise)
    nis = float(residual @ solve_psd(covariance, residual))
    return Innovation(residual=residual, covariance=covariance, nis=nis)


def batch_nis(
    measurement: FloatArray,
    predicted: FloatArray,
    matrices: FloatArray,
    covariances: FloatArray,
    noise: FloatArray,
) -> np.ndarray:
    """NIS of one measurement against K states at once.

    Args:
        measurement: z, shape (d,).
        predicted: h(x̂ₖ) per state, shape (K, d).
        matrices: Jacobians Hₖ, shape (K, d, n).
        covariances: State covariances Pₖ, shape (K, n, n).
        noise: Measurement noise R, shape (d, d).

    Returns:
        NIS per state, shape (K,): rₖᵀ Sₖ⁻¹ rₖ with Sₖ = Hₖ Pₖ Hₖᵀ + R.
    """
    residual = np.asarray(measurement, dtype=np.float64)[None, :] - predicted
    s = matrices @ covariances @ np.swapaxes(matrices, 1, 2) + noise
    s = 0.5 * (s + np.swapaxes(s, 1, 2))
    solved = np.linalg.solve(s, residual[..., None])[..., 0]
    return np.asarray(np.einsum("kd,kd->k", residual, solved))


def update(
    state: Gaussian,
    measurement: FloatArray,
    matrix: FloatArray,
    noise: FloatArray,
    predicted: FloatArray | None = None,
) -> tuple[Gaussian, Innovation]:
    """Kalman (or extended Kalman) measurement update in Joseph form.

    P⁺ = (I - KH) P (I - KH)ᵀ + K R Kᵀ stays symmetric positive semidefinite
    under rounding, unlike the short form (I - KH) P. For a nonlinear
    measurement pass ``predicted`` = h(x̂) and the Jacobian as ``matrix``.
    """
    innov = innovation(state, measurement, matrix, noise, predicted)
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
