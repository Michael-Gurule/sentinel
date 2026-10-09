"""Shared whitened nonlinear least-squares machinery."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from sentinel.core.errors import GeometryError
from sentinel.core.linalg import FloatArray, symmetrize, whitening_matrix

ModelFn = Callable[[FloatArray], tuple[FloatArray, FloatArray]]

# Condition number of the column-equilibrated whitened Jacobian (invariant to
# the units of each unknown). The covariance has its square, so beyond ~1e8
# it carries no precision.
_MAX_CONDITION = 1e8


@dataclass(frozen=True, eq=False)
class NLSSolution:
    state: FloatArray
    covariance: FloatArray
    chi2: float
    converged: bool
    systematic_covariance: FloatArray | None = None


def solve_whitened(
    model: ModelFn,
    observations: FloatArray,
    covariance: FloatArray,
    initial_state: FloatArray,
    max_nfev: int,
    systematic_covariance: FloatArray | None = None,
) -> NLSSolution:
    """Gauss-Newton/Levenberg-Marquardt ML estimate under Gaussian noise.

    Minimizes ‖W (z - h(x))‖² with W = L⁻¹, C = L Lᵀ. The returned covariance
    P = (Jᵀ C⁻¹ J)⁻¹ is evaluated at the solution.

    ``systematic_covariance`` is the part of C due to errors that are fixed
    over a scenario (already included in C). To first order the estimate
    moves by δx = A δz with A = P Jᵀ C⁻¹, so the systematic part of P is
    A Σ_sys Aᵀ; it is returned so a tracker can keep it as a floor.

    Raises:
        GeometryError: the Jacobian at the solution is rank deficient or
            ill-conditioned (unknowns not observable from the measurements).
    """
    whitening = whitening_matrix(covariance)

    def residuals(x: FloatArray) -> FloatArray:
        predicted, _ = model(x)
        return np.asarray(whitening @ (observations - predicted))

    def jacobian(x: FloatArray) -> FloatArray:
        _, jac = model(x)
        return np.asarray(-whitening @ jac)

    result = least_squares(
        residuals,
        initial_state,
        jac=jacobian,
        method="lm",
        max_nfev=max_nfev,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    state = np.asarray(result.x, dtype=np.float64)
    whitened_jac = jacobian(state)
    column_norms = np.linalg.norm(whitened_jac, axis=0)
    scale = np.where(column_norms > 0, column_norms, 1.0)
    equilibrated = whitened_jac / scale
    singular_values = np.linalg.svd(equilibrated, compute_uv=False)
    if singular_values[-1] <= singular_values[0] / _MAX_CONDITION:
        raise GeometryError(
            "unknowns are not observable from these measurements "
            f"(condition number {singular_values[0] / max(singular_values[-1], 1e-300):.2e})"
        )
    # (JᵀJ)⁻¹ = D⁻¹ (JₛᵀJₛ)⁻¹ D⁻¹ with Jₛ = J D⁻¹: invert the well-scaled form.
    scaled_cov = np.linalg.inv(equilibrated.T @ equilibrated)
    if np.linalg.eigvalsh(0.5 * (scaled_cov + scaled_cov.T))[0] <= 0.0:
        raise GeometryError("covariance at the solution is not positive definite")
    state_cov = scaled_cov / np.outer(scale, scale)
    systematic = None
    if systematic_covariance is not None:
        gain = state_cov @ (-whitened_jac.T) @ whitening  # A = P Jᵀ C⁻¹
        systematic = symmetrize(gain @ systematic_covariance @ gain.T)
    final = residuals(state)
    return NLSSolution(
        state=state,
        covariance=symmetrize(state_cov),
        chi2=float(final @ final),
        converged=bool(result.success),
        systematic_covariance=systematic,
    )
