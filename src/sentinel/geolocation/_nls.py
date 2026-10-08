"""Shared whitened nonlinear least-squares machinery."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from sentinel.core.errors import GeometryError
from sentinel.core.linalg import FloatArray, whitening_matrix

ModelFn = Callable[[FloatArray], tuple[FloatArray, FloatArray]]

_MAX_CONDITION = 1e12


@dataclass(frozen=True, eq=False)
class NLSSolution:
    state: FloatArray
    covariance: FloatArray
    chi2: float
    converged: bool


def solve_whitened(
    model: ModelFn,
    observations: FloatArray,
    covariance: FloatArray,
    initial_state: FloatArray,
    max_nfev: int,
) -> NLSSolution:
    """Gauss-Newton/Levenberg-Marquardt ML estimate under Gaussian noise.

    Minimizes ‖W (z - h(x))‖² with W = L⁻¹, C = L Lᵀ. The returned covariance
    (Jᵀ C⁻¹ J)⁻¹ is evaluated at the solution.

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
    singular_values = np.linalg.svd(whitened_jac, compute_uv=False)
    if singular_values[-1] <= singular_values[0] / _MAX_CONDITION:
        raise GeometryError(
            "unknowns are not observable from these measurements "
            f"(condition number {singular_values[0] / max(singular_values[-1], 1e-300):.2e})"
        )
    information = whitened_jac.T @ whitened_jac
    state_cov = np.linalg.inv(information)
    final = residuals(state)
    return NLSSolution(
        state=state,
        covariance=0.5 * (state_cov + state_cov.T),
        chi2=float(final @ final),
        converged=bool(result.success),
    )
