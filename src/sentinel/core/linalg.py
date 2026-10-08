"""Covariance and linear-algebra helpers.

Covariances are handled through Cholesky factorizations rather than explicit
inverses, which is both more accurate and an implicit positive-definiteness
check.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

from sentinel.core.errors import NotPositiveDefiniteError

FloatArray = NDArray[np.float64]


def symmetrize(matrix: FloatArray) -> FloatArray:
    """Return the symmetric part of a square matrix, (A + Aᵀ) / 2."""
    m = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (m + m.T)


def is_psd(matrix: FloatArray, tol: float = 1e-9) -> bool:
    """True if ``matrix`` is symmetric positive semidefinite.

    ``tol`` is relative to the largest eigenvalue magnitude.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    scale = max(float(np.max(np.abs(m))), 1.0)
    if not np.allclose(m, m.T, atol=tol * scale):
        return False
    eigenvalues = np.linalg.eigvalsh(symmetrize(m))
    return bool(eigenvalues.min() >= -tol * max(float(np.abs(eigenvalues).max()), 1.0))


def cholesky(matrix: FloatArray) -> tuple[FloatArray, bool]:
    """Cholesky factor in scipy ``cho_factor`` form.

    Raises:
        NotPositiveDefiniteError: if ``matrix`` is not positive definite.
    """
    try:
        factor, lower = cho_factor(symmetrize(matrix), lower=True)
    except np.linalg.LinAlgError as exc:
        raise NotPositiveDefiniteError("matrix is not positive definite") from exc
    return np.asarray(factor, dtype=np.float64), bool(lower)


def solve_psd(matrix: FloatArray, rhs: FloatArray) -> FloatArray:
    """Solve ``matrix @ x = rhs`` for a symmetric positive-definite matrix."""
    return np.asarray(cho_solve(cholesky(matrix), rhs), dtype=np.float64)


def whitening_matrix(covariance: FloatArray) -> FloatArray:
    """Return W with W @ C @ Wᵀ = I, i.e. W = L⁻¹ for C = L Lᵀ."""
    factor, _ = cholesky(covariance)
    lower = np.tril(factor)
    identity = np.eye(lower.shape[0])
    return np.asarray(np.linalg.solve(lower, identity), dtype=np.float64)


def mahalanobis_sq(residual: FloatArray, covariance: FloatArray) -> float:
    """Squared Mahalanobis distance rᵀ C⁻¹ r."""
    r = np.asarray(residual, dtype=np.float64)
    return float(r @ solve_psd(covariance, r))


def nees(error: FloatArray, covariance: FloatArray) -> float:
    """Normalized estimation error squared, eᵀ P⁻¹ e.

    For a consistent estimator NEES is χ²-distributed with ``len(error)``
    degrees of freedom.
    """
    return mahalanobis_sq(error, covariance)
