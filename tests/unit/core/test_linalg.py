import numpy as np
import pytest

from sentinel.core import NotPositiveDefiniteError, is_psd, mahalanobis_sq, nees
from sentinel.core.linalg import cholesky, solve_psd, symmetrize, whitening_matrix

SPD = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])


def test_symmetrize_returns_symmetric_part():
    a = np.array([[1.0, 2.0], [0.0, 3.0]])
    np.testing.assert_allclose(symmetrize(a), [[1.0, 1.0], [1.0, 3.0]])


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        (SPD, True),
        (np.zeros((2, 2)), True),  # PSD, not PD
        (np.diag([1.0, -1e-3]), False),  # indefinite
        (np.array([[1.0, 2.0], [0.0, 1.0]]), False),  # asymmetric
        (np.ones((2, 3)), False),  # not square
    ],
)
def test_is_psd(matrix, expected):
    assert is_psd(matrix) is expected


def test_cholesky_rejects_indefinite_matrix():
    with pytest.raises(NotPositiveDefiniteError):
        cholesky(np.diag([1.0, -1.0]))


def test_solve_psd_matches_dense_solve():
    rhs = np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(solve_psd(SPD, rhs), np.linalg.solve(SPD, rhs))


def test_whitening_matrix_whitens():
    w = whitening_matrix(SPD)
    np.testing.assert_allclose(w @ SPD @ w.T, np.eye(3), atol=1e-12)


def test_mahalanobis_and_nees_match_explicit_inverse():
    r = np.array([0.3, -1.0, 2.0])
    expected = r @ np.linalg.inv(SPD) @ r
    assert mahalanobis_sq(r, SPD) == pytest.approx(expected)
    assert nees(r, SPD) == pytest.approx(expected)
