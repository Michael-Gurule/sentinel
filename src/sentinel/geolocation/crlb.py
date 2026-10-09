"""Cramér-Rao lower bounds for TDOA and joint TDOA/FDOA geolocation.

For Gaussian measurements z = h(θ) + v, v ~ N(0, R), the Fisher information is
J = Hᵀ R⁻¹ H with H = ∂h/∂θ at the true state, and any unbiased estimator has
covariance ⪰ J⁻¹. With per-receiver TOA errors of standard deviation σ, the
reference-sensor range differences have R = (cσ)² (I + 11ᵀ). The bound is the
same for every choice of reference receiver (the differences are an invertible
linear map of one another).

The ML solvers report (Hᵀ R⁻¹ H)⁻¹ evaluated at their estimate, so at high SNR
their reported covariance converges to this bound.
"""

from collections.abc import Sequence

import numpy as np

from sentinel.core.constants import SPEED_OF_LIGHT
from sentinel.core.errors import GeometryError
from sentinel.core.linalg import FloatArray, solve_psd
from sentinel.geolocation.measurements import Receiver
from sentinel.geolocation.models import (
    range_difference_model,
    range_rate_difference_model,
)
from sentinel.geolocation.simulate import difference_covariance


def _split(
    receivers: Sequence[Receiver], reference_index: int
) -> tuple[Receiver, list[Receiver]]:
    if len(receivers) < 2:
        raise ValueError("at least two receivers are required")
    others = [r for i, r in enumerate(receivers) if i != reference_index]
    return receivers[reference_index], others


def _inverse_information(h: FloatArray, r: FloatArray) -> FloatArray:
    information = h.T @ solve_psd(r, h)
    eigenvalues = np.linalg.eigvalsh(information)
    if eigenvalues[0] <= 1e-12 * max(float(eigenvalues[-1]), 1e-300):
        raise GeometryError("Fisher information is singular: state not observable")
    bound = np.linalg.inv(information)
    return np.asarray(0.5 * (bound + bound.T))


def tdoa_crlb(
    emitter_position: FloatArray,
    receivers: Sequence[Receiver],
    toa_std: float,
    reference_index: int = 0,
) -> FloatArray:
    """CRLB (3×3, m²) on emitter position from reference-sensor TDOAs.

    Raises:
        GeometryError: position not observable (e.g. too few receivers).
    """
    reference, others = _split(receivers, reference_index)
    _, h = range_difference_model(
        np.asarray(emitter_position, dtype=np.float64), reference, others
    )
    r = difference_covariance(len(others), SPEED_OF_LIGHT * toa_std)
    return _inverse_information(h, r)


def tdoa_fdoa_crlb(
    emitter_position: FloatArray,
    emitter_velocity: FloatArray,
    receivers: Sequence[Receiver],
    toa_std: float,
    frequency_std: float,
    carrier_frequency: float,
    reference_index: int = 0,
) -> FloatArray:
    """CRLB (6×6) on ``[position (m), velocity (m/s)]`` from TDOA + FDOA.

    TDOA and FDOA errors are independent; each has the (I + 11ᵀ) structure of
    reference-sensor differences of iid per-receiver errors.
    """
    reference, others = _split(receivers, reference_index)
    position = np.asarray(emitter_position, dtype=np.float64)
    velocity = np.asarray(emitter_velocity, dtype=np.float64)
    m = len(others)
    _, h_rd = range_difference_model(position, reference, others)
    _, h_rrd = range_rate_difference_model(position, velocity, reference, others)
    h = np.zeros((2 * m, 6))
    h[:m, :3] = h_rd
    h[m:] = h_rrd
    range_rate_std = SPEED_OF_LIGHT / carrier_frequency * frequency_std
    r = np.zeros((2 * m, 2 * m))
    r[:m, :m] = difference_covariance(m, SPEED_OF_LIGHT * toa_std)
    r[m:, m:] = difference_covariance(m, range_rate_std)
    return _inverse_information(h, r)


def rms_bound(covariance: FloatArray) -> float:
    """√trace: the RMS error bound implied by a covariance (or CRLB) block."""
    return float(np.sqrt(np.trace(np.asarray(covariance))))
