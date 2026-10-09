"""Forward measurement models and their Jacobians.

TDOA and FDOA are modeled as range differences (m) and range-rate differences
(m/s) internally; that keeps both on comparable numeric scales. Conversion
helpers map them to and from seconds and Hz.
"""

from collections.abc import Sequence

import numpy as np

from locant.core.constants import SPEED_OF_LIGHT
from locant.core.linalg import FloatArray
from locant.geolocation.measurements import FDOAMeasurement, Receiver, TDOAMeasurement


def receiver_lookup(
    receivers: Sequence[Receiver], ids: Sequence[int]
) -> list[Receiver]:
    """Return receivers in the order of ``ids``.

    Raises:
        KeyError: if an id has no matching receiver.
    """
    by_id = {r.id: r for r in receivers}
    if len(by_id) != len(receivers):
        raise ValueError("receiver ids must be unique")
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown receiver ids: {missing}")
    return [by_id[i] for i in ids]


def _units_and_ranges(
    position: FloatArray, receiver_positions: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Unit vectors (n×3) from each receiver to the emitter, and ranges (n,)."""
    delta = np.asarray(position, dtype=np.float64)[None, :] - receiver_positions
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance == 0.0):
        raise ValueError("emitter coincides with a receiver")
    return delta / distance[:, None], distance


def range_difference_model(
    position: FloatArray, reference: Receiver, others: Sequence[Receiver]
) -> tuple[FloatArray, FloatArray]:
    """Predicted range differences r_k - r_0 (m) and Jacobian wrt position (m×3)."""
    receivers = np.array([reference.position, *(r.position for r in others)])
    u, r = _units_and_ranges(position, receivers)
    return r[1:] - r[0], u[1:] - u[0]


def range_rate_difference_model(
    position: FloatArray,
    velocity: FloatArray,
    reference: Receiver,
    others: Sequence[Receiver],
) -> tuple[FloatArray, FloatArray]:
    """Predicted range-rate differences (m/s) and Jacobian wrt [p, v] (m×6)."""
    receivers = [reference, *others]
    u, r = _units_and_ranges(position, np.array([x.position for x in receivers]))
    relative_velocity = np.asarray(velocity, dtype=np.float64)[None, :] - np.array(
        [x.velocity for x in receivers]
    )
    rate = np.einsum("nk,nk->n", u, relative_velocity)
    # d(uᵀw)/dp = wᵀ (I - u uᵀ) / r ;  d(uᵀw)/dv = uᵀ
    grad_position = (relative_velocity - u * rate[:, None]) / r[:, None]
    jacobian = np.hstack([grad_position[1:] - grad_position[0], u[1:] - u[0]])
    return rate[1:] - rate[0], jacobian


def tdoa_to_range_difference(
    measurement: TDOAMeasurement,
) -> tuple[FloatArray, FloatArray]:
    """TDOA (s, s²) → range differences (m) and covariance (m²)."""
    c = SPEED_OF_LIGHT
    return measurement.values * c, measurement.covariance * c**2


def fdoa_to_range_rate_difference(
    measurement: FDOAMeasurement,
) -> tuple[FloatArray, FloatArray]:
    """FDOA (Hz, Hz²) → range-rate differences (m/s) and covariance (m²/s²).

    A receiver closing on the emitter sees a positive Doppler shift, so
    f_d = -(f_c / c) · dr/dt and dr/dt = -(c / f_c) · f_d.
    """
    scale = -SPEED_OF_LIGHT / measurement.carrier_frequency
    return measurement.values * scale, measurement.covariance * scale**2
