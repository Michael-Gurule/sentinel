"""Forward measurement models and their Jacobians.

TDOA and FDOA are modeled as range differences (m) and range-rate differences
(m/s) internally; that keeps both on comparable numeric scales. Conversion
helpers map them to and from seconds and Hz.
"""

from collections.abc import Sequence

import numpy as np

from sentinel.core.constants import SPEED_OF_LIGHT
from sentinel.core.linalg import FloatArray
from sentinel.geolocation.measurements import FDOAMeasurement, Receiver, TDOAMeasurement


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


def _unit_and_range(
    position: FloatArray, receiver_position: FloatArray
) -> tuple[FloatArray, float]:
    delta = position - receiver_position
    distance = float(np.linalg.norm(delta))
    if distance == 0.0:
        raise ValueError("emitter coincides with a receiver")
    return delta / distance, distance


def range_difference_model(
    position: FloatArray, reference: Receiver, others: Sequence[Receiver]
) -> tuple[FloatArray, FloatArray]:
    """Predicted range differences r_k - r_0 (m) and Jacobian wrt position (m×3)."""
    u_ref, r_ref = _unit_and_range(position, reference.position)
    values = np.empty(len(others))
    jacobian = np.empty((len(others), 3))
    for k, receiver in enumerate(others):
        u_k, r_k = _unit_and_range(position, receiver.position)
        values[k] = r_k - r_ref
        jacobian[k] = u_k - u_ref
    return values, jacobian


def _range_rate_and_gradient(
    position: FloatArray, velocity: FloatArray, receiver: Receiver
) -> tuple[float, FloatArray, FloatArray]:
    u, distance = _unit_and_range(position, receiver.position)
    relative_velocity = velocity - receiver.velocity
    rate = float(u @ relative_velocity)
    # d(uᵀw)/dp = wᵀ (I - u uᵀ) / r ;  d(uᵀw)/dv = uᵀ
    grad_position = (relative_velocity - u * rate) / distance
    return rate, grad_position, u


def range_rate_difference_model(
    position: FloatArray,
    velocity: FloatArray,
    reference: Receiver,
    others: Sequence[Receiver],
) -> tuple[FloatArray, FloatArray]:
    """Predicted range-rate differences (m/s) and Jacobian wrt [p, v] (m×6)."""
    rate_ref, gp_ref, gv_ref = _range_rate_and_gradient(position, velocity, reference)
    values = np.empty(len(others))
    jacobian = np.empty((len(others), 6))
    for k, receiver in enumerate(others):
        rate_k, gp_k, gv_k = _range_rate_and_gradient(position, velocity, receiver)
        values[k] = rate_k - rate_ref
        jacobian[k, :3] = gp_k - gp_ref
        jacobian[k, 3:] = gv_k - gv_ref
    return values, jacobian


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
