"""Measurement simulation with physically consistent noise.

Each receiver has an independent time-of-arrival (or frequency) error.
Reference-sensor differences therefore share the reference error and have
covariance σ²(I + 11ᵀ), which the solvers use directly.
"""

from collections.abc import Sequence

import numpy as np

from locant.core.constants import SPEED_OF_LIGHT
from locant.core.linalg import FloatArray
from locant.geolocation.measurements import FDOAMeasurement, Receiver, TDOAMeasurement
from locant.geolocation.models import (
    range_difference_model,
    range_rate_difference_model,
)


def _split_reference(
    receivers: Sequence[Receiver], reference_index: int
) -> tuple[Receiver, list[Receiver]]:
    if len(receivers) < 2:
        raise ValueError("at least two receivers are required")
    reference = receivers[reference_index]
    others = [r for i, r in enumerate(receivers) if i != reference_index]
    return reference, others


def difference_covariance(num_others: int, per_receiver_std: float) -> FloatArray:
    """Covariance of reference-sensor differences of iid per-receiver errors."""
    if per_receiver_std <= 0:
        raise ValueError("per_receiver_std must be positive")
    return per_receiver_std**2 * (np.eye(num_others) + np.ones((num_others,) * 2))


def simulate_tdoa(
    emitter_position: FloatArray,
    receivers: Sequence[Receiver],
    toa_std: float,
    rng: np.random.Generator,
    reference_index: int = 0,
) -> TDOAMeasurement:
    """Simulate reference-sensor TDOAs with per-receiver TOA noise (seconds)."""
    reference, others = _split_reference(receivers, reference_index)
    true_rd, _ = range_difference_model(
        np.asarray(emitter_position, dtype=np.float64), reference, others
    )
    toa_noise = rng.normal(0.0, toa_std, len(receivers))
    noise_ref = toa_noise[reference_index]
    noise_others = np.delete(toa_noise, reference_index)
    return TDOAMeasurement(
        reference=reference.id,
        others=tuple(r.id for r in others),
        values=true_rd / SPEED_OF_LIGHT + (noise_others - noise_ref),
        covariance=difference_covariance(len(others), toa_std),
    )


def simulate_fdoa(
    emitter_position: FloatArray,
    emitter_velocity: FloatArray,
    receivers: Sequence[Receiver],
    carrier_frequency: float,
    frequency_std: float,
    rng: np.random.Generator,
    reference_index: int = 0,
) -> FDOAMeasurement:
    """Simulate reference-sensor FDOAs with per-receiver frequency noise (Hz)."""
    reference, others = _split_reference(receivers, reference_index)
    true_rrd, _ = range_rate_difference_model(
        np.asarray(emitter_position, dtype=np.float64),
        np.asarray(emitter_velocity, dtype=np.float64),
        reference,
        others,
    )
    true_fdoa = -true_rrd * carrier_frequency / SPEED_OF_LIGHT
    freq_noise = rng.normal(0.0, frequency_std, len(receivers))
    noise_ref = freq_noise[reference_index]
    noise_others = np.delete(freq_noise, reference_index)
    return FDOAMeasurement(
        reference=reference.id,
        others=tuple(r.id for r in others),
        values=true_fdoa + (noise_others - noise_ref),
        covariance=difference_covariance(len(others), frequency_std),
        carrier_frequency=carrier_frequency,
    )


def simulate_ranges(
    emitter_position: FloatArray,
    sensor_positions: FloatArray,
    range_std: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Simulate independent range measurements (m) from each sensor."""
    sensors = np.asarray(sensor_positions, dtype=np.float64)
    true_ranges = np.linalg.norm(sensors - np.asarray(emitter_position), axis=1)
    return np.asarray(true_ranges + rng.normal(0.0, range_std, len(sensors)))
