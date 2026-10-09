"""Conversion of sensor products into fusable measurements."""

from enum import StrEnum

import numpy as np

from sentinel.core.linalg import FloatArray
from sentinel.geolocation.measurements import GeolocationResult
from sentinel.tracking.models import ConstantVelocity
from sentinel.tracking.tracker import LinearMeasurement

FIT_PROBABILITY = 0.999
"""χ² goodness-of-fit level an RF fix must pass to be fused."""


class SensorType(StrEnum):
    """Sensor modalities, used as the measurement ``source``."""

    OPIR = "opir"
    RF = "rf"


def accept_fix(
    result: GeolocationResult, fit_probability: float = FIT_PROBABILITY
) -> bool:
    """Whether an RF fix is fit to fuse: the solver converged *and* the
    residuals pass the χ² fit test (a converged far-field fix can fail it)."""
    return result.converged and result.fits(fit_probability)


def rf_measurement(
    result: GeolocationResult, label: str | None = None
) -> LinearMeasurement:
    """RF geolocation → position (or position/velocity) measurement.

    The geolocation covariance is used as the measurement noise, including the
    position-velocity cross terms when FDOA made velocity observable. A fix
    solved with systematic-error levels carries their share of the covariance
    (see :class:`~sentinel.tracking.tracker.Track` ``bias_floor``).
    """
    systematic = result.systematic_covariance
    if result.velocity is None:
        return LinearMeasurement.position(
            result.position,
            result.position_covariance,
            SensorType.RF,
            label,
            systematic_covariance=None if systematic is None else systematic[:3, :3],
        )
    return LinearMeasurement.position_velocity(
        np.concatenate([result.position, result.velocity]),
        result.state_covariance,
        SensorType.RF,
        label,
        systematic_covariance=systematic,
    )


def extrapolate(
    measurement: LinearMeasurement, dt: float, model: ConstantVelocity
) -> LinearMeasurement:
    """Propagate a late position-velocity fix ``dt`` seconds forward.

    z' = F z and R' = F R Fᵀ + Q(dt), so a fix that arrives after newer
    measurements can still be fused in time order (E7: no loss up to 2 s of
    latency). The added Q is correlated with the track's own process noise
    over the same interval; ignoring that is slightly optimistic for large dt.

    Raises:
        ValueError: if the measurement is not a full position-velocity state
            (position-only fixes cannot be propagated).
    """
    if measurement.dim != 6 or not np.allclose(measurement.matrix, np.eye(6)):
        raise ValueError("only position-velocity measurements can be extrapolated")
    f = model.transition(dt)
    systematic = measurement.systematic_covariance
    return LinearMeasurement(
        f @ measurement.value,
        f @ measurement.covariance @ f.T + model.process_noise(dt),
        measurement.matrix,
        measurement.source,
        measurement.label,
        measurement.class_probabilities,
        None if systematic is None else f @ systematic @ f.T,
    )


def opir_measurement(
    position: FloatArray, covariance: FloatArray, label: str | None = None
) -> LinearMeasurement:
    """Geolocated OPIR event (e.g. a triangulated or altitude-intersected
    position from :mod:`sentinel.fusion.opir_geoloc`) → position measurement."""
    return LinearMeasurement.position(position, covariance, SensorType.OPIR, label)
