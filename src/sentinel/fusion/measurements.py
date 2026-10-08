"""Conversion of sensor products into fusable measurements."""

from enum import StrEnum

import numpy as np

from sentinel.core.linalg import FloatArray
from sentinel.geolocation.measurements import GeolocationResult
from sentinel.tracking.tracker import LinearMeasurement


class SensorType(StrEnum):
    """Sensor modalities, used as the measurement ``source``."""

    OPIR = "opir"
    RF = "rf"


def rf_measurement(
    result: GeolocationResult, label: str | None = None
) -> LinearMeasurement:
    """RF geolocation → position (or position/velocity) measurement.

    The geolocation covariance is used as the measurement noise, including the
    position-velocity cross terms when FDOA made velocity observable.
    """
    if result.velocity is None:
        return LinearMeasurement.position(
            result.position, result.position_covariance, SensorType.RF, label
        )
    return LinearMeasurement.position_velocity(
        np.concatenate([result.position, result.velocity]),
        result.state_covariance,
        SensorType.RF,
        label,
    )


def opir_measurement(
    position: FloatArray, covariance: FloatArray, label: str | None = None
) -> LinearMeasurement:
    """Geolocated OPIR event → position measurement.

    OPIR line-of-sight geolocation is introduced in Phase 5; until then the
    pipeline has no OPIR position to pass here (audit defect C1).
    """
    return LinearMeasurement.position(position, covariance, SensorType.OPIR, label)
