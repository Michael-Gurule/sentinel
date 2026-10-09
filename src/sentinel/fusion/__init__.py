"""Multi-sensor (OPIR + RF) fusion."""

from sentinel.fusion.engine import FusionEngine
from sentinel.fusion.measurements import SensorType, opir_measurement, rf_measurement
from sentinel.fusion.opir_geoloc import (
    LineOfSightMeasurement,
    associate_stereo,
    intersect_altitude,
    measurements_from_reports,
    triangulate,
)
from sentinel.fusion.t2t import (
    FusedEstimate,
    covariance_intersection,
    fuse_track_lists,
    naive_fusion,
)

__all__ = [
    "FusedEstimate",
    "FusionEngine",
    "LineOfSightMeasurement",
    "SensorType",
    "associate_stereo",
    "covariance_intersection",
    "fuse_track_lists",
    "intersect_altitude",
    "measurements_from_reports",
    "naive_fusion",
    "opir_measurement",
    "rf_measurement",
    "triangulate",
]
