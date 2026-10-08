"""OPIR event signatures and staring-sensor model."""

from sentinel.sim.opir.sensor import (
    OPIRSensor,
    PixelObservation,
    SceneConditions,
    frame_times,
    observe,
)
from sentinel.sim.opir.signatures import (
    AircraftSignature,
    ExplosionSignature,
    FireSignature,
    LaunchSignature,
    Signature,
)

__all__ = [
    "AircraftSignature",
    "ExplosionSignature",
    "FireSignature",
    "LaunchSignature",
    "OPIRSensor",
    "PixelObservation",
    "SceneConditions",
    "Signature",
    "frame_times",
    "observe",
]
