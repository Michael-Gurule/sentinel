"""OPIR event signatures and staring-sensor model."""

from locant.sim.opir.sensor import (
    OPIRSensor,
    PixelObservation,
    SceneConditions,
    frame_times,
    observe,
)
from locant.sim.opir.signatures import (
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
