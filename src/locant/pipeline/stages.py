"""Interfaces of the processing stages.

The pipeline depends only on these protocols, so any stage can be replaced
(another detector, a different classifier backend such as ONNX Runtime, a
mock in tests) without touching the orchestration. The default
implementations are :class:`~locant.detection.CFARDetector`,
:class:`~locant.classification.EventClassifier`,
:class:`TDOAFDOAGeolocator`, and :class:`~locant.fusion.FusionEngine`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from locant.classification.inference import Prediction
from locant.core.linalg import FloatArray
from locant.detection.base import Detector
from locant.geolocation.hybrid import solve_tdoa_fdoa
from locant.geolocation.measurements import (
    FDOAMeasurement,
    GeolocationResult,
    Receiver,
    TDOAMeasurement,
)
from locant.geolocation.systematic import SystematicErrors
from locant.tracking.tracker import Measurement, Track

__all__ = [
    "Classifier",
    "Detector",
    "RFGeolocator",
    "RFObservation",
    "TDOAFDOAGeolocator",
    "Tracker",
]


class Classifier(Protocol):
    """Calibrated window classifier."""

    @property
    def classes(self) -> tuple[str, ...]: ...

    def predict(self, signals: FloatArray) -> Prediction:
        """Classify a batch of raw pixel windows of shape (N, T)."""
        ...


@dataclass(frozen=True, eq=False)
class RFObservation:
    """One emitter's TDOA (and optionally FDOA) measurements.

    Attributes:
        receivers: Receiver states at measurement time (moving receivers);
            ``None`` uses the pipeline's network.
        time: Measurement time; ``None`` means the frame time. Older
            observations are late and handled by the configured policy.
    """

    tdoa: TDOAMeasurement
    fdoa: FDOAMeasurement | None = None
    receivers: Sequence[Receiver] | None = None
    time: float | None = None


class RFGeolocator(Protocol):
    """Turns RF observations into position(/velocity) fixes."""

    def locate(
        self, observation: RFObservation, receivers: Sequence[Receiver]
    ) -> GeolocationResult:
        """Raises ``GeometryError`` or ``InsufficientMeasurementsError`` when
        the observation cannot support a fix."""
        ...


class Tracker(Protocol):
    """Fuses scans of measurements into tracks."""

    @property
    def time(self) -> float | None: ...

    @property
    def tracks(self) -> list[Track]: ...

    @property
    def confirmed_tracks(self) -> list[Track]: ...

    def process(
        self, measurements: Sequence[Measurement], timestamp: float
    ) -> list[Track]: ...


@dataclass(frozen=True)
class TDOAFDOAGeolocator:
    """Maximum-likelihood TDOA/FDOA geolocation with optional consider
    covariance for the network's systematic errors."""

    systematic: SystematicErrors | None = None

    def locate(
        self, observation: RFObservation, receivers: Sequence[Receiver]
    ) -> GeolocationResult:
        return solve_tdoa_fdoa(
            receivers, observation.tdoa, observation.fdoa, systematic=self.systematic
        )
