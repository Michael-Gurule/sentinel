"""
Multi-sensor processing pipeline: OPIR detection/classification, RF
geolocation, and track fusion.

OPIR reports are detected and classified but not yet fused: OPIR has no
position until line-of-sight geolocation lands in Phase 5 (audit defect C1).
This module is replaced by a configurable runner in Phase 6.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sentinel.core.errors import GeometryError, InsufficientMeasurementsError
from sentinel.core.linalg import FloatArray
from sentinel.detection.opir_detectors import DetectionResult, MultiMethodDetector
from sentinel.fusion.engine import FusionEngine
from sentinel.fusion.measurements import rf_measurement
from sentinel.geolocation.hybrid import solve_tdoa_fdoa
from sentinel.geolocation.measurements import (
    FDOAMeasurement,
    Receiver,
    TDOAMeasurement,
)
from sentinel.models.cnn_classifier import ClassificationResult, OPIRClassifier
from sentinel.tracking.tracker import LinearMeasurement, Track

logger = logging.getLogger(__name__)

RFInput = TDOAMeasurement | tuple[TDOAMeasurement, FDOAMeasurement | None]


@dataclass(frozen=True, eq=False)
class OPIRReport:
    """Detected and classified OPIR event (not yet geolocated)."""

    timestamp: float
    detection: DetectionResult
    classification: ClassificationResult


def default_receiver_network() -> list[Receiver]:
    """Five stationary receivers at mixed altitudes.

    Altitude diversity keeps the vertical geometry observable; five receivers
    give four TDOAs, enough for the closed-form Chan-Ho initializer.
    """
    positions = [
        [0.0, 0.0, 500.0],  # ground station
        [10_000.0, 0.0, 1_500.0],  # low-altitude ISR aircraft
        [10_000.0, 10_000.0, 1_000.0],  # medium-altitude platform
        [0.0, 10_000.0, 2_000.0],  # high-altitude ISR
        [5_000.0, -4_000.0, 6_000.0],  # stand-off high-altitude platform
    ]
    return [Receiver(i, np.array(p)) for i, p in enumerate(positions)]


def track_to_dict(track: Track) -> dict[str, object]:
    return {
        "track_id": track.id,
        "position": track.position.tolist(),
        "velocity": track.velocity.tolist(),
        "position_rms_uncertainty_m": track.position_rms_uncertainty,
        "velocity_rms_uncertainty_mps": float(
            np.sqrt(np.trace(track.velocity_covariance))
        ),
        "label": track.label,
        "hits_by_source": dict(track.hits_by_source),
        "created": track.created,
        "last_update": track.last_update,
    }


class SENTINELPhase3Pipeline:
    """OPIR + RF processing with centralized track fusion."""

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
        receivers: Sequence[Receiver] | None = None,
        fusion_engine: FusionEngine | None = None,
    ) -> None:
        self.opir_detector = MultiMethodDetector()
        self.opir_classifier = OPIRClassifier(model_path=model_path, device=device)
        self.receivers = list(receivers or default_receiver_network())
        self.fusion_engine = fusion_engine or FusionEngine()
        self.processing_history: list[dict[str, object]] = []

    def process_opir_signal(
        self, signal: FloatArray, sampling_rate: float, timestamp: float
    ) -> OPIRReport | None:
        """Detect and classify an OPIR intensity time series."""
        detection = self.opir_detector.detect(signal, sampling_rate)
        if not detection.detected:
            return None
        return OPIRReport(
            timestamp=timestamp,
            detection=detection,
            classification=self.opir_classifier.classify(signal),
        )

    def process_rf_measurements(self, rf_input: RFInput) -> LinearMeasurement | None:
        """Geolocate an emitter; ``None`` if the geometry cannot support a fix."""
        tdoa, fdoa = rf_input if isinstance(rf_input, tuple) else (rf_input, None)
        try:
            result = solve_tdoa_fdoa(self.receivers, tdoa, fdoa)
        except (GeometryError, InsufficientMeasurementsError) as exc:
            logger.warning("RF geolocation failed: %s", exc)
            return None
        if not result.converged:
            logger.warning("RF geolocation did not converge")
            return None
        return rf_measurement(result)

    def process_multi_sensor_frame(
        self,
        opir_signals: Sequence[FloatArray],
        rf_measurements: Sequence[RFInput],
        sampling_rate: float,
        timestamp: float,
    ) -> dict[str, object]:
        """Process one synchronized frame from all sensors."""
        opir_reports = [
            report
            for signal in opir_signals
            if (report := self.process_opir_signal(signal, sampling_rate, timestamp))
        ]
        rf_fixes = [self.process_rf_measurements(m) for m in rf_measurements]
        measurements = [m for m in rf_fixes if m is not None]

        tracks = self.fusion_engine.process(measurements, timestamp)
        result: dict[str, object] = {
            "timestamp": timestamp,
            "opir_detections": len(opir_reports),
            "opir_labels": [r.classification.class_name for r in opir_reports],
            "rf_geolocations": len(measurements),
            "rf_failures": len(rf_fixes) - len(measurements),
            "fused_tracks": len(tracks),
            "tracks": [track_to_dict(t) for t in tracks],
        }
        self.processing_history.append(result)
        return result

    def get_situation_awareness(self) -> dict[str, object]:
        """Summary of the current track picture."""
        return self.fusion_engine.summary()

    def get_track_by_id(self, track_id: int) -> Track | None:
        return next((t for t in self.fusion_engine.tracks if t.id == track_id), None)

    def export_tracks_to_file(self, filepath: str | Path) -> None:
        data = {
            "tracks": [track_to_dict(t) for t in self.fusion_engine.tracks],
            "situation_awareness": self.get_situation_awareness(),
        }
        Path(filepath).write_text(json.dumps(data, indent=2))


def demo_phase3_system(seed: int = 0) -> None:
    """Track a moving emitter for 10 s and print the fused picture."""
    from sentinel.geolocation.simulate import simulate_tdoa
    from sentinel.models.signal_generator import OPIRSignalGenerator

    rng = np.random.default_rng(seed)
    pipeline = SENTINELPhase3Pipeline()
    generator = OPIRSignalGenerator(rng=rng)
    start, velocity = np.array([5_000.0, 5_000.0, 500.0]), np.array([100.0, 50.0, 0])

    print("SENTINEL multi-sensor demo")
    for frame in range(10):
        t = float(frame)
        result = pipeline.process_multi_sensor_frame(
            opir_signals=[generator.generate_launch_signature(start_time=2.0)],
            rf_measurements=[
                simulate_tdoa(start + velocity * t, pipeline.receivers, 10e-9, rng)
            ],
            sampling_rate=generator.sampling_rate,
            timestamp=t,
        )
        print(
            f"t={t:4.1f}s  OPIR detections={result['opir_detections']}  "
            f"RF fixes={result['rf_geolocations']}  tracks={result['fused_tracks']}"
        )

    truth = start + velocity * 9.0
    for track in pipeline.fusion_engine.tracks:
        error = np.linalg.norm(track.position - truth)
        print(
            f"track {track.id}: position error {error:.1f} m, "
            f"velocity {np.round(track.velocity, 1)} m/s, "
            f"RMS uncertainty {track.position_rms_uncertainty:.1f} m"
        )


if __name__ == "__main__":
    demo_phase3_system()
