"""
Multi-sensor processing pipeline: OPIR detection/classification, RF
geolocation, and track fusion.

OPIR windows go through a CFAR detector at a false-alarm-calibrated threshold,
then (optionally) a calibrated classifier that rejects background, including
sun glints, and returns a conformal prediction set. Detections that come with
their line of sight (:class:`OPIRObservation`) are geolocated, by stereo
triangulation when two satellites see the event or as an angle-only update
otherwise, and fused with RF fixes in the same tracker update (this closes
audit defect C1, where OPIR never reached the fused tracks). This module is
replaced by a configurable runner in Phase 6.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sentinel.classification.inference import EventClassifier
from sentinel.core.errors import GeometryError, InsufficientMeasurementsError
from sentinel.core.linalg import FloatArray
from sentinel.detection import CFARDetector, Detector
from sentinel.fusion.engine import FusionEngine
from sentinel.fusion.measurements import rf_measurement
from sentinel.fusion.opir_geoloc import measurements_from_reports
from sentinel.geolocation.hybrid import solve_tdoa_fdoa
from sentinel.geolocation.measurements import (
    FDOAMeasurement,
    Receiver,
    TDOAMeasurement,
)
from sentinel.taxonomy import EVENT_CLASSES
from sentinel.tracking.tracker import LinearMeasurement, Measurement, Track

logger = logging.getLogger(__name__)

RFInput = TDOAMeasurement | tuple[TDOAMeasurement, FDOAMeasurement | None]

CLASSIFIER_ONSET_RANGE_S = (2.0, 40.0)
"""Onset positions within the 64 s window that the classifier was trained on
(data card). Outside this range, e.g. once the window has slid past a launch's
onset, a boosting plume looks like a steady source, so class evidence is only
passed to tracks when the detected onset lies inside it (E6)."""


CFAR_THRESHOLD_PFA_1E2 = 5.27
"""CFAR threshold for a 1e-2 false-alarm rate per 64 s window on glint-free
background (E1, ``reports/phase3/e1_detection.json``; a test keeps them in
sync). Glint alarms are left to the classifier."""


@dataclass(frozen=True, eq=False)
class OPIRObservation:
    """One OPIR pixel window and, optionally, where it points.

    Attributes:
        signal: Radiometric samples ending at the frame time.
        sensor_index: Which OPIR satellite produced it (stereo pairs are
            formed across sensors).
        sensor_position: Satellite position in the local ENU frame, m.
        line_of_sight: Unit vector from the satellite to the pixel center.
        angle_std: Line-of-sight error per axis, rad.

    Without ``sensor_position`` and ``line_of_sight`` the detection is
    reported but cannot be geolocated or fused.
    """

    signal: FloatArray
    sensor_index: int = 0
    sensor_position: FloatArray | None = None
    line_of_sight: FloatArray | None = None
    angle_std: float = 10e-6

    @property
    def geolocatable(self) -> bool:
        return self.sensor_position is not None and self.line_of_sight is not None


@dataclass(frozen=True, eq=False)
class _Ray:
    sensor_index: int
    sensor_position: FloatArray
    line_of_sight: FloatArray
    angle_std: float


@dataclass(frozen=True, eq=False)
class OPIRReport:
    """A detected OPIR event (not yet geolocated).

    Attributes:
        score: Detector statistic (CFAR: peak standardized residual).
        onset_time: Estimated onset within the window, s.
        label: Most likely class, or ``None`` without a classifier.
        probabilities: Calibrated class probabilities, or ``None``.
        prediction_set: Conformal set of plausible classes (empty without a
            classifier).
    """

    timestamp: float
    score: float
    onset_time: float
    label: str | None = None
    probabilities: dict[str, float] | None = None
    prediction_set: tuple[str, ...] = ()


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
        classifier: EventClassifier | str | Path | None = None,
        receivers: Sequence[Receiver] | None = None,
        fusion_engine: FusionEngine | None = None,
        detector: Detector | None = None,
        detection_threshold: float = CFAR_THRESHOLD_PFA_1E2,
    ) -> None:
        """
        Args:
            classifier: An :class:`EventClassifier` or a path to a saved
                artifact; without one, detections are reported unclassified.
            detector: OPIR detector (default CFAR); ``detection_threshold``
                must be calibrated for it.
        """
        self.opir_detector: Detector = detector or CFARDetector()
        self.detection_threshold = detection_threshold
        self.classifier = (
            EventClassifier.load(classifier)
            if isinstance(classifier, str | Path)
            else classifier
        )
        self.receivers = list(receivers or default_receiver_network())
        self.fusion_engine = fusion_engine or FusionEngine()
        self.processing_history: list[dict[str, object]] = []

    def process_opir_signal(
        self, signal: FloatArray, sampling_rate: float, timestamp: float
    ) -> OPIRReport | None:
        """Detect, then classify, one pixel window.

        Returns ``None`` if the detector does not fire or the classifier
        labels the window background (e.g. a sun glint).
        """
        scores = self.opir_detector.score(signal, sampling_rate)
        score = float(scores.score[0])
        if score <= self.detection_threshold:
            return None
        onset = float(scores.onset_index[0]) / sampling_rate
        if self.classifier is None:
            return OPIRReport(timestamp=timestamp, score=score, onset_time=onset)
        prediction = self.classifier.predict(np.asarray(signal)[None, :])
        label = prediction.labels[0]
        if label == "background":
            return None
        classes = self.classifier.classes
        return OPIRReport(
            timestamp=timestamp,
            score=score,
            onset_time=onset,
            label=label,
            probabilities=dict(
                zip(classes, map(float, prediction.probabilities[0]), strict=True)
            ),
            prediction_set=tuple(
                c
                for c, member in zip(
                    classes, prediction.prediction_sets[0], strict=True
                )
                if member
            ),
        )

    def process_rf_measurements(
        self, rf_input: RFInput, receivers: Sequence[Receiver] | None = None
    ) -> LinearMeasurement | None:
        """Geolocate an emitter; ``None`` if the geometry cannot support a fix.

        ``receivers`` overrides the pipeline's network for this measurement
        (e.g. moving receivers reported with each scan).
        """
        tdoa, fdoa = rf_input if isinstance(rf_input, tuple) else (rf_input, None)
        try:
            result = solve_tdoa_fdoa(receivers or self.receivers, tdoa, fdoa)
        except (GeometryError, InsufficientMeasurementsError) as exc:
            logger.warning("RF geolocation failed: %s", exc)
            return None
        if not result.converged or not result.fits():
            logger.warning(
                "RF geolocation rejected (converged=%s, chi2=%.1f on %d dof)",
                result.converged,
                result.chi2,
                result.dof,
            )
            return None
        return rf_measurement(result)

    def process_multi_sensor_frame(
        self,
        opir_signals: Sequence[FloatArray | OPIRObservation],
        rf_measurements: Sequence[RFInput],
        sampling_rate: float,
        timestamp: float,
        receivers: Sequence[Receiver] | None = None,
    ) -> dict[str, object]:
        """Process one synchronized frame from all sensors.

        Args:
            opir_signals: Pixel windows; plain arrays are detected and
                classified only, :class:`OPIRObservation` with geometry is
                also geolocated and fused.
            receivers: RF receiver states for this frame (moving receivers);
                defaults to the pipeline's network.
        """
        observations = [
            s if isinstance(s, OPIRObservation) else OPIRObservation(np.asarray(s))
            for s in opir_signals
        ]
        opir_reports: list[OPIRReport] = []
        rays: list[_Ray] = []
        probabilities: list[FloatArray | None] = []
        for observation in observations:
            report = self.process_opir_signal(
                observation.signal, sampling_rate, timestamp
            )
            if report is None:
                continue
            opir_reports.append(report)
            if observation.geolocatable:
                assert observation.sensor_position is not None
                assert observation.line_of_sight is not None
                rays.append(
                    _Ray(
                        observation.sensor_index,
                        observation.sensor_position,
                        observation.line_of_sight,
                        observation.angle_std,
                    )
                )
                in_domain = (
                    CLASSIFIER_ONSET_RANGE_S[0]
                    <= report.onset_time
                    <= CLASSIFIER_ONSET_RANGE_S[1]
                )
                probabilities.append(
                    np.array(list(report.probabilities.values()))
                    if report.probabilities is not None and in_domain
                    else None
                )
        opir_measurements = measurements_from_reports(rays, probabilities)
        rf_fixes = [self.process_rf_measurements(m, receivers) for m in rf_measurements]
        rf_valid = [m for m in rf_fixes if m is not None]
        measurements: list[Measurement] = [*rf_valid, *opir_measurements]

        tracks = self.fusion_engine.process(measurements, timestamp)
        result: dict[str, object] = {
            "timestamp": timestamp,
            "opir_detections": len(opir_reports),
            "opir_labels": [r.label for r in opir_reports],
            "opir_measurements": len(opir_measurements),
            "rf_geolocations": len(rf_valid),
            "rf_failures": len(rf_fixes) - len(rf_valid),
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


DEFAULT_CLASSIFIER = (
    Path(__file__).resolve().parents[3] / "models" / "opir_event_classifier"
)


def demo_phase3_system(seed: int = 0, classifier: str | Path | None = None) -> None:
    """Run the example scenario frame by frame through detection,
    classification, geolocation, and fusion, then print the track picture."""
    from sentinel.sim import load_scenario, simulate_scenario
    from sentinel.sim.opir.reports import to_local

    root = Path(__file__).resolve().parents[3]
    config = load_scenario(root / "configs" / "scenario" / "launch_with_radar.yaml")
    result = simulate_scenario(config, seed)
    frame = config.origin.frame()
    model = classifier or (DEFAULT_CLASSIFIER if DEFAULT_CLASSIFIER.exists() else None)
    pipeline = SENTINELPhase3Pipeline(classifier=model)
    fs = config.sensor.frame_rate_hz
    window = round(64.0 * fs)

    print("SENTINEL multi-sensor demo")
    print(f"classifier: {model or 'none (detections reported unclassified)'}")
    scans_by_time: dict[float, list[object]] = {}
    for _, scan in result.rf_scans:
        scans_by_time.setdefault(scan.t, []).append(scan)
    for t in np.arange(1.0, config.duration_s, 1.0):
        k = int(np.argmin(np.abs(result.times - t)))
        observations = []
        for pixel in result.opir.values():
            position, los = to_local(frame, pixel, k)
            observations.append(
                OPIRObservation(
                    pixel.measured[max(0, k + 1 - window) : k + 1],
                    sensor_position=position,
                    line_of_sight=los,
                )
            )
        scans = scans_by_time.get(float(t), [])
        pipeline.process_multi_sensor_frame(
            observations,
            [(s.tdoa, s.fdoa) for s in scans],  # type: ignore[attr-defined]
            fs,
            float(t),
            receivers=scans[0].receivers if scans else None,  # type: ignore[attr-defined]
        )

    for event_id, pixel in result.opir.items():
        report = pipeline.process_opir_signal(
            pixel.measured[-window:], fs, float(result.times[-1])
        )
        if report is None:
            print(f"OPIR {event_id:11s} no alarm")
        else:
            print(
                f"OPIR {event_id:11s} alarm (score {report.score:.1f}) -> {report.label}"
                f"  set {list(report.prediction_set)}"
            )
    for track in pipeline.fusion_engine.confirmed_tracks:
        posterior = track.class_posterior
        label = (
            f"{EVENT_CLASSES[int(np.argmax(posterior))]} ({np.max(posterior):.2f})"
            if posterior is not None
            else "unclassified"
        )
        print(
            f"track {track.id}: velocity {np.round(track.velocity, 1)} m/s, "
            f"RMS uncertainty {track.position_rms_uncertainty:.1f} m, "
            f"class {label}, hits {track.hits_by_source}"
        )


if __name__ == "__main__":
    demo_phase3_system()
