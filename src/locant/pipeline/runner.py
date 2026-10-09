"""The Locant processing chain, frame by frame.

For each :class:`SensorFrame`:

1. **OPIR detection.** Every pixel window is scored by the detector at a
   false-alarm-calibrated threshold.
2. **Classification** (optional). Each detection gets calibrated class
   probabilities and a conformal set. Detections labeled background are
   dropped. Class evidence is passed on only when the detected onset lies in
   the range the classifier was trained on.
3. **OPIR geolocation.** Detections with a line of sight are paired across
   satellites into stereo positions; the rest become angle-only updates.
4. **RF geolocation.** Each emitter's TDOA/FDOA observation becomes a fix,
   fused only if it converged and passes the χ² fit test. A late fix is
   extrapolated to the frame time with its own velocity, or dropped.
5. **Fusion.** All measurements update the tracks in one centralized step.

Each stage is an injectable component (:mod:`locant.pipeline.stages`), and
every number comes from :class:`~locant.pipeline.config.PipelineConfig`.
"""

import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from locant.classification.inference import EventClassifier
from locant.core.errors import GeometryError, InsufficientMeasurementsError
from locant.core.linalg import FloatArray
from locant.core.logs import log_event
from locant.detection import CFARDetector, CUSUMDetector, StepGLRTDetector
from locant.detection.base import Detector
from locant.fusion.engine import FusionEngine
from locant.fusion.measurements import accept_fix, extrapolate, rf_measurement
from locant.fusion.opir_geoloc import measurements_from_reports
from locant.geolocation.measurements import Receiver
from locant.geolocation.systematic import SystematicErrors
from locant.pipeline.config import (
    ClassificationConfig,
    DetectionConfig,
    PipelineConfig,
    TrackingConfig,
    load_pipeline_config,
)
from locant.pipeline.stages import (
    Classifier,
    RFGeolocator,
    RFObservation,
    TDOAFDOAGeolocator,
    Tracker,
)
from locant.sim.rf.network import default_receiver_network
from locant.tracking.models import ConstantVelocity
from locant.tracking.tracker import LinearMeasurement, Measurement, Track

logger = logging.getLogger(__name__)

_LATE_TOLERANCE_S = 1e-6


@dataclass(frozen=True, eq=False)
class OPIRObservation:
    """One OPIR pixel window and, optionally, where it points.

    Attributes:
        signal: Radiometric samples ending at the frame time.
        sensor_index: Which OPIR satellite produced it (stereo pairs are
            formed across sensors).
        sensor_position: Satellite position in the local ENU frame, m.
        line_of_sight: Unit vector from the satellite to the pixel center.
        angle_std: Line-of-sight error per axis, rad; ``None`` uses the
            configured value.

    Without ``sensor_position`` and ``line_of_sight`` the detection is
    reported but cannot be geolocated or fused.
    """

    signal: FloatArray
    sensor_index: int = 0
    sensor_position: FloatArray | None = None
    line_of_sight: FloatArray | None = None
    angle_std: float | None = None

    @property
    def geolocatable(self) -> bool:
        return self.sensor_position is not None and self.line_of_sight is not None


@dataclass(frozen=True, eq=False)
class OPIRReport:
    """A detected OPIR event.

    Attributes:
        score: Detector statistic (CFAR: peak standardized residual).
        onset_time: Estimated onset within the window, s.
        label: Most likely class, or ``None`` without a classifier.
        probabilities: Calibrated class probabilities, or ``None``.
        prediction_set: Conformal set of plausible classes (empty without a
            classifier).
        in_domain: Whether the onset lies in the classifier's trained range
            (only then is class evidence fused).
    """

    timestamp: float
    score: float
    onset_time: float
    sensor_index: int = 0
    label: str | None = None
    probabilities: dict[str, float] | None = None
    prediction_set: tuple[str, ...] = ()
    in_domain: bool = False


@dataclass(frozen=True, eq=False)
class SensorFrame:
    """Everything the sensors delivered for one processing time."""

    time: float
    opir: Sequence[OPIRObservation] = ()
    rf: Sequence[RFObservation] = ()
    sampling_rate: float = 10.0
    """OPIR frame rate, Hz."""


@dataclass(frozen=True, eq=False)
class FrameResult:
    """Outcome of one frame. ``tracks`` are the confirmed tracks."""

    time: float
    opir_reports: tuple[OPIRReport, ...]
    opir_measurements: int
    rf_fixes: int
    rf_rejected: int
    rf_late_dropped: int
    tracks: tuple[Track, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "opir_detections": len(self.opir_reports),
            "opir_labels": [r.label for r in self.opir_reports],
            "opir_measurements": self.opir_measurements,
            "rf_fixes": self.rf_fixes,
            "rf_rejected": self.rf_rejected,
            "rf_late_dropped": self.rf_late_dropped,
            "tracks": [track_to_dict(t) for t in self.tracks],
        }


def track_to_dict(track: Track, classes: Sequence[str] | None = None) -> dict[str, Any]:
    posterior = track.class_posterior
    out: dict[str, Any] = {
        "track_id": track.id,
        "status": str(track.status),
        "position": track.position.tolist(),
        "velocity": track.velocity.tolist(),
        "position_rms_uncertainty_m": track.position_rms_uncertainty,
        "velocity_rms_uncertainty_mps": float(
            np.sqrt(np.trace(track.velocity_covariance))
        ),
        "hits_by_source": dict(track.hits_by_source),
        "created": track.created,
        "last_update": track.last_update,
    }
    if posterior is not None:
        names = classes or [str(i) for i in range(len(posterior))]
        out["class_posterior"] = dict(zip(names, map(float, posterior), strict=True))
    return out


def build_detector(config: DetectionConfig) -> Detector:
    detectors: dict[str, Detector] = {
        "cfar": CFARDetector(),
        "cusum": CUSUMDetector(),
        "glrt": StepGLRTDetector(),
    }
    return detectors[config.method]


def build_classifier(config: ClassificationConfig) -> Classifier | None:
    if config.artifact is None:
        return None
    if config.backend == "onnx":
        from locant.classification.onnx_backend import OnnxEventClassifier

        return OnnxEventClassifier.load(config.artifact)
    return EventClassifier.load(config.artifact)


def build_tracker(config: TrackingConfig) -> FusionEngine:
    return FusionEngine(
        ConstantVelocity(noise_intensity=config.imm_noise[0]),
        gate_probability=config.gate_probability,
        max_coast_time=config.max_coast_s,
        initial_velocity_std=config.initial_velocity_std,
        confirm_hits=config.confirm_hits,
        confirm_window=config.confirm_window,
        class_weight=config.class_weight,
        merge_probability=config.merge_probability,
        imm=config.imm,
        imm_noise=config.imm_noise,
        imm_sojourn_s=config.imm_sojourn_s,
    )


def _full_window(signal: FloatArray, length: int) -> np.ndarray:
    """The last ``length`` samples; a short history is padded at the front
    with its median (a pixel that has only just been observed)."""
    x = np.asarray(signal, dtype=np.float64)[-length:]
    if x.size < length:
        x = np.concatenate([np.full(length - x.size, np.median(x)), x])
    return x


@dataclass(frozen=True, eq=False)
class _Ray:
    sensor_index: int
    sensor_position: FloatArray
    line_of_sight: FloatArray
    angle_std: float


class LocantPipeline:
    """OPIR + RF detection, classification, geolocation, and fusion.

    Components default to those described by ``config``; pass any of them
    explicitly to override (e.g. a pre-loaded classifier or a test double).
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        detector: Detector | None = None,
        classifier: Classifier | None = None,
        geolocator: RFGeolocator | None = None,
        tracker: Tracker | None = None,
        receivers: Sequence[Receiver] | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.detector = detector or build_detector(self.config.detection)
        self.classifier = (
            classifier
            if classifier is not None
            else build_classifier(self.config.classification)
        )
        rf = self.config.rf
        self.geolocator = geolocator or TDOAFDOAGeolocator(
            SystematicErrors(rf.survey_m, rf.clock_bias_ns * 1e-9, rf.lo_offset_hz)
            if rf.has_systematics
            else None
        )
        self.tracker: Tracker = tracker or build_tracker(self.config.tracking)
        self.receivers = list(receivers or default_receiver_network())
        self.history: list[FrameResult] = []
        self._late_model = ConstantVelocity(
            noise_intensity=self.config.tracking.imm_noise[0]
        )

    @classmethod
    def from_yaml(cls, path: str | Path, **components: Any) -> "LocantPipeline":
        return cls(load_pipeline_config(path), **components)

    # -- OPIR ---------------------------------------------------------------

    def detect(
        self, signals: Sequence[FloatArray], sampling_rate: float, timestamp: float
    ) -> list[OPIRReport | None]:
        """Detect and classify a batch of pixel windows.

        Returns one entry per window: ``None`` if the detector did not fire or
        the classifier labeled the window background (e.g. a sun glint).
        """
        if not signals:
            return []
        length = round(self.config.detection.window_s * sampling_rate)
        raw = [np.asarray(x, dtype=np.float64)[-length:] for x in signals]
        score = np.empty(len(raw))
        onset_index = np.zeros(len(raw), dtype=int)
        # Score windows of equal length together; a pixel observed for less
        # than a full window is scored as is (CFAR needs real background in
        # its reference, so it must not be padded).
        for size in {x.size for x in raw}:
            group = [i for i, x in enumerate(raw) if x.size == size]
            scores = self.detector.score(
                np.stack([raw[i] for i in group]), sampling_rate
            )
            score[group] = scores.score
            onset_index[group] = scores.onset_index
        fired = np.flatnonzero(score > self.config.detection.threshold)
        reports: list[OPIRReport | None] = [None] * len(signals)
        # The classifier takes full windows; a short history is padded at the
        # front with its median (validated in E6), and onsets are measured in
        # the padded window, as in training.
        prediction = (
            self.classifier.predict(
                np.stack([_full_window(raw[i], length) for i in fired])
            )
            if self.classifier is not None and fired.size
            else None
        )
        low, high = self.config.classification.onset_range_s
        for k, i in enumerate(fired):
            onset = float(onset_index[i]) / sampling_rate
            if prediction is None:
                reports[i] = OPIRReport(timestamp, float(score[i]), onset)
                continue
            assert self.classifier is not None
            label = prediction.labels[k]
            if label == "background" and self.config.classification.reject_background:
                continue
            classes = self.classifier.classes
            reports[i] = OPIRReport(
                timestamp=timestamp,
                score=float(score[i]),
                onset_time=onset,
                label=label,
                probabilities=dict(
                    zip(classes, map(float, prediction.probabilities[k]), strict=True)
                ),
                prediction_set=tuple(
                    c
                    for c, member in zip(
                        classes, prediction.prediction_sets[k], strict=True
                    )
                    if member
                ),
                in_domain=low <= onset + (length - raw[i].size) / sampling_rate <= high,
            )
        return reports

    def _opir_measurements(
        self,
        observations: Sequence[OPIRObservation],
        reports: Sequence[OPIRReport | None],
    ) -> list[Measurement]:
        rays: list[_Ray] = []
        probabilities: list[FloatArray | None] = []
        for observation, report in zip(observations, reports, strict=True):
            if report is None or not observation.geolocatable:
                continue
            assert observation.sensor_position is not None
            assert observation.line_of_sight is not None
            rays.append(
                _Ray(
                    observation.sensor_index,
                    np.asarray(observation.sensor_position, dtype=np.float64),
                    np.asarray(observation.line_of_sight, dtype=np.float64),
                    observation.angle_std or self.config.opir.angle_std_rad,
                )
            )
            probabilities.append(
                np.array(list(report.probabilities.values()))
                if report.probabilities is not None and report.in_domain
                else None
            )
        return list(
            measurements_from_reports(
                rays,
                probabilities,
                gate_probability=self.config.opir.stereo_gate_probability,
                reject_ambiguous=self.config.opir.reject_ambiguous,
            )
        )

    # -- RF -----------------------------------------------------------------

    def locate(
        self, observation: RFObservation, frame_time: float
    ) -> tuple[LinearMeasurement | None, str]:
        """RF observation → measurement at ``frame_time``.

        Returns the measurement (or ``None``) and an outcome: ``"fused"``,
        ``"rejected"`` (no fix, or a fix failing the convergence/χ² checks),
        or ``"late"`` (too old and cannot be extrapolated).
        """
        receivers = observation.receivers or self.receivers
        try:
            fix = self.geolocator.locate(observation, receivers)
        except (GeometryError, InsufficientMeasurementsError) as exc:
            log_event(logger, logging.WARNING, "rf_fix_failed", reason=str(exc))
            return None, "rejected"
        if not accept_fix(fix, self.config.rf.fit_probability):
            log_event(
                logger,
                logging.WARNING,
                "rf_fix_rejected",
                converged=fix.converged,
                chi2=fix.chi2,
                dof=fix.dof,
            )
            return None, "rejected"
        measurement = rf_measurement(fix)
        age = frame_time - (
            frame_time if observation.time is None else observation.time
        )
        if age <= _LATE_TOLERANCE_S:
            return measurement, "fused"
        if self.config.rf.late_fixes == "extrapolate" and measurement.dim == 6:
            return extrapolate(measurement, age, self._late_model), "fused"
        log_event(logger, logging.INFO, "rf_fix_late_dropped", age_s=age)
        return None, "late"

    # -- frames -------------------------------------------------------------

    def process_frame(self, frame: SensorFrame) -> FrameResult:
        """Run every stage on one frame and update the tracks."""
        reports = self.detect(
            [o.signal for o in frame.opir], frame.sampling_rate, frame.time
        )
        opir = self._opir_measurements(frame.opir, reports)
        outcomes = Counter[str]()
        rf: list[Measurement] = []
        for observation in frame.rf:
            measurement, outcome = self.locate(observation, frame.time)
            outcomes[outcome] += 1
            if measurement is not None:
                rf.append(measurement)
        self.tracker.process([*rf, *opir], frame.time)
        result = FrameResult(
            time=frame.time,
            opir_reports=tuple(r for r in reports if r is not None),
            opir_measurements=len(opir),
            rf_fixes=len(rf),
            rf_rejected=outcomes["rejected"],
            rf_late_dropped=outcomes["late"],
            tracks=tuple(self.tracker.confirmed_tracks),
        )
        log_event(
            logger,
            logging.DEBUG,
            "frame_processed",
            time=frame.time,
            opir_detections=len(result.opir_reports),
            opir_measurements=result.opir_measurements,
            rf_fixes=result.rf_fixes,
            rf_rejected=result.rf_rejected,
            confirmed_tracks=len(result.tracks),
        )
        self.history.append(result)
        return result

    # -- outputs ------------------------------------------------------------

    @property
    def tracks(self) -> list[Track]:
        return self.tracker.tracks

    @property
    def confirmed_tracks(self) -> list[Track]:
        return self.tracker.confirmed_tracks

    @property
    def classes(self) -> tuple[str, ...] | None:
        return None if self.classifier is None else self.classifier.classes

    def summary(self) -> dict[str, Any]:
        """Counts and uncertainty over the confirmed tracks."""
        tracks = self.confirmed_tracks
        sources: Counter[str] = Counter()
        for track in tracks:
            sources.update(track.hits_by_source.keys())
        uncertainty = [t.position_rms_uncertainty for t in tracks]
        return {
            "time": self.tracker.time,
            "confirmed_tracks": len(tracks),
            "tentative_tracks": len(self.tracks) - len(tracks),
            "tracks_with_source": dict(sources),
            "multi_source_tracks": sum(len(t.hits_by_source) > 1 for t in tracks),
            "mean_position_rms_uncertainty_m": (
                float(np.mean(uncertainty)) if uncertainty else None
            ),
        }

    def track_by_id(self, track_id: int) -> Track | None:
        return next((t for t in self.tracks if t.id == track_id), None)

    def export_tracks(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(
            json.dumps(
                {
                    "config_sha256": self.config.sha256(),
                    "summary": self.summary(),
                    "tracks": [
                        track_to_dict(t, self.classes) for t in self.confirmed_tracks
                    ],
                },
                indent=2,
            )
        )
        return out
