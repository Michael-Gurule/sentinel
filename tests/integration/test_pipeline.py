"""End-to-end tests of the multi-sensor pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError

from locant.classification import (
    ModelArtifact,
    TrainConfig,
    save_artifact,
    train_model,
)
from locant.classification.artifact import ConformalSpec
from locant.classification.inference import Prediction
from locant.data.build import generate_samples
from locant.data.config import Priors
from locant.detection import CFAR_THRESHOLD_PFA_1E2, DetectionScores
from locant.geolocation import simulate_fdoa, simulate_tdoa
from locant.pipeline import (
    ClassificationConfig,
    LocantPipeline,
    OPIRObservation,
    PipelineConfig,
    RFConfig,
    RFObservation,
    SensorFrame,
    TrackingConfig,
    load_pipeline_config,
    run_scenario,
)
from locant.sim import load_scenario, simulate_scenario
from locant.taxonomy import EVENT_CLASSES

ROOT = Path(__file__).resolve().parents[2]
START = np.array([5_000.0, 5_000.0, 500.0])
VELOCITY = np.array([100.0, 50.0, 0.0])
GEO, HEO = np.array([0.0, -30e6, 25e6]), np.array([10e6, 15e6, 35e6])
TARGET = np.array([2_000.0, 3_000.0, 12_000.0])


@pytest.fixture
def pipeline() -> LocantPipeline:
    return LocantPipeline()


@pytest.fixture(scope="module")
def windows() -> dict[str, np.ndarray]:
    """A few high-SNR simulated windows per class."""
    priors = Priors.model_validate(
        {
            "sensor": {"nei": {"low": 0.5, "high": 0.6}},
            "scene": {"glint_probability": 0.0},
        }
    )
    return {
        label: generate_samples(7, f"pipeline_{label}", label, 6, priors, 64.0)[0]
        for label in EVENT_CLASSES
    }


@pytest.fixture(scope="module")
def tiny_classifier(tmp_path_factory, windows) -> Path:
    signals = np.concatenate(list(windows.values()))
    labels = np.repeat(np.arange(len(EVENT_CLASSES)), 6)
    config = TrainConfig(model="cnn", epochs=10, batch_size=16, seed=0)
    result = train_model(
        config, signals, labels, signals, labels, 5, torch.device("cpu")
    )
    path = tmp_path_factory.mktemp("model") / "classifier"
    artifact = ModelArtifact(
        name="tiny",
        model="cnn",
        preprocess=config.preprocess,
        conformal=ConformalSpec(alpha=0.1, threshold=0.99),
    )
    save_artifact(path, artifact, result.model)
    return path


def stereo_pair(signal: np.ndarray, rng: np.random.Generator) -> list[OPIRObservation]:
    observations = []
    for index, sensor in enumerate((GEO, HEO)):
        u = (TARGET - sensor) / np.linalg.norm(TARGET - sensor)
        u = u + rng.normal(0.0, 10e-6, 3)
        observations.append(
            OPIRObservation(
                signal,
                sensor_index=index,
                sensor_position=sensor,
                line_of_sight=u / np.linalg.norm(u),
            )
        )
    return observations


# -- configuration -------------------------------------------------------------


def test_default_yaml_matches_code_defaults_except_deployment_choices():
    config = load_pipeline_config(ROOT / "configs/pipeline/default.yaml")
    assert config.tracking == TrackingConfig()
    assert config.detection.threshold == CFAR_THRESHOLD_PFA_1E2
    assert config.classification.artifact == Path("models/opir_event_classifier")
    assert config.rf.has_systematics
    assert len(config.sha256()) == 64
    assert config.sha256() != PipelineConfig().sha256()


def test_invalid_configs_are_rejected(tmp_path):
    with pytest.raises(ValidationError):
        TrackingConfig(confirm_hits=4, confirm_window=3)
    with pytest.raises(ValidationError):
        ClassificationConfig(onset_range_s=(40.0, 2.0))
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"tracking": {"confirm_hit": 3}}))
    with pytest.raises(ValidationError, match="confirm_hit"):
        load_pipeline_config(path)


# -- RF --------------------------------------------------------------------------


@pytest.mark.parametrize("with_fdoa", [False, True])
def test_moving_emitter_is_tracked_within_its_uncertainty(pipeline, rng, with_fdoa):
    for frame in range(15):
        t = float(frame)
        truth = START + VELOCITY * t
        tdoa = simulate_tdoa(truth, pipeline.receivers, 10e-9, rng)
        fdoa = (
            simulate_fdoa(truth, VELOCITY, pipeline.receivers, 1e9, 1.0, rng)
            if with_fdoa
            else None
        )
        result = pipeline.process_frame(SensorFrame(t, rf=[RFObservation(tdoa, fdoa)]))
    assert len(result.tracks) == 1
    (track,) = result.tracks
    assert np.linalg.norm(track.position - truth) < 4 * track.position_rms_uncertainty
    np.testing.assert_allclose(track.velocity, VELOCITY, atol=15.0)


def test_rf_failures_are_counted_not_raised(pipeline, rng):
    too_few = simulate_tdoa(START, pipeline.receivers[:3], 10e-9, rng)
    result = pipeline.process_frame(SensorFrame(0.0, rf=[RFObservation(too_few)]))
    assert result.rf_rejected == 1
    assert pipeline.tracks == []


def test_late_fixes_are_extrapolated_or_dropped(rng):
    truth = START + VELOCITY * 10.0
    late = RFObservation(
        simulate_tdoa(START + VELOCITY * 8.0, LocantPipeline().receivers, 10e-9, rng),
        simulate_fdoa(
            START + VELOCITY * 8.0,
            VELOCITY,
            LocantPipeline().receivers,
            1e9,
            1.0,
            rng,
        ),
        time=8.0,
    )
    extrapolating = LocantPipeline()
    measurement, outcome = extrapolating.locate(late, frame_time=10.0)
    assert outcome == "fused"
    assert measurement is not None
    assert np.linalg.norm(measurement.value[:3] - truth) < 60.0
    dropping = LocantPipeline(PipelineConfig(rf=RFConfig(late_fixes="drop")))
    result = dropping.process_frame(SensorFrame(10.0, rf=[late]))
    assert result.rf_late_dropped == 1


def test_systematic_levels_give_tracks_a_bias_floor(rng):
    pipeline = LocantPipeline(PipelineConfig(rf=RFConfig(clock_bias_ns=20.0)))
    for t in range(3):
        truth = START + VELOCITY * t
        tdoa = simulate_tdoa(truth, pipeline.receivers, 10e-9, rng)
        pipeline.process_frame(SensorFrame(float(t), rf=[RFObservation(tdoa)]))
    (track,) = pipeline.tracks
    assert track.bias_floor is not None
    assert np.trace(track.position_covariance) > np.trace(
        track.state.covariance[:3, :3]
    )


# -- OPIR ------------------------------------------------------------------------


def test_opir_detection_without_classifier(pipeline, windows):
    launch, constant = pipeline.detect(
        [windows["launch"][0], np.full(640, 200.0)], 10.0, 0.0
    )
    assert launch is not None
    assert launch.label is None
    assert launch.score > CFAR_THRESHOLD_PFA_1E2
    assert constant is None
    assert pipeline.detect([], 10.0, 0.0) == []


def test_short_windows_are_padded_without_false_alarms(pipeline, rng):
    """A pixel observed for only 20 s is padded with its median to a full
    window; the padding must not look like a step."""
    short = [200.0 + rng.normal(0.0, 1.0, 200) for _ in range(20)]
    assert all(report is None for report in pipeline.detect(short, 10.0, 0.0))


def test_opir_classification_and_background_rejection(windows, tiny_classifier):
    pipeline = LocantPipeline(
        PipelineConfig.model_validate(
            {
                "detection": {"threshold": 0.0},
                "classification": {"artifact": str(tiny_classifier)},
            }
        )
    )
    (report,) = pipeline.detect([windows["explosion"][0]], 10.0, 0.0)
    assert report is not None
    assert report.label in EVENT_CLASSES
    assert report.probabilities is not None
    assert sum(report.probabilities.values()) == pytest.approx(1.0)
    assert report.label in report.prediction_set
    result = pipeline.process_frame(
        SensorFrame(1.0, opir=[OPIRObservation(w) for w in windows["launch"][:2]])
    )
    assert all(r.label != "background" for r in result.opir_reports)


def test_geolocated_opir_detections_start_fused_tracks(pipeline, windows, rng):
    """Two satellites detecting the same event triangulate it; the stereo
    position starts and confirms a track without any RF."""
    for t in range(5):
        result = pipeline.process_frame(
            SensorFrame(float(t), opir=stereo_pair(windows["launch"][0], rng))
        )
    assert result.opir_measurements == 1
    (track,) = result.tracks
    assert track.hits_by_source == {"opir": 5}
    assert np.linalg.norm(track.position - TARGET) < 4 * track.position_rms_uncertainty


def test_unlocated_opir_detections_are_reported_not_fused(pipeline, windows):
    result = pipeline.process_frame(
        SensorFrame(0.0, opir=[OPIRObservation(windows["launch"][0])])
    )
    assert len(result.opir_reports) == 1
    assert result.opir_measurements == 0
    assert pipeline.tracks == []


class _FixedOnsetDetector:
    """Test double: every window fires, with the onset at ``onset_s``."""

    name = "fixed"

    def __init__(self, onset_s: float) -> None:
        self.onset_s = onset_s

    def score(self, signals, fs):
        n = len(signals)
        return DetectionScores(
            np.full(n, 10.0), np.full(n, round(self.onset_s * fs), dtype=int)
        )

    def analytic_threshold(self, pfa, num_frames, fs):
        return 0.0


class _UniformClassifier:
    """Test double: uniform probabilities, label launch."""

    classes = EVENT_CLASSES

    def predict(self, signals):
        n, c = len(signals), len(EVENT_CLASSES)
        return Prediction(
            labels=("launch",) * n,
            probabilities=np.full((n, c), 1.0 / c),
            prediction_sets=np.ones((n, c), dtype=bool),
            energy=np.zeros(n),
        )


@pytest.mark.parametrize(("onset", "classified"), [(10.0, True), (55.0, False)])
def test_class_evidence_only_when_onset_is_in_the_trained_range(rng, onset, classified):
    """Detections whose onset lies outside the classifier's training range
    still update the track kinematically but add no class evidence."""
    pipeline = LocantPipeline(
        PipelineConfig(tracking=TrackingConfig(confirm_hits=1)),
        detector=_FixedOnsetDetector(onset),
        classifier=_UniformClassifier(),
    )
    pipeline.process_frame(SensorFrame(0.0, opir=stereo_pair(np.zeros(640), rng)))
    (track,) = pipeline.tracks
    assert (track.class_posterior is not None) is classified


# -- outputs and scenarios ------------------------------------------------------


def test_summary_and_export(pipeline, rng, tmp_path):
    for t in range(5):
        tdoa = simulate_tdoa(START + VELOCITY * t, pipeline.receivers, 10e-9, rng)
        pipeline.process_frame(SensorFrame(float(t), rf=[RFObservation(tdoa)]))
    assert pipeline.summary()["confirmed_tracks"] == 1
    track_id = pipeline.tracks[0].id
    assert pipeline.track_by_id(track_id) is not None
    assert pipeline.track_by_id(track_id + 100) is None
    exported = json.loads(pipeline.export_tracks(tmp_path / "tracks.json").read_text())
    assert exported["tracks"][0]["hits_by_source"] == {"rf": 5}
    assert exported["config_sha256"] == pipeline.config.sha256()
    assert len(pipeline.history) == 5
    assert pipeline.history[-1].as_dict()["rf_fixes"] == 1


def test_stereo_scenario_end_to_end():
    """GEO + HEO + RF: every target is tracked and the picture is consistent."""
    result = simulate_scenario(
        load_scenario(ROOT / "configs/scenario/multi_int.yaml"), 0
    )
    pipeline = LocantPipeline(
        load_pipeline_config(ROOT / "configs/pipeline/default.yaml")
    )
    run = run_scenario(pipeline, result)
    metrics = run.metrics()
    assert all(np.isfinite(v) for v in metrics["confirmation_latency_s"].values())
    assert metrics["false_tracks_per_scan"] < 0.3
    assert metrics["nees_mean"] < 6.0
    fused = [
        t for t in pipeline.confirmed_tracks if {"rf", "opir"} & t.hits_by_source.keys()
    ]
    assert any(
        "rf" in t.hits_by_source and any(s.startswith("opir") for s in t.hits_by_source)
        for t in fused
    )
