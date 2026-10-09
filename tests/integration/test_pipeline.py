"""End-to-end tests of the multi-sensor pipeline on simulated scenarios."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sentinel.classification import (
    ModelArtifact,
    TrainConfig,
    save_artifact,
    train_model,
)
from sentinel.classification.artifact import ConformalSpec
from sentinel.data.build import generate_samples
from sentinel.data.config import Priors
from sentinel.fusion import FusionEngine
from sentinel.geolocation import simulate_fdoa, simulate_tdoa
from sentinel.pipeline.phase3_pipeline import (
    CFAR_THRESHOLD_PFA_1E2,
    OPIRObservation,
    OPIRReport,
    SENTINELPhase3Pipeline,
    demo_phase3_system,
)
from sentinel.taxonomy import EVENT_CLASSES

ROOT = Path(__file__).resolve().parents[2]
START = np.array([5_000.0, 5_000.0, 500.0])
VELOCITY = np.array([100.0, 50.0, 0.0])


@pytest.fixture
def pipeline() -> SENTINELPhase3Pipeline:
    return SENTINELPhase3Pipeline()


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


@pytest.mark.parametrize("with_fdoa", [False, True])
def test_moving_emitter_is_tracked_within_its_uncertainty(pipeline, rng, with_fdoa):
    for frame in range(15):
        t = float(frame)
        truth = START + VELOCITY * t
        tdoa = simulate_tdoa(truth, pipeline.receivers, 10e-9, rng)
        rf = (
            (tdoa, simulate_fdoa(truth, VELOCITY, pipeline.receivers, 1e9, 1.0, rng))
            if with_fdoa
            else tdoa
        )
        result = pipeline.process_multi_sensor_frame([], [rf], 10.0, t)
    assert result["fused_tracks"] == 1
    (track,) = pipeline.fusion_engine.tracks
    assert np.linalg.norm(track.position - truth) < 4 * track.position_rms_uncertainty
    np.testing.assert_allclose(track.velocity, VELOCITY, atol=15.0)


def test_rf_failures_are_reported_not_raised(pipeline, rng):
    too_few = simulate_tdoa(START, pipeline.receivers[:3], 10e-9, rng)
    result = pipeline.process_multi_sensor_frame([], [too_few], 10.0, 0.0)
    assert result["rf_failures"] == 1
    assert result["fused_tracks"] == 0


def test_opir_detection_without_classifier(pipeline, windows):
    launch = pipeline.process_opir_signal(windows["launch"][0], 10.0, 0.0)
    assert launch is not None
    assert launch.label is None
    assert launch.score > CFAR_THRESHOLD_PFA_1E2
    assert pipeline.process_opir_signal(np.full(640, 200.0), 10.0, 0.0) is None


def test_opir_classification_and_background_rejection(windows, tiny_classifier):
    pipeline = SENTINELPhase3Pipeline(
        classifier=tiny_classifier, detection_threshold=0.0
    )
    report = pipeline.process_opir_signal(windows["explosion"][0], 10.0, 0.0)
    assert report is not None
    assert report.label in EVENT_CLASSES
    assert report.probabilities is not None
    assert sum(report.probabilities.values()) == pytest.approx(1.0)
    assert report.label in report.prediction_set
    frame = pipeline.process_multi_sensor_frame(
        list(windows["launch"][:2]), [], 10.0, 1.0
    )
    assert frame["opir_detections"] == len(frame["opir_labels"])


def test_geolocated_opir_detections_start_fused_tracks(pipeline, windows, rng):
    """Two satellites detecting the same event triangulate it; the stereo
    position starts and confirms a track without any RF."""
    geo, heo = np.array([0.0, -30e6, 25e6]), np.array([10e6, 15e6, 35e6])
    target = np.array([2_000.0, 3_000.0, 12_000.0])

    def observe(sensor_index: int, sensor: np.ndarray) -> OPIRObservation:
        u = (target - sensor) / np.linalg.norm(target - sensor)
        u = u + rng.normal(0.0, 10e-6, 3)
        return OPIRObservation(
            windows["launch"][0],
            sensor_index=sensor_index,
            sensor_position=sensor,
            line_of_sight=u / np.linalg.norm(u),
        )

    for t in range(5):
        result = pipeline.process_multi_sensor_frame(
            [observe(0, geo), observe(1, heo)], [], 10.0, float(t)
        )
    assert result["opir_measurements"] == 1
    (track,) = pipeline.fusion_engine.confirmed_tracks
    assert track.hits_by_source == {"opir": 5}
    assert np.linalg.norm(track.position - target) < 4 * track.position_rms_uncertainty


def test_unlocated_opir_detections_are_reported_not_fused(pipeline, windows):
    result = pipeline.process_multi_sensor_frame([windows["launch"][0]], [], 10.0, 0.0)
    assert result["opir_detections"] == 1
    assert result["opir_measurements"] == 0
    assert result["fused_tracks"] == 0


def test_situation_awareness_and_export(pipeline, rng, tmp_path):
    for t in range(5):
        tdoa = simulate_tdoa(START + VELOCITY * t, pipeline.receivers, 10e-9, rng)
        pipeline.process_multi_sensor_frame([], [tdoa], 10.0, float(t))
    assert pipeline.get_situation_awareness()["total_tracks"] == 1
    track_id = pipeline.fusion_engine.tracks[0].id
    assert pipeline.get_track_by_id(track_id) is not None
    assert pipeline.get_track_by_id(track_id + 100) is None
    path = tmp_path / "tracks.json"
    pipeline.export_tracks_to_file(path)
    assert json.loads(path.read_text())["tracks"][0]["hits_by_source"] == {"rf": 5}


def test_demo_runs(capsys, tiny_classifier):
    demo_phase3_system(seed=1, classifier=tiny_classifier)
    out = capsys.readouterr().out
    assert "OPIR launch-1" in out
    assert "track 0" in out


@pytest.mark.parametrize(("onset", "classified"), [(10.0, True), (55.0, False)])
def test_class_evidence_only_when_onset_is_in_the_trained_range(
    monkeypatch, onset, classified
):
    """Windows whose onset lies outside the classifier's training range
    still update the track kinematically but add no class evidence."""
    pipeline = SENTINELPhase3Pipeline(fusion_engine=FusionEngine(confirm_hits=1))
    report = OPIRReport(
        timestamp=0.0,
        score=10.0,
        onset_time=onset,
        label="launch",
        probabilities=dict.fromkeys(EVENT_CLASSES, 0.2),
    )
    monkeypatch.setattr(pipeline, "process_opir_signal", lambda *_: report)
    geo, heo = np.array([0.0, -30e6, 25e6]), np.array([10e6, 15e6, 35e6])
    target = np.array([2_000.0, 3_000.0, 12_000.0])
    observations = [
        OPIRObservation(
            np.zeros(640),
            sensor_index=i,
            sensor_position=s,
            line_of_sight=(target - s) / np.linalg.norm(target - s),
        )
        for i, s in enumerate((geo, heo))
    ]
    pipeline.process_multi_sensor_frame(observations, [], 10.0, 0.0)
    (track,) = pipeline.fusion_engine.tracks
    assert (track.class_posterior is not None) is classified
