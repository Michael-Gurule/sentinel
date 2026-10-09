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
from sentinel.geolocation import simulate_fdoa, simulate_tdoa
from sentinel.pipeline.phase3_pipeline import (
    CFAR_THRESHOLD_PFA_1E2,
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
