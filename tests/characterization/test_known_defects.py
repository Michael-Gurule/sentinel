"""
Regression tests for defects found in the v1 audit (REFACTOR_PLAN.md, Part 2).

Each test asserts the *correct* behavior and is named after its defect ID.
Open defects carry a strict ``xfail`` marker; when a fix lands the test starts
passing, strict xfail turns that into a failure, and the marker is removed in
the same change. Fixed defects keep their test as a regression guard.

All tests are deterministic, so they cannot pass or fail by chance.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sentinel.classification import ModelArtifact, TrainConfig
from sentinel.core import GeometryError
from sentinel.core.constants import SPEED_OF_LIGHT
from sentinel.data import build_dataset, load_dataset_config
from sentinel.data.build import generate_samples
from sentinel.data.config import Priors
from sentinel.detection import CFARDetector, CUSUMDetector, StepGLRTDetector
from sentinel.fusion import FusionEngine
from sentinel.geolocation import (
    Receiver,
    TDOAMeasurement,
    chan_ho,
    difference_covariance,
    solve_ranges,
    tdoa_dop,
)
from sentinel.geolocation.models import range_difference_model
from sentinel.pipeline.phase3_pipeline import (
    CFAR_THRESHOLD_PFA_1E2,
    SENTINELPhase3Pipeline,
)
from sentinel.sim import load_scenario, simulate_scenario
from sentinel.taxonomy import EVENT_CLASSES
from sentinel.tracking import ConstantVelocity, LinearMeasurement

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "reports" / "phase3"

# Non-coplanar sensor layout (meters, local ENU) and an emitter inside it.
SENSORS = np.array(
    [
        [0.0, 0.0, 0.0],
        [10_000.0, 0.0, 200.0],
        [10_000.0, 10_000.0, 50.0],
        [0.0, 10_000.0, 400.0],
        [5_000.0, -3_000.0, 1_000.0],
    ]
)
EMITTER = np.array([3_000.0, 4_000.0, 800.0])
FS = 10.0


def known_defect(defect_id: str, reason: str, **kwargs):
    """Strict xfail tagged with an audit defect ID."""
    return pytest.mark.xfail(strict=True, reason=f"{defect_id}: {reason}", **kwargs)


def _windows(
    label: str, count: int, **scene: object
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    priors = Priors.model_validate({"scene": scene}) if scene else Priors()
    return generate_samples(11, f"characterization_{label}", label, count, priors, 64.0)


# --------------------------------------------------------------------------
# C1 (open, Phase 5): OPIR detections never contribute to fused tracks
# --------------------------------------------------------------------------
@known_defect(
    "C1",
    "OPIR has no position until line-of-sight geolocation (Phase 5), so OPIR "
    "reports cannot update fused tracks",
)
def test_c1_opir_contributes_to_fused_tracks():
    result = simulate_scenario(
        load_scenario(REPO_ROOT / "configs/scenario/launch_with_radar.yaml"), seed=0
    )
    pipeline = SENTINELPhase3Pipeline()
    launch = result.opir["launch-1"].measured
    for emitter, scan in result.rf_scans:
        if emitter != "radar-1":
            continue
        end = round(scan.t * FS) + 1
        window = launch[max(0, end - 640) : end] if end > 20 else launch[:640]
        fix = pipeline.process_rf_measurements(scan.tdoa, scan.receivers)
        pipeline.process_multi_sensor_frame([window], [], FS, scan.t)
        if fix is not None:
            pipeline.fusion_engine.process([fix], scan.t)

    tracks = pipeline.fusion_engine.tracks
    assert tracks, "expected at least one fused track"
    assert any({"opir", "rf"} <= t.hits_by_source.keys() for t in tracks)


# --------------------------------------------------------------------------
# C3 (fixed in Phases 1-2): the dataset could not be regenerated
# --------------------------------------------------------------------------
def test_c3_dataset_is_reproducible_from_committed_code(tmp_path):
    config = load_dataset_config(REPO_ROOT / "configs/dataset/opir_v2.yaml").scaled(
        0.002
    )
    first = build_dataset(config, tmp_path / "a")
    second = build_dataset(config, tmp_path / "b", workers=2)
    for name, split in first["splits"].items():
        assert split["content_sha256"] == second["splits"][name]["content_sha256"]


def test_c3_versioned_manifest_matches_committed_config():
    """The checked-in manifest must be rebuilt whenever the dataset config changes."""
    manifest = json.loads((REPO_ROOT / "data/manifests/opir_v2.json").read_text())
    config = load_dataset_config(REPO_ROOT / "configs/dataset/opir_v2.yaml")
    config_json = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    assert manifest["config_sha256"] == hashlib.sha256(config_json.encode()).hexdigest()


# --------------------------------------------------------------------------
# C4 (fixed in Phases 1, 3): classifier contracts disagreed
# --------------------------------------------------------------------------
def test_c4_artifact_carries_the_training_contract():
    """Class order and preprocessing travel with the weights, so inference
    cannot silently use a different taxonomy or transform than training."""
    config = TrainConfig(
        model="tcn", preprocess={"normalization": "zscore", "length": 100}
    )
    artifact = ModelArtifact(name="x", model=config.model, preprocess=config.preprocess)
    restored = ModelArtifact.model_validate_json(artifact.model_dump_json())
    assert restored.classes == EVENT_CLASSES
    assert restored.preprocess == config.preprocess
    shipped = REPO_ROOT / "models/opir_event_classifier/artifact.json"
    if shipped.exists():
        exported = ModelArtifact.model_validate_json(shipped.read_text())
        assert exported.classes == EVENT_CLASSES


# --------------------------------------------------------------------------
# C5 (fixed in Phase 3): detectors raised false alarms on background
# --------------------------------------------------------------------------
def test_c5_false_alarm_rate_on_glint_free_background():
    """The pipeline's CFAR threshold holds its design rate (1e-2) on new background."""
    background, _ = _windows("background", 400, glint_probability=0.0)
    alarms = CFARDetector().score(background, FS).score > CFAR_THRESHOLD_PFA_1E2
    assert alarms.mean() <= 0.03


@pytest.mark.parametrize(
    "detector", [CFARDetector(), CUSUMDetector(), StepGLRTDetector()]
)
def test_c5_detectors_ignore_constant_signal(detector):
    threshold = detector.analytic_threshold(1e-2, 640, FS)
    assert detector.score(np.full(640, 280.0), FS).score[0] < threshold


def test_c5_detection_does_not_precede_the_event():
    launches, meta = _windows("launch", 20, glint_probability=0.0)
    scores = CFARDetector().score(launches, FS)
    detected = scores.score > CFAR_THRESHOLD_PFA_1E2
    assert detected.mean() > 0.5
    onset_s = scores.onset_index[detected] / FS
    assert np.all(onset_s >= meta["onset_s"][detected] - 0.5)


# --------------------------------------------------------------------------
# H11 (fixed in Phase 3): the v1 temporal-difference detector missed launches
# and explosions; the replacement detectors catch them
# --------------------------------------------------------------------------
@pytest.mark.parametrize("event", ["launch", "explosion"])
def test_h11_detectors_catch_target_events(event):
    signals, meta = _windows(event, 30, glint_probability=0.0)
    bright = meta["peak_snr"] > 10.0
    assert bright.sum() >= 10
    scores = CFARDetector().score(signals[bright], FS).score
    assert np.mean(scores > CFAR_THRESHOLD_PFA_1E2) >= 0.9


# --------------------------------------------------------------------------
# C8 (fixed in Phase 3): the headline result was 100% on a toy task
# --------------------------------------------------------------------------
def test_c8_reported_results_are_honest():
    """Results come with baselines, seed CIs, and shift sets, and are not
    the trivially perfect accuracy of a separable toy problem."""
    report = json.loads((REPORTS / "e2_classification.json").read_text())
    summary = report["summary"]
    assert {"logistic", "gbm", "cnn", "tcn"} <= set(summary)
    best = summary[report["selected_model"]]["test"]["macro_f1"]
    assert best["value"] < 0.99
    assert best["ci_low"] <= best["value"] <= best["ci_high"]
    for shift in ("shift_low_snr", "shift_params", "shift_clutter"):
        assert shift in summary[report["selected_model"]]


def test_pipeline_threshold_matches_e1_report():
    report = json.loads((REPORTS / "e1_detection.json").read_text())
    calibrated = report["results"]["cfar"]["pfa_0.01"]["calibrated_glint_free"][
        "threshold"
    ]
    assert calibrated == pytest.approx(CFAR_THRESHOLD_PFA_1E2, abs=0.01)


# --------------------------------------------------------------------------
# C6 (fixed in Phase 1): weighted least squares diverged yet reported success
# --------------------------------------------------------------------------
def test_c6_wls_recovers_noiseless_position():
    ranges = np.linalg.norm(SENSORS - EMITTER, axis=1)
    position, covariance, converged = solve_ranges(
        SENSORS, ranges, np.eye(len(SENSORS)), initial=SENSORS.mean(axis=0)
    )
    assert converged
    np.testing.assert_allclose(position, EMITTER, atol=1e-3)
    assert np.all(np.isfinite(covariance))


# --------------------------------------------------------------------------
# C7 (fixed in Phase 1): "Chan's algorithm" was not Chan-Ho
# --------------------------------------------------------------------------
def test_c7_chan_recovers_noiseless_position():
    receivers = [Receiver(i, p) for i, p in enumerate(SENSORS)]
    rd, _ = range_difference_model(EMITTER, receivers[0], receivers[1:])
    measurement = TDOAMeasurement(
        reference=0,
        others=(1, 2, 3, 4),
        values=rd / SPEED_OF_LIGHT,
        covariance=difference_covariance(4, 1e-9),
    )
    result = chan_ho(receivers, measurement)
    np.testing.assert_allclose(result.position, EMITTER, atol=1e-3)


# --------------------------------------------------------------------------
# H1 (fixed in Phase 1): fusion prediction ignored elapsed time
# --------------------------------------------------------------------------
def test_h1_fusion_prediction_uses_elapsed_time():
    engine = FusionEngine()
    p0, v0 = np.array([1_000.0, 2_000.0, 0.0]), np.array([10.0, -5.0, 0.0])
    state = np.concatenate([p0, v0])
    engine.process(
        [LinearMeasurement.position_velocity(state, np.eye(6), "rf")], timestamp=0.0
    )

    engine.process([], timestamp=5.0)

    (track,) = engine.tracks
    np.testing.assert_allclose(track.position, p0 + v0 * 5.0, atol=1e-6)


# --------------------------------------------------------------------------
# H2 (fixed in Phase 1): the association gate accepted distant measurements
# --------------------------------------------------------------------------
def test_h2_gate_rejects_distant_measurement():
    engine = SENTINELPhase3Pipeline().fusion_engine
    sigma_10m = np.eye(3) * 100.0
    engine.process([LinearMeasurement.position(np.zeros(3), sigma_10m, "rf")], 0.0)
    (track,) = engine.tracks
    far_away = LinearMeasurement.position(np.array([5e3, 0.0, 0.0]), sigma_10m, "rf")

    assert not engine.in_gate(track, far_away)


# --------------------------------------------------------------------------
# H4 (fixed in Phase 1): process noise was not a valid covariance
# --------------------------------------------------------------------------
def test_h4_process_noise_is_positive_semidefinite():
    q = ConstantVelocity(noise_intensity=1.0).process_noise(1.0)
    np.testing.assert_allclose(q, q.T)
    assert np.linalg.eigvalsh(q).min() >= -1e-12


# --------------------------------------------------------------------------
# H5 (fixed in Phase 1): DOP returned a magic 999 for unsolvable geometry
# --------------------------------------------------------------------------
def test_h5_unobservable_geometry_raises_instead_of_magic_value():
    with pytest.raises(GeometryError):
        tdoa_dop(EMITTER, SENSORS[:3])
