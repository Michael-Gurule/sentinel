"""
Regression tests for defects found in the v1 audit (REFACTOR_PLAN.md, Part 2).

Each test asserts the *correct* behavior and is named after its defect ID.
Open defects carry a strict ``xfail`` marker; when a fix lands the test starts
passing, strict xfail turns that into a failure, and the marker is removed in
the same change. Fixed defects keep their test as a regression guard.

All tests are deterministic, so they cannot pass or fail by chance.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from sentinel.core import GeometryError
from sentinel.core.constants import SPEED_OF_LIGHT
from sentinel.detection.opir_detectors import (
    AnomalyDetector,
    MultiMethodDetector,
    RiseTimeDetector,
    TemporalDifferenceDetector,
)
from sentinel.fusion import FusionEngine
from sentinel.geolocation import (
    Receiver,
    TDOAMeasurement,
    chan_ho,
    difference_covariance,
    simulate_tdoa,
    solve_ranges,
    tdoa_dop,
)
from sentinel.geolocation.models import range_difference_model
from sentinel.inference.opir_inference import OPIRInference
from sentinel.models.cnn_classifier import OPIRClassifier
from sentinel.models.signal_generator import OPIRSignalGenerator
from sentinel.pipeline.phase3_pipeline import SENTINELPhase3Pipeline
from sentinel.tracking import ConstantVelocity, LinearMeasurement

REPO_ROOT = Path(__file__).resolve().parents[2]

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


def known_defect(defect_id: str, reason: str, **kwargs):
    """Strict xfail tagged with an audit defect ID."""
    return pytest.mark.xfail(strict=True, reason=f"{defect_id}: {reason}", **kwargs)


@pytest.fixture(autouse=True)
def _seed_torch():
    torch.manual_seed(1234)


# --------------------------------------------------------------------------
# C1 (open, Phase 5): OPIR detections never contribute to fused tracks
# --------------------------------------------------------------------------
@known_defect(
    "C1",
    "OPIR has no position until line-of-sight geolocation (Phase 5), so OPIR "
    "reports cannot update fused tracks",
)
def test_c1_opir_contributes_to_fused_tracks():
    rng = np.random.default_rng(0)
    pipeline = SENTINELPhase3Pipeline()
    generator = OPIRSignalGenerator(rng=rng)
    start, velocity = np.array([5_000.0, 5_000.0, 500.0]), np.array([100.0, 50.0, 0])

    for frame in range(5):
        t = float(frame)
        pipeline.process_multi_sensor_frame(
            opir_signals=[generator.generate_launch_signature(start_time=2.0)],
            rf_measurements=[
                simulate_tdoa(start + velocity * t, pipeline.receivers, 10e-9, rng)
            ],
            sampling_rate=generator.sampling_rate,
            timestamp=t,
        )

    tracks = pipeline.fusion_engine.tracks
    assert tracks, "expected at least one fused track"
    assert any({"opir", "rf"} <= t.hits_by_source.keys() for t in tracks)
    assert any(t.label is not None for t in tracks)


# --------------------------------------------------------------------------
# C3 (fixed in Phase 1): the dataset could not be regenerated
# --------------------------------------------------------------------------
def test_c3_generator_produces_background_scenario():
    generator = OPIRSignalGenerator(duration_s=10.0, sample_rate_hz=10)
    scenario, events = generator.generate_scenario([])
    assert scenario.shape == (generator.num_samples,)
    assert events == []


def test_c3_dataset_script_runs_and_is_reproducible(tmp_path):
    script = REPO_ROOT / "scripts" / "generate_opir_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_opir_dataset", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ("a", "b"):
        module.generate_dataset(
            num_samples_per_class=1,
            output_dir=str(tmp_path / name),
            sequence_length=100,
            seed=42,
        )
    assert (tmp_path / "a" / "metadata.json").exists()
    files_a = sorted((tmp_path / "a").rglob("*.npy"))
    files_b = sorted((tmp_path / "b").rglob("*.npy"))
    assert len(files_a) == 5
    for fa, fb in zip(files_a, files_b, strict=True):
        np.testing.assert_array_equal(np.load(fa), np.load(fb))


# --------------------------------------------------------------------------
# C4 (fixed in Phase 1): classifier contracts disagreed
# --------------------------------------------------------------------------
def test_c4_classifier_contracts_agree():
    assert list(OPIRClassifier.CLASS_NAMES) == list(OPIRInference.CLASS_NAMES)
    assert OPIRClassifier().model.input_length == 100


# --------------------------------------------------------------------------
# C5 (open, Phase 3): v1 detectors raise false alarms on background
# --------------------------------------------------------------------------
@known_defect(
    "C5",
    "RiseTimeDetector uses an absolute rate threshold, TemporalDifference "
    "estimates its baseline from 5 samples, and MAD scores use 3-sample windows",
)
def test_c5_false_alarm_rate_on_background_noise():
    detector = MultiMethodDetector()
    rng = np.random.default_rng(7)
    trials = 100
    false_alarms = sum(
        detector.detect(280.0 + rng.normal(0.0, 5.0, 1_000), 100.0).detected
        for _ in range(trials)
    )
    # Generous bound; the design target (Pfa ~1e-3) is set in Phase 3.
    assert false_alarms / trials <= 0.05


@known_defect(
    "C5",
    "moving-average smoothing with mode='same' pulls edge samples down, "
    "creating a spurious rise at t=0 on a constant signal",
)
def test_c5_rise_time_detector_ignores_constant_signal():
    assert not RiseTimeDetector().detect(np.full(500, 280.0), 100.0).detected


@known_defect(
    "C5",
    "MAD scores start after only 3 samples, so the first frames of pure "
    "noise exceed the threshold",
)
def test_c5_mad_detector_does_not_fire_before_the_event():
    generator = OPIRSignalGenerator(rng=np.random.default_rng(0))
    signal = generator.generate_launch_signature(start_time=2.0)
    result = AnomalyDetector(method="mad").detect(signal, 100.0)
    assert result.detection_time >= 2.0


# --------------------------------------------------------------------------
# H11 (open, Phase 3): temporal differencing misses the events it targets
# --------------------------------------------------------------------------
@known_defect(
    "H11",
    "requires 3 consecutive above-threshold frame differences; a launch rise "
    "gives at most 2 at 1 Hz and an explosion step gives 1",
)
@pytest.mark.parametrize("event", ["launch", "explosion"])
def test_h11_temporal_difference_detects_target_events(event):
    gen = OPIRSignalGenerator(
        duration_s=100.0, sample_rate_hz=1.0, rng=np.random.default_rng(0)
    )
    signal = (
        gen.generate_launch_signature(start_time=20.0)
        if event == "launch"
        else gen.generate_explosion_signature(start_time=20.0)
    )
    assert TemporalDifferenceDetector().detect(signal, 1.0).detected


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
