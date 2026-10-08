"""
Characterization tests for defects found in the v1 audit (REFACTOR_PLAN.md, Part 2).

Each test asserts the *correct* behavior and is marked as a strict expected
failure tied to its defect ID. When a fix lands, the test starts passing,
strict xfail turns that into a failure, and the marker must be removed in the
same change, which records the fix in the test suite.

All tests are deterministic: they fail because of the defect, not because of
random draws, so they cannot pass by chance.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from sentinel.detection.opir_detectors import MultiMethodDetector
from sentinel.fusion.sensor_fusion import (
    SensorFusionEngine,
    SensorMeasurement,
    SensorType,
)
from sentinel.geolocation.multilateration import (
    HyperbolicMultilateration,
    RangeDifferenceMeasurement,
    RangeMeasurement,
    WeightedLeastSquares,
)
from sentinel.geolocation.tdoa_fdoa import simulate_tdoa_measurements
from sentinel.inference.opir_inference import OPIRInference
from sentinel.models.cnn_classifier import OPIRClassifier
from sentinel.models.signal_generator import OPIRSignalGenerator
from sentinel.pipeline.phase3_pipeline import SENTINELPhase3Pipeline
from sentinel.tracking.kalman_tracker import KalmanFilter

REPO_ROOT = Path(__file__).resolve().parents[2]

# Non-coplanar sensor layout (meters, local ENU) and an emitter inside it.
SENSORS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([10_000.0, 0.0, 200.0]),
    np.array([10_000.0, 10_000.0, 50.0]),
    np.array([0.0, 10_000.0, 400.0]),
    np.array([5_000.0, -3_000.0, 1_000.0]),
]
EMITTER = np.array([3_000.0, 4_000.0, 800.0])


def known_defect(defect_id: str, reason: str, **kwargs):
    """Strict xfail tagged with an audit defect ID."""
    return pytest.mark.xfail(strict=True, reason=f"{defect_id}: {reason}", **kwargs)


@pytest.fixture(autouse=True)
def _seed_global_rng():
    # v1 code draws from the global numpy/torch RNGs (debt NPY002, Phase 2).
    np.random.seed(1234)  # noqa: NPY002 - pins legacy global RNG used by v1
    torch.manual_seed(1234)


# --------------------------------------------------------------------------
# C1: OPIR detections never contribute to fused tracks
# --------------------------------------------------------------------------
@known_defect(
    "C1",
    "OPIR measurements have position=None and are dropped by the fusion engine",
)
def test_c1_opir_contributes_to_fused_tracks():
    pipeline = SENTINELPhase3Pipeline()
    generator = OPIRSignalGenerator()
    start, velocity = np.array([5_000.0, 5_000.0, 500.0]), np.array([100.0, 50.0, 0])

    for frame in range(5):
        t = float(frame)
        pipeline.process_multi_sensor_frame(
            opir_signals=[generator.generate_launch_signature(start_time=2.0)],
            rf_measurements=[
                simulate_tdoa_measurements(
                    start + velocity * t, pipeline.sensors, noise_std=1e-8
                )
            ],
            sampling_rate=generator.sampling_rate,
            timestamp=t,
        )

    tracks = pipeline.fusion_engine.get_tracks()
    assert tracks, "expected at least one fused track"
    assert any(t.opir_detections > 0 and t.rf_detections > 0 for t in tracks)
    assert any(t.event_type is not None for t in tracks)


# --------------------------------------------------------------------------
# C3: the training dataset cannot be regenerated from committed code
# --------------------------------------------------------------------------
@known_defect(
    "C3",
    "generate_background() references self.t, which is never defined",
    raises=AttributeError,
)
def test_c3_generator_produces_background_scenario():
    generator = OPIRSignalGenerator(duration=10.0, sampling_rate=10)
    scenario, events = generator.generate_scenario([])
    assert scenario.shape == (generator.num_samples,)
    assert events == []


@known_defect(
    "C3",
    "generate_opir_dataset.py passes sample_rate_hz/duration_s, which "
    "OPIRSignalGenerator has never accepted",
    raises=TypeError,
)
def test_c3_dataset_script_runs(tmp_path):
    script = REPO_ROOT / "scripts" / "generate_opir_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_opir_dataset", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.generate_dataset(
        num_samples_per_class=1, output_dir=str(tmp_path), sequence_length=100
    )
    assert (tmp_path / "metadata.json").exists()


# --------------------------------------------------------------------------
# C4: classifier contracts disagree between pipeline and inference
# --------------------------------------------------------------------------
@known_defect(
    "C4",
    "OPIRClassifier uses 4 classes / 256 samples; OPIRInference uses "
    "5 classes / 100 samples",
)
def test_c4_classifier_contracts_agree():
    assert list(OPIRClassifier.CLASS_NAMES) == list(OPIRInference.CLASS_NAMES)
    assert OPIRClassifier().model.input_length == 100


# --------------------------------------------------------------------------
# C5: the ensemble detector fires on pure background noise
# --------------------------------------------------------------------------
@known_defect(
    "C5",
    "RiseTimeDetector uses an absolute rate threshold and TemporalDifference "
    "estimates its baseline from 5 samples",
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


# --------------------------------------------------------------------------
# C6: weighted least squares diverges yet reports success
# --------------------------------------------------------------------------
@known_defect(
    "C6",
    "Jacobian sign is inverted, so Gauss-Newton steps away from the solution",
)
def test_c6_wls_recovers_noiseless_position():
    measurements = [
        RangeMeasurement(
            sensor_id=i,
            sensor_position=p,
            range=float(np.linalg.norm(EMITTER - p)),
            std=1.0,
        )
        for i, p in enumerate(SENSORS)
    ]
    position, covariance, success = WeightedLeastSquares().solve_wls(measurements)

    assert success
    np.testing.assert_allclose(position, EMITTER, atol=1e-3)
    assert np.all(np.isfinite(covariance))


# --------------------------------------------------------------------------
# C7: "Chan's algorithm" is not Chan-Ho and is wrong on noiseless data
# --------------------------------------------------------------------------
@known_defect(
    "C7",
    "linearization drops the unknown reference range; not the Chan-Ho estimator",
)
def test_c7_chan_recovers_noiseless_position():
    ref = SENSORS[0]
    measurements = [
        RangeDifferenceMeasurement(
            sensor_1_id=0,
            sensor_2_id=i,
            sensor_1_position=ref,
            sensor_2_position=p,
            range_difference=float(
                np.linalg.norm(EMITTER - p) - np.linalg.norm(EMITTER - ref)
            ),
            std=1.0,
        )
        for i, p in enumerate(SENSORS[1:], start=1)
    ]
    position, success, _ = HyperbolicMultilateration().solve(
        measurements, method="chan"
    )

    assert success
    np.testing.assert_allclose(position, EMITTER, atol=1e-3)


# --------------------------------------------------------------------------
# H1: fusion-engine prediction ignores the elapsed time
# --------------------------------------------------------------------------
@known_defect(
    "H1",
    "predict_tracks(dt) uses a KalmanFilter built with a fixed dt=1.0",
)
def test_h1_fusion_prediction_uses_elapsed_time():
    engine = SensorFusionEngine()
    p0, v0 = np.array([1_000.0, 2_000.0, 0.0]), np.array([10.0, -5.0, 0.0])
    engine.initialize_track(
        [
            SensorMeasurement(
                sensor_type=SensorType.RF,
                timestamp=0.0,
                position=p0,
                velocity=v0,
                covariance=np.eye(6),
                confidence=0.9,
            )
        ]
    )

    engine.process_measurements([], timestamp=5.0)

    (track,) = engine.get_tracks()
    np.testing.assert_allclose(track.position, p0 + v0 * 5.0, atol=1e-6)


# --------------------------------------------------------------------------
# H2: the association gate accepts measurements kilometers away
# --------------------------------------------------------------------------
@known_defect(
    "H2",
    "gate uses track covariance instead of innovation covariance, and the "
    "pipeline threshold (2000) is not a chi-square value",
)
def test_h2_gate_rejects_distant_measurement():
    engine = SENTINELPhase3Pipeline().fusion_engine
    engine.initialize_track(
        [
            SensorMeasurement(
                sensor_type=SensorType.RF,
                timestamp=0.0,
                position=np.zeros(3),
                covariance=np.eye(6) * 100.0,  # sigma = 10 m per axis
                confidence=0.9,
            )
        ]
    )
    (track,) = engine.get_tracks()
    far_away = SensorMeasurement(
        sensor_type=SensorType.RF,
        timestamp=0.0,
        position=np.array([5_000.0, 0.0, 0.0]),
        covariance=np.eye(6) * 100.0,
    )

    assert not engine.data_association.gate_measurement(far_away, track)


# --------------------------------------------------------------------------
# H4: Kalman process-noise matrix is not a valid covariance
# --------------------------------------------------------------------------
@known_defect(
    "H4",
    "position term uses dt^4/6 instead of dt^4/4, making Q indefinite "
    "(per-axis block [[1/6, 1/2], [1/2, 1]] has determinant -1/12)",
)
def test_h4_process_noise_is_positive_semidefinite():
    q = KalmanFilter(process_noise=1.0, dt=1.0).Q
    np.testing.assert_allclose(q, q.T)
    assert np.linalg.eigvalsh(q).min() >= -1e-12
