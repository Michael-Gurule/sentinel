import numpy as np

from sentinel.fusion import FusionEngine, SensorType, opir_measurement, rf_measurement
from sentinel.geolocation import (
    simulate_fdoa,
    simulate_tdoa,
    solve_tdoa,
    solve_tdoa_fdoa,
)


def test_rf_measurement_position_only(receivers, emitter, rng):
    result = solve_tdoa(receivers, simulate_tdoa(emitter, receivers, 10e-9, rng))
    m = rf_measurement(result)
    assert m.source == SensorType.RF
    assert m.matrix.shape == (3, 6)
    np.testing.assert_allclose(m.covariance, result.position_covariance)


def test_rf_measurement_keeps_position_velocity_cross_covariance(
    moving_receivers, emitter, emitter_velocity, rng
):
    tdoa = simulate_tdoa(emitter, moving_receivers, 10e-9, rng)
    fdoa = simulate_fdoa(emitter, emitter_velocity, moving_receivers, 1e9, 1.0, rng)
    result = solve_tdoa_fdoa(moving_receivers, tdoa, fdoa)
    m = rf_measurement(result, label="radar")
    assert m.matrix.shape == (6, 6)
    np.testing.assert_allclose(m.covariance, result.state_covariance)
    assert np.any(m.covariance[:3, 3:] != 0.0)
    assert m.label == "radar"


def test_opir_measurement_is_a_position_measurement():
    m = opir_measurement(np.ones(3), np.eye(3) * 400.0, label="launch")
    assert m.source == SensorType.OPIR
    assert m.label == "launch"


def test_engine_fuses_two_modalities_into_one_track():
    engine = FusionEngine()
    truth = np.array([1_000.0, 2_000.0, 500.0])
    rf = opir_measurement  # same constructor shape; source differs below
    engine.process([rf_like(truth, 100.0), rf(truth + 5.0, np.eye(3) * 900.0)], 0.0)
    (track,) = engine.tracks
    assert track.hits_by_source == {"rf": 1, "opir": 1}
    # Fused uncertainty is below the better single sensor (100 m² per axis).
    assert np.all(np.diag(track.position_covariance) < 100.0)
    summary = engine.summary()
    assert summary["total_tracks"] == 1
    assert summary["multi_source_tracks"] == 1


def test_engine_reports_time_and_gate():
    engine = FusionEngine(max_coast_time=2.0)
    engine.process([rf_like(np.zeros(3), 25.0)], 1.0)
    assert engine.time == 1.0
    (track,) = engine.tracks
    assert engine.in_gate(track, rf_like(np.ones(3), 25.0))
    assert not engine.in_gate(track, rf_like(np.full(3, 1e4), 25.0))
    engine.process([], 4.0)
    assert engine.summary()["mean_position_rms_uncertainty_m"] is None


def rf_like(position, variance):
    from sentinel.tracking import LinearMeasurement

    return LinearMeasurement.position(position, np.eye(3) * variance, SensorType.RF)
