"""The bias floor: shared, time-constant measurement error on tracks."""

import numpy as np
import pytest

from locant.core import mean_nees_bounds, nees
from locant.fusion import extrapolate
from locant.tracking import ConstantVelocity, LinearMeasurement, MultiTargetTracker

MODEL = ConstantVelocity(noise_intensity=1e-6)
RANDOM = np.eye(3) * 100.0  # 10 m per axis, independent scan to scan
BIAS = np.eye(3) * 400.0  # 20 m per axis, fixed over a run


def run_with_bias(rng: np.random.Generator, scans: int) -> tuple[float, float]:
    """NEES of the filter and reported covariances after ``scans`` updates of a
    stationary target whose fixes share one bias draw."""
    tracker = MultiTargetTracker(MODEL, confirm_hits=1, gate_probability=0.9999)
    truth = np.array([1_000.0, 2_000.0, 3_000.0])
    bias = rng.multivariate_normal(np.zeros(3), BIAS)
    for t in range(scans):
        z = truth + bias + rng.multivariate_normal(np.zeros(3), RANDOM)
        tracker.step(
            [
                LinearMeasurement.position(
                    z, RANDOM + BIAS, "rf", systematic_covariance=BIAS
                )
            ],
            float(t),
        )
    (track,) = tracker.tracks
    error = track.position - truth
    return (
        nees(error, track.state.covariance[:3, :3]),
        nees(error, track.position_covariance),
    )


def test_floor_keeps_long_tracks_consistent(rng):
    runs = 200
    filtered, reported = zip(
        *(run_with_bias(rng, 40) for _ in range(runs)), strict=True
    )
    low, high = mean_nees_bounds(3, runs, confidence=0.99)
    assert np.mean(filtered) > 3 * high  # averaging cannot remove a shared bias
    assert low <= np.mean(reported) <= high


def test_floor_is_reported_but_not_used_for_gating():
    tracker = MultiTargetTracker(MODEL, confirm_hits=1)
    tracker.step(
        [
            LinearMeasurement.position(
                np.zeros(3), RANDOM + BIAS, "rf", systematic_covariance=BIAS
            )
        ],
        0.0,
    )
    (track,) = tracker.tracks
    assert track.bias_floor is not None
    np.testing.assert_allclose(track.bias_floor[:3, :3], BIAS)
    np.testing.assert_allclose(track.bias_floor[3:, 3:], 0.0)
    np.testing.assert_allclose(
        track.reported.covariance, track.state.covariance + track.bias_floor
    )
    np.testing.assert_allclose(track.position_covariance, (RANDOM + BIAS) + BIAS)


def test_measurement_validation_and_extrapolation():
    with pytest.raises(ValueError, match="systematic_covariance"):
        LinearMeasurement.position(
            np.zeros(3), RANDOM, "rf", systematic_covariance=np.eye(2)
        )
    fix = LinearMeasurement.position_velocity(
        np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0]),
        np.eye(6),
        "rf",
        systematic_covariance=np.eye(6) * 0.5,
    )
    late = extrapolate(fix, 2.0, ConstantVelocity(noise_intensity=25.0))
    np.testing.assert_allclose(late.value[:3], [200.0, 0.0, 0.0])
    assert np.trace(late.covariance) > np.trace(fix.covariance)
    assert late.systematic_covariance is not None
    assert late.systematic_covariance[0, 0] == pytest.approx(0.5 + 4 * 0.5)
    with pytest.raises(ValueError, match="position-velocity"):
        extrapolate(LinearMeasurement.position(np.zeros(3), RANDOM, "rf"), 1.0, MODEL)
