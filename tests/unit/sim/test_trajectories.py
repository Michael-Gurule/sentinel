import numpy as np
import pytest

from locant.sim.trajectories import (
    GRAVITY,
    BallisticBoost,
    ConstantVelocity,
    Stationary,
)


def test_stationary_and_constant_velocity():
    t = np.array([0.0, 1.0, 10.0])
    s = Stationary(np.array([1.0, 2.0, 3.0])).sample(t)
    np.testing.assert_allclose(s.position, [[1, 2, 3]] * 3)
    assert not s.thrusting.any()
    cv = ConstantVelocity(np.zeros(3), np.array([10.0, 0, -1])).sample(t)
    np.testing.assert_allclose(cv.position[2], [100.0, 0, -10])
    np.testing.assert_allclose(cv.velocity, [[10, 0, -1]] * 3)


@pytest.fixture
def boost():
    return BallisticBoost(
        np.array([100.0, 200.0, 0.0]),
        azimuth=np.radians(90.0),
        thrust_acceleration=30.0,
        burn_time=60.0,
        vertical_rise_s=10.0,
        pitch_elevation=np.radians(45.0),
    )


def test_boost_sits_on_the_pad_before_ignition(boost):
    s = boost.sample(np.array([-5.0, 0.0]))
    np.testing.assert_allclose(s.position, [[100, 200, 0]] * 2)
    assert not s.thrusting[0]
    assert s.thrusting[1]


def test_vertical_rise_matches_closed_form(boost):
    t = 6.0
    s = boost.sample(np.array([t]))
    net = 30.0 - GRAVITY
    np.testing.assert_allclose(s.position[0], [100, 200, 0.5 * net * t**2], atol=1e-6)
    np.testing.assert_allclose(s.velocity[0], [0, 0, net * t], atol=1e-9)


def test_pitch_over_heads_east_and_coast_is_ballistic(boost):
    s = boost.sample(np.array([30.0, 61.0, 70.0, 80.0]))
    assert s.velocity[0, 0] > 0
    assert abs(s.velocity[0, 1]) < 1e-6
    assert not s.thrusting[1:].any()
    accel = (s.velocity[3] - s.velocity[2]) / 10.0
    np.testing.assert_allclose(accel, [0, 0, -GRAVITY], atol=1e-9)


def test_velocity_is_the_derivative_of_position(boost):
    t = np.linspace(15.0, 120.0, 400)
    s = boost.sample(t)
    numeric = np.gradient(s.position, t, axis=0)
    # Skip samples adjacent to the burnout kink, where acceleration jumps.
    smooth = np.abs(t - boost.burn_time) > 2 * (t[1] - t[0])
    smooth[[0, -1]] = False
    np.testing.assert_allclose(numeric[smooth], s.velocity[smooth], atol=0.2)


def test_boost_validation():
    with pytest.raises(ValueError, match="gravity"):
        BallisticBoost(np.zeros(3), 0.0, 5.0, 60.0)
    with pytest.raises(ValueError, match="burnout"):
        BallisticBoost(np.zeros(3), 0.0, 30.0, 5.0, vertical_rise_s=10.0)
