import numpy as np
import pytest

from sentinel.core import (
    GeometryError,
    InsufficientMeasurementsError,
    mean_nees_bounds,
    nees,
)
from sentinel.geolocation import simulate_fdoa, simulate_tdoa, solve_tdoa_fdoa
from sentinel.geolocation._nls import solve_whitened

CARRIER = 1e9


def test_recovers_position_and_velocity_on_noiseless_data(
    moving_receivers, emitter, emitter_velocity, rng
):
    tdoa = simulate_tdoa(emitter, moving_receivers, 1e-30, rng)
    fdoa = simulate_fdoa(
        emitter, emitter_velocity, moving_receivers, CARRIER, 1e-30, rng
    )
    result = solve_tdoa_fdoa(moving_receivers, tdoa, fdoa)
    assert result.converged
    np.testing.assert_allclose(result.position, emitter, atol=1e-5)
    assert result.velocity is not None
    np.testing.assert_allclose(result.velocity, emitter_velocity, atol=1e-6)
    assert result.state_covariance.shape == (6, 6)
    assert result.dof == 2


def test_monte_carlo_nees_is_consistent(
    moving_receivers, emitter, emitter_velocity, rng
):
    runs = 300
    truth = np.concatenate([emitter, emitter_velocity])
    values = []
    for _ in range(runs):
        tdoa = simulate_tdoa(emitter, moving_receivers, 10e-9, rng)
        fdoa = simulate_fdoa(
            emitter, emitter_velocity, moving_receivers, CARRIER, 1.0, rng
        )
        r = solve_tdoa_fdoa(moving_receivers, tdoa, fdoa)
        assert r.velocity is not None
        state = np.concatenate([r.position, r.velocity])
        values.append(nees(state - truth, r.state_covariance))
    low, high = mean_nees_bounds(6, runs, confidence=0.99)
    assert low <= np.mean(values) <= high


def test_velocity_is_none_without_fdoa(moving_receivers, emitter, rng):
    tdoa = simulate_tdoa(emitter, moving_receivers, 10e-9, rng)
    result = solve_tdoa_fdoa(moving_receivers, tdoa, initial=np.zeros(6) + 1e3)
    assert result.velocity is None
    assert result.velocity_covariance is None


def test_needs_six_measurements(moving_receivers, emitter, emitter_velocity, rng):
    three = moving_receivers[:3]
    tdoa = simulate_tdoa(emitter, three, 1e-9, rng)
    fdoa = simulate_fdoa(emitter, emitter_velocity, three, CARRIER, 1.0, rng)
    with pytest.raises(InsufficientMeasurementsError):
        solve_tdoa_fdoa(three, tdoa, fdoa)


def test_unobservable_state_raises_geometry_error():
    # z depends only on x[0] + x[1]; the difference is unobservable.
    def model(x):
        return np.array([x[0] + x[1], 2 * (x[0] + x[1])]), np.array(
            [[1.0, 1.0], [2.0, 2.0]]
        )

    with pytest.raises(GeometryError, match="not observable"):
        solve_whitened(model, np.array([1.0, 2.0]), np.eye(2), np.zeros(2), 50)
