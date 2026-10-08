import numpy as np
import pytest

from sentinel.core import (
    GeometryError,
    InsufficientMeasurementsError,
    mean_nees_bounds,
    nees,
)
from sentinel.geolocation import Receiver, chan_ho, simulate_tdoa

EXTRA_RECEIVERS = np.array([[2_000.0, 8_000.0, 1_500.0], [8_000.0, 3_000.0, 700.0]])


def test_exact_on_noiseless_data(receivers, emitter, rng):
    result = chan_ho(receivers, simulate_tdoa(emitter, receivers, 1e-30, rng))
    np.testing.assert_allclose(result.position, emitter, atol=1e-4)
    assert result.method == "chan_ho"
    assert result.dof == 1


def test_translation_invariant(receivers, emitter, rng):
    shift = np.array([123_456.0, -7_890.0, 321.0])
    moved = [Receiver(r.id, r.position + shift) for r in receivers]
    m = simulate_tdoa(emitter, receivers, 5e-9, rng)
    a = chan_ho(receivers, m).position
    b = chan_ho(moved, m).position
    np.testing.assert_allclose(b - shift, a, atol=1e-5)


def test_covariance_is_consistent_with_redundant_geometry(receivers, emitter, rng):
    seven = receivers + [Receiver(5 + i, p) for i, p in enumerate(EXTRA_RECEIVERS)]
    runs = 300
    values = []
    for _ in range(runs):
        result = chan_ho(seven, simulate_tdoa(emitter, seven, 3e-9, rng))
        values.append(nees(result.position - emitter, result.position_covariance))
    low, high = mean_nees_bounds(3, runs, confidence=0.99)
    assert low <= np.mean(values) <= high


def test_needs_four_tdoas(receivers, emitter, rng):
    m = simulate_tdoa(emitter, receivers[:4], 1e-9, rng)
    with pytest.raises(InsufficientMeasurementsError):
        chan_ho(receivers[:4], m)


def test_rejects_coplanar_receivers(rng):
    flat = [
        Receiver(i, np.array(p, dtype=float))
        for i, p in enumerate(
            [[0, 0, 0], [1e4, 0, 0], [1e4, 1e4, 0], [0, 1e4, 0], [5e3, -3e3, 0]]
        )
    ]
    m = simulate_tdoa(np.array([3e3, 4e3, 800.0]), flat, 1e-9, rng)
    with pytest.raises(GeometryError):
        chan_ho(flat, m)
