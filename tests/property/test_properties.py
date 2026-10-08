"""Property-based tests (Hypothesis) for numerical invariants."""

import itertools

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sentinel.core import is_psd
from sentinel.geolocation import (
    Receiver,
    TDOAMeasurement,
    chan_ho,
    simulate_tdoa,
    solve_tdoa,
)
from sentinel.tracking import ConstantVelocity, Gaussian, assign_gnn, update

finite = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)
H = np.hstack([np.eye(3), np.zeros((3, 3))])


def spd(seed: int, n: int, scale: float = 1.0) -> np.ndarray:
    a = np.random.default_rng(seed).normal(size=(n, n))
    return scale * (a @ a.T + 0.1 * np.eye(n))


@given(st.floats(0.0, 1e3), st.floats(0.0, 60.0))
def test_process_noise_is_psd(intensity, dt):
    assert is_psd(ConstantVelocity(intensity).process_noise(dt))


@given(
    st.integers(0, 10_000), st.integers(0, 10_000), arrays(float, 3, elements=finite)
)
def test_update_keeps_covariance_psd_and_never_increases_it(seed_p, seed_r, z):
    prior = Gaussian(np.zeros(6), spd(seed_p, 6, 100.0))
    posterior, innov = update(prior, z, H, spd(seed_r, 3))
    assert is_psd(posterior.covariance)
    assert is_psd(prior.covariance - posterior.covariance, tol=1e-7)
    assert innov.nis >= 0.0


@given(arrays(float, (3, 3), elements=st.floats(0.0, 100.0)))
def test_gnn_is_optimal_against_brute_force(cost):
    pairs = assign_gnn(cost, unassigned_cost=1e9)
    total = sum(cost[i, j] for i, j in pairs)
    best = min(
        sum(cost[i, p[i]] for i in range(3)) for p in itertools.permutations(range(3))
    )
    assert len(pairs) == 3
    assert np.isclose(total, best)


positions = np.array(
    [
        [0.0, 0.0, 0.0],
        [10_000.0, 0.0, 200.0],
        [10_000.0, 10_000.0, 50.0],
        [0.0, 10_000.0, 400.0],
        [5_000.0, -3_000.0, 1_000.0],
    ]
)
RECEIVERS = [Receiver(i, p) for i, p in enumerate(positions)]
EMITTER = np.array([3_000.0, 4_000.0, 800.0])


@given(st.permutations(range(4)), st.integers(0, 2**32 - 1))
def test_tdoa_estimate_ignores_measurement_order(order, seed):
    m = simulate_tdoa(EMITTER, RECEIVERS, 10e-9, np.random.default_rng(seed))
    idx = list(order)
    shuffled = TDOAMeasurement(
        m.reference,
        tuple(m.others[i] for i in idx),
        m.values[idx],
        m.covariance[np.ix_(idx, idx)],
    )
    np.testing.assert_allclose(
        solve_tdoa(RECEIVERS, shuffled).position,
        solve_tdoa(RECEIVERS, m).position,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        chan_ho(RECEIVERS, shuffled).position,
        chan_ho(RECEIVERS, m).position,
        atol=1e-4,
    )


@given(arrays(float, 3, elements=st.floats(-1e5, 1e5)))
def test_tdoa_estimate_is_translation_equivariant(shift):
    m = simulate_tdoa(EMITTER, RECEIVERS, 10e-9, np.random.default_rng(0))
    moved = [Receiver(r.id, r.position + shift) for r in RECEIVERS]
    np.testing.assert_allclose(
        solve_tdoa(moved, m).position - shift,
        solve_tdoa(RECEIVERS, m).position,
        atol=1e-3,
    )
