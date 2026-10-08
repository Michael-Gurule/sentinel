import numpy as np
import pytest

from sentinel.core import InsufficientMeasurementsError, mean_nees_bounds, nees
from sentinel.core.constants import SPEED_OF_LIGHT
from sentinel.geolocation import (
    TDOAMeasurement,
    difference_covariance,
    simulate_tdoa,
    solve_tdoa,
    tdoa_dop,
)


def rereference(m: TDOAMeasurement, new_reference: int) -> TDOAMeasurement:
    """Express the same TDOA information relative to another receiver."""
    ids = [m.reference, *m.others]
    values = np.concatenate([[0.0], m.values])  # arrival times relative to old ref
    full_cov = np.zeros((len(ids), len(ids)))
    full_cov[1:, 1:] = m.covariance
    keep = [i for i in range(len(ids)) if ids[i] != new_reference]
    ref = ids.index(new_reference)
    transform = np.zeros((len(keep), len(ids)))
    for row, col in enumerate(keep):
        transform[row, col] = 1.0
        transform[row, ref] = -1.0
    return TDOAMeasurement(
        reference=new_reference,
        others=tuple(ids[i] for i in keep),
        values=transform @ values,
        covariance=transform @ full_cov @ transform.T,
    )


def test_exact_on_noiseless_data(receivers, emitter, rng):
    result = solve_tdoa(receivers, simulate_tdoa(emitter, receivers, 1e-30, rng))
    assert result.converged
    np.testing.assert_allclose(result.position, emitter, atol=1e-6)
    assert result.velocity is None
    assert result.state_covariance.shape == (3, 3)


@pytest.mark.parametrize("toa_std", [3e-9, 30e-9])
def test_monte_carlo_nees_is_consistent(receivers, emitter, rng, toa_std):
    runs = 300
    values = []
    for _ in range(runs):
        result = solve_tdoa(receivers, simulate_tdoa(emitter, receivers, toa_std, rng))
        values.append(nees(result.position - emitter, result.position_covariance))
    low, high = mean_nees_bounds(3, runs, confidence=0.99)
    assert low <= np.mean(values) <= high


def test_covariance_matches_dop(receivers, emitter, rng):
    toa_std = 10e-9
    noiseless = simulate_tdoa(emitter, receivers, 1e-30, rng)
    m = TDOAMeasurement(
        noiseless.reference,
        noiseless.others,
        noiseless.values,
        difference_covariance(len(noiseless), toa_std),
    )
    result = solve_tdoa(receivers, m)
    gdop = tdoa_dop(emitter, np.array([r.position for r in receivers])).gdop
    rms = np.sqrt(np.trace(result.position_covariance))
    assert rms == pytest.approx(gdop * SPEED_OF_LIGHT * toa_std, rel=1e-6)


def test_estimate_does_not_depend_on_reference_choice(receivers, emitter, rng):
    m = simulate_tdoa(emitter, receivers, 20e-9, rng)
    a = solve_tdoa(receivers, m)
    b = solve_tdoa(receivers, rereference(m, 3))
    np.testing.assert_allclose(a.position, b.position, atol=1e-4)
    np.testing.assert_allclose(a.position_covariance, b.position_covariance, rtol=1e-5)


def test_chi2_follows_its_distribution(receivers, emitter, rng):
    runs = 300
    chi2 = [
        solve_tdoa(receivers, simulate_tdoa(emitter, receivers, 10e-9, rng)).chi2
        for _ in range(runs)
    ]
    low, high = mean_nees_bounds(1, runs, confidence=0.99)  # dof = 4 - 3
    assert low <= np.mean(chi2) <= high


def test_works_with_three_tdoas_and_explicit_initial(receivers, emitter, rng):
    four = receivers[:4]
    m = simulate_tdoa(emitter, four, 1e-30, rng)
    result = solve_tdoa(four, m, initial=emitter + 200.0)
    np.testing.assert_allclose(result.position, emitter, atol=1e-5)


def test_needs_three_tdoas(receivers, emitter, rng):
    m = simulate_tdoa(emitter, receivers[:3], 1e-9, rng)
    with pytest.raises(InsufficientMeasurementsError):
        solve_tdoa(receivers, m)
