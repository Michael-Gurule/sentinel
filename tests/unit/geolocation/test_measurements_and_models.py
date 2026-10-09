import numpy as np
import pytest

from locant.core.constants import SPEED_OF_LIGHT
from locant.geolocation import (
    FDOAMeasurement,
    Receiver,
    TDOAMeasurement,
    difference_covariance,
    simulate_fdoa,
    simulate_tdoa,
)
from locant.geolocation.models import (
    fdoa_to_range_rate_difference,
    range_difference_model,
    range_rate_difference_model,
    receiver_lookup,
)


def numerical_jacobian(fn, x, step=1e-3):
    base = fn(x)
    jac = np.empty((base.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = step
        jac[:, i] = (fn(x + dx) - fn(x - dx)) / (2 * step)
    return jac


class TestValidation:
    def test_receiver_requires_3d_vectors(self):
        with pytest.raises(ValueError, match="shape"):
            Receiver(0, np.zeros(2))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"others": ()}, "at least one"),
            ({"others": (0, 1)}, "reference"),
            ({"others": (1, 1)}, "duplicate"),
            ({"values": np.zeros(3)}, "shape"),
            ({"covariance": np.eye(3)}, "shape"),
            ({"covariance": np.diag([1.0, -1.0])}, "positive semidefinite"),
        ],
    )
    def test_difference_measurement_validation(self, kwargs, message):
        args = {
            "reference": 0,
            "others": (1, 2),
            "values": np.zeros(2),
            "covariance": np.eye(2),
        } | kwargs
        with pytest.raises(ValueError, match=message):
            TDOAMeasurement(**args)

    def test_fdoa_requires_positive_carrier(self):
        with pytest.raises(ValueError, match="carrier"):
            FDOAMeasurement(0, (1,), np.zeros(1), np.eye(1), carrier_frequency=0.0)

    def test_receiver_lookup_errors(self, receivers):
        with pytest.raises(KeyError, match="unknown"):
            receiver_lookup(receivers, [99])
        with pytest.raises(ValueError, match="unique"):
            receiver_lookup([*receivers, receivers[0]], [0])


def test_range_difference_jacobian_matches_finite_differences(receivers, emitter):
    ref, others = receivers[0], receivers[1:]
    _, analytic = range_difference_model(emitter, ref, others)
    numeric = numerical_jacobian(
        lambda p: range_difference_model(p, ref, others)[0], emitter
    )
    np.testing.assert_allclose(analytic, numeric, atol=1e-6)


def test_range_rate_jacobian_matches_finite_differences(
    moving_receivers, emitter, emitter_velocity
):
    ref, others = moving_receivers[0], moving_receivers[1:]
    state = np.concatenate([emitter, emitter_velocity])
    _, analytic = range_rate_difference_model(emitter, emitter_velocity, ref, others)
    numeric = numerical_jacobian(
        lambda s: range_rate_difference_model(s[:3], s[3:], ref, others)[0], state
    )
    np.testing.assert_allclose(analytic, numeric, atol=1e-6)


def test_closing_receiver_sees_positive_doppler():
    receivers = [Receiver(0, np.zeros(3)), Receiver(1, np.array([1000.0, 0, 0]))]
    # Emitter between them moving toward receiver 1, away from receiver 0.
    m = simulate_fdoa(
        np.array([500.0, 0, 0]),
        np.array([10.0, 0, 0]),
        receivers,
        carrier_frequency=1e9,
        frequency_std=1e-12,
        rng=np.random.default_rng(0),
    )
    expected = 2 * 10.0 * 1e9 / SPEED_OF_LIGHT  # +f·v/c at rx1, -f·v/c at rx0
    assert m.values[0] == pytest.approx(expected, rel=1e-6)
    rate, _ = fdoa_to_range_rate_difference(m)
    assert rate[0] == pytest.approx(-20.0, rel=1e-6)


def test_simulated_tdoa_noise_has_declared_covariance(receivers, emitter, rng):
    std = 10e-9
    noiseless = simulate_tdoa(emitter, receivers, 1e-30, rng).values
    draws = np.array(
        [simulate_tdoa(emitter, receivers, std, rng).values for _ in range(4000)]
    )
    declared = difference_covariance(len(receivers) - 1, std)
    np.testing.assert_allclose(draws.mean(axis=0), noiseless, atol=4 * std / 30)
    np.testing.assert_allclose(np.cov(draws.T), declared, atol=0.1 * std**2)


def test_simulation_respects_reference_index(receivers, emitter, rng):
    m = simulate_tdoa(emitter, receivers, 1e-9, rng, reference_index=2)
    assert m.reference == 2
    assert m.others == (0, 1, 3, 4)


def test_difference_covariance_rejects_nonpositive_std():
    with pytest.raises(ValueError, match="positive"):
        difference_covariance(3, 0.0)
