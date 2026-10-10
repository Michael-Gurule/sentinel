import numpy as np
import pytest

from locant.core import GeometryError, mean_nees_bounds, nees
from locant.core.constants import SPEED_OF_LIGHT
from locant.geolocation import (
    FDOAMeasurement,
    SystematicErrors,
    TDOAMeasurement,
    difference_covariance,
    inflate_fdoa,
    inflate_tdoa,
    rms_bound,
    simulate_fdoa,
    simulate_tdoa,
    solve_tdoa,
    solve_tdoa_fdoa,
    tdoa_crlb,
    tdoa_dop,
    tdoa_fdoa_crlb,
)
from locant.sim.rf.network import ReceiverModel, RFNetwork, default_receiver_network
from locant.sim.trajectories import Stationary


class TestCRLB:
    def test_equals_dop_scaled_by_timing_noise(self, receivers, emitter):
        sigma = 10e-9
        bound = rms_bound(tdoa_crlb(emitter, receivers, sigma))
        gdop = tdoa_dop(emitter, np.array([r.position for r in receivers])).gdop
        assert bound == pytest.approx(gdop * SPEED_OF_LIGHT * sigma, rel=1e-9)

    def test_is_reference_invariant_and_scales_with_noise(self, receivers, emitter):
        a = tdoa_crlb(emitter, receivers, 10e-9, reference_index=0)
        b = tdoa_crlb(emitter, receivers, 10e-9, reference_index=3)
        np.testing.assert_allclose(a, b, rtol=1e-9)
        np.testing.assert_allclose(
            tdoa_crlb(emitter, receivers, 20e-9), 4 * a, rtol=1e-9
        )

    def test_ml_estimator_attains_the_bound_at_high_snr(self, receivers, emitter, rng):
        sigma = 3e-9
        errors = np.array(
            [
                solve_tdoa(
                    receivers, simulate_tdoa(emitter, receivers, sigma, rng)
                ).position
                - emitter
                for _ in range(600)
            ]
        )
        empirical = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
        assert empirical == pytest.approx(
            rms_bound(tdoa_crlb(emitter, receivers, sigma)), rel=0.08
        )

    def test_hybrid_bound_matches_solver_covariance_at_truth(
        self, moving_receivers, emitter, emitter_velocity, rng
    ):
        tdoa = simulate_tdoa(emitter, moving_receivers, 1e-30, rng)
        fdoa = simulate_fdoa(
            emitter, emitter_velocity, moving_receivers, 1e9, 1e-30, rng
        )
        tdoa = type(tdoa)(
            tdoa.reference, tdoa.others, tdoa.values, difference_covariance(4, 10e-9)
        )
        fdoa = FDOAMeasurement(
            fdoa.reference, fdoa.others, fdoa.values, difference_covariance(4, 1.0), 1e9
        )
        result = solve_tdoa_fdoa(moving_receivers, tdoa, fdoa)
        bound = tdoa_fdoa_crlb(
            emitter, emitter_velocity, moving_receivers, 10e-9, 1.0, 1e9
        )
        np.testing.assert_allclose(result.state_covariance, bound, rtol=1e-4, atol=1e-9)

    def test_unobservable_geometry_raises(self, receivers, emitter):
        with pytest.raises(GeometryError):
            tdoa_crlb(emitter, receivers[:3], 10e-9)
        with pytest.raises(ValueError, match="two receivers"):
            tdoa_crlb(emitter, receivers[:1], 10e-9)


class TestSystematic:
    def test_inflation_structure(self, receivers, emitter, rng):
        m = simulate_tdoa(emitter, receivers, 10e-9, rng)
        sys = SystematicErrors(receiver_position_std=3.0, clock_bias_std=5e-9)
        inflated = inflate_tdoa(m, sys)
        expected = m.covariance + sys.tdoa_variance * (np.eye(4) + np.ones((4, 4)))
        np.testing.assert_allclose(inflated.covariance, expected)
        assert sys.tdoa_variance == pytest.approx(25e-18 + (3.0 / SPEED_OF_LIGHT) ** 2)
        np.testing.assert_array_equal(inflated.values, m.values)
        f = simulate_fdoa(emitter, np.zeros(3), receivers, 1e9, 1.0, rng)
        assert np.trace(
            inflate_fdoa(f, SystematicErrors(lo_offset_std=2.0)).covariance
        ) > np.trace(f.covariance)
        with pytest.raises(ValueError, match="non-negative"):
            SystematicErrors(clock_bias_std=-1.0)

    @pytest.mark.parametrize(
        ("clock_bias_std", "position_std"), [(30e-9, 0.0), (0.0, 20.0)]
    )
    def test_consider_covariance_restores_consistency(
        self, rng, clock_bias_std, position_std
    ):
        # Well-conditioned network: the first-order inflation is only valid
        # while errors stay small relative to the geometry (E4 shows the
        # breakdown when vertical DOP is large).
        receivers = default_receiver_network()
        emitter = np.array([5_000.0, 5_000.0, 500.0])
        runs, naive, consider = 300, [], []
        systematic = SystematicErrors(position_std, clock_bias_std)
        for _ in range(runs):
            network = RFNetwork(
                [
                    ReceiverModel(
                        r.id,
                        Stationary(r.position),
                        toa_std=10e-9,
                        clock_bias_std=clock_bias_std,
                        position_error_std=position_std,
                    )
                    for r in receivers
                ],
                rng,
            )
            scan = network.scan(0.0, emitter, np.zeros(3), rng)
            a = solve_tdoa(scan.receivers, scan.tdoa)
            b = solve_tdoa(scan.receivers, scan.tdoa, systematic=systematic)
            naive.append(nees(a.position - emitter, a.position_covariance))
            consider.append(nees(b.position - emitter, b.position_covariance))
        low, high = mean_nees_bounds(3, runs, confidence=0.99)
        assert np.mean(naive) > 3 * high  # ignoring systematics is overconfident
        assert low <= np.mean(consider) <= high

    def test_hybrid_accepts_systematics(
        self, moving_receivers, emitter, emitter_velocity, rng
    ):
        tdoa = simulate_tdoa(emitter, moving_receivers, 10e-9, rng)
        fdoa = simulate_fdoa(emitter, emitter_velocity, moving_receivers, 1e9, 1.0, rng)
        sys = SystematicErrors(5.0, 5e-9, 1.0)
        plain = solve_tdoa_fdoa(moving_receivers, tdoa, fdoa)
        robust = solve_tdoa_fdoa(moving_receivers, tdoa, fdoa, systematic=sys)
        assert np.trace(robust.state_covariance) > np.trace(plain.state_covariance)
        only_tdoa = solve_tdoa_fdoa(moving_receivers, tdoa, systematic=sys)
        assert only_tdoa.velocity is None

    def test_systematic_part_of_the_covariance(self, receivers, emitter, rng):
        """Clock bias has the random noise's (I + 11ᵀ) structure, so its share
        of the TDOA covariance is k / (1 + k), k = σ_sys² / σ_toa²."""
        m = simulate_tdoa(emitter, receivers, 10e-9, rng)
        sys = SystematicErrors(clock_bias_std=20e-9)
        result = solve_tdoa(receivers, m, systematic=sys)
        k = sys.tdoa_variance / (10e-9) ** 2
        assert result.systematic_covariance is not None
        np.testing.assert_allclose(
            result.systematic_covariance,
            k / (1 + k) * result.position_covariance,
            rtol=1e-6,
            atol=1e-9,
        )
        assert solve_tdoa(receivers, m).systematic_covariance is None

    def test_systematic_covariance_predicts_bias_only_errors(self, rng):
        """Data with clock bias but (almost) no random noise: the error is the
        systematic part alone and is consistent with ``systematic_covariance``."""
        receivers = default_receiver_network()
        emitter = np.array([5_000.0, 5_000.0, 500.0])
        systematic = SystematicErrors(clock_bias_std=30e-9)
        runs, values = 300, []
        for _ in range(runs):
            network = RFNetwork(
                [
                    ReceiverModel(
                        r.id,
                        Stationary(r.position),
                        toa_std=1e-12,
                        clock_bias_std=30e-9,
                    )
                    for r in receivers
                ],
                rng,
            )
            scan = network.scan(0.0, emitter, np.zeros(3), rng)
            measurement = TDOAMeasurement(
                reference=scan.tdoa.reference,
                others=scan.tdoa.others,
                values=scan.tdoa.values,
                covariance=difference_covariance(len(scan.tdoa), 10e-9),
            )
            fix = solve_tdoa(scan.receivers, measurement, systematic=systematic)
            assert fix.systematic_covariance is not None
            values.append(nees(fix.position - emitter, fix.systematic_covariance))
        low, high = mean_nees_bounds(3, runs, confidence=0.99)
        assert low <= np.mean(values) <= high

    def test_hybrid_systematic_covariance_is_part_of_the_total(
        self, moving_receivers, emitter, emitter_velocity, rng
    ):
        tdoa = simulate_tdoa(emitter, moving_receivers, 10e-9, rng)
        fdoa = simulate_fdoa(emitter, emitter_velocity, moving_receivers, 1e9, 1.0, rng)
        fix = solve_tdoa_fdoa(
            moving_receivers, tdoa, fdoa, systematic=SystematicErrors(5.0, 5e-9, 1.0)
        )
        assert fix.systematic_covariance is not None
        assert fix.systematic_covariance.shape == (6, 6)
        assert np.linalg.eigvalsh(fix.systematic_covariance)[0] > -1e-9
        assert (
            np.linalg.eigvalsh(fix.state_covariance - fix.systematic_covariance)[0]
            > -1e-9
        )
