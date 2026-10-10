import numpy as np
import pytest

from locant.core import (
    GeometryError,
    InsufficientMeasurementsError,
    mean_nees_bounds,
    nees,
)
from locant.geolocation import (
    range_dop,
    simulate_ranges,
    solve_ranges,
    solve_ranges_linear,
    tdoa_dop,
)

SENSORS = np.array(
    [
        [0.0, 0.0, 0.0],
        [10_000.0, 0.0, 200.0],
        [10_000.0, 10_000.0, 50.0],
        [0.0, 10_000.0, 400.0],
        [5_000.0, -3_000.0, 1_000.0],
    ]
)
EMITTER_POSITION = np.array([3_000.0, 4_000.0, 800.0])
COPLANAR = np.array(
    [[0, 0, 0], [1e4, 0, 0], [1e4, 1e4, 0], [0, 1e4, 0], [5e3, -3e3, 0]], dtype=float
)


class TestRanges:
    def test_linear_solution_is_exact_on_noiseless_data(self):
        z = np.linalg.norm(SENSORS - EMITTER_POSITION, axis=1)
        np.testing.assert_allclose(
            solve_ranges_linear(SENSORS, z), EMITTER_POSITION, atol=1e-6
        )

    def test_wls_converges_from_a_distant_start(self):
        """Regression for C6: the inverted Jacobian made Gauss-Newton diverge."""
        z = np.linalg.norm(SENSORS - EMITTER_POSITION, axis=1)
        position, covariance, converged = solve_ranges(
            SENSORS, z, np.eye(5), initial=np.array([9_000.0, -2_000.0, 3_000.0])
        )
        assert converged
        np.testing.assert_allclose(position, EMITTER_POSITION, atol=1e-6)
        assert np.all(np.isfinite(covariance))

    def test_wls_nees_is_consistent(self, rng):
        runs, std = 300, 5.0
        values = []
        for _ in range(runs):
            z = simulate_ranges(EMITTER_POSITION, SENSORS, std, rng)
            position, cov, _ = solve_ranges(SENSORS, z, std**2 * np.eye(5))
            values.append(nees(position - EMITTER_POSITION, cov))
        low, high = mean_nees_bounds(3, runs, confidence=0.99)
        assert low <= np.mean(values) <= high

    def test_wls_covariance_matches_range_dop(self):
        z = np.linalg.norm(SENSORS - EMITTER_POSITION, axis=1)
        _, cov, _ = solve_ranges(SENSORS, z, np.eye(5))
        dop = range_dop(EMITTER_POSITION, SENSORS)
        assert np.sqrt(np.trace(cov)) == pytest.approx(dop.gdop, rel=1e-6)

    def test_linear_rejects_coplanar_sensors(self):
        z = np.linalg.norm(COPLANAR - EMITTER_POSITION, axis=1)
        with pytest.raises(GeometryError):
            solve_ranges_linear(COPLANAR, z)

    def test_wls_with_coplanar_sensors(self):
        z = np.linalg.norm(COPLANAR - EMITTER_POSITION, axis=1)
        # The default start (centroid) lies in the sensor plane, where the
        # out-of-plane coordinate is unobservable.
        with pytest.raises(GeometryError, match="singular"):
            solve_ranges(COPLANAR, z, np.eye(5))
        # Off-plane starts converge, up to the mirror ambiguity about the plane.
        position, _, converged = solve_ranges(
            COPLANAR, z, np.eye(5), initial=np.array([5e3, 5e3, -500.0])
        )
        assert converged
        np.testing.assert_allclose(
            np.abs(position), np.abs(EMITTER_POSITION), atol=1e-5
        )

    @pytest.mark.parametrize("n", [2, 3])
    def test_minimum_sensor_counts(self, n):
        z = np.ones(n)
        with pytest.raises(InsufficientMeasurementsError):
            solve_ranges_linear(SENSORS[:n], z)
        if n < 3:
            with pytest.raises(InsufficientMeasurementsError):
                solve_ranges(SENSORS[:n], z, np.eye(n))

    def test_input_shapes_are_validated(self):
        with pytest.raises(ValueError, match="shape"):
            solve_ranges_linear(SENSORS[:, :2], np.ones(5))
        with pytest.raises(ValueError, match="one entry"):
            solve_ranges_linear(SENSORS, np.ones(4))


class TestDOP:
    def test_range_dop_analytic_octahedron(self):
        sensors = np.vstack([np.eye(3), -np.eye(3)]) * 1_000.0
        dop = range_dop(np.zeros(3), sensors)  # HᵀH = 2I
        assert dop.gdop == pytest.approx(np.sqrt(1.5))
        assert dop.hdop == pytest.approx(1.0)
        assert dop.vdop == pytest.approx(np.sqrt(0.5))

    def test_tdoa_dop_is_independent_of_reference(self):
        values = [tdoa_dop(EMITTER_POSITION, SENSORS, i).gdop for i in range(5)]
        np.testing.assert_allclose(values, values[0], rtol=1e-10)

    def test_gdop_combines_horizontal_and_vertical(self):
        dop = tdoa_dop(EMITTER_POSITION, SENSORS)
        assert dop.gdop**2 == pytest.approx(dop.hdop**2 + dop.vdop**2)

    def test_singular_geometries_raise(self):
        in_plane = np.array([3e3, 4e3, 0.0])
        with pytest.raises(GeometryError):
            range_dop(in_plane, COPLANAR)
        with pytest.raises(GeometryError):
            tdoa_dop(EMITTER_POSITION, SENSORS[:3])
        with pytest.raises(GeometryError):
            range_dop(EMITTER_POSITION, SENSORS[:2])
        with pytest.raises(GeometryError):
            range_dop(SENSORS[0], SENSORS)
