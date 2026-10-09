import numpy as np
import pytest

from sentinel.core import GeometryError, mean_nees_bounds, nees
from sentinel.fusion import (
    LineOfSightMeasurement,
    associate_stereo,
    covariance_intersection,
    fuse_track_lists,
    intersect_altitude,
    measurements_from_reports,
    naive_fusion,
    triangulate,
)
from sentinel.fusion.opir_geoloc import perpendicular_basis, stereo_miss_distances
from sentinel.sim.opir.reports import OPIRReport
from sentinel.tracking import (
    ConstantVelocity,
    Gaussian,
    LinearMeasurement,
    MultiTargetTracker,
)

GEO = np.array([0.0, -30e6, 25e6])
HEO = np.array([10e6, 15e6, 35e6])
TARGET = np.array([2_000.0, 3_000.0, 12_000.0])
SIGMA = 10e-6


def noisy_los(sensor, target, rng, sigma=SIGMA):
    u = (target - sensor) / np.linalg.norm(target - sensor)
    u = u + perpendicular_basis(u).T @ rng.normal(0.0, sigma, 2)
    return u / np.linalg.norm(u)


def test_basis_is_orthonormal_and_perpendicular():
    d = np.array([0.3, -0.4, 0.866])
    d /= np.linalg.norm(d)
    e = perpendicular_basis(d)
    np.testing.assert_allclose(e @ e.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(e @ d, 0.0, atol=1e-12)


def test_triangulation_is_exact_and_consistent(rng):
    exact = [(TARGET - s) / np.linalg.norm(TARGET - s) for s in (GEO, HEO)]
    x, _ = triangulate([GEO, HEO], exact, [SIGMA, SIGMA])
    np.testing.assert_allclose(x, TARGET, atol=1e-3)
    values = []
    for _ in range(400):
        x, p = triangulate(
            [GEO, HEO],
            [noisy_los(GEO, TARGET, rng), noisy_los(HEO, TARGET, rng)],
            [SIGMA] * 2,
        )
        values.append(nees(x - TARGET, p))
    low, high = mean_nees_bounds(3, 400, confidence=0.99)
    assert low <= np.mean(values) <= high
    with pytest.raises(GeometryError):
        triangulate([GEO], [exact[0]], [SIGMA])
    with pytest.raises(GeometryError, match="parallel"):
        triangulate([GEO, GEO + 1.0], [exact[0], exact[0]], [SIGMA] * 2)


def test_altitude_intersection_is_exact_and_consistent(rng):
    ground = np.array([2_000.0, 3_000.0, 0.0])
    exact = (ground - GEO) / np.linalg.norm(ground - GEO)
    x, _ = intersect_altitude(GEO, exact, SIGMA, 0.0, 1e-6)
    np.testing.assert_allclose(x, ground, atol=1e-3)
    values = []
    for _ in range(400):
        altitude = rng.normal(0.0, 50.0)
        target = np.array([2_000.0, 3_000.0, altitude])
        x, p = intersect_altitude(GEO, noisy_los(GEO, target, rng), SIGMA, 0.0, 50.0)
        values.append(nees(x - target, p))
    low, high = mean_nees_bounds(3, 400, confidence=0.99)
    assert low <= np.mean(values) <= high
    with pytest.raises(GeometryError):
        intersect_altitude(GEO, -exact, SIGMA)


def test_line_of_sight_jacobian_and_ekf_update(rng):
    m = LineOfSightMeasurement(GEO, noisy_los(GEO, TARGET, rng), SIGMA)
    state = np.concatenate([TARGET + 500.0, np.zeros(3)])
    _, jac = m.linearize(state)
    step = 1e-1
    for i in range(3):
        dx = np.zeros(6)
        dx[i] = step
        numeric = (m.linearize(state + dx)[0] - m.linearize(state - dx)[0]) / (2 * step)
        np.testing.assert_allclose(jac[:, i], numeric, atol=1e-12)
    assert m.initial_state(100.0) is None
    tracker = MultiTargetTracker(ConstantVelocity(1.0))
    tracker.step(
        [LinearMeasurement.position(TARGET + 300.0, np.eye(3) * 1e6, "opir")], 0.0
    )
    before = tracker.tracks[0].position_rms_uncertainty
    tracker.step([m], 0.0)
    assert tracker.tracks[0].position_rms_uncertainty < before
    with pytest.raises(ValueError, match="position"):
        tracker.initiate(m, 0.0)


def test_stereo_association_and_measurement_building(rng):
    targets = [TARGET, TARGET + np.array([30_000.0, 0, 0])]
    reports = []
    for sensor_index, sensor in enumerate((GEO, HEO)):
        for target in targets[:: 1 if sensor_index == 0 else -1]:  # shuffled order
            reports.append(
                OPIRReport(
                    0.0,
                    sensor_index,
                    sensor,
                    noisy_los(sensor, target, rng),
                    SIGMA,
                    "e",
                    10.0,
                )
            )
    pairs = associate_stereo(
        [(r.sensor_position, r.line_of_sight, r.angle_std) for r in reports[:2]],
        [(r.sensor_position, r.line_of_sight, r.angle_std) for r in reports[2:]],
    )
    assert sorted(pairs) == [(0, 1), (1, 0)]
    extra = OPIRReport(0.0, 0, GEO, noisy_los(GEO, TARGET + 9e4, rng), SIGMA, None, 0.0)
    built = measurements_from_reports(
        [*reports, extra], class_probabilities=[np.ones(5) / 5] * 5
    )
    sources = sorted(m.source for m in built)
    assert sources == ["opir", "opir", "opir_los/0"]
    positions = [m.value for m in built if m.source == "opir"]
    assert min(np.linalg.norm(positions[0] - t) for t in targets) < 2_000.0
    assert all(m.class_probabilities is not None for m in built)


def test_t2t_fusion_consistency_under_correlation(rng):
    truth = np.zeros(6)
    p = np.diag([100.0, 100.0, 100.0, 4.0, 4.0, 4.0])
    common = np.linalg.cholesky(p * 0.8)
    own = np.linalg.cholesky(p * 0.2)
    naive_nees, ci_nees = [], []
    for _ in range(400):
        shared = common @ rng.normal(size=6)
        a = Gaussian(truth + shared + own @ rng.normal(size=6), p)
        b = Gaussian(truth + shared + own @ rng.normal(size=6), p)
        naive_nees.append(nees(naive_fusion(a, b).mean, naive_fusion(a, b).covariance))
        fused, omega = covariance_intersection(a, b)
        assert 0.0 <= omega <= 1.0
        ci_nees.append(nees(fused.mean, fused.covariance))
    _, high = mean_nees_bounds(6, 400, confidence=0.99)
    assert np.mean(naive_nees) > high  # independence assumption is overconfident
    assert np.mean(ci_nees) <= high


def test_fuse_track_lists():
    p = np.eye(6) * 100.0
    a = [Gaussian(np.zeros(6), p), Gaussian(np.full(6, 1e4), p)]
    b = [Gaussian(np.ones(6), p)]
    fused = fuse_track_lists(a, b, method="naive", labels=("opir", "rf"))
    assert sorted(f.sources for f in fused) == [("opir",), ("opir", "rf")]
    joint = next(f for f in fused if f.sources == ("opir", "rf"))
    assert np.trace(joint.state.covariance) < np.trace(p)
    assert joint.indices == (0, 0)
    assert next(f for f in fused if f.sources == ("opir",)).indices == (1, None)
    with pytest.raises(ValueError, match="method"):
        fuse_track_lists(a, b, method="mean")


def _reference_miss(a, b):
    """Definition: weighted squared perpendicular distances at the WLS point."""
    x, _ = triangulate([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])
    total = 0.0
    for s, u, std in (a, b):
        d = x - s
        perpendicular = d - (d @ u) * u
        total += perpendicular @ perpendicular / (std * np.linalg.norm(d)) ** 2
    return total


def test_vectorized_miss_distances_match_the_definition(rng):
    targets = rng.uniform(-30_000, 30_000, (6, 3)) + np.array([0, 0, 35_000])
    rays_a = [(GEO, noisy_los(GEO, t, rng), SIGMA) for t in targets]
    rays_b = [(HEO, noisy_los(HEO, t, rng), SIGMA) for t in targets[::-1]]
    fast = stereo_miss_distances(rays_a, rays_b)
    slow = np.array([[_reference_miss(a, b) for b in rays_b] for a in rays_a])
    np.testing.assert_allclose(fast, slow, rtol=1e-6)
    parallel = (HEO, rays_a[0][1], SIGMA)
    assert np.isinf(stereo_miss_distances(rays_a[:1], [parallel])[0, 0])
    assert stereo_miss_distances([], rays_b).shape == (0, 6)


def test_batched_linearization_matches_single(rng):
    los = LineOfSightMeasurement(GEO, noisy_los(GEO, TARGET, rng), SIGMA)
    means = np.column_stack(
        [TARGET + rng.normal(0, 1_000, (4, 3)), rng.normal(0, 100, (4, 3))]
    )
    h, jac = los.linearize_many(means)
    for k, mean in enumerate(means):
        h1, jac1 = los.linearize(mean)
        np.testing.assert_allclose(h[k], h1, atol=1e-15)
        np.testing.assert_allclose(jac[k], jac1, rtol=1e-9, atol=1e-20)
