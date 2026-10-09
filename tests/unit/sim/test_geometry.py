import numpy as np
import pytest

from locant.sim.geometry import (
    GEO_RADIUS,
    WGS84_A,
    WGS84_F,
    GeostationaryPlatform,
    KeplerianPlatform,
    LocalFrame,
    ecef_to_lla,
    elevation_angle,
    geodetic_up,
    line_of_sight,
    lla_to_ecef,
)


@pytest.mark.parametrize("lat_deg", [-89.9, -45.0, 0.0, 30.0, 89.9])
@pytest.mark.parametrize("alt", [-100.0, 0.0, 10_000.0, 36_000_000.0])
def test_lla_ecef_round_trip(lat_deg, alt):
    lat, lon = np.radians(lat_deg), np.radians(123.0)
    back = ecef_to_lla(lla_to_ecef(lat, lon, alt))
    assert back[0] == pytest.approx(lat, abs=1e-11)
    assert back[1] == pytest.approx(lon, abs=1e-12)
    assert back[2] == pytest.approx(alt, abs=1e-3)


def test_ecef_reference_points():
    np.testing.assert_allclose(lla_to_ecef(0.0, 0.0, 0.0), [WGS84_A, 0, 0])
    polar = WGS84_A * (1 - WGS84_F)
    np.testing.assert_allclose(
        lla_to_ecef(np.pi / 2, 0.0, 0.0), [0, 0, polar], atol=1e-6
    )


def test_local_frame_round_trip_and_up_axis():
    frame = LocalFrame.from_degrees(40.0, -100.0, 500.0)
    points = np.array([[1_000.0, -2_000.0, 300.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        frame.from_ecef(frame.to_ecef(points)), points, atol=1e-6
    )
    np.testing.assert_allclose(frame.rotation @ frame.rotation.T, np.eye(3), atol=1e-12)
    above = frame.to_ecef(np.array([0.0, 0.0, 1_000.0]))
    lat, _, alt = ecef_to_lla(above)
    assert alt == pytest.approx(1_500.0, abs=1e-3)
    assert np.degrees(lat) == pytest.approx(40.0, abs=1e-9)
    np.testing.assert_allclose(
        frame.vector_to_ecef(np.array([0.0, 0.0, 1.0])),
        geodetic_up(frame.origin_ecef)[0],
    )


def test_geostationary_platform_is_fixed_on_the_equator():
    platform = GeostationaryPlatform(np.radians(-75.0))
    positions = platform.position_ecef(np.array([0.0, 3_600.0, 86_400.0]))
    np.testing.assert_allclose(np.linalg.norm(positions, axis=1), GEO_RADIUS)
    np.testing.assert_allclose(positions[0], positions[2])
    assert positions[0, 2] == 0.0


def test_molniya_orbit_period_and_apsides():
    orbit = KeplerianPlatform.molniya(mean_anomaly=0.0)
    assert orbit.period / 3_600 == pytest.approx(11.99, abs=0.01)
    t = np.linspace(0.0, orbit.period, 2_001)
    radius = np.linalg.norm(orbit.position_eci(t), axis=1)
    a, e = orbit.semi_major_axis, orbit.eccentricity
    assert radius.min() == pytest.approx(a * (1 - e), rel=1e-6)
    assert radius.max() == pytest.approx(a * (1 + e), rel=1e-4)
    np.testing.assert_allclose(
        orbit.position_eci(0.0), orbit.position_eci(orbit.period), atol=1e-3
    )
    # Apogee over the northern hemisphere (argument of perigee 270°).
    apogee = orbit.position_eci(orbit.period / 2)[0]
    assert apogee[2] > 0


def test_keplerian_validation():
    with pytest.raises(ValueError, match="eccentricity"):
        KeplerianPlatform(3e7, 1.2, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="perigee"):
        KeplerianPlatform(7e6, 0.5, 0, 0, 0, 0)


def test_line_of_sight_and_elevation():
    sat = GeostationaryPlatform(0.0).position_ecef(0.0)[0]
    below = lla_to_ecef(0.0, 0.0, 0.0)
    direction, distance = line_of_sight(sat, below)
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert distance == pytest.approx(GEO_RADIUS - WGS84_A)
    assert np.degrees(elevation_angle(below, sat))[0] == pytest.approx(90.0, abs=1e-6)
    far = lla_to_ecef(np.radians(40.0), 0.0, 0.0)
    assert np.degrees(elevation_angle(far, sat))[0] == pytest.approx(43.76, abs=0.05)
