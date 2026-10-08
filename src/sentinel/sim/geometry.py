"""Earth geometry: WGS-84 frames, sensor platforms, and lines of sight.

Frames:

* **LLA**: geodetic latitude/longitude (rad) and height above the ellipsoid (m).
* **ECEF**: Earth-centered, Earth-fixed Cartesian (m).
* **ENU**: local east-north-up Cartesian (m) at a scenario origin.

Platforms report ECEF position as a function of time (s since scenario epoch).
Earth rotation is modeled as a constant rate about the z axis, with the
Greenwich meridian aligned with inertial x at the epoch. That is adequate for
scenario geometry over minutes to hours, not for precise orbit determination.
"""

from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np

from sentinel.core.linalg import FloatArray

WGS84_A: Final[float] = 6_378_137.0
"""Semi-major axis, m."""
WGS84_F: Final[float] = 1.0 / 298.257223563
"""Flattening."""
WGS84_E2: Final[float] = WGS84_F * (2.0 - WGS84_F)
"""First eccentricity squared."""
EARTH_MU: Final[float] = 3.986004418e14
"""Gravitational parameter GM, m³/s²."""
EARTH_ROTATION_RATE: Final[float] = 7.2921150e-5
"""Sidereal rotation rate, rad/s."""
GEO_RADIUS: Final[float] = 42_164_000.0
"""Geostationary orbit radius, m."""


def lla_to_ecef(lat: float, lon: float, alt: float) -> FloatArray:
    """Geodetic (rad, rad, m) → ECEF (m)."""
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    return np.array(
        [
            (n + alt) * np.cos(lat) * np.cos(lon),
            (n + alt) * np.cos(lat) * np.sin(lon),
            (n * (1.0 - WGS84_E2) + alt) * np.sin(lat),
        ]
    )


def ecef_to_lla(ecef: FloatArray, iterations: int = 6) -> tuple[float, float, float]:
    """ECEF (m) → geodetic (rad, rad, m) by fixed-point iteration on latitude.

    Converges to sub-millimeter height for terrestrial and orbital points in
    a few iterations.
    """
    x, y, z = (float(v) for v in ecef)
    lon = float(np.arctan2(y, x))
    p = float(np.hypot(x, y))
    lat = float(np.arctan2(z, p * (1.0 - WGS84_E2)))
    alt = 0.0
    for _ in range(iterations):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        alt = (
            p / np.cos(lat) - n
            if abs(lat) < np.pi / 4
            else z / np.sin(lat) - n * (1.0 - WGS84_E2)
        )
        lat = float(np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + alt))))
    return lat, lon, float(alt)


def enu_rotation(lat: float, lon: float) -> FloatArray:
    """Rotation matrix R with ENU = R · (ECEF - origin_ECEF)."""
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ]
    )


@dataclass(frozen=True)
class LocalFrame:
    """ENU frame anchored at a geodetic origin."""

    lat: float
    lon: float
    alt: float = 0.0

    @classmethod
    def from_degrees(
        cls, lat_deg: float, lon_deg: float, alt: float = 0.0
    ) -> "LocalFrame":
        return cls(np.radians(lat_deg), np.radians(lon_deg), alt)

    @property
    def origin_ecef(self) -> FloatArray:
        return lla_to_ecef(self.lat, self.lon, self.alt)

    @property
    def rotation(self) -> FloatArray:
        return enu_rotation(self.lat, self.lon)

    def to_ecef(self, enu: FloatArray) -> FloatArray:
        """ENU (…, 3) → ECEF (…, 3)."""
        return np.asarray(np.asarray(enu) @ self.rotation + self.origin_ecef)

    def from_ecef(self, ecef: FloatArray) -> FloatArray:
        """ECEF (…, 3) → ENU (…, 3)."""
        return np.asarray((np.asarray(ecef) - self.origin_ecef) @ self.rotation.T)

    def vector_to_ecef(self, enu_vector: FloatArray) -> FloatArray:
        """Rotate a direction or velocity (…, 3) from ENU to ECEF."""
        return np.asarray(np.asarray(enu_vector) @ self.rotation)


class Platform(Protocol):
    """Anything that can report its ECEF position over time."""

    def position_ecef(self, t: FloatArray) -> FloatArray:
        """ECEF positions (len(t), 3) at times ``t`` (s)."""
        ...


@dataclass(frozen=True)
class GeostationaryPlatform:
    """Satellite fixed above the equator at a given longitude."""

    longitude: float
    """Sub-satellite longitude, rad."""

    def position_ecef(self, t: FloatArray) -> FloatArray:
        times = np.atleast_1d(np.asarray(t, dtype=np.float64))
        point = GEO_RADIUS * np.array(
            [np.cos(self.longitude), np.sin(self.longitude), 0]
        )
        return np.tile(point, (times.size, 1))


@dataclass(frozen=True)
class KeplerianPlatform:
    """Two-body Keplerian orbit (no perturbations), e.g. HEO/Molniya.

    Angles in radians; ``mean_anomaly`` is at the scenario epoch.
    """

    semi_major_axis: float
    eccentricity: float
    inclination: float
    raan: float
    arg_perigee: float
    mean_anomaly: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("eccentricity must be in [0, 1)")
        if self.semi_major_axis * (1.0 - self.eccentricity) <= WGS84_A:
            raise ValueError("perigee is below the Earth's surface")

    @classmethod
    def molniya(
        cls, raan: float = 0.0, mean_anomaly: float = np.pi
    ) -> "KeplerianPlatform":
        """Classic 12-hour Molniya orbit, apogee over the northern hemisphere."""
        return cls(
            semi_major_axis=26_600_000.0,
            eccentricity=0.74,
            inclination=np.radians(63.4),
            raan=raan,
            arg_perigee=np.radians(270.0),
            mean_anomaly=mean_anomaly,
        )

    @property
    def period(self) -> float:
        return float(2.0 * np.pi * np.sqrt(self.semi_major_axis**3 / EARTH_MU))

    def position_eci(self, t: FloatArray) -> FloatArray:
        times = np.atleast_1d(np.asarray(t, dtype=np.float64))
        a, e = self.semi_major_axis, self.eccentricity
        mean = self.mean_anomaly + np.sqrt(EARTH_MU / a**3) * times
        ecc = mean.copy()
        for _ in range(30):  # Newton on Kepler's equation E - e sin E = M
            ecc -= (ecc - e * np.sin(ecc) - mean) / (1.0 - e * np.cos(ecc))
        x_pf = a * (np.cos(ecc) - e)
        y_pf = a * np.sqrt(1.0 - e**2) * np.sin(ecc)
        co, so = np.cos(self.raan), np.sin(self.raan)
        cw, sw = np.cos(self.arg_perigee), np.sin(self.arg_perigee)
        ci, si = np.cos(self.inclination), np.sin(self.inclination)
        rotation = np.array(
            [
                [co * cw - so * sw * ci, -co * sw - so * cw * ci],
                [so * cw + co * sw * ci, -so * sw + co * cw * ci],
                [sw * si, cw * si],
            ]
        )
        return np.asarray((rotation @ np.vstack([x_pf, y_pf])).T)

    def position_ecef(self, t: FloatArray) -> FloatArray:
        times = np.atleast_1d(np.asarray(t, dtype=np.float64))
        eci = self.position_eci(times)
        theta = EARTH_ROTATION_RATE * times
        c, s = np.cos(theta), np.sin(theta)
        return np.column_stack(
            [c * eci[:, 0] + s * eci[:, 1], -s * eci[:, 0] + c * eci[:, 1], eci[:, 2]]
        )


def line_of_sight(
    sensor_ecef: FloatArray, target_ecef: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Unit vectors sensor→target and ranges (m); inputs broadcast over (…, 3)."""
    delta = np.asarray(target_ecef) - np.asarray(sensor_ecef)
    ranges = np.linalg.norm(delta, axis=-1)
    return delta / ranges[..., None], ranges


def geodetic_up(ecef: FloatArray, iterations: int = 6) -> FloatArray:
    """Unit "up" (ellipsoid normal) vectors at ECEF points (…, 3), vectorized."""
    points = np.atleast_2d(np.asarray(ecef, dtype=np.float64))
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(iterations):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + WGS84_E2 * n * np.sin(lat), p)
    lon = np.arctan2(y, x)
    return np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def elevation_angle(target_ecef: FloatArray, toward_ecef: FloatArray) -> FloatArray:
    """Elevation (rad) of ``toward`` as seen from ``target``, above the local horizon.

    Uses the geodetic up direction at the target.
    """
    targets = np.atleast_2d(np.asarray(target_ecef, dtype=np.float64))
    direction, _ = line_of_sight(targets, np.atleast_2d(toward_ecef))
    up = geodetic_up(targets)
    return np.asarray(np.arcsin(np.clip(np.sum(direction * up, axis=1), -1.0, 1.0)))
