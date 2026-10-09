"""Geolocating OPIR detections from lines of sight.

An OPIR detection is a direction, not a position. Three ways to use it, all
with an explicit covariance:

* :class:`LineOfSightMeasurement`: the direction itself as a 2-D angular
  measurement in an extended Kalman update of an existing track. It cannot
  start a track (range is unobservable from one direction).
* :func:`triangulate`: two or more sensors viewing the same event (e.g. GEO +
  HEO) → 3-D position by weighted least squares on the rays.
* :func:`intersect_altitude`: one sensor and a target at a known altitude
  (ground fire, explosion, launch pad) → position on that altitude plane.

Positions and directions are in the scenario's local ENU frame; the flat
altitude plane is adequate within a few hundred kilometers of the origin.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sentinel.core.errors import GeometryError
from sentinel.core.linalg import FloatArray
from sentinel.core.stats import chi2_gate
from sentinel.tracking.assignment import assign_gnn
from sentinel.tracking.kalman import Gaussian
from sentinel.tracking.tracker import LinearMeasurement


def perpendicular_basis(direction: FloatArray) -> FloatArray:
    """Two unit vectors (2×3) perpendicular to ``direction`` and to each other."""
    d = np.asarray(direction, dtype=np.float64)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, helper)
    e1 /= np.linalg.norm(e1)
    return np.vstack([e1, np.cross(d, e1)])


@dataclass(frozen=True, eq=False)
class LineOfSightMeasurement:
    """Angular measurement of a target direction from a known sensor position.

    The measurement is the target's angular offset from the measured line of
    sight, along two perpendicular axes: z = 0 by construction, and
    h(x) = E·u(x) with u(x) the unit vector from the sensor to the target and
    E the perpendicular basis of the measured line of sight.
    """

    sensor_position: FloatArray
    line_of_sight: FloatArray
    angle_std: float
    source: str = "opir_los"
    label: str | None = None
    class_probabilities: FloatArray | None = None

    @property
    def basis(self) -> FloatArray:
        return perpendicular_basis(self.line_of_sight)

    @property
    def value(self) -> FloatArray:
        return np.zeros(2)

    @property
    def covariance(self) -> FloatArray:
        return np.eye(2) * self.angle_std**2

    @property
    def dim(self) -> int:
        return 2

    def linearize(self, mean: FloatArray) -> tuple[FloatArray, FloatArray]:
        delta = np.asarray(mean[:3]) - self.sensor_position
        r = float(np.linalg.norm(delta))
        u = delta / r
        e = self.basis
        jacobian = np.zeros((2, 6))
        jacobian[:, :3] = e @ (np.eye(3) - np.outer(u, u)) / r
        return e @ u, jacobian

    def initial_state(self, velocity_std: float) -> Gaussian | None:
        del velocity_std
        return None


def triangulate(
    sensor_positions: Sequence[FloatArray],
    lines_of_sight: Sequence[FloatArray],
    angle_stds: Sequence[float],
    iterations: int = 3,
) -> tuple[FloatArray, FloatArray]:
    """Weighted least-squares intersection of two or more rays.

    Each ray constrains the target to lie on it, with cross-range error
    σᵢ·rᵢ (angle × range). Minimizing Σ ‖Pᵢ(x - sᵢ)‖² / (σᵢrᵢ)², with Pᵢ the
    projector perpendicular to ray i, gives information J = Σ Pᵢ / (σᵢrᵢ)² and
    covariance J⁻¹. Ranges come from the previous iterate.

    Raises:
        GeometryError: fewer than two rays, or rays (nearly) parallel.
    """
    if len(sensor_positions) < 2:
        raise GeometryError("triangulation needs at least two lines of sight")
    s = [np.asarray(p, dtype=np.float64) for p in sensor_positions]
    u = [np.asarray(d, dtype=np.float64) / np.linalg.norm(d) for d in lines_of_sight]
    projectors = [np.eye(3) - np.outer(ui, ui) for ui in u]
    weights = [1.0] * len(s)
    position = np.zeros(3)
    information = np.zeros((3, 3))
    for _ in range(iterations):
        information = sum(
            (w * p for w, p in zip(weights, projectors, strict=True)), np.zeros((3, 3))
        )
        eigenvalues = np.linalg.eigvalsh(information)
        if eigenvalues[0] <= 1e-10 * eigenvalues[-1]:
            raise GeometryError("lines of sight are (nearly) parallel")
        rhs = sum(
            (w * p @ si for w, p, si in zip(weights, projectors, s, strict=True)),
            np.zeros(3),
        )
        position = np.linalg.solve(information, rhs)
        weights = [
            1.0 / (std * np.linalg.norm(position - si)) ** 2
            for std, si in zip(angle_stds, s, strict=True)
        ]
    information = sum(
        (w * p for w, p in zip(weights, projectors, strict=True)), np.zeros((3, 3))
    )
    covariance = np.linalg.inv(information)
    return position, 0.5 * (covariance + covariance.T)


def intersect_altitude(
    sensor_position: FloatArray,
    line_of_sight: FloatArray,
    angle_std: float,
    altitude: float = 0.0,
    altitude_std: float = 50.0,
) -> tuple[FloatArray, FloatArray]:
    """Where a ray crosses the plane z = ``altitude``, with covariance.

    Angular errors move the intersection by r·(eₖ - (eₖ_z / u_z) u) per unit
    angle; altitude uncertainty slides it along the ray by u / u_z.

    Raises:
        GeometryError: the ray does not descend to the plane.
    """
    s = np.asarray(sensor_position, dtype=np.float64)
    u = np.asarray(line_of_sight, dtype=np.float64) / np.linalg.norm(line_of_sight)
    if u[2] >= -1e-6 or s[2] <= altitude:
        raise GeometryError("line of sight does not intersect the altitude plane")
    t = (altitude - s[2]) / u[2]
    position = s + t * u
    covariance = np.zeros((3, 3))
    for e in perpendicular_basis(u):
        g = t * (e - (e[2] / u[2]) * u)
        covariance += angle_std**2 * np.outer(g, g)
    g_alt = u / u[2]
    covariance += altitude_std**2 * np.outer(g_alt, g_alt)
    return position, 0.5 * (covariance + covariance.T)


def stereo_miss_distance_sq(
    position_a: FloatArray,
    los_a: FloatArray,
    std_a: float,
    position_b: FloatArray,
    los_b: FloatArray,
    std_b: float,
) -> float:
    """Normalized squared miss distance of two rays at their triangulated point
    (χ² with 1 dof when both rays see the same target)."""
    try:
        x, _ = triangulate([position_a, position_b], [los_a, los_b], [std_a, std_b])
    except GeometryError:
        return float("inf")
    chi2 = 0.0
    for s, u, std in ((position_a, los_a, std_a), (position_b, los_b, std_b)):
        d = x - s
        r = float(np.linalg.norm(d))
        perpendicular = d - (d @ u) * u
        chi2 += float(perpendicular @ perpendicular) / (std * r) ** 2
    return chi2


def associate_stereo(
    rays_a: Sequence[tuple[FloatArray, FloatArray, float]],
    rays_b: Sequence[tuple[FloatArray, FloatArray, float]],
    gate_probability: float = 0.99,
    reject_ambiguous: bool = True,
) -> list[tuple[int, int]]:
    """GNN pairing of detections from two sensors by stereo consistency.

    Two sensors constrain a pair of rays only within their epipolar plane, so
    rays from *different* targets that lie near a common epipolar plane also
    pass the gate and triangulate to a "ghost". With ``reject_ambiguous``, a
    pair is kept only if neither ray gates with any other ray; ambiguous rays
    are left for the tracker, which resolves them against predicted tracks.
    """
    if not rays_a or not rays_b:
        return []
    threshold = chi2_gate(1, gate_probability)
    cost = np.full((len(rays_a), len(rays_b)), np.inf)
    for i, a in enumerate(rays_a):
        for j, b in enumerate(rays_b):
            value = stereo_miss_distance_sq(*a, *b)
            if value <= threshold:
                cost[i, j] = value
    pairs = assign_gnn(cost, unassigned_cost=threshold)
    if not reject_ambiguous:
        return pairs
    gated = np.isfinite(cost)
    return [(i, j) for i, j in pairs if gated[i].sum() == 1 and gated[:, j].sum() == 1]


def measurements_from_reports(
    reports: Sequence["OPIRReportLike"],
    class_probabilities: Sequence[FloatArray | None] | None = None,
    gate_probability: float = 0.99,
    reject_ambiguous: bool = True,
) -> list[LinearMeasurement | LineOfSightMeasurement]:
    """Turn one scan of OPIR reports into fusable measurements.

    Reports from sensor 0 are paired with reports from each other sensor by
    stereo consistency; pairs become triangulated position measurements
    (source ``"opir"``, can start tracks), and unpaired reports become
    line-of-sight measurements (source ``"opir_los/<sensor index>"``,
    update-only; one source per sensor so that a track can take a ray from
    every sensor in the same scan).
    Ambiguous pairings are deferred to line of sight (see
    :func:`associate_stereo`).
    """
    probs = (
        list(class_probabilities)
        if class_probabilities is not None
        else [None] * len(reports)
    )
    by_sensor: dict[int, list[int]] = {}
    for i, report in enumerate(reports):
        by_sensor.setdefault(report.sensor_index, []).append(i)
    used: set[int] = set()
    out: list[LinearMeasurement | LineOfSightMeasurement] = []
    primary = by_sensor.get(0, [])
    for sensor, indices in sorted(by_sensor.items()):
        if sensor == 0:
            continue
        free_primary = [i for i in primary if i not in used]

        def rays(ids: list[int]) -> list[tuple[FloatArray, FloatArray, float]]:
            return [
                (
                    reports[i].sensor_position,
                    reports[i].line_of_sight,
                    reports[i].angle_std,
                )
                for i in ids
            ]

        for a, b in associate_stereo(
            rays(free_primary), rays(indices), gate_probability, reject_ambiguous
        ):
            ia, ib = free_primary[a], indices[b]
            ra, rb = reports[ia], reports[ib]
            position, covariance = triangulate(
                [ra.sensor_position, rb.sensor_position],
                [ra.line_of_sight, rb.line_of_sight],
                [ra.angle_std, rb.angle_std],
            )
            out.append(
                LinearMeasurement.position(
                    position,
                    covariance,
                    "opir",
                    class_probabilities=probs[ia]
                    if probs[ia] is not None
                    else probs[ib],
                )
            )
            used.update((ia, ib))
    for i, report in enumerate(reports):
        if i not in used:
            out.append(
                LineOfSightMeasurement(
                    report.sensor_position,
                    report.line_of_sight,
                    report.angle_std,
                    source=f"opir_los/{report.sensor_index}",
                    class_probabilities=probs[i],
                )
            )
    return out


class OPIRReportLike(Protocol):
    """The fields of :class:`sentinel.sim.opir.reports.OPIRReport` used here."""

    @property
    def sensor_index(self) -> int: ...

    @property
    def sensor_position(self) -> FloatArray: ...

    @property
    def line_of_sight(self) -> FloatArray: ...

    @property
    def angle_std(self) -> float: ...
