"""Ground-truth target trajectories in a local ENU frame.

All trajectories are deterministic functions of time; randomness enters only
through their parameters. Times are seconds relative to the trajectory's own
start (t = 0 at ignition, first sighting, etc.); before t = 0 the target sits
at its initial state.
"""

from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np

from locant.core.linalg import FloatArray

GRAVITY: Final[float] = 9.80665
"""Standard gravity, m/s². Constant over the boost-phase altitudes modeled here."""


@dataclass(frozen=True, eq=False)
class TrajectorySample:
    """Positions and velocities (len(t), 3) plus an "engine on" flag per time."""

    position: FloatArray
    velocity: FloatArray
    thrusting: np.ndarray


class Trajectory(Protocol):
    def sample(self, t: FloatArray) -> TrajectorySample:
        """Evaluate the trajectory at times ``t`` (s)."""
        ...


def _times(t: FloatArray) -> FloatArray:
    return np.atleast_1d(np.asarray(t, dtype=np.float64))


@dataclass(frozen=True, eq=False)
class Stationary:
    """A fixed point (fire, explosion, ground site)."""

    position: FloatArray

    def sample(self, t: FloatArray) -> TrajectorySample:
        n = _times(t).size
        return TrajectorySample(
            position=np.tile(np.asarray(self.position, dtype=np.float64), (n, 1)),
            velocity=np.zeros((n, 3)),
            thrusting=np.zeros(n, dtype=bool),
        )


@dataclass(frozen=True, eq=False)
class ConstantVelocity:
    """Straight-line motion, e.g. an aircraft in cruise."""

    position: FloatArray
    velocity: FloatArray

    def sample(self, t: FloatArray) -> TrajectorySample:
        times = _times(t)
        v = np.asarray(self.velocity, dtype=np.float64)
        return TrajectorySample(
            position=np.asarray(self.position) + np.outer(times, v),
            velocity=np.tile(v, (times.size, 1)),
            thrusting=np.ones(times.size, dtype=bool),
        )


@dataclass(frozen=True, eq=False)
class BallisticBoost:
    """Powered ascent followed by unpowered ballistic coast (flat Earth).

    The vehicle rises vertically for ``vertical_rise_s``, then pitches over
    to a fixed flight-path elevation and accelerates along it until burnout.
    The thrust acceleration is constant, gravity is constant, and drag and
    Earth curvature are ignored: an adequate model of the first few minutes
    over a few hundred kilometers, which is what an OPIR detection window
    covers.

    Attributes:
        position: Launch point (ENU, m).
        azimuth: Ground-track heading, rad clockwise from north.
        thrust_acceleration: Engine acceleration (m/s²), excluding gravity.
        burn_time: Seconds from ignition to burnout.
        vertical_rise_s: Seconds of vertical flight before pitch-over.
        pitch_elevation: Flight-path elevation after pitch-over, rad.
    """

    position: FloatArray
    azimuth: float
    thrust_acceleration: float
    burn_time: float
    vertical_rise_s: float = 10.0
    pitch_elevation: float = np.radians(45.0)

    def __post_init__(self) -> None:
        if self.thrust_acceleration <= GRAVITY:
            raise ValueError("thrust acceleration must exceed gravity to lift off")
        if not 0.0 < self.vertical_rise_s < self.burn_time:
            raise ValueError("vertical rise must end before burnout")

    def _thrust_direction(self, t: FloatArray) -> FloatArray:
        heading = np.array(
            [
                np.sin(self.azimuth) * np.cos(self.pitch_elevation),
                np.cos(self.azimuth) * np.cos(self.pitch_elevation),
                np.sin(self.pitch_elevation),
            ]
        )
        directions = np.tile(heading, (t.size, 1))
        directions[t < self.vertical_rise_s] = [0.0, 0.0, 1.0]
        return directions

    def sample(self, t: FloatArray, step: float = 0.05) -> TrajectorySample:
        """Integrate the equations of motion and interpolate at ``t``.

        The powered phase is integrated with a fixed ``step`` (velocity Verlet,
        exact for piecewise-constant acceleration); the coast phase is
        evaluated in closed form.
        """
        times = _times(t)
        grid = np.arange(0.0, self.burn_time + step, step)
        grid[-1] = self.burn_time
        accel = self.thrust_acceleration * self._thrust_direction(grid)
        accel[:, 2] -= GRAVITY
        pos = np.zeros((grid.size, 3))
        vel = np.zeros((grid.size, 3))
        pos[0] = self.position
        for k in range(1, grid.size):
            dt = grid[k] - grid[k - 1]
            a = accel[k - 1]
            pos[k] = pos[k - 1] + vel[k - 1] * dt + 0.5 * a * dt**2
            vel[k] = vel[k - 1] + a * dt

        clipped = np.clip(times, 0.0, None)
        powered = clipped <= self.burn_time
        out_pos = np.empty((times.size, 3))
        out_vel = np.empty((times.size, 3))
        for axis in range(3):
            out_pos[powered, axis] = np.interp(clipped[powered], grid, pos[:, axis])
            out_vel[powered, axis] = np.interp(clipped[powered], grid, vel[:, axis])
        tau = clipped[~powered] - self.burn_time
        gravity = np.array([0.0, 0.0, -GRAVITY])
        out_pos[~powered] = (
            pos[-1] + np.outer(tau, vel[-1]) + 0.5 * np.outer(tau**2, gravity)
        )
        out_vel[~powered] = vel[-1] + np.outer(tau, gravity)
        return TrajectorySample(
            position=out_pos,
            velocity=out_vel,
            thrusting=(times >= 0.0) & powered,
        )
