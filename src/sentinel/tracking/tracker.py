"""Timestamp-driven multi-target, multi-sensor tracker.

Each call to :meth:`MultiTargetTracker.step` handles one scan:

1. predict every track to the scan time (Δt from timestamps);
2. for each measurement source in turn, gate on the normalized innovation
   squared (NIS) against a χ² threshold, assign by global nearest neighbor,
   and apply a Kalman update;
3. start tracks from measurements no track claimed;
4. delete tracks that have coasted longer than ``max_coast_time``.

Processing sources sequentially is exact centralized fusion when sensor errors
are independent: each measurement updates the track once with its own noise
covariance, so no information is double counted.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from sentinel.core.linalg import FloatArray
from sentinel.core.stats import chi2_gate
from sentinel.tracking.assignment import assign_gnn
from sentinel.tracking.kalman import Gaussian, innovation, predict, update
from sentinel.tracking.models import ConstantVelocity


@dataclass(frozen=True, eq=False)
class LinearMeasurement:
    """Linear-Gaussian measurement z = Hx + v, v ~ N(0, R).

    Attributes:
        value: Measured vector z.
        covariance: Measurement noise covariance R.
        matrix: Measurement matrix H mapping the 6-D state to z.
        source: Sensor or modality name; measurements are processed per source.
        label: Optional class label carried to the track (e.g. event type).
    """

    value: FloatArray
    covariance: FloatArray
    matrix: FloatArray
    source: str
    label: str | None = None

    def __post_init__(self) -> None:
        value = np.asarray(self.value, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        matrix = np.asarray(self.matrix, dtype=np.float64)
        k = value.size
        if value.shape != (k,) or covariance.shape != (k, k) or matrix.shape[0] != k:
            raise ValueError("value, covariance, and matrix dimensions disagree")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "source", str(self.source))  # plain str keys

    @property
    def dim(self) -> int:
        return int(self.value.size)

    @classmethod
    def position(
        cls,
        position: FloatArray,
        covariance: FloatArray,
        source: str,
        label: str | None = None,
    ) -> "LinearMeasurement":
        """A 3-D position measurement."""
        matrix = np.hstack([np.eye(3), np.zeros((3, 3))])
        return cls(position, covariance, matrix, source, label)

    @classmethod
    def position_velocity(
        cls,
        state: FloatArray,
        covariance: FloatArray,
        source: str,
        label: str | None = None,
    ) -> "LinearMeasurement":
        """A joint position/velocity measurement ``[x, y, z, vx, vy, vz]``."""
        return cls(state, covariance, np.eye(6), source, label)


@dataclass(eq=False)
class Track:
    """A tracked target."""

    id: int
    state: Gaussian
    created: float
    last_update: float
    label: str | None = None
    hits_by_source: dict[str, int] = field(default_factory=dict)

    @property
    def position(self) -> FloatArray:
        return self.state.mean[:3]

    @property
    def velocity(self) -> FloatArray:
        return self.state.mean[3:]

    @property
    def position_covariance(self) -> FloatArray:
        return self.state.covariance[:3, :3]

    @property
    def velocity_covariance(self) -> FloatArray:
        return self.state.covariance[3:, 3:]

    @property
    def position_rms_uncertainty(self) -> float:
        """RMS position uncertainty, √trace(P_pos), in meters."""
        return float(np.sqrt(np.trace(self.position_covariance)))

    @property
    def hits(self) -> int:
        return sum(self.hits_by_source.values())


class MultiTargetTracker:
    """Centralized multi-sensor tracker with χ² gating and GNN assignment."""

    def __init__(
        self,
        motion_model: ConstantVelocity,
        gate_probability: float = 0.99,
        max_coast_time: float = 5.0,
        initial_velocity_std: float = 300.0,
    ) -> None:
        """
        Args:
            motion_model: Track dynamics.
            gate_probability: Probability that a correct measurement falls in
                the χ² gate.
            max_coast_time: Seconds without an update before a track is deleted.
            initial_velocity_std: Prior velocity standard deviation (m/s) for
                tracks started from position-only measurements.
        """
        if max_coast_time < 0 or initial_velocity_std <= 0:
            raise ValueError("max_coast_time >= 0 and initial_velocity_std > 0")
        self.motion_model = motion_model
        self.gate_probability = gate_probability
        self.max_coast_time = max_coast_time
        self.initial_velocity_std = initial_velocity_std
        self.time: float | None = None
        self._tracks: list[Track] = []
        self._next_id = 0

    @property
    def tracks(self) -> list[Track]:
        """Current tracks (a copy of the list; tracks are live objects)."""
        return list(self._tracks)

    def gate_threshold(self, dof: int) -> float:
        return chi2_gate(dof, self.gate_probability)

    def in_gate(self, track: Track, measurement: LinearMeasurement) -> bool:
        """Whether ``measurement`` falls in ``track``'s χ² gate."""
        nis = innovation(
            track.state, measurement.value, measurement.matrix, measurement.covariance
        ).nis
        return nis <= self.gate_threshold(measurement.dim)

    def step(
        self, measurements: Sequence[LinearMeasurement], timestamp: float
    ) -> list[Track]:
        """Process one scan of measurements taken at ``timestamp``.

        Raises:
            ValueError: if ``timestamp`` is earlier than the previous scan
                (out-of-sequence measurements are not supported), or if
                measurements from one source have different dimensions.
        """
        self.advance(timestamp)
        by_source: dict[str, list[LinearMeasurement]] = defaultdict(list)
        for measurement in measurements:
            by_source[measurement.source].append(measurement)
        for source in sorted(by_source):
            self._process_source(by_source[source], timestamp)
        self._prune(timestamp)
        return self.tracks

    def advance(self, timestamp: float) -> None:
        """Predict all tracks to ``timestamp`` without measurements."""
        if self.time is not None and timestamp < self.time:
            raise ValueError(
                f"out-of-sequence timestamp {timestamp} < current time {self.time}"
            )
        dt = 0.0 if self.time is None else timestamp - self.time
        if dt > 0.0:
            for track in self._tracks:
                track.state = predict(track.state, self.motion_model, dt)
        self.time = timestamp

    def initiate(self, measurement: LinearMeasurement, timestamp: float) -> Track:
        """Start a track from a position or position/velocity measurement."""
        if measurement.matrix.shape == (6, 6) and np.allclose(
            measurement.matrix, np.eye(6)
        ):
            state = Gaussian(measurement.value, measurement.covariance)
        elif measurement.matrix.shape == (3, 6) and np.allclose(
            measurement.matrix, self.motion_model.position_matrix()
        ):
            covariance = np.zeros((6, 6))
            covariance[:3, :3] = measurement.covariance
            covariance[3:, 3:] = np.eye(3) * self.initial_velocity_std**2
            state = Gaussian(
                np.concatenate([measurement.value, np.zeros(3)]), covariance
            )
        else:
            raise ValueError("tracks can only start from position(/velocity) data")
        track = Track(
            id=self._next_id,
            state=state,
            created=timestamp,
            last_update=timestamp,
            label=measurement.label,
            hits_by_source={measurement.source: 1},
        )
        self._next_id += 1
        self._tracks.append(track)
        return track

    def _process_source(
        self, measurements: list[LinearMeasurement], timestamp: float
    ) -> None:
        dims = {m.dim for m in measurements}
        if len(dims) != 1:
            raise ValueError("measurements from one source must share a dimension")
        threshold = self.gate_threshold(dims.pop())

        cost = np.full((len(self._tracks), len(measurements)), np.inf)
        for i, track in enumerate(self._tracks):
            for j, m in enumerate(measurements):
                nis = innovation(track.state, m.value, m.matrix, m.covariance).nis
                if nis <= threshold:
                    cost[i, j] = nis

        assigned_measurements: set[int] = set()
        for i, j in assign_gnn(cost, unassigned_cost=threshold):
            track, m = self._tracks[i], measurements[j]
            track.state, _ = update(track.state, m.value, m.matrix, m.covariance)
            track.last_update = timestamp
            track.hits_by_source[m.source] = track.hits_by_source.get(m.source, 0) + 1
            if m.label is not None:
                track.label = m.label
            assigned_measurements.add(j)

        for j, m in enumerate(measurements):
            if j not in assigned_measurements:
                self.initiate(m, timestamp)

    def _prune(self, timestamp: float) -> None:
        self._tracks = [
            t for t in self._tracks if timestamp - t.last_update <= self.max_coast_time
        ]
