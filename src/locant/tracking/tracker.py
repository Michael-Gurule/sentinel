"""Timestamp-driven multi-target, multi-sensor tracker.

Each call to :meth:`MultiTargetTracker.step` handles one scan:

1. predict every track to the scan time (Δt from timestamps);
2. for each measurement source in turn, gate on the normalized innovation
   squared (NIS) against a χ² threshold, assign by global nearest neighbor,
   and apply a Kalman update;
3. start tentative tracks from unclaimed measurements that can initiate one
   (positions, not bearings);
4. merge duplicate tracks (statistically indistinguishable positions; the
   track with more hits survives);
5. confirm tentative tracks with M hits in their last N scans, delete
   tentatives that cannot reach M, and delete any track that has coasted
   longer than ``max_coast_time``.

Measurements may be nonlinear (e.g. an OPIR line of sight): they expose
``linearize(x) -> (h(x), H)`` and the update is an extended Kalman update.

Processing sources sequentially is exact centralized fusion when sensor errors
are independent: each measurement updates the track once with its own noise
covariance, so no information is double counted.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np

from locant.core.linalg import FloatArray
from locant.core.stats import chi2_gate
from locant.tracking.assignment import assign_gnn
from locant.tracking.classes import pool_class_evidence
from locant.tracking.imm import IMMState, imm_predict, imm_update
from locant.tracking.kalman import Gaussian, batch_nis, predict, update
from locant.tracking.models import ConstantVelocity


@runtime_checkable
class Measurement(Protocol):
    """Anything the tracker can gate, associate, and fuse."""

    @property
    def value(self) -> FloatArray: ...

    @property
    def covariance(self) -> FloatArray: ...

    @property
    def source(self) -> str: ...

    @property
    def label(self) -> str | None: ...

    @property
    def class_probabilities(self) -> FloatArray | None: ...

    @property
    def systematic_covariance(self) -> FloatArray | None:
        """Part of ``covariance`` from errors shared by every measurement of
        this sensor (e.g. RF network bias); ``None`` if there is none."""
        ...

    @property
    def dim(self) -> int: ...

    def linearize(self, mean: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Predicted measurement h(x) and Jacobian H at state ``mean``."""
        ...

    def linearize_many(self, means: FloatArray) -> tuple[FloatArray, FloatArray]:
        """``linearize`` for K states at once: h (K, d) and H (K, d, 6)."""
        ...

    def initial_state(self, velocity_std: float) -> Gaussian | None:
        """Track-initial state, or ``None`` if this measurement cannot start one."""
        ...


@dataclass(frozen=True, eq=False)
class LinearMeasurement:
    """Linear-Gaussian measurement z = Hx + v, v ~ N(0, R).

    Attributes:
        value: Measured vector z.
        covariance: Measurement noise covariance R.
        matrix: Measurement matrix H mapping the 6-D state to z.
        source: Sensor or modality name; measurements are processed per source.
        label: Optional class label carried to the track (e.g. event type).
        class_probabilities: Optional calibrated class probabilities, pooled
            into the track's class posterior.
        systematic_covariance: Part of R that is fixed over a scenario and
            shared by every measurement from this sensor (it does not average
            out); kept on the track as a floor on its reported covariance.
    """

    value: FloatArray
    covariance: FloatArray
    matrix: FloatArray
    source: str
    label: str | None = None
    class_probabilities: FloatArray | None = None
    systematic_covariance: FloatArray | None = None

    def __post_init__(self) -> None:
        value = np.asarray(self.value, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        matrix = np.asarray(self.matrix, dtype=np.float64)
        k = value.size
        if value.shape != (k,) or covariance.shape != (k, k) or matrix.shape[0] != k:
            raise ValueError("value, covariance, and matrix dimensions disagree")
        if self.systematic_covariance is not None:
            systematic = np.asarray(self.systematic_covariance, dtype=np.float64)
            if systematic.shape != (k, k):
                raise ValueError("systematic_covariance must match covariance")
            object.__setattr__(self, "systematic_covariance", systematic)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "source", str(self.source))  # plain str keys

    @property
    def dim(self) -> int:
        return int(self.value.size)

    def linearize(self, mean: FloatArray) -> tuple[FloatArray, FloatArray]:
        return self.matrix @ mean, self.matrix

    def linearize_many(self, means: FloatArray) -> tuple[FloatArray, FloatArray]:
        means = np.asarray(means, dtype=np.float64)
        return means @ self.matrix.T, np.broadcast_to(
            self.matrix, (means.shape[0], *self.matrix.shape)
        )

    def initial_state(self, velocity_std: float) -> Gaussian | None:
        """Start from a position or position/velocity measurement."""
        position_only = np.hstack([np.eye(3), np.zeros((3, 3))])
        if self.matrix.shape == (6, 6) and np.allclose(self.matrix, np.eye(6)):
            return Gaussian(self.value, self.covariance)
        if self.matrix.shape == (3, 6) and np.allclose(self.matrix, position_only):
            covariance = np.zeros((6, 6))
            covariance[:3, :3] = self.covariance
            covariance[3:, 3:] = np.eye(3) * velocity_std**2
            return Gaussian(np.concatenate([self.value, np.zeros(3)]), covariance)
        return None

    @classmethod
    def position(
        cls,
        position: FloatArray,
        covariance: FloatArray,
        source: str,
        label: str | None = None,
        class_probabilities: FloatArray | None = None,
        systematic_covariance: FloatArray | None = None,
    ) -> "LinearMeasurement":
        """A 3-D position measurement."""
        matrix = np.hstack([np.eye(3), np.zeros((3, 3))])
        return cls(
            position,
            covariance,
            matrix,
            source,
            label,
            class_probabilities,
            systematic_covariance,
        )

    @classmethod
    def position_velocity(
        cls,
        state: FloatArray,
        covariance: FloatArray,
        source: str,
        label: str | None = None,
        systematic_covariance: FloatArray | None = None,
    ) -> "LinearMeasurement":
        """A joint position/velocity measurement ``[x, y, z, vx, vy, vz]``."""
        return cls(
            state,
            covariance,
            np.eye(6),
            source,
            label,
            systematic_covariance=systematic_covariance,
        )


class TrackStatus(StrEnum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"


@dataclass(eq=False)
class Track:
    """A tracked target.

    Attributes:
        status: Tentative until M of the last N scans updated it.
        scan_hits: Per-scan update history (most recent last).
        class_posterior: Pooled class posterior, or ``None`` without class evidence.
        confirmed_at: Time of confirmation, or ``None``.
        bias_floor: State-space covariance of the systematic error carried by
            the most recent measurement that had one (e.g. RF network bias).
            The filter averages measurements as if their errors were
            independent, so its covariance ``state.covariance`` shrinks below
            this shared error; the *reported* covariance adds it back.
    """

    id: int
    state: Gaussian
    created: float
    last_update: float
    label: str | None = None
    hits_by_source: dict[str, int] = field(default_factory=dict)
    status: TrackStatus = TrackStatus.TENTATIVE
    scan_hits: list[bool] = field(default_factory=list)
    class_posterior: FloatArray | None = None
    confirmed_at: float | None = None
    imm: IMMState | None = None
    """Per-model estimates when the tracker runs an IMM; ``state`` is their combination."""
    bias_floor: FloatArray | None = None

    @property
    def position(self) -> FloatArray:
        return self.state.mean[:3]

    @property
    def velocity(self) -> FloatArray:
        return self.state.mean[3:]

    @property
    def covariance(self) -> FloatArray:
        """Reported covariance: filter covariance plus the bias floor."""
        if self.bias_floor is None:
            return self.state.covariance
        return self.state.covariance + self.bias_floor

    @property
    def reported(self) -> Gaussian:
        """The track estimate with its reported covariance (for output and
        track-to-track fusion; gating uses the filter's ``state``)."""
        return Gaussian(self.state.mean, self.covariance)

    @property
    def position_covariance(self) -> FloatArray:
        return self.covariance[:3, :3]

    @property
    def velocity_covariance(self) -> FloatArray:
        return self.covariance[3:, 3:]

    @property
    def position_rms_uncertainty(self) -> float:
        """RMS position uncertainty, √trace(P_pos), in meters."""
        return float(np.sqrt(np.trace(self.position_covariance)))

    @property
    def hits(self) -> int:
        return sum(self.hits_by_source.values())


class MultiTargetTracker:
    """Centralized multi-sensor tracker: χ² gating, GNN, EKF updates, M-of-N."""

    def __init__(
        self,
        motion_model: ConstantVelocity,
        gate_probability: float = 0.99,
        max_coast_time: float = 5.0,
        initial_velocity_std: float = 300.0,
        confirm_hits: int = 3,
        confirm_window: int = 5,
        class_weight: float = 0.3,
        merge_probability: float | None = 0.9,
        imm_models: Sequence[ConstantVelocity] | None = None,
        imm_sojourn_s: Sequence[float] = (60.0, 10.0),
        imm_initial: Sequence[float] = (0.5, 0.5),
    ) -> None:
        """
        Args:
            motion_model: Track dynamics.
            gate_probability: Probability that a correct measurement falls in
                the χ² gate.
            max_coast_time: Seconds without an update before a track is deleted.
            initial_velocity_std: Prior velocity standard deviation (m/s) for
                tracks started from position-only measurements.
            confirm_hits, confirm_window: M-of-N confirmation; a tentative
                track is confirmed after M updates in its last N scans and
                deleted once it can no longer reach M.
            class_weight: Tempering weight for pooling class evidence.
            merge_probability: Two tracks whose position difference falls in
                this χ²(3) gate under the sum of their covariances are
                duplicates; ``None`` disables merging.
            imm_models: Run an IMM over these motion models instead of the
                single ``motion_model`` (which is then used only for
                reporting-time prediction).
            imm_sojourn_s: Mean time each IMM mode persists, s.
            imm_initial: Mode probabilities of a new track.
        """
        if max_coast_time < 0 or initial_velocity_std <= 0:
            raise ValueError("max_coast_time >= 0 and initial_velocity_std > 0")
        if not 1 <= confirm_hits <= confirm_window:
            raise ValueError("need 1 <= confirm_hits <= confirm_window")
        self.motion_model = motion_model
        self.gate_probability = gate_probability
        self.max_coast_time = max_coast_time
        self.initial_velocity_std = initial_velocity_std
        self.confirm_hits = confirm_hits
        self.confirm_window = confirm_window
        self.class_weight = class_weight
        self.merge_threshold = (
            None if merge_probability is None else chi2_gate(3, merge_probability)
        )
        self.imm_models = list(imm_models) if imm_models else None
        if self.imm_models is not None and not (
            len(self.imm_models) == len(imm_sojourn_s) == len(imm_initial)
        ):
            raise ValueError(
                "IMM models, sojourn times, and initial probabilities must align"
            )
        self.imm_sojourn_s = tuple(imm_sojourn_s)
        self.imm_initial = np.asarray(imm_initial, dtype=np.float64) / np.sum(
            imm_initial
        )
        self.time: float | None = None
        self._tracks: list[Track] = []
        self._next_id = 0

    @property
    def tracks(self) -> list[Track]:
        """All current tracks, tentative and confirmed (a copy of the list)."""
        return list(self._tracks)

    @property
    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.status is TrackStatus.CONFIRMED]

    def gate_threshold(self, dof: int) -> float:
        return chi2_gate(dof, self.gate_probability)

    @staticmethod
    def _nis_matrix(
        tracks: Sequence[Track], measurements: Sequence[Measurement]
    ) -> np.ndarray:
        """NIS of every measurement for every track, shape (tracks, measurements).

        For an IMM track this is the smallest NIS over its modes: a
        measurement consistent with *any* motion mode is admitted, otherwise
        the quiet mode's narrow gate would reject the very measurements that
        should shift probability to the maneuver mode. Each measurement is
        linearized against all track/mode states in one vectorized call.
        """
        out = np.full((len(tracks), len(measurements)), np.inf)
        owners, means, covariances = [], [], []
        for i, track in enumerate(tracks):
            for state in track.imm.states if track.imm is not None else [track.state]:
                owners.append(i)
                means.append(state.mean)
                covariances.append(state.covariance)
        if not owners:
            return out
        owner = np.asarray(owners)
        mean, covariance = np.asarray(means), np.asarray(covariances)
        for j, m in enumerate(measurements):
            predicted, matrices = m.linearize_many(mean)
            nis = batch_nis(m.value, predicted, matrices, covariance, m.covariance)
            np.minimum.at(out[:, j], owner, nis)
        return out

    def in_gate(self, track: Track, measurement: Measurement) -> bool:
        """Whether ``measurement`` falls in ``track``'s χ² gate."""
        nis = self._nis_matrix([track], [measurement])[0, 0]
        return bool(nis <= self.gate_threshold(measurement.dim))

    def step(
        self, measurements: Sequence[Measurement], timestamp: float
    ) -> list[Track]:
        """Process one scan of measurements taken at ``timestamp``.

        Measurements are grouped by source and dimension (e.g. RF fixes with
        and without FDOA velocity); each group is gated with its own χ²
        threshold and assigned by GNN, so a track takes at most one
        measurement per group per scan.

        Raises:
            ValueError: if ``timestamp`` is earlier than the previous scan
                (out-of-sequence measurements must be reordered upstream).
        """
        self.advance(timestamp)
        groups: dict[tuple[str, int], list[Measurement]] = defaultdict(list)
        for measurement in measurements:
            groups[(measurement.source, measurement.dim)].append(measurement)
        updated: set[int] = set()
        for key in sorted(groups):
            updated |= self._process_source(groups[key], timestamp)
        self._end_scan(updated, timestamp)
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
                if track.imm is not None and self.imm_models is not None:
                    track.imm = imm_predict(
                        track.imm, self.imm_models, self.imm_sojourn_s, dt
                    )
                    track.state = track.imm.combined()
                else:
                    track.state = predict(track.state, self.motion_model, dt)
        self.time = timestamp

    def initiate(self, measurement: Measurement, timestamp: float) -> Track:
        """Start a tentative track from a measurement that determines position."""
        state = measurement.initial_state(self.initial_velocity_std)
        if state is None:
            raise ValueError("tracks can only start from position(/velocity) data")
        track = Track(
            id=self._next_id,
            state=state,
            created=timestamp,
            last_update=timestamp,
            label=measurement.label,
            hits_by_source={measurement.source: 1},
        )
        if self.imm_models is not None:
            track.imm = IMMState(
                [state] * len(self.imm_models), self.imm_initial.copy()
            )
        self._absorb_class(track, measurement)
        self._absorb_floor(track, measurement)
        if self.confirm_hits == 1:
            track.status, track.confirmed_at = TrackStatus.CONFIRMED, timestamp
        self._next_id += 1
        self._tracks.append(track)
        return track

    @staticmethod
    def _absorb_floor(track: Track, measurement: Measurement) -> None:
        if measurement.systematic_covariance is None:
            return
        # Hᵀ Σ H places Σ in the state components the measurement observes;
        # exact for the selection matrices of position(/velocity) fixes.
        _, matrix = measurement.linearize(track.state.mean)
        track.bias_floor = matrix.T @ measurement.systematic_covariance @ matrix

    def _absorb_class(self, track: Track, measurement: Measurement) -> None:
        if measurement.label is not None:
            track.label = measurement.label
        if measurement.class_probabilities is not None:
            track.class_posterior = pool_class_evidence(
                track.class_posterior,
                measurement.class_probabilities,
                self.class_weight,
            )

    def _process_source(
        self, measurements: list[Measurement], timestamp: float
    ) -> set[int]:
        threshold = self.gate_threshold(measurements[0].dim)

        nis = self._nis_matrix(self._tracks, measurements)
        cost = np.where(nis <= threshold, nis, np.inf)

        assigned: set[int] = set()
        updated: set[int] = set()
        for i, j in assign_gnn(cost, unassigned_cost=threshold):
            track, m = self._tracks[i], measurements[j]
            if track.imm is not None:
                track.imm = imm_update(track.imm, m.value, m.linearize, m.covariance)
                track.state = track.imm.combined()
            else:
                predicted, matrix = m.linearize(track.state.mean)
                track.state, _ = update(
                    track.state, m.value, matrix, m.covariance, predicted
                )
            track.last_update = timestamp
            track.hits_by_source[m.source] = track.hits_by_source.get(m.source, 0) + 1
            self._absorb_class(track, m)
            self._absorb_floor(track, m)
            assigned.add(j)
            updated.add(track.id)

        for j, m in enumerate(measurements):
            if (
                j not in assigned
                and m.initial_state(self.initial_velocity_std) is not None
            ):
                updated.add(self.initiate(m, timestamp).id)
        return updated

    def _merge_duplicates(self) -> None:
        """Delete tracks that duplicate a better-established track.

        Two tracks on one target alternate in claiming its single measurement
        per scan, so neither coasts long enough to be deleted. Treating the
        pair as independent (summing covariances) makes the test conservative.
        Filter covariances are used, not reported ones: a bias shared by both
        tracks' measurements cancels in their difference.
        """
        if self.merge_threshold is None or len(self._tracks) < 2:
            return
        ranked = sorted(self._tracks, key=lambda t: (-t.hits, t.created, t.id))
        positions = np.array([t.position for t in ranked])
        covariances = np.array([t.state.covariance[:3, :3] for t in ranked])
        difference = positions[:, None, :] - positions[None, :, :]
        summed = covariances[:, None] + covariances[None, :]
        distance = np.einsum(
            "ijk,ijk->ij",
            difference,
            np.linalg.solve(summed, difference[..., None])[..., 0],
        )
        close = distance <= self.merge_threshold
        kept: list[int] = []
        for i in range(len(ranked)):
            if not close[i, kept].any():
                kept.append(i)
        kept_ids = {ranked[i].id for i in kept}
        self._tracks = [t for t in self._tracks if t.id in kept_ids]

    def _end_scan(self, updated: set[int], timestamp: float) -> None:
        self._merge_duplicates()
        survivors = []
        for track in self._tracks:
            track.scan_hits.append(track.id in updated)
            del track.scan_hits[: -self.confirm_window]
            recent = sum(track.scan_hits)
            if track.status is TrackStatus.TENTATIVE:
                if recent >= self.confirm_hits:
                    track.status, track.confirmed_at = TrackStatus.CONFIRMED, timestamp
                elif len(track.scan_hits) >= self.confirm_window:
                    continue  # cannot reach M of the last N: drop the tentative
            if timestamp - track.last_update <= self.max_coast_time:
                survivors.append(track)
        self._tracks = survivors
