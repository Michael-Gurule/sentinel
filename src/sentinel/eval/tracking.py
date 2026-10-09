"""Multi-target tracking metrics.

* **GOSPA** (Rahmathullah, García-Fernández & Svensson 2017), α = 2: a metric
  on sets of positions that decomposes into localization error of assigned
  targets plus a fixed cost c^p/2 per missed target and per false track.
* **OSPA** (Schuhmacher, Vo & Vo 2008): per-target normalized variant.
* Track-level measures from a per-scan truth-to-track assignment: purity,
  fragmentation, identity switches, confirmation latency, and localization
  RMSE / NEES of matched tracks.
"""

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel.core.linalg import FloatArray, nees


@dataclass(frozen=True)
class GOSPA:
    """GOSPA distance and its decomposition (α = 2)."""

    distance: float
    localization: float
    """Sum over assigned pairs of d^p (before the 1/p root)."""
    missed: int
    false: int
    assignments: tuple[tuple[int, int], ...] = ()


def _distances(truth: FloatArray, estimates: FloatArray) -> FloatArray:
    t = np.asarray(truth, dtype=np.float64).reshape(-1, 3)
    e = np.asarray(estimates, dtype=np.float64).reshape(-1, 3)
    return np.asarray(np.linalg.norm(t[:, None, :] - e[None, :, :], axis=2))


def gospa(
    truth: FloatArray, estimates: FloatArray, cutoff: float, p: float = 2.0
) -> GOSPA:
    """GOSPA between a set of true and estimated positions (rows)."""
    d = _distances(truth, estimates)
    n, m = d.shape
    pairs: list[tuple[int, int]] = []
    localization = 0.0
    if n and m:
        rows, cols = linear_sum_assignment(np.minimum(d, cutoff) ** p)
        for r, c in zip(rows, cols, strict=True):
            if d[r, c] < cutoff:
                pairs.append((int(r), int(c)))
                localization += float(d[r, c] ** p)
    missed, false = n - len(pairs), m - len(pairs)
    total = localization + cutoff**p / 2.0 * (missed + false)
    return GOSPA(float(total ** (1.0 / p)), localization, missed, false, tuple(pairs))


def ospa(
    truth: FloatArray, estimates: FloatArray, cutoff: float, p: float = 2.0
) -> float:
    """OSPA distance (0 when both sets are empty)."""
    d = _distances(truth, estimates)
    n, m = d.shape
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return float(cutoff)
    rows, cols = linear_sum_assignment(np.minimum(d, cutoff) ** p)
    cost = float(np.sum(np.minimum(d[rows, cols], cutoff) ** p)) + cutoff**p * abs(
        n - m
    )
    return float((cost / max(n, m)) ** (1.0 / p))


@dataclass(frozen=True, eq=False)
class Snapshot:
    """Truth and confirmed-track estimates at one scan."""

    time: float
    truth: dict[str, FloatArray]
    """Truth id → position, for targets that exist (and should be tracked)."""
    track_ids: tuple[int, ...] = ()
    positions: FloatArray = field(default_factory=lambda: np.zeros((0, 3)))
    covariances: FloatArray = field(default_factory=lambda: np.zeros((0, 3, 3)))


@dataclass(frozen=True)
class TrackingEvaluation:
    gospa_mean: float
    localization_rmse: float
    missed_mean: float
    false_mean: float
    ospa_mean: float
    nees_mean: float
    purity: float
    fragmentations: int
    id_switches: int
    latency: dict[str, float]
    """Truth id → seconds from first appearance to first confirmed match
    (``inf`` if never tracked)."""
    scans: int

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def evaluate_tracking(
    snapshots: Sequence[Snapshot], cutoff: float
) -> TrackingEvaluation:
    """Score a tracker run against truth, scan by scan."""
    gospa_values, ospa_values, missed, false = [], [], [], []
    sq_errors: list[float] = []
    nees_values: list[float] = []
    track_truths: dict[int, list[str]] = defaultdict(list)
    truth_track_sequence: dict[str, list[int | None]] = defaultdict(list)
    first_seen: dict[str, float] = {}
    first_tracked: dict[str, float] = {}
    for snap in snapshots:
        truth_ids = list(snap.truth)
        truth_pos = np.array([snap.truth[t] for t in truth_ids]).reshape(-1, 3)
        g = gospa(truth_pos, snap.positions, cutoff)
        gospa_values.append(g.distance)
        ospa_values.append(ospa(truth_pos, snap.positions, cutoff))
        missed.append(g.missed)
        false.append(g.false)
        matched = {truth_ids[r]: c for r, c in g.assignments}
        for tid in truth_ids:
            first_seen.setdefault(tid, snap.time)
            col = matched.get(tid)
            truth_track_sequence[tid].append(
                None if col is None else snap.track_ids[col]
            )
            if col is None:
                continue
            first_tracked.setdefault(tid, snap.time)
            error = snap.positions[col] - snap.truth[tid]
            sq_errors.append(float(error @ error))
            nees_values.append(nees(error, snap.covariances[col]))
            track_truths[snap.track_ids[col]].append(tid)

    purity_values = [
        Counter(truths).most_common(1)[0][1] / len(truths)
        for truths in track_truths.values()
    ]
    # Fragmentation: tracking of a truth resumes after a gap.
    # Identity switch: the matched track id changes.
    fragmentations = id_switches = 0
    for sequence in truth_track_sequence.values():
        previous: int | None = None
        in_gap = False
        for track_id in sequence:
            if track_id is None:
                in_gap = previous is not None
                continue
            if previous is not None and track_id != previous:
                id_switches += 1
            if in_gap:
                fragmentations += 1
            previous, in_gap = track_id, False
    latency = {
        tid: first_tracked.get(tid, float("inf")) - first_seen[tid]
        for tid in first_seen
    }
    return TrackingEvaluation(
        gospa_mean=float(np.mean(gospa_values)) if gospa_values else 0.0,
        localization_rmse=float(np.sqrt(np.mean(sq_errors)))
        if sq_errors
        else float("nan"),
        missed_mean=float(np.mean(missed)) if missed else 0.0,
        false_mean=float(np.mean(false)) if false else 0.0,
        ospa_mean=float(np.mean(ospa_values)) if ospa_values else 0.0,
        nees_mean=float(np.mean(nees_values)) if nees_values else float("nan"),
        purity=float(np.mean(purity_values)) if purity_values else float("nan"),
        fragmentations=fragmentations,
        id_switches=id_switches,
        latency=latency,
        scans=len(snapshots),
    )
