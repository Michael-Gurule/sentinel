"""Track-to-track fusion: combining estimates from separate trackers.

When OPIR and RF are tracked separately (decentralized architecture), their
track estimates are correlated through the shared target dynamics (common
process noise), so the cross-covariance is unknown.

* :func:`naive_fusion` assumes independence: P = (Pa⁻¹ + Pb⁻¹)⁻¹. It is
  optimal only for independent estimates and overconfident otherwise.
* :func:`covariance_intersection` (Julier & Uhlmann 1997) is consistent for
  *any* cross-correlation: P⁻¹ = ω Pa⁻¹ + (1-ω) Pb⁻¹, with ω minimizing
  trace(P). The price is conservatism when the estimates really are
  independent.

E6 compares both with centralized measurement-level fusion
(docs/adr/0001-fusion-architecture.md).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from sentinel.core.linalg import FloatArray, mahalanobis_sq
from sentinel.core.stats import chi2_gate
from sentinel.tracking.assignment import assign_gnn
from sentinel.tracking.kalman import Gaussian


def naive_fusion(a: Gaussian, b: Gaussian) -> Gaussian:
    """Information-form fusion assuming independent estimates."""
    info_a, info_b = np.linalg.inv(a.covariance), np.linalg.inv(b.covariance)
    covariance = np.linalg.inv(info_a + info_b)
    mean = covariance @ (info_a @ a.mean + info_b @ b.mean)
    return Gaussian(mean, 0.5 * (covariance + covariance.T))


def covariance_intersection(a: Gaussian, b: Gaussian) -> tuple[Gaussian, float]:
    """CI fusion with ω chosen to minimize the fused covariance trace."""
    info_a, info_b = np.linalg.inv(a.covariance), np.linalg.inv(b.covariance)

    def fused(omega: float) -> tuple[FloatArray, FloatArray]:
        information = omega * info_a + (1.0 - omega) * info_b
        covariance = np.linalg.inv(information)
        mean = covariance @ (omega * info_a @ a.mean + (1.0 - omega) * info_b @ b.mean)
        return mean, covariance

    result = minimize_scalar(
        lambda w: float(np.trace(fused(w)[1])), bounds=(0.0, 1.0), method="bounded"
    )
    omega = float(result.x)
    mean, covariance = fused(omega)
    return Gaussian(mean, 0.5 * (covariance + covariance.T)), omega


@dataclass(frozen=True, eq=False)
class FusedEstimate:
    state: Gaussian
    sources: tuple[str, ...]
    """Which trackers contributed (e.g. ``("opir", "rf")``)."""
    indices: tuple[int | None, int | None] = (None, None)
    """Index of the contributing track in each input list."""


def fuse_track_lists(
    tracks_a: Sequence[Gaussian],
    tracks_b: Sequence[Gaussian],
    method: str = "ci",
    labels: tuple[str, str] = ("a", "b"),
    gate_probability: float = 0.99,
) -> list[FusedEstimate]:
    """Associate two track lists by position and fuse matched pairs.

    Pairs are gated on the χ² distance of their position difference under
    the sum of their position covariances (conservative for correlated
    tracks) and assigned by GNN; unmatched tracks pass through unchanged.
    """
    if method not in ("ci", "naive"):
        raise ValueError("method must be 'ci' or 'naive'")
    threshold = chi2_gate(3, gate_probability)
    cost = np.full((len(tracks_a), len(tracks_b)), np.inf)
    for i, ta in enumerate(tracks_a):
        for j, tb in enumerate(tracks_b):
            d2 = mahalanobis_sq(
                ta.mean[:3] - tb.mean[:3], ta.covariance[:3, :3] + tb.covariance[:3, :3]
            )
            if d2 <= threshold:
                cost[i, j] = d2
    pairs = assign_gnn(cost, unassigned_cost=threshold)
    matched_a = {i for i, _ in pairs}
    matched_b = {j for _, j in pairs}
    out = [
        FusedEstimate(
            naive_fusion(tracks_a[i], tracks_b[j])
            if method == "naive"
            else covariance_intersection(tracks_a[i], tracks_b[j])[0],
            labels,
            (i, j),
        )
        for i, j in pairs
    ]
    out += [
        FusedEstimate(t, (labels[0],), (i, None))
        for i, t in enumerate(tracks_a)
        if i not in matched_a
    ]
    out += [
        FusedEstimate(t, (labels[1],), (None, j))
        for j, t in enumerate(tracks_b)
        if j not in matched_b
    ]
    return out
