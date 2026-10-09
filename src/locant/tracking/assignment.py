"""Measurement-to-track assignment."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from locant.core.linalg import FloatArray


def assign_gnn(
    cost: FloatArray, unassigned_cost: float | FloatArray
) -> list[tuple[int, int]]:
    """Global nearest neighbor assignment.

    Finds the jointly optimal one-to-one assignment of tracks (rows) to
    measurements (columns), where each track may instead stay unassigned at
    ``unassigned_cost``. Infinite costs mark gated-out pairs. Unlike greedy
    nearest neighbor, the total cost is minimized over all pairs at once.

    Args:
        cost: ``(num_tracks, num_measurements)`` cost matrix, ``inf`` where a
            pair is infeasible.
        unassigned_cost: Cost of leaving a track unassigned (scalar or one per
            track), typically the gate threshold.

    Returns:
        ``(track_index, measurement_index)`` pairs, sorted by track index.
    """
    cost = np.asarray(cost, dtype=np.float64)
    num_tracks, num_measurements = cost.shape
    if num_tracks == 0 or num_measurements == 0:
        return []
    missed = np.broadcast_to(np.asarray(unassigned_cost, dtype=np.float64), num_tracks)
    dummy = np.full((num_tracks, num_tracks), np.inf)
    np.fill_diagonal(dummy, missed)
    augmented = np.hstack([cost, dummy])
    rows, cols = linear_sum_assignment(augmented)
    return [
        (int(r), int(c))
        for r, c in zip(rows, cols, strict=True)
        if c < num_measurements and np.isfinite(cost[r, c])
    ]
