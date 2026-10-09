"""Split-conformal prediction sets.

Calibrating the (1-α) quantile of a conformity score on held-out data gives
prediction sets that contain the true label with probability ≥ 1-α whenever
calibration and test data are exchangeable. That guarantee is exactly what
the domain-shift experiments test.

Two scores are provided:

* ``lac`` (default): s = 1 - p(true label); the set is every label with
  p ≥ 1 - q̂. "Least ambiguous set-valued classifiers" (Sadinle, Lei &
  Wasserman, 2019): the smallest average sets among conformal methods using
  the same probabilities.
* ``aps``: non-randomized Adaptive Prediction Sets (Romano, Sesia & Candès,
  2020); s = total probability of labels at least as likely as the true one.
  With a confident model, misclassified samples score close to 1, so the
  quantile approaches 1 and sets become nearly the full label set. E3 shows
  this on opir_v2 and therefore ships LAC.
"""

from typing import Literal

import numpy as np

from locant.core.linalg import FloatArray

Method = Literal["lac", "aps"]


def conformal_scores(
    probs: FloatArray, labels: np.ndarray, method: Method = "lac"
) -> np.ndarray:
    """Nonconformity score of each true label (higher = less conforming)."""
    p = np.asarray(probs)
    true_p = p[np.arange(len(labels)), labels]
    if method == "lac":
        return np.asarray(1.0 - true_p)
    if method == "aps":
        return np.asarray(np.sum(np.where(p >= true_p[:, None], p, 0.0), axis=1))
    raise ValueError(f"unknown conformal method {method!r}")


def calibrate(
    probs: FloatArray, labels: np.ndarray, alpha: float, method: Method = "lac"
) -> float:
    """Threshold q̂: the ⌈(n+1)(1-α)⌉-th smallest calibration score."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.sort(conformal_scores(probs, labels, method))
    n = scores.size
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(scores[rank - 1])


def prediction_sets(
    probs: FloatArray, threshold: float, method: Method = "lac"
) -> np.ndarray:
    """Boolean (N, C) membership: label k is in the set if its score ≤ q̂.

    The most likely label is always included, so sets are never empty.
    """
    p = np.asarray(probs)
    if method == "lac":
        members = (1.0 - p) <= threshold
    elif method == "aps":
        order = np.argsort(-p, axis=1)
        sorted_p = np.take_along_axis(p, order, axis=1)
        # In descending-probability order, a label's APS score is the
        # cumulative mass up to and including it.
        include_sorted = np.cumsum(sorted_p, axis=1) <= threshold
        members = np.zeros_like(include_sorted)
        np.put_along_axis(members, order, include_sorted, axis=1)
    else:
        raise ValueError(f"unknown conformal method {method!r}")
    members[np.arange(p.shape[0]), p.argmax(axis=1)] = True
    return np.asarray(members)


def coverage(sets: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(sets[np.arange(len(labels)), labels]))
