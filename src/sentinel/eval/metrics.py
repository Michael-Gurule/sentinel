"""Evaluation metrics with uncertainty: classification, calibration, detection."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from sentinel.core.linalg import FloatArray

Metric = Callable[[np.ndarray, np.ndarray], float]


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(np.asarray(labels) == np.asarray(predictions)))


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(f1_score(labels, predictions, average="macro", zero_division=0))


def per_class_f1(
    labels: np.ndarray, predictions: np.ndarray, num_classes: int
) -> list[float]:
    scores = f1_score(
        labels,
        predictions,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )
    return [float(s) for s in scores]


def confusion(
    labels: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    """Row-normalizable counts: rows true class, columns predicted class."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(labels), np.asarray(predictions)), 1)
    return matrix


@dataclass(frozen=True)
class Estimate:
    """Point estimate with a two-sided confidence interval."""

    value: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"value": self.value, "ci_low": self.low, "ci_high": self.high}


def bootstrap(
    metric: Metric,
    labels: np.ndarray,
    predictions: np.ndarray,
    rng: np.random.Generator,
    resamples: int = 1_000,
    confidence: float = 0.95,
) -> Estimate:
    """Percentile bootstrap CI of ``metric`` over test samples."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    n = labels.size
    draws = np.array(
        [metric(labels[i], predictions[i]) for i in rng.integers(0, n, (resamples, n))]
    )
    tail = (1.0 - confidence) / 2.0
    return Estimate(
        value=metric(labels, predictions),
        low=float(np.quantile(draws, tail)),
        high=float(np.quantile(draws, 1.0 - tail)),
    )


def seed_summary(values: list[float], confidence: float = 0.95) -> Estimate:
    """Mean over training seeds with a Student-t interval (seed variance)."""
    from scipy.stats import t

    v = np.asarray(values, dtype=np.float64)
    if v.size == 1:
        return Estimate(float(v[0]), float(v[0]), float(v[0]))
    half = float(
        t.ppf(0.5 + confidence / 2.0, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
    )
    return Estimate(float(v.mean()), float(v.mean() - half), float(v.mean() + half))


def expected_calibration_error(
    probs: FloatArray, labels: np.ndarray, bins: int = 15
) -> tuple[float, dict[str, list[float]]]:
    """Top-label ECE with equal-width confidence bins, plus the reliability table."""
    p = np.asarray(probs)
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == np.asarray(labels)).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    table: dict[str, list[float]] = {
        "bin_confidence": [],
        "bin_accuracy": [],
        "bin_count": [],
    }
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        gap = abs(correct[mask].mean() - confidence[mask].mean())
        ece += mask.mean() * gap
        table["bin_confidence"].append(float(confidence[mask].mean()))
        table["bin_accuracy"].append(float(correct[mask].mean()))
        table["bin_count"].append(float(mask.sum()))
    return float(ece), table


def auroc(in_distribution: FloatArray, out_of_distribution: FloatArray) -> float:
    """AUROC for separating in-distribution (higher score) from OOD samples."""
    y = np.concatenate(
        [np.ones(len(in_distribution)), np.zeros(len(out_of_distribution))]
    )
    s = np.concatenate([in_distribution, out_of_distribution])
    return float(roc_auc_score(y, s))


def detection_roc(
    background_scores: FloatArray, event_scores: FloatArray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Pfa, Pd, thresholds) over all score thresholds."""
    y = np.concatenate([np.zeros(len(background_scores)), np.ones(len(event_scores))])
    s = np.concatenate([background_scores, event_scores])
    pfa, pd, thresholds = roc_curve(y, s)
    return np.asarray(pfa), np.asarray(pd), np.asarray(thresholds)


def binomial_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> Estimate:
    """Clopper-Pearson interval for a rate (e.g. empirical Pfa or Pd)."""
    from scipy.stats import beta

    alpha = 1.0 - confidence
    low = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    )
    high = (
        1.0
        if successes == trials
        else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    )
    return Estimate(successes / trials, low, high)
