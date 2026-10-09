"""Post-hoc probability calibration and uncertainty scores."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, logsumexp, softmax

from sentinel.core.linalg import FloatArray


def nll(logits: FloatArray, labels: np.ndarray, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of ``labels`` under softmax(logits / T)."""
    logp = log_softmax(np.asarray(logits) / temperature, axis=1)
    return float(-np.mean(logp[np.arange(len(labels)), labels]))


def fit_temperature(logits: FloatArray, labels: np.ndarray) -> float:
    """Temperature T > 0 minimizing validation NLL (Guo et al., 2017).

    Scaling by a single scalar changes confidence but never the argmax, so
    accuracy is unchanged while calibration improves.
    """
    result = minimize_scalar(
        lambda log_t: nll(logits, labels, float(np.exp(log_t))),
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )
    return float(np.exp(result.x))


def probabilities(logits: FloatArray, temperature: float = 1.0) -> np.ndarray:
    return np.asarray(softmax(np.asarray(logits) / temperature, axis=1))


def energy_score(logits: FloatArray, temperature: float = 1.0) -> np.ndarray:
    """Negative free energy T·logsumexp(f/T); higher means more in-distribution
    (Liu et al., 2020)."""
    return np.asarray(temperature * logsumexp(np.asarray(logits) / temperature, axis=1))


def max_softmax(logits: FloatArray, temperature: float = 1.0) -> np.ndarray:
    """Maximum softmax probability (Hendrycks & Gimpel, 2017)."""
    return np.asarray(probabilities(logits, temperature).max(axis=1))
