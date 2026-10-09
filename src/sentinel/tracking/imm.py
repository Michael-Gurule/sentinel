"""Interacting Multiple Model (IMM) filtering.

A single constant-velocity model must trade agility for smoothness: low process
noise tracks steady targets (fires, cruising aircraft) well but loses
accelerating ones (boosting launches); high process noise does the reverse.
The IMM (Blom & Bar-Shalom 1988) runs one filter per motion model and mixes
them with Markov mode probabilities, so each target is followed by whichever
model currently explains it.

Each cycle: mix the per-model estimates by the mode transition probabilities,
predict each model, update each with the measurement, reweight the modes by
their measurement likelihoods, and combine into one Gaussian for gating and
reporting.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sentinel.core.linalg import FloatArray, symmetrize
from sentinel.tracking.kalman import Gaussian, predict, update
from sentinel.tracking.models import ConstantVelocity


@dataclass(eq=False)
class IMMState:
    """Per-model estimates and mode probabilities."""

    states: list[Gaussian]
    probabilities: FloatArray

    def combined(self) -> Gaussian:
        """Moment-matched single Gaussian of the mixture."""
        mean = sum(
            (p * s.mean for p, s in zip(self.probabilities, self.states, strict=True)),
            np.zeros_like(self.states[0].mean),
        )
        covariance = np.zeros_like(self.states[0].covariance)
        for p, s in zip(self.probabilities, self.states, strict=True):
            d = s.mean - mean
            covariance += p * (s.covariance + np.outer(d, d))
        return Gaussian(mean, symmetrize(covariance))


def transition_matrix(sojourn_times: Sequence[float], dt: float) -> FloatArray:
    """Markov mode-switch matrix for a scan interval ``dt``.

    Model i persists with probability exp(-dt / τᵢ); otherwise it switches
    to one of the other models uniformly.
    """
    n = len(sojourn_times)
    out = np.zeros((n, n))
    for i, tau in enumerate(sojourn_times):
        stay = float(np.exp(-dt / tau)) if n > 1 else 1.0
        out[i] = (1.0 - stay) / max(n - 1, 1)
        out[i, i] = stay
    return out


def imm_predict(
    imm: IMMState,
    models: Sequence[ConstantVelocity],
    sojourn_times: Sequence[float],
    dt: float,
) -> IMMState:
    """Mixing step followed by per-model prediction."""
    if dt == 0.0:
        return imm
    pi = transition_matrix(sojourn_times, dt)
    predicted_probs = pi.T @ imm.probabilities
    states = []
    for j, model in enumerate(models):
        weights = pi[:, j] * imm.probabilities / max(predicted_probs[j], 1e-300)
        mixed = IMMState(imm.states, weights).combined()
        states.append(predict(mixed, model, dt))
    return IMMState(states, predicted_probs)


def imm_update(
    imm: IMMState,
    measurement: FloatArray,
    linearize: object,
    noise: FloatArray,
) -> IMMState:
    """Per-model (E)KF update and mode reweighting by measurement likelihood.

    ``linearize`` maps a state mean to ``(h(x), H)``.
    """
    states, log_likelihoods = [], []
    for state in imm.states:
        predicted, matrix = linearize(state.mean)  # type: ignore[operator]
        posterior, innov = update(state, measurement, matrix, noise, predicted)
        _, logdet = np.linalg.slogdet(2.0 * np.pi * innov.covariance)
        states.append(posterior)
        log_likelihoods.append(-0.5 * (innov.nis + logdet))
    log_weights = np.log(np.maximum(imm.probabilities, 1e-300)) + np.asarray(
        log_likelihoods
    )
    log_weights -= log_weights.max()
    probabilities = np.exp(log_weights)
    return IMMState(states, probabilities / probabilities.sum())
