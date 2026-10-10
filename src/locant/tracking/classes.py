"""Track-level class posteriors from calibrated per-window classifier outputs.

Each classifier output p(c | window) is turned into a likelihood by dividing
out the training class prior, then pooled into the track posterior in the log
domain with a tempering weight w ≤ 1:

    log π_t(c) = log π_{t-1}(c) + w · [log p_t(c) - log prior(c)] + const.

w = 1 is naive Bayes, correct only for independent observations. Consecutive
OPIR windows overlap almost entirely, so their outputs are highly correlated
and naive Bayes would drive the posterior to certainty after a few frames.
w < 1 discounts that redundancy (log-linear pooling); E6 checks the
calibration of the resulting track posteriors.
"""

import numpy as np

from locant.core.linalg import FloatArray

_EPS = 1e-6


def pool_class_evidence(
    posterior: FloatArray | None,
    probabilities: FloatArray,
    weight: float,
    prior: FloatArray | None = None,
) -> np.ndarray:
    """Updated class posterior after one more classifier output.

    Args:
        posterior: Current track posterior, or ``None`` to start from the prior.
        probabilities: Calibrated classifier probabilities for this window.
        weight: Tempering weight w in (0, 1].
        prior: Class prior the classifier was trained with (uniform default).
    """
    if not 0.0 < weight <= 1.0:
        raise ValueError("weight must be in (0, 1]")
    p = np.clip(np.asarray(probabilities, dtype=np.float64), _EPS, 1.0)
    base = (
        np.full(p.size, 1.0 / p.size)
        if prior is None
        else np.clip(np.asarray(prior, dtype=np.float64), _EPS, 1.0)
    )
    start = base if posterior is None else np.clip(np.asarray(posterior), _EPS, 1.0)
    log_post = np.log(start) + weight * (np.log(p) - np.log(base))
    log_post -= log_post.max()
    out = np.exp(log_post)
    return np.asarray(out / out.sum())
