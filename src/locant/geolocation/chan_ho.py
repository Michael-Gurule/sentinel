"""Two-stage closed-form TDOA estimator of Chan and Ho (1994).

Y. T. Chan and K. C. Ho, "A simple and efficient estimator for hyperbolic
location," IEEE Trans. Signal Processing 42(8), 1994.

With the reference receiver s₀ moved to the origin (s'ₖ = sₖ - s₀, x' = x - s₀),
range differences dₖ = rₖ - r₀ satisfy the pseudo-linear equations

    2 s'ₖᵀ x' + 2 dₖ r₀ = ‖s'ₖ‖² - dₖ²,

linear in θ = [x', r₀]. Stage 1 solves them by weighted least squares, treating
r₀ as independent. Stage 2 enforces r₀² = ‖x'‖². The estimator attains the
CRLB at moderate noise and is used here as a closed-form initializer for the
iterative ML solver.
"""

from collections.abc import Sequence

import numpy as np

from locant.core.errors import (
    GeometryError,
    InsufficientMeasurementsError,
    NotPositiveDefiniteError,
)
from locant.core.linalg import FloatArray, solve_psd
from locant.geolocation.measurements import (
    GeolocationResult,
    Receiver,
    TDOAMeasurement,
)
from locant.geolocation.models import (
    range_difference_model,
    receiver_lookup,
    tdoa_to_range_difference,
)

_STAGE2_MIN_SIGMAS = 3.0
"""Stage 2 linearizes θ² around the stage-1 estimate, which is only valid when
each element of θ is well away from zero relative to its uncertainty. Below this
many standard deviations stage 2 is skipped and the stage-1 estimate returned
(Chan & Ho 1994, §IV discuss the same small-noise requirement)."""


def _wls(
    design: FloatArray, observations: FloatArray, noise_cov: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Weighted least squares; returns estimate and its covariance."""
    weighted = solve_psd(noise_cov, design)  # Ψ⁻¹ G
    information = design.T @ weighted
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise GeometryError("singular geometry in Chan-Ho stage") from exc
    estimate = covariance @ (weighted.T @ observations)
    return np.asarray(estimate), np.asarray(covariance)


def chan_ho(
    receivers: Sequence[Receiver], measurement: TDOAMeasurement
) -> GeolocationResult:
    """Closed-form 3-D emitter position from reference-sensor TDOAs.

    Requires at least four non-reference receivers (five in total).

    The returned covariance is the first-order (small-noise) approximation.
    It is consistent when the geometry is redundant (more than five
    receivers). With exactly five, stage 1 has no redundancy and stage 2
    becomes biased along poorly determined axes, so the covariance is
    optimistic. Use :func:`~locant.geolocation.tdoa.solve_tdoa`, which is
    initialized from this estimate, for the final position and covariance.

    Raises:
        InsufficientMeasurementsError: fewer than four TDOAs.
        GeometryError: receivers are degenerate (e.g. coplanar with the
            reference in a way that makes the stage-1 system singular).
    """
    m = len(measurement)
    if m < 4:
        raise InsufficientMeasurementsError(
            f"Chan-Ho in 3-D needs >= 4 TDOAs (5 receivers), got {m}"
        )
    (reference,) = receiver_lookup(receivers, [measurement.reference])
    others = receiver_lookup(receivers, measurement.others)
    d, q = tdoa_to_range_difference(measurement)

    s = np.array([r.position for r in others]) - reference.position  # s'ₖ
    g1 = np.hstack([2.0 * s, 2.0 * d[:, None]])
    h1 = np.sum(s**2, axis=1) - d**2
    if np.linalg.matrix_rank(g1) < 4:
        raise GeometryError("receiver geometry is degenerate for Chan-Ho stage 1")

    # Stage 1: start with B = I, then refine B = diag(rₖ) from the estimate.
    theta, _ = _wls(g1, h1, q)
    ranges = np.maximum(np.linalg.norm(theta[:3] - s, axis=1), 1e-9)
    b1 = np.diag(ranges)
    theta, cov_theta = _wls(g1, h1, 4.0 * b1 @ q @ b1)

    position_rel = theta[:3]
    position_cov = cov_theta[:3, :3]

    # Stage 2: exploit r₀² = ‖x'‖².
    theta_sigma = np.sqrt(np.diag(cov_theta))
    stage2_ok = bool(np.all(np.abs(theta) > _STAGE2_MIN_SIGMAS * theta_sigma))
    if stage2_ok:
        g2 = np.vstack([np.eye(3), np.ones((1, 3))])
        h2 = theta**2
        b2 = np.diag(theta)
        try:
            z, cov_z = _wls(g2, h2, 4.0 * b2 @ cov_theta @ b2)
        except (GeometryError, NotPositiveDefiniteError):
            stage2_ok = False
        else:
            position_rel = np.sign(theta[:3]) * np.sqrt(np.abs(z))
            b3_inv = np.diag(1.0 / position_rel)
            position_cov = b3_inv @ cov_z @ b3_inv / 4.0

    position = position_rel + reference.position
    predicted, _ = range_difference_model(position, reference, others)
    residual = d - predicted
    chi2 = float(residual @ solve_psd(q, residual))
    return GeolocationResult(
        position=position,
        position_covariance=0.5 * (position_cov + position_cov.T),
        velocity=None,
        velocity_covariance=None,
        state_covariance=0.5 * (position_cov + position_cov.T),
        chi2=chi2,
        dof=m - 3,
        num_measurements=m,
        converged=True,
        method="chan_ho" if stage2_ok else "chan_ho_stage1",
    )
