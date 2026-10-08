"""Joint TDOA/FDOA estimation of emitter position and velocity."""

from collections.abc import Sequence

import numpy as np
from scipy.linalg import block_diag

from sentinel.core.errors import InsufficientMeasurementsError
from sentinel.core.linalg import FloatArray
from sentinel.geolocation._nls import solve_whitened
from sentinel.geolocation.measurements import (
    FDOAMeasurement,
    GeolocationResult,
    Receiver,
    TDOAMeasurement,
)
from sentinel.geolocation.models import (
    fdoa_to_range_rate_difference,
    range_difference_model,
    range_rate_difference_model,
    receiver_lookup,
    tdoa_to_range_difference,
)
from sentinel.geolocation.tdoa import solve_tdoa


def solve_tdoa_fdoa(
    receivers: Sequence[Receiver],
    tdoa: TDOAMeasurement,
    fdoa: FDOAMeasurement | None = None,
    initial: FloatArray | None = None,
    max_nfev: int = 400,
) -> GeolocationResult:
    """Emitter position and, when FDOA is available, velocity.

    Without FDOA the velocity is not observable, so the result is a
    position-only TDOA solution with ``velocity=None``. TDOA and FDOA errors
    are assumed independent of each other.

    Args:
        initial: Optional starting state ``[x, y, z, vx, vy, vz]``.

    Raises:
        InsufficientMeasurementsError: fewer than six scalar measurements for
            the six unknowns.
        GeometryError: state not observable from the given geometry and
            receiver motion.
    """
    if fdoa is None:
        start = None if initial is None else np.asarray(initial)[:3]
        return solve_tdoa(receivers, tdoa, initial=start)

    num_measurements = len(tdoa) + len(fdoa)
    if num_measurements < 6:
        raise InsufficientMeasurementsError(
            f"joint position/velocity needs >= 6 measurements, got {num_measurements}"
        )
    (tdoa_ref,) = receiver_lookup(receivers, [tdoa.reference])
    tdoa_others = receiver_lookup(receivers, tdoa.others)
    (fdoa_ref,) = receiver_lookup(receivers, [fdoa.reference])
    fdoa_others = receiver_lookup(receivers, fdoa.others)

    rd, rd_cov = tdoa_to_range_difference(tdoa)
    rrd, rrd_cov = fdoa_to_range_rate_difference(fdoa)
    observations = np.concatenate([rd, rrd])
    covariance = block_diag(rd_cov, rrd_cov)

    def model(state: FloatArray) -> tuple[FloatArray, FloatArray]:
        position, velocity = state[:3], state[3:]
        rd_pred, rd_jac = range_difference_model(position, tdoa_ref, tdoa_others)
        rrd_pred, rrd_jac = range_rate_difference_model(
            position, velocity, fdoa_ref, fdoa_others
        )
        jac = np.zeros((len(rd_pred) + len(rrd_pred), 6))
        jac[: len(rd_pred), :3] = rd_jac
        jac[len(rd_pred) :] = rrd_jac
        return np.concatenate([rd_pred, rrd_pred]), jac

    if initial is None:
        if len(tdoa) >= 3:
            position0 = solve_tdoa(receivers, tdoa).position
        else:
            position0 = np.mean([r.position for r in tdoa_others], axis=0)
        start = np.concatenate([position0, np.zeros(3)])
    else:
        start = np.asarray(initial, dtype=np.float64)

    solution = solve_whitened(model, observations, covariance, start, max_nfev)
    return GeolocationResult(
        position=solution.state[:3],
        position_covariance=solution.covariance[:3, :3],
        velocity=solution.state[3:],
        velocity_covariance=solution.covariance[3:, 3:],
        state_covariance=solution.covariance,
        chi2=solution.chi2,
        dof=num_measurements - 6,
        num_measurements=num_measurements,
        converged=solution.converged,
        method="tdoa_fdoa_ml",
    )
