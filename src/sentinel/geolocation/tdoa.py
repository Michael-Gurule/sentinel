"""Maximum-likelihood TDOA geolocation with full measurement covariance."""

from collections.abc import Sequence

import numpy as np

from sentinel.core.errors import GeometryError, InsufficientMeasurementsError
from sentinel.core.linalg import FloatArray
from sentinel.geolocation._nls import solve_whitened
from sentinel.geolocation.chan_ho import chan_ho
from sentinel.geolocation.measurements import (
    GeolocationResult,
    Receiver,
    TDOAMeasurement,
)
from sentinel.geolocation.models import (
    range_difference_model,
    receiver_lookup,
    tdoa_to_range_difference,
)


def initial_position(
    receivers: Sequence[Receiver], measurement: TDOAMeasurement
) -> FloatArray:
    """Chan-Ho closed form when available, else the receiver centroid."""
    if len(measurement) >= 4:
        try:
            return chan_ho(receivers, measurement).position
        except GeometryError:
            pass
    used = receiver_lookup(receivers, [measurement.reference, *measurement.others])
    return np.asarray(np.mean([r.position for r in used], axis=0))


def solve_tdoa(
    receivers: Sequence[Receiver],
    measurement: TDOAMeasurement,
    initial: FloatArray | None = None,
    max_nfev: int = 200,
) -> GeolocationResult:
    """Emitter position from reference-sensor TDOAs (iterative ML).

    The measurement covariance is used in full, so correlated reference-sensor
    differences are weighted correctly. Initialized with Chan-Ho when at least
    four TDOAs are available.

    Raises:
        InsufficientMeasurementsError: fewer than three TDOAs.
        GeometryError: position not observable from the given geometry.
    """
    m = len(measurement)
    if m < 3:
        raise InsufficientMeasurementsError(f"3-D TDOA needs >= 3 TDOAs, got {m}")
    (reference,) = receiver_lookup(receivers, [measurement.reference])
    others = receiver_lookup(receivers, measurement.others)
    observations, covariance = tdoa_to_range_difference(measurement)
    start = (
        initial_position(receivers, measurement)
        if initial is None
        else np.asarray(initial, dtype=np.float64)
    )

    solution = solve_whitened(
        lambda x: range_difference_model(x, reference, others),
        observations,
        covariance,
        start,
        max_nfev,
    )
    return GeolocationResult(
        position=solution.state,
        position_covariance=solution.covariance,
        velocity=None,
        velocity_covariance=None,
        state_covariance=solution.covariance,
        chi2=solution.chi2,
        dof=m - 3,
        num_measurements=m,
        converged=solution.converged,
        method="tdoa_ml",
    )
