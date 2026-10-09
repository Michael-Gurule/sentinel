"""Maximum-likelihood TDOA geolocation with full measurement covariance."""

from collections.abc import Sequence
from contextlib import suppress

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
from sentinel.geolocation.systematic import SystematicErrors, inflate_tdoa


def initial_positions(
    receivers: Sequence[Receiver], measurement: TDOAMeasurement
) -> list[FloatArray]:
    """Starting points in order of preference: Chan-Ho closed form when
    available, then the receiver centroid."""
    starts = []
    if len(measurement) >= 4:
        with suppress(GeometryError):
            starts.append(chan_ho(receivers, measurement).position)
    used = receiver_lookup(receivers, [measurement.reference, *measurement.others])
    starts.append(np.asarray(np.mean([r.position for r in used], axis=0)))
    return starts


def solve_tdoa(
    receivers: Sequence[Receiver],
    measurement: TDOAMeasurement,
    initial: FloatArray | None = None,
    max_nfev: int = 200,
    systematic: SystematicErrors | None = None,
) -> GeolocationResult:
    """Emitter position from reference-sensor TDOAs (iterative ML).

    The measurement covariance is used in full, so correlated reference-sensor
    differences are weighted correctly. Initialized with Chan-Ho when at least
    four TDOAs are available. Chan-Ho can pick the wrong root when one
    coordinate is poorly observed (e.g. altitude from a near-planar network)
    and the iteration then runs off along a hyperboloid asymptote; if that
    solution fails the χ² fit test, the solver restarts from the receiver
    centroid and keeps the better fit.

    Args:
        systematic: Receiver clock-bias / survey-error levels to fold into the
            measurement covariance (see :mod:`sentinel.geolocation.systematic`).
            Without it the reported covariance assumes perfectly synchronized,
            perfectly surveyed receivers.

    Raises:
        InsufficientMeasurementsError: fewer than three TDOAs.
        GeometryError: position not observable from the given geometry.
    """
    if systematic is not None:
        measurement = inflate_tdoa(measurement, systematic)
    m = len(measurement)
    if m < 3:
        raise InsufficientMeasurementsError(f"3-D TDOA needs >= 3 TDOAs, got {m}")
    (reference,) = receiver_lookup(receivers, [measurement.reference])
    others = receiver_lookup(receivers, measurement.others)
    observations, covariance = tdoa_to_range_difference(measurement)
    starts = (
        initial_positions(receivers, measurement)
        if initial is None
        else [np.asarray(initial, dtype=np.float64)]
    )
    best: GeolocationResult | None = None
    error: GeometryError | None = None
    for start in starts:
        try:
            solution = solve_whitened(
                lambda x: range_difference_model(x, reference, others),
                observations,
                covariance,
                start,
                max_nfev,
            )
        except GeometryError as exc:
            error = exc
            continue
        result = GeolocationResult(
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
        if best is None or result.chi2 < best.chi2:
            best = result
        if best.converged and best.fits():
            break
    if best is None:
        assert error is not None
        raise error
    return best
