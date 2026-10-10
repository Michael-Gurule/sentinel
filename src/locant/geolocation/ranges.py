"""Range-based (spherical) multilateration.

Used when absolute ranges are measured (e.g. two-way ranging), as opposed to
the range *differences* of TDOA.
"""

import numpy as np

from locant.core.errors import GeometryError, InsufficientMeasurementsError
from locant.core.linalg import FloatArray, solve_psd


def _validate(
    sensor_positions: FloatArray, ranges: FloatArray
) -> tuple[FloatArray, FloatArray]:
    sensors = np.asarray(sensor_positions, dtype=np.float64)
    z = np.asarray(ranges, dtype=np.float64)
    if sensors.ndim != 2 or sensors.shape[1] != 3:
        raise ValueError("sensor_positions must have shape (n, 3)")
    if z.shape != (len(sensors),):
        raise ValueError("ranges must have one entry per sensor")
    return sensors, z


def solve_ranges_linear(sensor_positions: FloatArray, ranges: FloatArray) -> FloatArray:
    """Closed-form position by differencing the sphere equations.

    Subtracting sensor 0's equation from the others gives the linear system
    2 (sᵢ - s₀)ᵀ x = ‖sᵢ‖² - ‖s₀‖² + r₀² - rᵢ². Exact on noiseless data;
    not statistically efficient (use :func:`solve_ranges` for that).

    Raises:
        InsufficientMeasurementsError: fewer than four sensors.
        GeometryError: sensors are coplanar or collinear.
    """
    sensors, z = _validate(sensor_positions, ranges)
    if len(sensors) < 4:
        raise InsufficientMeasurementsError(
            f"linear 3-D multilateration needs >= 4 sensors, got {len(sensors)}"
        )
    s0, r0 = sensors[0], z[0]
    a = 2.0 * (sensors[1:] - s0)
    b = np.sum(sensors[1:] ** 2, axis=1) - np.sum(s0**2) + r0**2 - z[1:] ** 2
    if np.linalg.matrix_rank(a) < 3:
        raise GeometryError("sensors are coplanar or collinear")
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.asarray(solution)


def solve_ranges(
    sensor_positions: FloatArray,
    ranges: FloatArray,
    covariance: FloatArray,
    initial: FloatArray | None = None,
    max_iterations: int = 50,
    tolerance: float = 1e-9,
) -> tuple[FloatArray, FloatArray, bool]:
    """Weighted least-squares position from ranges (Gauss-Newton).

    With residual r(x) = z - h(x) and H = ∂h/∂x (rows are unit vectors from each
    sensor to x), each step is δ = (Hᵀ C⁻¹ H)⁻¹ Hᵀ C⁻¹ r.

    Args:
        covariance: Range measurement covariance, shape (n, n).
        initial: Starting position; defaults to the linear solution when
            available, otherwise the sensor centroid.

    Returns:
        ``(position, covariance, converged)``. The covariance is
        (Hᵀ C⁻¹ H)⁻¹ at the solution.

    Raises:
        InsufficientMeasurementsError: fewer than three sensors.
        GeometryError: the normal equations are singular.
    """
    sensors, z = _validate(sensor_positions, ranges)
    if len(sensors) < 3:
        raise InsufficientMeasurementsError(
            f"3-D range multilateration needs >= 3 sensors, got {len(sensors)}"
        )
    cov = np.asarray(covariance, dtype=np.float64)
    if initial is not None:
        x = np.asarray(initial, dtype=np.float64).copy()
    else:
        try:
            x = solve_ranges_linear(sensors, z)
        except (InsufficientMeasurementsError, GeometryError):
            x = sensors.mean(axis=0)

    def linearize(position: FloatArray) -> tuple[FloatArray, FloatArray]:
        delta = position - sensors
        distances = np.linalg.norm(delta, axis=1)
        if np.any(distances == 0.0):
            raise GeometryError("estimate coincides with a sensor")
        return distances, delta / distances[:, None]

    converged = False
    for _ in range(max_iterations):
        predicted, h = linearize(x)
        information = h.T @ solve_psd(cov, h)
        if np.linalg.matrix_rank(information) < 3:
            raise GeometryError("singular normal equations")
        step = np.linalg.solve(information, h.T @ solve_psd(cov, z - predicted))
        x = x + step
        if np.linalg.norm(step) <= tolerance * (1.0 + np.linalg.norm(x)):
            converged = True
            break

    _, h = linearize(x)
    position_cov = np.linalg.inv(h.T @ solve_psd(cov, h))
    return x, 0.5 * (position_cov + position_cov.T), converged
