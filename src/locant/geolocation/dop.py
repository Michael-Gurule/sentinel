"""Dilution of precision for TDOA and range geometries.

DOP maps unit measurement noise to position error: for TDOA with iid
per-receiver TOA error σₜ, the position RMS error is GDOP · c · σₜ; for ranges
with iid error σᵣ it is GDOP · σᵣ.
"""

from dataclasses import dataclass

import numpy as np

from locant.core.errors import GeometryError
from locant.core.linalg import FloatArray, solve_psd


@dataclass(frozen=True)
class DilutionOfPrecision:
    """Geometric (3-D), horizontal (x, y), and vertical (z) DOP."""

    gdop: float
    hdop: float
    vdop: float


def _unit_vectors(position: FloatArray, sensor_positions: FloatArray) -> FloatArray:
    delta = np.asarray(position, dtype=np.float64) - np.asarray(
        sensor_positions, dtype=np.float64
    )
    distances = np.linalg.norm(delta, axis=1)
    if np.any(distances == 0.0):
        raise GeometryError("position coincides with a sensor")
    return np.asarray(delta / distances[:, None])


def _dop_from_cofactor(cofactor: FloatArray) -> DilutionOfPrecision:
    return DilutionOfPrecision(
        gdop=float(np.sqrt(np.trace(cofactor))),
        hdop=float(np.sqrt(cofactor[0, 0] + cofactor[1, 1])),
        vdop=float(np.sqrt(cofactor[2, 2])),
    )


def _invert_information(information: FloatArray) -> FloatArray:
    eigenvalues = np.linalg.eigvalsh(information)
    if eigenvalues[0] <= 1e-12 * max(eigenvalues[-1], 1e-300):
        raise GeometryError("geometry is singular; DOP is unbounded")
    return np.asarray(np.linalg.inv(information))


def tdoa_dop(
    position: FloatArray, sensor_positions: FloatArray, reference_index: int = 0
) -> DilutionOfPrecision:
    """DOP of reference-sensor TDOA positioning.

    Rows of the geometry matrix are differenced unit vectors uₖ - u₀, and the
    differences carry the correlated covariance I + 11ᵀ of unit iid TOA errors.
    The result does not depend on which receiver is the reference.

    Raises:
        GeometryError: fewer than four receivers or a singular geometry.
    """
    units = _unit_vectors(position, sensor_positions)
    if len(units) < 4:
        raise GeometryError("3-D TDOA needs >= 4 receivers")
    others = np.delete(units, reference_index, axis=0)
    h = others - units[reference_index]
    m = len(h)
    noise = np.eye(m) + np.ones((m, m))
    return _dop_from_cofactor(_invert_information(h.T @ solve_psd(noise, h)))


def range_dop(
    position: FloatArray, sensor_positions: FloatArray
) -> DilutionOfPrecision:
    """DOP of range (spherical) positioning with iid unit range errors.

    Raises:
        GeometryError: fewer than three sensors or a singular geometry.
    """
    units = _unit_vectors(position, sensor_positions)
    if len(units) < 3:
        raise GeometryError("3-D range positioning needs >= 3 sensors")
    return _dop_from_cofactor(_invert_information(units.T @ units))
