"""First-order handling of systematic receiver errors ("consider" covariance).

Fielded TDOA networks carry errors that are fixed over a scenario but unknown
to the solver: receiver clock bias (synchronization) and receiver survey
(position) error. Ignoring them makes the solver's covariance overconfident.

Treating them as random over network realizations and adding their effect to
the measurement covariance (Schmidt's consider covariance) restores
consistency. For reference-sensor range differences d_k = r_k - r_0:

* a clock bias b_i (s) per receiver adds c(b_k - b_0), covariance
  (cσ_b)² (I + 11ᵀ);
* a survey error e_i per receiver moves d_k by -u_kᵀe_k + u_0ᵀe_0 with unit
  line-of-sight vectors u, so isotropic per-axis σ_p contributes
  σ_p² (uₖᵀuₖ δₖₗ + u₀ᵀu₀) = σ_p² (I + 11ᵀ), independent of geometry.

Both therefore inflate the TDOA covariance by
(σ_b² + σ_p² / c²)(I + 11ᵀ) in s². The inflation is exact to first order and
needs no position estimate.

Caveat: over a single realization the bias is *constant*, so averaging many
scans of the same network does not reduce it. A tracker fusing consecutive
scans must model the bias as correlated across time: E7 shows track NEES
growing with track age unless the systematic part of the fix covariance is
kept as a floor on the reported track covariance (docs/fusion.md).
"""

from dataclasses import dataclass

import numpy as np

from locant.core.constants import SPEED_OF_LIGHT
from locant.geolocation.measurements import FDOAMeasurement, TDOAMeasurement


@dataclass(frozen=True)
class SystematicErrors:
    """Standard deviations of per-receiver errors the solver cannot estimate.

    Attributes:
        receiver_position_std: Survey error per axis, m.
        clock_bias_std: Clock bias, s.
        lo_offset_std: Local-oscillator frequency offset, Hz (FDOA only).
    """

    receiver_position_std: float = 0.0
    clock_bias_std: float = 0.0
    lo_offset_std: float = 0.0

    def __post_init__(self) -> None:
        if min(self.receiver_position_std, self.clock_bias_std, self.lo_offset_std) < 0:
            raise ValueError("standard deviations must be non-negative")

    @property
    def tdoa_variance(self) -> float:
        """Per-receiver timing variance equivalent (s²)."""
        return (
            self.clock_bias_std**2 + (self.receiver_position_std / SPEED_OF_LIGHT) ** 2
        )


def _structure(m: int) -> np.ndarray:
    return np.eye(m) + np.ones((m, m))


def inflate_tdoa(
    measurement: TDOAMeasurement, systematic: SystematicErrors
) -> TDOAMeasurement:
    """TDOA measurement with the systematic-error consider covariance added."""
    m = len(measurement)
    return TDOAMeasurement(
        reference=measurement.reference,
        others=measurement.others,
        values=measurement.values,
        covariance=measurement.covariance + systematic.tdoa_variance * _structure(m),
    )


def inflate_fdoa(
    measurement: FDOAMeasurement, systematic: SystematicErrors
) -> FDOAMeasurement:
    """FDOA measurement with the LO-offset consider covariance added."""
    m = len(measurement)
    return FDOAMeasurement(
        reference=measurement.reference,
        others=measurement.others,
        values=measurement.values,
        covariance=measurement.covariance + systematic.lo_offset_std**2 * _structure(m),
        carrier_frequency=measurement.carrier_frequency,
    )
