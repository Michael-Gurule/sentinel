"""RF receiver network with realistic error sources.

Each network realization draws, once, the errors a fielded system would carry
for the whole scenario:

* receiver position error (surveyed or navigation position vs. truth);
* clock bias (synchronization error) per receiver;
* local-oscillator frequency offset per receiver;

and each scan adds independent TOA and frequency noise. Scans report the
*nominal* receiver states and a covariance for the random noise only, exactly
as an operational system would; the systematic errors are unmodeled by the
solver. That mismatch is what Phase 4's sensitivity study measures.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from locant.core.constants import SPEED_OF_LIGHT
from locant.core.linalg import FloatArray
from locant.geolocation.measurements import FDOAMeasurement, Receiver, TDOAMeasurement
from locant.geolocation.models import (
    range_difference_model,
    range_rate_difference_model,
)
from locant.geolocation.simulate import difference_covariance
from locant.sim.trajectories import Trajectory


@dataclass(frozen=True, eq=False)
class ReceiverModel:
    """A receiver's motion and error budget.

    Attributes:
        id: Receiver id.
        trajectory: True motion in the scenario ENU frame.
        toa_std: Per-scan time-of-arrival noise, s.
        frequency_std: Per-scan frequency measurement noise, Hz.
        position_error_std: Per-axis error of the reported position, m.
        clock_bias_std: Standard deviation of the fixed clock bias, s.
        lo_offset_std: Standard deviation of the fixed LO offset, Hz.
    """

    id: int
    trajectory: Trajectory
    toa_std: float = 10e-9
    frequency_std: float = 1.0
    position_error_std: float = 0.0
    clock_bias_std: float = 0.0
    lo_offset_std: float = 0.0


@dataclass(frozen=True, eq=False)
class RFScan:
    """One TDOA (and optionally FDOA) measurement set at time ``t``."""

    t: float
    receivers: list[Receiver]
    """Nominal (reported) receiver states, as the solver sees them."""
    tdoa: TDOAMeasurement
    fdoa: FDOAMeasurement | None
    true_receivers: list[Receiver] = field(repr=False, default_factory=list)


class RFNetwork:
    """A realization of a receiver network's systematic errors."""

    def __init__(
        self, receivers: Sequence[ReceiverModel], rng: np.random.Generator
    ) -> None:
        if len(receivers) < 2:
            raise ValueError("an RF network needs at least two receivers")
        self.models = list(receivers)
        n = len(self.models)
        self.position_errors = np.array(
            [rng.normal(0.0, m.position_error_std, 3) for m in self.models]
        )
        self.clock_biases = np.array(
            [rng.normal(0.0, m.clock_bias_std) for m in self.models]
        )
        self.lo_offsets = np.array(
            [rng.normal(0.0, m.lo_offset_std) for m in self.models]
        )
        assert self.position_errors.shape == (n, 3)

    def true_receivers(self, t: float) -> list[Receiver]:
        out = []
        for m in self.models:
            sample = m.trajectory.sample(np.array([t]))
            out.append(Receiver(m.id, sample.position[0], sample.velocity[0]))
        return out

    def nominal_receivers(self, t: float) -> list[Receiver]:
        return [
            Receiver(r.id, r.position + err, r.velocity)
            for r, err in zip(self.true_receivers(t), self.position_errors, strict=True)
        ]

    def scan(
        self,
        t: float,
        emitter_position: FloatArray,
        emitter_velocity: FloatArray,
        rng: np.random.Generator,
        carrier_frequency: float | None = None,
        reference_index: int = 0,
    ) -> RFScan:
        """Measure an emitter at time ``t``; FDOA only if ``carrier_frequency`` is set."""
        truth = self.true_receivers(t)
        ref = truth[reference_index]
        others = [r for i, r in enumerate(truth) if i != reference_index]
        keep = [i for i in range(len(truth)) if i != reference_index]
        m = len(others)

        true_rd, _ = range_difference_model(np.asarray(emitter_position), ref, others)
        toa_std = np.array([mod.toa_std for mod in self.models])
        noise = rng.normal(0.0, toa_std)
        timing_error = (self.clock_biases + noise)[keep] - (self.clock_biases + noise)[
            reference_index
        ]
        declared_toa = float(np.sqrt(np.mean(toa_std**2)))
        tdoa = TDOAMeasurement(
            reference=ref.id,
            others=tuple(r.id for r in others),
            values=true_rd / SPEED_OF_LIGHT + timing_error,
            covariance=difference_covariance(m, declared_toa),
        )

        fdoa = None
        if carrier_frequency is not None:
            true_rrd, _ = range_rate_difference_model(
                np.asarray(emitter_position), np.asarray(emitter_velocity), ref, others
            )
            freq_std = np.array([mod.frequency_std for mod in self.models])
            f_err = self.lo_offsets + rng.normal(0.0, freq_std)
            fdoa = FDOAMeasurement(
                reference=ref.id,
                others=tuple(r.id for r in others),
                values=-true_rrd * carrier_frequency / SPEED_OF_LIGHT
                + f_err[keep]
                - f_err[reference_index],
                covariance=difference_covariance(
                    m, float(np.sqrt(np.mean(freq_std**2)))
                ),
                carrier_frequency=carrier_frequency,
            )
        return RFScan(
            t=t,
            receivers=self.nominal_receivers(t),
            tdoa=tdoa,
            fdoa=fdoa,
            true_receivers=truth,
        )


def default_receiver_network() -> list[Receiver]:
    """Five stationary receivers at mixed altitudes (an example network).

    Altitude diversity keeps the vertical geometry observable; five receivers
    give four TDOAs, enough for the closed-form Chan-Ho initializer.
    """
    positions = [
        [0.0, 0.0, 500.0],  # ground station
        [10_000.0, 0.0, 1_500.0],  # low-altitude ISR aircraft
        [10_000.0, 10_000.0, 1_000.0],  # medium-altitude platform
        [0.0, 10_000.0, 2_000.0],  # high-altitude ISR
        [5_000.0, -4_000.0, 6_000.0],  # stand-off high-altitude platform
    ]
    return [Receiver(i, np.array(p)) for i, p in enumerate(positions)]
