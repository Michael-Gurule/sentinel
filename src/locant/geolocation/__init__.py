"""RF emitter geolocation: TDOA/FDOA, range multilateration, and DOP."""

from locant.geolocation.chan_ho import chan_ho
from locant.geolocation.crlb import rms_bound, tdoa_crlb, tdoa_fdoa_crlb
from locant.geolocation.dop import DilutionOfPrecision, range_dop, tdoa_dop
from locant.geolocation.hybrid import solve_tdoa_fdoa
from locant.geolocation.measurements import (
    FDOAMeasurement,
    GeolocationResult,
    Receiver,
    TDOAMeasurement,
)
from locant.geolocation.ranges import solve_ranges, solve_ranges_linear
from locant.geolocation.simulate import (
    difference_covariance,
    simulate_fdoa,
    simulate_ranges,
    simulate_tdoa,
)
from locant.geolocation.systematic import SystematicErrors, inflate_fdoa, inflate_tdoa
from locant.geolocation.tdoa import solve_tdoa

__all__ = [
    "DilutionOfPrecision",
    "FDOAMeasurement",
    "GeolocationResult",
    "Receiver",
    "SystematicErrors",
    "TDOAMeasurement",
    "chan_ho",
    "difference_covariance",
    "inflate_fdoa",
    "inflate_tdoa",
    "range_dop",
    "rms_bound",
    "simulate_fdoa",
    "simulate_ranges",
    "simulate_tdoa",
    "solve_ranges",
    "solve_ranges_linear",
    "solve_tdoa",
    "solve_tdoa_fdoa",
    "tdoa_crlb",
    "tdoa_dop",
    "tdoa_fdoa_crlb",
]
