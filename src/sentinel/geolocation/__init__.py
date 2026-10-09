"""RF emitter geolocation: TDOA/FDOA, range multilateration, and DOP."""

from sentinel.geolocation.chan_ho import chan_ho
from sentinel.geolocation.crlb import rms_bound, tdoa_crlb, tdoa_fdoa_crlb
from sentinel.geolocation.dop import DilutionOfPrecision, range_dop, tdoa_dop
from sentinel.geolocation.hybrid import solve_tdoa_fdoa
from sentinel.geolocation.measurements import (
    FDOAMeasurement,
    GeolocationResult,
    Receiver,
    TDOAMeasurement,
)
from sentinel.geolocation.ranges import solve_ranges, solve_ranges_linear
from sentinel.geolocation.simulate import (
    difference_covariance,
    simulate_fdoa,
    simulate_ranges,
    simulate_tdoa,
)
from sentinel.geolocation.systematic import SystematicErrors, inflate_fdoa, inflate_tdoa
from sentinel.geolocation.tdoa import solve_tdoa

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
