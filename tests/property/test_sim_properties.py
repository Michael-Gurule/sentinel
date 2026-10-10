"""Property-based tests for simulator invariants."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from locant.sim.geometry import ecef_to_lla, lla_to_ecef
from locant.sim.opir.sensor import ensquared_energy


@given(
    st.floats(-89.0, 89.0),
    st.floats(-179.0, 179.0),
    st.floats(-400.0, 4e7),
)
def test_lla_round_trip(lat_deg, lon_deg, alt):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    back_lat, back_lon, back_alt = ecef_to_lla(lla_to_ecef(lat, lon, alt))
    assert abs(back_lat - lat) < 1e-10
    assert abs(back_lon - lon) < 1e-10
    assert abs(back_alt - alt) < 1e-2


@given(st.floats(-0.5, 0.5), st.floats(-0.5, 0.5), st.floats(0.05, 2.0))
def test_ensquared_energy_is_a_fraction_and_peaks_at_center(u, v, sigma):
    value = float(ensquared_energy(np.array([u, v]), sigma))
    assert 0.0 <= value <= 1.0
    assert value <= float(ensquared_energy(np.zeros(2), sigma)) + 1e-12
