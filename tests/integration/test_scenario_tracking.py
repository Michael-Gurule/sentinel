"""Phase 1 + Phase 2 end to end: simulated RF network → geolocation → fusion."""

from pathlib import Path

import numpy as np

from sentinel.fusion import FusionEngine, rf_measurement
from sentinel.geolocation import solve_tdoa_fdoa
from sentinel.sim import load_scenario, simulate_scenario

EXAMPLE = (
    Path(__file__).resolve().parents[2] / "configs/scenario/launch_with_radar.yaml"
)


def test_aircraft_datalink_is_tracked_through_systematic_errors():
    config = load_scenario(EXAMPLE)
    result = simulate_scenario(config, seed=2026)
    engine = FusionEngine()
    for emitter, scan in result.rf_scans:
        if emitter != "datalink-1":
            continue
        fix = solve_tdoa_fdoa(scan.receivers, scan.tdoa, scan.fdoa)
        engine.process([rf_measurement(fix)], scan.t)

    (track,) = engine.tracks
    aircraft = next(e for e in config.events if e.id == "aircraft-1")
    t_end = engine.time
    assert t_end is not None
    truth = np.array(aircraft.position_m) + np.array(aircraft.velocity_mps) * (
        t_end - aircraft.onset_s
    )
    # Clock biases (5 ns) and survey errors (2 m) are unmodeled by the solver,
    # so the error is dominated by bias, not by the reported covariance.
    assert np.linalg.norm(track.position - truth) < 150.0
    np.testing.assert_allclose(track.velocity, aircraft.velocity_mps, atol=15.0)
    assert track.hits_by_source == {"rf": 120}
