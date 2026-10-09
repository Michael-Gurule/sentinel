import numpy as np

from locant.sim import load_scenario, simulate_scenario
from locant.sim.opir.reports import clutter_reports, event_reports
from locant.sim.scenario import PlatformConfig, SensorConfig


def test_stereo_scenario_reports(rng):
    config = load_scenario("configs/scenario/launch_with_radar.yaml")
    config = config.model_copy(
        update={
            "extra_sensors": [
                SensorConfig(platform=PlatformConfig(kind="molniya", raan_deg=-160))
            ]
        }
    )
    result = simulate_scenario(config, seed=1)
    assert len(result.opir_by_sensor) == 2
    single = simulate_scenario(
        load_scenario("configs/scenario/launch_with_radar.yaml"), seed=1
    )
    np.testing.assert_array_equal(
        result.opir["fire-1"].measured, single.opir["fire-1"].measured
    )

    reports = event_reports(
        config.origin.frame(),
        result.opir_by_sensor,
        result.times,
        np.array([60.0]),
        10e-6,
    )
    assert {(r.sensor_index, r.event_id) for r in reports} >= {
        (0, "launch-1"),
        (1, "launch-1"),
    }
    for r in reports:
        assert r.snr >= 5.0
        np.testing.assert_allclose(np.linalg.norm(r.line_of_sight), 1.0)
        assert r.line_of_sight[2] < 0  # looking down from orbit
    strict = event_reports(
        config.origin.frame(),
        result.opir_by_sensor,
        result.times,
        np.array([60.0]),
        10e-6,
        1e9,
    )
    assert strict == []


def test_clutter_rate(rng):
    sensors = [np.array([0.0, -3e7, 2.5e7])]
    reports = clutter_reports(sensors, np.arange(400.0), 2.0, 50_000.0, 10e-6, rng)
    assert abs(len(reports) / 400 - 2.0) < 0.2
    assert all(r.event_id is None for r in reports)
