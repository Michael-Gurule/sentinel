from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from sentinel.sim import load_scenario, simulate_scenario
from sentinel.sim.geometry import KeplerianPlatform
from sentinel.sim.scenario import PlatformConfig, ScenarioConfig

EXAMPLE = (
    Path(__file__).resolve().parents[3] / "configs/scenario/launch_with_radar.yaml"
)


@pytest.fixture(scope="module")
def config() -> ScenarioConfig:
    return load_scenario(EXAMPLE)


def test_example_scenario_simulates_all_components(config):
    result = simulate_scenario(config, seed=3)
    assert set(result.opir) == {"launch-1", "aircraft-1", "fire-1"}
    assert result.times.size == 1_200
    for event_id, obs in result.opir.items():
        assert obs.measured.shape == result.times.shape
        np.testing.assert_allclose(obs.target_enu, result.truth[event_id].position)
    radar = [scan for emitter, scan in result.rf_scans if emitter == "radar-1"]
    datalink = [scan for emitter, scan in result.rf_scans if emitter == "datalink-1"]
    assert [s.t for s in radar] == [
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        22.0,
        24.0,
    ]
    assert all(s.fdoa is None for s in radar)
    assert len(datalink) == 120
    assert all(s.fdoa is not None for s in datalink)


def test_same_seed_same_result_different_seed_different_result(config):
    a, b, c = (simulate_scenario(config, seed) for seed in (5, 5, 6))
    for key in a.opir:
        np.testing.assert_array_equal(a.opir[key].measured, b.opir[key].measured)
        assert not np.array_equal(a.opir[key].measured, c.opir[key].measured)
    np.testing.assert_array_equal(
        a.rf_scans[0][1].tdoa.values, b.rf_scans[0][1].tdoa.values
    )


def test_launch_rises_while_aircraft_cruises(config):
    result = simulate_scenario(config, seed=1)
    launch = result.truth["launch-1"]
    assert launch.position[0, 2] == 0.0  # on the pad before onset
    assert launch.position[-1, 2] > 20_000.0
    aircraft = result.truth["aircraft-1"]
    np.testing.assert_allclose(aircraft.position[:, 2], 9_000.0)


def test_molniya_platform_config_builds():
    platform = PlatformConfig(kind="molniya", raan_deg=30.0).build()
    assert isinstance(platform, KeplerianPlatform)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"events": [{"kind": "fire", "id": "a"}, {"kind": "fire", "id": "a"}]},
            "unique",
        ),
        ({"bogus": 1}, "Extra inputs"),
        ({"duration_s": -1.0}, "greater than 0"),
        (
            {
                "rf": {
                    "receivers": [
                        {"id": 0, "position_m": [0, 0, 0]},
                        {"id": 1, "position_m": [1, 0, 0]},
                    ],
                    "emitters": [
                        {"id": "e", "attached_to": "nope", "profile": "datalink"}
                    ],
                }
            },
            "unknown event",
        ),
        (
            {
                "events": [{"kind": "fire", "id": "f"}],
                "rf": {
                    "receivers": [
                        {"id": 0, "position_m": [0, 0, 0]},
                        {"id": 1, "position_m": [1, 0, 0]},
                    ],
                    "emitters": [{"id": "e", "attached_to": "f", "profile": "jammer"}],
                },
            },
            "unknown emitter profile",
        ),
    ],
)
def test_config_validation(change, message):
    base = {"name": "x", "origin": {"lat_deg": 0, "lon_deg": 0}, "duration_s": 10.0}
    with pytest.raises(ValidationError, match=message):
        ScenarioConfig.model_validate(base | change)
