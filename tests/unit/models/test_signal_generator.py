import numpy as np
import pytest

from sentinel.models.signal_generator import EVENT_TYPES, OPIRSignalGenerator


def test_time_axis_and_sizes():
    gen = OPIRSignalGenerator(duration_s=12.0, sample_rate_hz=5.0)
    assert gen.num_samples == 60
    np.testing.assert_allclose(gen.t[:3], [0.0, 0.2, 0.4])
    assert gen.sampling_rate == gen.fs == 5.0


def test_seeded_generators_reproduce_scenarios():
    events = [{"type": "launch", "start_time": 2.0}, {"type": "fire", "start_time": 1}]
    a, _ = OPIRSignalGenerator(rng=np.random.default_rng(7)).generate_scenario(events)
    b, _ = OPIRSignalGenerator(rng=np.random.default_rng(7)).generate_scenario(events)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("event_type", EVENT_TYPES)
def test_each_event_type_rises_above_background(event_type):
    gen = OPIRSignalGenerator(duration_s=100.0, sample_rate_hz=1.0)
    gen.rng = np.random.default_rng(0)
    scenario, records = gen.generate_scenario(
        [{"type": event_type, "start_time": 10.0, "lat": 1.0, "lon": 2.0}]
    )
    background, _ = gen.generate_scenario([])
    assert scenario.max() > background.max() + 300.0
    (record,) = records
    assert record.event_type == event_type
    assert (record.latitude, record.longitude) == (1.0, 2.0)


def test_background_scenario_is_near_base_temperature():
    gen = OPIRSignalGenerator(duration_s=50.0, sample_rate_hz=10.0)
    scenario, records = gen.generate_scenario([])
    assert records == []
    assert scenario.mean() == pytest.approx(280.0, abs=1.0)


def test_unknown_event_type_and_bad_arguments_raise():
    gen = OPIRSignalGenerator()
    with pytest.raises(ValueError, match="unknown event type"):
        gen.generate_scenario([{"type": "volcano", "start_time": 0.0}])
    with pytest.raises(ValueError, match="positive"):
        OPIRSignalGenerator(duration_s=0.0)
