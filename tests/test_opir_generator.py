"""
Test OPIR signal generator
"""

import numpy as np

from sentinel.models.signal_generator import OPIRSignalGenerator
from sentinel.utils.visualization import plot_thermal_scenario


def test_all_event_types():
    """Test all OPIR event types"""

    print("\n" + "=" * 60)
    print("TESTING OPIR SIGNAL GENERATOR")
    print("=" * 60)

    generator = OPIRSignalGenerator(sample_rate_hz=1.0, duration_s=300)

    # Define test scenario with all event types
    events = [
        {
            "type": "launch",
            "start_time": 30,
            "lat": 40.0,
            "lon": -100.0,
            "peak_temp": 3800,
            "rise_time": 4,
        },
        {
            "type": "explosion",
            "start_time": 100,
            "lat": 40.2,
            "lon": -100.3,
            "peak_temp": 5200,
        },
        {
            "type": "fire",
            "start_time": 50,
            "lat": 39.8,
            "lon": -99.8,
            "peak_temp": 1400,
            "growth_time": 40,
            "sustain_duration": 150,
        },
        {
            "type": "aircraft",
            "start_time": 200,
            "lat": 41.0,
            "lon": -101.0,
            "peak_temp": 950,
            "transit_duration": 40,
        },
    ]

    print(f"\n📊 Generating scenario with {len(events)} events...")
    scenario, event_records = generator.generate_scenario(events)

    print(f"\n✓ Generated {len(event_records)} thermal events:")
    for i, event in enumerate(event_records, 1):
        print(
            f"   {i}. {event.event_type:12s} at t={event.timestamp:6.1f}s, "
            f"peak={event.peak_temperature:6.0f}K, duration={event.duration:5.1f}s"
        )

    # Statistics
    print("\n📈 Scenario Statistics:")
    print(f"   Duration: {generator.duration:.1f} seconds")
    print(f"   Sample rate: {generator.fs:.1f} Hz")
    print(f"   Total samples: {len(scenario)}")
    print(f"   Temperature range: {np.min(scenario):.1f}K to {np.max(scenario):.1f}K")
    print(f"   Mean temperature: {np.mean(scenario):.1f}K")

    # Plot
    print("\n📊 Creating visualization...")
    plot_thermal_scenario(
        generator.t, scenario, event_records, save_path="tests/opir_test_scenario.png"
    )

    print("\n" + "=" * 60)
    print("✓ OPIR GENERATOR TEST COMPLETE")
    print("=" * 60)

    return scenario, event_records


if __name__ == "__main__":
    scenario, events = test_all_event_types()
