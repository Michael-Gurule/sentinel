"""
Test RF signal generator
"""

from sentinel.models.rf_generator import RFScenarioGenerator
from sentinel.utils.visualization import plot_rf_pulse_train, plot_spectrogram


def test_all_emitter_types():
    """Test all RF emitter types"""

    print("\n" + "=" * 60)
    print("TESTING RF SIGNAL GENERATOR")
    print("=" * 60)

    scenario_gen = RFScenarioGenerator()

    # Test 1: Early Warning Radar
    print("\n1️⃣  Early Warning Radar")
    print("-" * 40)
    radar_sig, t_radar, radar_emitter = scenario_gen.generate_early_warning_radar(
        start_time=0, duration_s=0.01, lat=40.0, lon=-100.0
    )
    print(f"   ✓ Carrier frequency: {radar_emitter.carrier_freq_hz / 1e6:.1f} MHz")
    print(f"   ✓ Power: {radar_emitter.power_dbm:.1f} dBm")
    print(f"   ✓ Type: {radar_emitter.emitter_type}")
    print(f"   ✓ Signal length: {len(radar_sig):,} samples")

    # Test 2: Fire Control Radar
    print("\n2️⃣  Fire Control Radar")
    print("-" * 40)
    fc_sig, t_fc, fc_emitter = scenario_gen.generate_fire_control_radar(
        start_time=0, duration_s=0.001, lat=40.0, lon=-100.0, target_velocity_mps=300
    )
    print(f"   ✓ Carrier frequency: {fc_emitter.carrier_freq_hz / 1e9:.2f} GHz")
    print(f"   ✓ Bandwidth: {fc_emitter.bandwidth_hz / 1e6:.1f} MHz")
    print(f"   ✓ Modulation: {fc_emitter.modulation}")
    print(f"   ✓ Signal length: {len(fc_sig):,} samples")

    # Test 3: Tactical Radio
    print("\n3️⃣  Tactical Radio (FM)")
    print("-" * 40)
    radio_sig, t_radio, radio_emitter = scenario_gen.generate_tactical_radio(
        start_time=0, duration_s=0.01, lat=40.0, lon=-100.0
    )
    print(f"   ✓ Carrier frequency: {radio_emitter.carrier_freq_hz / 1e6:.1f} MHz")
    print(f"   ✓ Modulation: {radio_emitter.modulation}")
    print(f"   ✓ Bandwidth: {radio_emitter.bandwidth_hz / 1e3:.1f} kHz")
    print(f"   ✓ Signal length: {len(radio_sig):,} samples")

    # Test 4: Satellite Uplink
    print("\n4️⃣  Satellite Uplink (QPSK)")
    print("-" * 40)
    sat_sig, t_sat, sat_emitter = scenario_gen.generate_satellite_uplink(
        start_time=0, duration_s=0.001, lat=40.0, lon=-100.0
    )
    print(f"   ✓ Carrier frequency: {sat_emitter.carrier_freq_hz / 1e9:.2f} GHz")
    print(f"   ✓ Modulation: {sat_emitter.modulation}")
    print(f"   ✓ Bandwidth: {sat_emitter.bandwidth_hz / 1e6:.1f} MHz")
    print(f"   ✓ Signal length: {len(sat_sig):,} samples")

    # Create visualizations
    print("\n📊 Creating visualizations...")

    # Pulse train
    plot_rf_pulse_train(
        t_radar,
        radar_sig,
        title="Early Warning Radar Pulse Train",
        save_path="tests/rf_test_pulse_train.png",
    )

    # Spectrogram for fire control radar
    plot_spectrogram(
        fc_sig,
        scenario_gen.radar_gen.fs,
        title="Fire Control Radar Spectrogram (Chirped)",
        save_path="tests/rf_test_spectrogram.png",
    )

    print("\n" + "=" * 60)
    print("✓ RF GENERATOR TEST COMPLETE")
    print("=" * 60)

    return {
        "radar": (radar_sig, t_radar, radar_emitter),
        "fire_control": (fc_sig, t_fc, fc_emitter),
        "radio": (radio_sig, t_radio, radio_emitter),
        "satellite": (sat_sig, t_sat, sat_emitter),
    }


if __name__ == "__main__":
    signals = test_all_emitter_types()
