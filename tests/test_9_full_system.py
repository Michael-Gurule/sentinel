"""
Test 9: Complete Phase 3 System Integration
"""

import numpy as np

from sentinel.pipeline.phase3_pipeline import SENTINELPhase3Pipeline


def test_full_system():
    print("=" * 70)
    print("TEST: COMPLETE SENTINEL PHASE 3 SYSTEM")
    print("=" * 70)

    # Initialize
    print("\n1. Initializing SENTINEL system...")
    pipeline = SENTINELPhase3Pipeline()
    print("   ✓ System initialized")
    print(f"   ✓ Sensors: {len(pipeline.sensors)}")

    # Test OPIR processing
    print("\n2. Testing OPIR signal processing...")
    from sentinel.models.signal_generator import OPIRSignalGenerator

    generator = OPIRSignalGenerator()
    opir_signal = generator.generate_launch_signature(start_time=2.0)

    opir_meas = pipeline.process_opir_signal(
        signal=opir_signal, sampling_rate=generator.sampling_rate, timestamp=0.0
    )

    if opir_meas:
        print("   ✓ OPIR signal detected and classified")
        print(f"      Event type: {opir_meas.event_type}")
        print(f"      Confidence: {opir_meas.confidence:.3f}")
    else:
        print("   ✗ No OPIR detection")

    # Test RF geolocation
    print("\n3. Testing RF geolocation...")
    from sentinel.geolocation.tdoa_fdoa import simulate_tdoa_measurements

    emitter_pos = np.array([5000.0, 5000.0, 500.0])
    rf_measurements = simulate_tdoa_measurements(
        emitter_pos=emitter_pos, sensors=pipeline.sensors, noise_std=1e-8
    )

    rf_meas = pipeline.process_rf_measurements(
        rf_measurements=rf_measurements, timestamp=0.0
    )

    if rf_meas:
        print("   ✓ RF emitter geolocated")
        print(f"      Position: {rf_meas.position}")
        print(f"      Confidence: {rf_meas.confidence:.3f}")
        pos_error = np.linalg.norm(rf_meas.position - emitter_pos)
        print(f"      Position error: {pos_error:.1f} m")
    else:
        print("   ✗ RF geolocation failed")

    # Test multi-sensor frame processing
    print("\n4. Testing multi-sensor frame processing...")

    opir_signals = [
        generator.generate_launch_signature(start_time=2.0) for _ in range(2)
    ]
    rf_meas_sets = [rf_measurements]

    result = pipeline.process_multi_sensor_frame(
        opir_signals=opir_signals,
        rf_measurements=rf_meas_sets,
        sampling_rate=generator.sampling_rate,
        timestamp=0.0,
    )

    print("   ✓ Frame processed")
    print(f"      OPIR detections: {result['opir_detections']}")
    print(f"      RF geolocations: {result['rf_geolocations']}")
    print(f"      Fused tracks: {result['fused_tracks']}")

    # Test situation awareness
    print("\n5. Testing situation awareness...")
    sa = pipeline.get_situation_awareness()

    print("   Situation Awareness:")
    for key, value in sa.items():
        print(f"      {key}: {value}")

    print("\n" + "=" * 70)
    print("COMPLETE SYSTEM TEST FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    test_full_system()
