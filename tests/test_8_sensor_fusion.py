"""
Test 8: Sensor Fusion Engine
"""

import numpy as np

from sentinel.fusion.sensor_fusion import (
    MeasurementFusion,
    SensorFusionEngine,
    SensorMeasurement,
    SensorType,
)


def test_data_association():
    print("=" * 60)
    print("TEST: Data Association")
    print("=" * 60)

    # Create fusion engine
    print("\n1. Creating fusion engine...")
    fusion = SensorFusionEngine()

    # Initialize tracks
    print("\n2. Initializing tracks...")
    measurements = [
        SensorMeasurement(
            sensor_type=SensorType.RF,
            timestamp=0.0,
            position=np.array([1000.0, 2000.0, 500.0]),
            confidence=0.8,
        ),
        SensorMeasurement(
            sensor_type=SensorType.OPIR,
            timestamp=0.0,
            position=np.array([5000.0, 6000.0, 800.0]),
            confidence=0.7,
        ),
    ]

    fusion.process_measurements(measurements, timestamp=0.0)
    print(f"   Created {len(fusion.get_tracks())} initial tracks")

    # New measurements close to existing tracks
    print("\n3. Testing association...")
    new_measurements = [
        SensorMeasurement(
            sensor_type=SensorType.RF,
            timestamp=1.0,
            position=np.array([1050.0, 2050.0, 505.0]),  # Close to track 0
            confidence=0.8,
        ),
        SensorMeasurement(
            sensor_type=SensorType.OPIR,
            timestamp=1.0,
            position=np.array([5100.0, 6100.0, 810.0]),  # Close to track 1
            confidence=0.7,
        ),
        SensorMeasurement(
            sensor_type=SensorType.RF,
            timestamp=1.0,
            position=np.array([10000.0, 10000.0, 1000.0]),  # New track
            confidence=0.9,
        ),
    ]

    fusion.process_measurements(new_measurements, timestamp=1.0)
    tracks = fusion.get_tracks()

    print(f"   Total tracks: {len(tracks)}")
    print("   Track details:")
    for track in tracks:
        print(
            f"      Track {track.track_id}: pos={track.position}, conf={track.confidence:.3f}"
        )

    print("\n" + "=" * 60)
    print("Data association test complete!")
    print("=" * 60)


def test_measurement_fusion():
    print("\n" + "=" * 60)
    print("TEST: Measurement Fusion")
    print("=" * 60)

    # Create measurements from different sensors
    print("\n1. Creating multi-sensor measurements...")
    measurements = [
        SensorMeasurement(
            sensor_type=SensorType.OPIR,
            timestamp=0.0,
            position=np.array([5000.0, 5000.0, 500.0]),
            covariance=np.eye(6) * 100,
            confidence=0.7,
        ),
        SensorMeasurement(
            sensor_type=SensorType.RF,
            timestamp=0.0,
            position=np.array([5050.0, 5050.0, 510.0]),
            covariance=np.eye(6) * 50,
            confidence=0.9,
        ),
    ]

    print(f"   Created {len(measurements)} measurements")

    # Fuse measurements
    print("\n2. Fusing measurements...")
    fusion = MeasurementFusion()
    fused_pos, fused_cov = fusion.fuse_positions(measurements)

    print(f"   OPIR position: {measurements[0].position}")
    print(f"   RF position: {measurements[1].position}")
    print(f"   Fused position: {fused_pos}")
    print(f"   Fused uncertainty: σ={np.sqrt(np.trace(fused_cov)):.1f} m")

    # Compute fused confidence
    confidence = fusion.compute_fusion_confidence(measurements)
    print(f"   Fused confidence: {confidence:.3f}")

    print("\n" + "=" * 60)
    print("Measurement fusion test complete!")
    print("=" * 60)


def test_multi_sensor_tracking():
    print("\n" + "=" * 60)
    print("TEST: Multi-Sensor Tracking")
    print("=" * 60)

    # Create fusion engine
    fusion = SensorFusionEngine()

    # Simulate moving emitter with both OPIR and RF detections
    print("\n1. Simulating multi-sensor tracking scenario...")
    true_position = np.array([1000.0, 2000.0, 500.0])
    true_velocity = np.array([100.0, 50.0, 0.0])

    num_steps = 10

    for step in range(num_steps):
        timestamp = step * 1.0
        true_position = true_position + true_velocity * 1.0

        # Create measurements (alternating OPIR and RF)
        measurements = []

        if step % 2 == 0:
            # OPIR detection
            opir_pos = true_position + np.random.randn(3) * 50
            measurements.append(
                SensorMeasurement(
                    sensor_type=SensorType.OPIR,
                    timestamp=timestamp,
                    position=opir_pos,
                    confidence=0.7,
                    event_type="launch",
                )
            )

        if step % 3 == 0:
            # RF geolocation
            rf_pos = true_position + np.random.randn(3) * 30
            measurements.append(
                SensorMeasurement(
                    sensor_type=SensorType.RF,
                    timestamp=timestamp,
                    position=rf_pos,
                    velocity=true_velocity + np.random.randn(3) * 10,
                    confidence=0.9,
                )
            )

        fusion.process_measurements(measurements, timestamp)

        if step % 3 == 0:
            tracks = fusion.get_tracks()
            if tracks:
                track = tracks[0]
                pos_error = np.linalg.norm(track.position - true_position)
                print(f"\n   Step {step}: t={timestamp:.1f}s")
                print(f"      Position error: {pos_error:.1f} m")
                print(f"      Confidence: {track.confidence:.3f}")
                print(f"      Quality: {track.track_quality:.3f}")
                print(f"      OPIR detections: {track.opir_detections}")
                print(f"      RF detections: {track.rf_detections}")

    # Final tracks
    print("\n2. Final tracking results:")
    tracks = fusion.get_tracks()
    print(f"   Active tracks: {len(tracks)}")

    if tracks:
        track = tracks[0]
        pos_error = np.linalg.norm(track.position - true_position)
        vel_error = np.linalg.norm(track.velocity - true_velocity)

        print(f"\n   Track {track.track_id}:")
        print(f"      Position error: {pos_error:.1f} m")
        print(f"      Velocity error: {vel_error:.1f} m/s")
        print(f"      Track quality: {track.track_quality:.3f}")
        print(f"      Event type: {track.event_type}")

    print("\n" + "=" * 60)
    print("Multi-sensor tracking test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_data_association()
    test_measurement_fusion()
    test_multi_sensor_tracking()
