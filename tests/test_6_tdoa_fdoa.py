"""
Test 6: TDOA/FDOA Geolocation
"""

import numpy as np
from sentinel.geolocation.tdoa_fdoa import (
    TDOAGeolocation,
    FDOAGeolocation,
    HybridTDOAFDOA,
    SensorPosition,
    simulate_tdoa_measurements,
    simulate_fdoa_measurements
)


def test_tdoa_geolocation():
    print("=" * 60)
    print("TEST: TDOA Geolocation")
    print("=" * 60)
    
    # Create sensor network
    print("\n1. Setting up sensor network...")
    sensors = [
        SensorPosition(id=0, position=np.array([0.0, 0.0, 0.0])),
        SensorPosition(id=1, position=np.array([10000.0, 0.0, 0.0])),
        SensorPosition(id=2, position=np.array([10000.0, 10000.0, 0.0])),
        SensorPosition(id=3, position=np.array([0.0, 10000.0, 0.0]))
    ]
    print(f"   Created {len(sensors)} sensors")
    
    # True emitter position
    true_position = np.array([5000.0, 5000.0, 1000.0])
    print(f"   True emitter position: {true_position}")
    
    # Simulate measurements
    print("\n2. Simulating TDOA measurements...")
    measurements = simulate_tdoa_measurements(
        emitter_pos=true_position,
        sensors=sensors,
        noise_std=1e-8
    )
    print(f"   Generated {len(measurements)} measurements")
    
    # Estimate position
    print("\n3. Estimating position with TDOA...")
    tdoa_solver = TDOAGeolocation()
    result = tdoa_solver.estimate_position(sensors, measurements)
    
    print(f"   Converged: {result.converged}")
    print(f"   Estimated position: {result.position}")
    print(f"   Position error: {np.linalg.norm(result.position - true_position):.2f} m")
    print(f"   GDOP: {result.gdop:.3f}")
    print(f"   Residual: {result.residual:.6f}")
    
    print("\n" + "=" * 60)
    print("TDOA test complete!")
    print("=" * 60)


def test_fdoa_geolocation():
    print("\n" + "=" * 60)
    print("TEST: FDOA Geolocation")
    print("=" * 60)
    
    # Create sensor network with velocities
    print("\n1. Setting up sensor network...")
    sensors = [
        SensorPosition(
            id=0,
            position=np.array([0.0, 0.0, 1000.0]),
            velocity=np.array([10.0, 0.0, 0.0])
        ),
        SensorPosition(
            id=1,
            position=np.array([10000.0, 0.0, 1000.0]),
            velocity=np.array([0.0, 10.0, 0.0])
        ),
        SensorPosition(
            id=2,
            position=np.array([10000.0, 10000.0, 1000.0]),
            velocity=np.array([-10.0, 0.0, 0.0])
        ),
        SensorPosition(
            id=3,
            position=np.array([0.0, 10000.0, 1000.0]),
            velocity=np.array([0.0, -10.0, 0.0])
        )
    ]
    
    # Moving emitter
    true_position = np.array([5000.0, 5000.0, 500.0])
    true_velocity = np.array([100.0, 50.0, 0.0])
    print(f"   True position: {true_position}")
    print(f"   True velocity: {true_velocity}")
    
    # Simulate FDOA measurements
    print("\n2. Simulating FDOA measurements...")
    measurements = simulate_fdoa_measurements(
        emitter_pos=true_position,
        emitter_vel=true_velocity,
        sensors=sensors,
        carrier_freq=1e9,
        noise_std=5.0
    )
    print(f"   Generated {len(measurements)} measurements")
    
    # Estimate position and velocity
    print("\n3. Estimating position and velocity with FDOA...")
    fdoa_solver = FDOAGeolocation(carrier_freq=1e9)
    result = fdoa_solver.estimate_position_velocity(sensors, measurements)
    
    print(f"   Converged: {result.converged}")
    print(f"   Estimated position: {result.position}")
    print(f"   Position error: {np.linalg.norm(result.position - true_position):.2f} m")
    print(f"   Estimated velocity: {result.velocity}")
    print(f"   Velocity error: {np.linalg.norm(result.velocity - true_velocity):.2f} m/s")
    
    print("\n" + "=" * 60)
    print("FDOA test complete!")
    print("=" * 60)


def test_hybrid_tdoa_fdoa():
    print("\n" + "=" * 60)
    print("TEST: Hybrid TDOA/FDOA Geolocation")
    print("=" * 60)
    
    # Create sensor network
    sensors = [
        SensorPosition(
            id=0,
            position=np.array([0.0, 0.0, 1000.0]),
            velocity=np.array([10.0, 0.0, 0.0])
        ),
        SensorPosition(
            id=1,
            position=np.array([10000.0, 0.0, 1000.0]),
            velocity=np.array([0.0, 10.0, 0.0])
        ),
        SensorPosition(
            id=2,
            position=np.array([10000.0, 10000.0, 1000.0]),
            velocity=np.array([-10.0, 0.0, 0.0])
        ),
        SensorPosition(
            id=3,
            position=np.array([0.0, 10000.0, 1000.0]),
            velocity=np.array([0.0, -10.0, 0.0])
        )
    ]
    
    # True state
    true_position = np.array([5000.0, 5000.0, 500.0])
    true_velocity = np.array([100.0, 50.0, 0.0])
    
    print(f"\n1. True state:")
    print(f"   Position: {true_position}")
    print(f"   Velocity: {true_velocity}")
    
    # Simulate both TDOA and FDOA measurements
    print("\n2. Simulating combined measurements...")
    tdoa_meas = simulate_tdoa_measurements(true_position, sensors, noise_std=1e-8)
    fdoa_meas = simulate_fdoa_measurements(
        true_position, true_velocity, sensors, noise_std=5.0
    )
    
    # Combine measurements
    from sentinel.geolocation.tdoa_fdoa import GeolocationMeasurement
    combined_meas = []
    for i, (tm, fm) in enumerate(zip(tdoa_meas, fdoa_meas)):
        combined_meas.append(GeolocationMeasurement(
            sensor_1_id=tm.sensor_1_id,
            sensor_2_id=tm.sensor_2_id,
            tdoa=tm.tdoa,
            fdoa=fm.fdoa,
            std_tdoa=tm.std_tdoa,
            std_fdoa=fm.std_fdoa
        ))
    
    print(f"   Combined {len(combined_meas)} measurement pairs")
    
    # Estimate with hybrid solver
    print("\n3. Estimating with hybrid solver...")
    hybrid_solver = HybridTDOAFDOA(carrier_freq=1e9)
    result = hybrid_solver.estimate(sensors, combined_meas)
    
    print(f"   Converged: {result.converged}")
    print(f"   Estimated position: {result.position}")
    print(f"   Position error: {np.linalg.norm(result.position - true_position):.2f} m")
    print(f"   Estimated velocity: {result.velocity}")
    print(f"   Velocity error: {np.linalg.norm(result.velocity - true_velocity):.2f} m/s")
    print(f"   GDOP: {result.gdop:.3f}")
    
    print("\n" + "=" * 60)
    print("Hybrid TDOA/FDOA test complete!")
    print("=" * 60)


if __name__ == '__main__':
    test_tdoa_geolocation()
    test_fdoa_geolocation()
    test_hybrid_tdoa_fdoa()