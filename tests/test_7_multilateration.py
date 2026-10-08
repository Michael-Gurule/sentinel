"""
Test 7: Multilateration Algorithms
"""

import numpy as np
from sentinel.geolocation.multilateration import (
    SphericalMultilateration,
    HyperbolicMultilateration,
    WeightedLeastSquares,
    GeometricDilution,
    RangeMeasurement,
    RangeDifferenceMeasurement,
    create_sensor_network,
    simulate_range_measurements
)


def test_spherical_multilateration():
    print("=" * 60)
    print("TEST: Spherical Multilateration")
    print("=" * 60)
    
    # Create sensor network
    print("\n1. Creating sensor network...")
    sensor_positions = create_sensor_network(num_sensors=5, area_size=10000.0)
    print(f"   Created {len(sensor_positions)} sensors")
    
    # True emitter position
    true_position = np.array([5000.0, 5000.0, 500.0])
    print(f"   True position: {true_position}")
    
    # Simulate range measurements
    print("\n2. Simulating range measurements...")
    measurements = simulate_range_measurements(
        emitter_pos=true_position,
        sensor_positions=sensor_positions,
        noise_std=10.0
    )
    print(f"   Generated {len(measurements)} measurements")
    
    # Test linear solution
    print("\n3. Testing linear solution...")
    solver = SphericalMultilateration(method='linear')
    pos_linear, success_linear = solver.solve_linear(measurements)
    
    if success_linear:
        error_linear = np.linalg.norm(pos_linear - true_position)
        print(f"   Estimated position: {pos_linear}")
        print(f"   Position error: {error_linear:.2f} m")
    else:
        print("   Linear solution failed")
    
    # Test nonlinear solution
    print("\n4. Testing nonlinear solution...")
    solver_nl = SphericalMultilateration(method='nonlinear')
    pos_nl, success_nl, residual_nl = solver_nl.solve(measurements)
    
    if success_nl:
        error_nl = np.linalg.norm(pos_nl - true_position)
        print(f"   Estimated position: {pos_nl}")
        print(f"   Position error: {error_nl:.2f} m")
        print(f"   RMS residual: {residual_nl:.3f}")
    else:
        print("   Nonlinear solution failed")
    
    print("\n" + "=" * 60)
    print("Spherical multilateration test complete!")
    print("=" * 60)


def test_hyperbolic_multilateration():
    print("\n" + "=" * 60)
    print("TEST: Hyperbolic Multilateration")
    print("=" * 60)
    
    # Create sensor network
    sensor_positions = create_sensor_network(num_sensors=4, area_size=10000.0)
    true_position = np.array([5000.0, 5000.0, 500.0])
    
    print(f"\n1. True position: {true_position}")
    print(f"   Sensors: {len(sensor_positions)}")
    
    # Create range difference measurements
    print("\n2. Creating range difference measurements...")
    measurements = []
    
    for i in range(len(sensor_positions)):
        for j in range(i + 1, len(sensor_positions)):
            r1 = np.linalg.norm(true_position - sensor_positions[i])
            r2 = np.linalg.norm(true_position - sensor_positions[j])
            rd = (r2 - r1) + np.random.randn() * 10.0  # Add noise
            
            measurements.append(RangeDifferenceMeasurement(
                sensor_1_id=i,
                sensor_2_id=j,
                sensor_1_position=sensor_positions[i],
                sensor_2_position=sensor_positions[j],
                range_difference=rd,
                std=10.0
            ))
    
    print(f"   Generated {len(measurements)} range difference measurements")
    
    # Solve with hyperbolic multilateration
    print("\n3. Solving with hyperbolic multilateration...")
    solver = HyperbolicMultilateration()
    pos, success, residual = solver.solve(measurements)
    
    if success:
        error = np.linalg.norm(pos - true_position)
        print(f"   Estimated position: {pos}")
        print(f"   Position error: {error:.2f} m")
        print(f"   RMS residual: {residual:.3f}")
    else:
        print("   Solution failed")
    
    print("\n" + "=" * 60)
    print("Hyperbolic multilateration test complete!")
    print("=" * 60)


def test_weighted_least_squares():
    print("\n" + "=" * 60)
    print("TEST: Weighted Least Squares with Covariance")
    print("=" * 60)
    
    # Create scenario
    sensor_positions = create_sensor_network(num_sensors=6, area_size=10000.0)
    true_position = np.array([5000.0, 5000.0, 500.0])
    
    print(f"\n1. True position: {true_position}")
    
    # Simulate measurements with varying uncertainties
    print("\n2. Simulating measurements with varying uncertainties...")
    measurements = []
    for i, sensor_pos in enumerate(sensor_positions):
        true_range = np.linalg.norm(true_position - sensor_pos)
        uncertainty = 5.0 + i * 2.0  # Varying uncertainty
        noisy_range = true_range + np.random.randn() * uncertainty
        
        measurements.append(RangeMeasurement(
            sensor_id=i,
            sensor_position=sensor_pos,
            range=noisy_range,
            std=uncertainty
        ))
    
    print(f"   Generated {len(measurements)} measurements")
    
    # Solve with WLS
    print("\n3. Solving with weighted least squares...")
    solver = WeightedLeastSquares()
    pos, cov, success = solver.solve_wls(measurements)
    
    if success:
        error = np.linalg.norm(pos - true_position)
        print(f"   Estimated position: {pos}")
        print(f"   Position error: {error:.2f} m")
        
        sigma_x, sigma_y, sigma_z = solver.get_position_uncertainty()
        print(f"   Position uncertainty: σx={sigma_x:.1f}, σy={sigma_y:.1f}, σz={sigma_z:.1f} m")
        
        cep = solver.get_cep()
        print(f"   CEP (50% confidence): {cep:.1f} m")
    else:
        print("   Solution failed")
    
    print("\n" + "=" * 60)
    print("Weighted least squares test complete!")
    print("=" * 60)


def test_geometric_dilution():
    print("\n" + "=" * 60)
    print("TEST: Geometric Dilution of Precision")
    print("=" * 60)
    
    # Test with good geometry
    print("\n1. Testing with good sensor geometry...")
    good_sensors = [
        np.array([0.0, 0.0, 0.0]),
        np.array([10000.0, 0.0, 0.0]),
        np.array([10000.0, 10000.0, 0.0]),
        np.array([0.0, 10000.0, 0.0]),
        np.array([5000.0, 5000.0, 5000.0])  # Vertical diversity
    ]
    
    emitter_pos = np.array([5000.0, 5000.0, 1000.0])
    
    gdop_good = GeometricDilution.compute_gdop(emitter_pos, good_sensors)
    pdop_good = GeometricDilution.compute_pdop(emitter_pos, good_sensors)
    hdop_good = GeometricDilution.compute_hdop(emitter_pos, good_sensors)
    
    print(f"   GDOP: {gdop_good:.3f}")
    print(f"   PDOP: {pdop_good:.3f}")
    print(f"   HDOP: {hdop_good:.3f}")
    
    # Test with poor geometry (all sensors in line)
    print("\n2. Testing with poor sensor geometry...")
    poor_sensors = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2500.0, 0.0, 0.0]),
        np.array([5000.0, 0.0, 0.0]),
        np.array([7500.0, 0.0, 0.0]),
        np.array([10000.0, 0.0, 0.0])
    ]
    
    gdop_poor = GeometricDilution.compute_gdop(emitter_pos, poor_sensors)
    pdop_poor = GeometricDilution.compute_pdop(emitter_pos, poor_sensors)
    hdop_poor = GeometricDilution.compute_hdop(emitter_pos, poor_sensors)
    
    print(f"   GDOP: {gdop_poor:.3f} (higher is worse)")
    print(f"   PDOP: {pdop_poor:.3f}")
    print(f"   HDOP: {hdop_poor:.3f}")
    
    print("\n   Good geometry has lower DOP values!")
    
    print("\n" + "=" * 60)
    print("Geometric dilution test complete!")
    print("=" * 60)


if __name__ == '__main__':
    test_spherical_multilateration()
    test_hyperbolic_multilateration()
    test_weighted_least_squares()
    test_geometric_dilution()