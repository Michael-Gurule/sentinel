"""
Test 4: Kalman Filter Only
"""

import numpy as np

from sentinel.tracking.kalman_tracker import KalmanFilter


def test_kalman():
    print("=" * 60)
    print("TEST: Kalman Filter")
    print("=" * 60)

    # Initialize filter
    print("\n1. Initializing Kalman filter...")
    kf = KalmanFilter(dt=1.0)
    print("   Filter initialized")

    # Initialize track
    print("\n2. Initializing track...")
    position = np.array([100.0, 200.0, 50.0])
    velocity = np.array([10.0, 5.0, 2.0])
    state, covariance = kf.initialize_track(position, velocity)
    print(f"   Initial position: {state[:3]}")
    print(f"   Initial velocity: {state[3:]}")

    # Test prediction
    print("\n3. Testing prediction...")
    state_pred, cov_pred = kf.predict(state, covariance)
    print(f"   Predicted position: {state_pred[:3]}")
    print(f"   Predicted velocity: {state_pred[3:]}")

    # Test update
    print("\n4. Testing update...")
    measurement = np.array([110.0, 205.0, 52.0])
    state_updated, _cov_updated = kf.update(state_pred, cov_pred, measurement)
    print(f"   Measurement: {measurement}")
    print(f"   Updated position: {state_updated[:3]}")
    print(f"   Updated velocity: {state_updated[3:]}")

    # Test tracking sequence
    print("\n5. Testing tracking sequence...")
    true_pos = np.array([0.0, 0.0, 0.0])
    true_vel = np.array([10.0, 5.0, 2.0])

    state, covariance = kf.initialize_track(true_pos, true_vel)

    errors = []
    for _t in range(10):
        true_pos = true_pos + true_vel
        measurement = true_pos + np.random.randn(3) * 1.0

        state_pred, cov_pred = kf.predict(state, covariance)
        state, covariance = kf.update(state_pred, cov_pred, measurement)

        error = np.linalg.norm(true_pos - state[:3])
        errors.append(error)

    mean_error = np.mean(errors)
    print(f"   Mean tracking error: {mean_error:.3f}")
    print(f"   Final position: {state[:3]}")
    print(f"   True position: {true_pos}")

    print("\n" + "=" * 60)
    print("Kalman filter test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_kalman()
