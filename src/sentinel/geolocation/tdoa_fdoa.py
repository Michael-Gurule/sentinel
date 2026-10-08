"""
TDOA/FDOA Geolocation Algorithms
Locates RF emitters using time and frequency difference measurements
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import least_squares, minimize
from scipy.constants import speed_of_light


@dataclass
class SensorPosition:
    """Position of a receiving sensor"""
    id: int
    position: np.ndarray  # [x, y, z] in meters
    velocity: Optional[np.ndarray] = None  # [vx, vy, vz] in m/s
    time_bias: float = 0.0  # Clock bias in seconds


@dataclass
class GeolocationMeasurement:
    """Measurement from sensor pair"""
    sensor_1_id: int
    sensor_2_id: int
    tdoa: Optional[float] = None  # Time difference in seconds
    fdoa: Optional[float] = None  # Frequency difference in Hz
    std_tdoa: float = 1e-6  # TDOA standard deviation (seconds)
    std_fdoa: float = 1.0   # FDOA standard deviation (Hz)


@dataclass
class GeolocationResult:
    """Result of geolocation estimation"""
    position: np.ndarray  # Estimated [x, y, z]
    velocity: Optional[np.ndarray] = None  # Estimated [vx, vy, vz]
    covariance: Optional[np.ndarray] = None  # Position covariance
    residual: float = 0.0  # RMS residual error
    gdop: float = 0.0  # Geometric Dilution of Precision
    num_measurements: int = 0
    converged: bool = False


class TDOAGeolocation:
    """
    Time Difference of Arrival (TDOA) geolocation
    Locates emitter position using time difference measurements
    """
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Args:
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.c = speed_of_light  # Speed of light
    
    def compute_tdoa(
        self,
        emitter_pos: np.ndarray,
        sensor_1: SensorPosition,
        sensor_2: SensorPosition
    ) -> float:
        """
        Compute theoretical TDOA for given emitter and sensor positions
        
        Args:
            emitter_pos: Emitter position [x, y, z]
            sensor_1: First sensor
            sensor_2: Second sensor
            
        Returns:
            TDOA in seconds (positive if closer to sensor_1)
        """
        r1 = np.linalg.norm(emitter_pos - sensor_1.position)
        r2 = np.linalg.norm(emitter_pos - sensor_2.position)
        tdoa = (r2 - r1) / self.c
        return tdoa
    
    def tdoa_residuals(
        self,
        position: np.ndarray,
        sensors: List[SensorPosition],
        measurements: List[GeolocationMeasurement]
    ) -> np.ndarray:
        """
        Compute residuals between measured and predicted TDOAs
        
        Args:
            position: Candidate emitter position
            sensors: List of sensor positions
            measurements: List of TDOA measurements
            
        Returns:
            Array of weighted residuals
        """
        residuals = []
        sensor_dict = {s.id: s for s in sensors}
        
        for meas in measurements:
            if meas.tdoa is None:
                continue
            
            sensor_1 = sensor_dict[meas.sensor_1_id]
            sensor_2 = sensor_dict[meas.sensor_2_id]
            
            predicted_tdoa = self.compute_tdoa(position, sensor_1, sensor_2)
            residual = (meas.tdoa - predicted_tdoa) / meas.std_tdoa
            residuals.append(residual)
        
        return np.array(residuals)
    
    def estimate_position(
        self,
        sensors: List[SensorPosition],
        measurements: List[GeolocationMeasurement],
        initial_guess: Optional[np.ndarray] = None
    ) -> GeolocationResult:
        """
        Estimate emitter position using TDOA measurements
        
        Args:
            sensors: List of sensor positions
            measurements: List of TDOA measurements
            initial_guess: Initial position estimate [x, y, z]
            
        Returns:
            GeolocationResult with position estimate
        """
        # Filter measurements with TDOA data
        tdoa_measurements = [m for m in measurements if m.tdoa is not None]
        
        if len(tdoa_measurements) < 3:
            return GeolocationResult(
                position=np.zeros(3),
                converged=False,
                num_measurements=len(tdoa_measurements)
            )
        
        # Initial guess: centroid of sensors if not provided
        if initial_guess is None:
            initial_guess = np.mean([s.position for s in sensors], axis=0)
        
        # Solve using least squares
        result = least_squares(
            self.tdoa_residuals,
            initial_guess,
            args=(sensors, tdoa_measurements),
            max_nfev=self.max_iterations,
            ftol=self.tolerance,
            xtol=self.tolerance
        )
        
        # Compute residual
        residuals = self.tdoa_residuals(result.x, sensors, tdoa_measurements)
        rms_residual = np.sqrt(np.mean(residuals**2))
        
        # Estimate covariance (simplified)
        if result.success and hasattr(result, 'jac'):
            try:
                # Covariance = (J^T J)^-1
                jac = result.jac
                cov = np.linalg.inv(jac.T @ jac)
            except:
                cov = None
        else:
            cov = None
        
        # Compute GDOP
        gdop = self.compute_gdop(result.x, sensors)
        
        return GeolocationResult(
            position=result.x,
            covariance=cov,
            residual=rms_residual,
            gdop=gdop,
            num_measurements=len(tdoa_measurements),
            converged=result.success
        )
    
    def compute_gdop(
        self,
        position: np.ndarray,
        sensors: List[SensorPosition]
    ) -> float:
        """
        Compute Geometric Dilution of Precision
        
        Args:
            position: Estimated position
            sensors: Sensor positions
            
        Returns:
            GDOP value (lower is better)
        """
        # Compute unit vectors to each sensor
        unit_vectors = []
        for sensor in sensors:
            diff = sensor.position - position
            dist = np.linalg.norm(diff)
            if dist > 0:
                unit_vectors.append(diff / dist)
        
        if len(unit_vectors) < 4:
            return 999.0  # Invalid GDOP
        
        # Geometry matrix
        H = np.array(unit_vectors)
        
        try:
            # GDOP = sqrt(trace((H^T H)^-1))
            Q = np.linalg.inv(H.T @ H)
            gdop = np.sqrt(np.trace(Q))
        except:
            gdop = 999.0
        
        return gdop


class FDOAGeolocation:
    """
    Frequency Difference of Arrival (FDOA) geolocation
    Locates moving emitter using Doppler frequency differences
    """
    
    def __init__(self, carrier_freq: float = 1e9, max_iterations: int = 100):
        """
        Args:
            carrier_freq: Carrier frequency in Hz
            max_iterations: Maximum optimization iterations
        """
        self.carrier_freq = carrier_freq
        self.max_iterations = max_iterations
        self.c = speed_of_light
    
    def compute_fdoa(
        self,
        emitter_pos: np.ndarray,
        emitter_vel: np.ndarray,
        sensor_1: SensorPosition,
        sensor_2: SensorPosition
    ) -> float:
        """
        Compute theoretical FDOA
        
        Args:
            emitter_pos: Emitter position [x, y, z]
            emitter_vel: Emitter velocity [vx, vy, vz]
            sensor_1: First sensor
            sensor_2: Second sensor
            
        Returns:
            FDOA in Hz
        """
        # Range vectors
        r1_vec = emitter_pos - sensor_1.position
        r2_vec = emitter_pos - sensor_2.position
        
        r1 = np.linalg.norm(r1_vec)
        r2 = np.linalg.norm(r2_vec)
        
        # Unit vectors
        u1 = r1_vec / r1
        u2 = r2_vec / r2
        
        # Relative velocities
        v_rel_1 = emitter_vel
        v_rel_2 = emitter_vel
        
        if sensor_1.velocity is not None:
            v_rel_1 = emitter_vel - sensor_1.velocity
        if sensor_2.velocity is not None:
            v_rel_2 = emitter_vel - sensor_2.velocity
        
        # Doppler shifts
        doppler_1 = -np.dot(u1, v_rel_1) / self.c * self.carrier_freq
        doppler_2 = -np.dot(u2, v_rel_2) / self.c * self.carrier_freq
        
        fdoa = doppler_2 - doppler_1
        return fdoa
    
    def fdoa_residuals(
        self,
        state: np.ndarray,
        sensors: List[SensorPosition],
        measurements: List[GeolocationMeasurement]
    ) -> np.ndarray:
        """
        Compute FDOA residuals
        
        Args:
            state: [x, y, z, vx, vy, vz]
            sensors: Sensor positions
            measurements: FDOA measurements
            
        Returns:
            Array of weighted residuals
        """
        position = state[:3]
        velocity = state[3:6]
        
        residuals = []
        sensor_dict = {s.id: s for s in sensors}
        
        for meas in measurements:
            if meas.fdoa is None:
                continue
            
            sensor_1 = sensor_dict[meas.sensor_1_id]
            sensor_2 = sensor_dict[meas.sensor_2_id]
            
            predicted_fdoa = self.compute_fdoa(position, velocity, sensor_1, sensor_2)
            residual = (meas.fdoa - predicted_fdoa) / meas.std_fdoa
            residuals.append(residual)
        
        return np.array(residuals)
    
    def estimate_position_velocity(
        self,
        sensors: List[SensorPosition],
        measurements: List[GeolocationMeasurement],
        initial_guess: Optional[np.ndarray] = None
    ) -> GeolocationResult:
        """
        Estimate emitter position and velocity using FDOA
        
        Args:
            sensors: Sensor positions
            measurements: FDOA measurements
            initial_guess: Initial state [x, y, z, vx, vy, vz]
            
        Returns:
            GeolocationResult with position and velocity
        """
        fdoa_measurements = [m for m in measurements if m.fdoa is not None]
        
        if len(fdoa_measurements) < 4:
            return GeolocationResult(
                position=np.zeros(3),
                velocity=np.zeros(3),
                converged=False,
                num_measurements=len(fdoa_measurements)
            )
        
        # Initial guess
        if initial_guess is None:
            pos_guess = np.mean([s.position for s in sensors], axis=0)
            vel_guess = np.zeros(3)
            initial_guess = np.concatenate([pos_guess, vel_guess])
        
        # Solve
        result = least_squares(
            self.fdoa_residuals,
            initial_guess,
            args=(sensors, fdoa_measurements),
            max_nfev=self.max_iterations
        )
        
        # Extract results
        position = result.x[:3]
        velocity = result.x[3:6]
        
        residuals = self.fdoa_residuals(result.x, sensors, fdoa_measurements)
        rms_residual = np.sqrt(np.mean(residuals**2))
        
        return GeolocationResult(
            position=position,
            velocity=velocity,
            residual=rms_residual,
            num_measurements=len(fdoa_measurements),
            converged=result.success
        )


class HybridTDOAFDOA:
    """
    Hybrid TDOA/FDOA geolocation
    Combines time and frequency measurements for improved accuracy
    """
    
    def __init__(
        self,
        carrier_freq: float = 1e9,
        max_iterations: int = 100,
        tdoa_weight: float = 1.0,
        fdoa_weight: float = 1.0
    ):
        """
        Args:
            carrier_freq: Carrier frequency in Hz
            max_iterations: Maximum optimization iterations
            tdoa_weight: Weight for TDOA measurements
            fdoa_weight: Weight for FDOA measurements
        """
        self.tdoa_solver = TDOAGeolocation(max_iterations=max_iterations)
        self.fdoa_solver = FDOAGeolocation(
            carrier_freq=carrier_freq,
            max_iterations=max_iterations
        )
        self.tdoa_weight = tdoa_weight
        self.fdoa_weight = fdoa_weight
    
    def hybrid_residuals(
        self,
        state: np.ndarray,
        sensors: List[SensorPosition],
        measurements: List[GeolocationMeasurement]
    ) -> np.ndarray:
        """
        Compute combined TDOA and FDOA residuals
        
        Args:
            state: [x, y, z, vx, vy, vz]
            sensors: Sensor positions
            measurements: Combined measurements
            
        Returns:
            Combined residual array
        """
        position = state[:3]
        velocity = state[3:6]
        
        residuals = []
        sensor_dict = {s.id: s for s in sensors}
        
        for meas in measurements:
            sensor_1 = sensor_dict[meas.sensor_1_id]
            sensor_2 = sensor_dict[meas.sensor_2_id]
            
            # TDOA residual
            if meas.tdoa is not None:
                predicted_tdoa = self.tdoa_solver.compute_tdoa(
                    position, sensor_1, sensor_2
                )
                residual = (meas.tdoa - predicted_tdoa) / meas.std_tdoa
                residuals.append(residual * self.tdoa_weight)
            
            # FDOA residual
            if meas.fdoa is not None:
                predicted_fdoa = self.fdoa_solver.compute_fdoa(
                    position, velocity, sensor_1, sensor_2
                )
                residual = (meas.fdoa - predicted_fdoa) / meas.std_fdoa
                residuals.append(residual * self.fdoa_weight)
        
        return np.array(residuals)
    
    def estimate(
        self,
        sensors: List[SensorPosition],
        measurements: List[GeolocationMeasurement],
        initial_guess: Optional[np.ndarray] = None
    ) -> GeolocationResult:
        """
        Estimate position and velocity using hybrid TDOA/FDOA
        
        Args:
            sensors: Sensor positions
            measurements: Combined TDOA/FDOA measurements
            initial_guess: Initial state [x, y, z, vx, vy, vz]
            
        Returns:
            GeolocationResult with position and velocity
        """
        # Count measurements
        num_tdoa = sum(1 for m in measurements if m.tdoa is not None)
        num_fdoa = sum(1 for m in measurements if m.fdoa is not None)
        
        if num_tdoa + num_fdoa < 4:
            return GeolocationResult(
                position=np.zeros(3),
                velocity=np.zeros(3),
                converged=False,
                num_measurements=num_tdoa + num_fdoa
            )
        
        # Initial guess
        if initial_guess is None:
            pos_guess = np.mean([s.position for s in sensors], axis=0)
            vel_guess = np.zeros(3)
            initial_guess = np.concatenate([pos_guess, vel_guess])
        
        # Solve
        result = least_squares(
            self.hybrid_residuals,
            initial_guess,
            args=(sensors, measurements),
            max_nfev=self.tdoa_solver.max_iterations
        )
        
        # Extract results
        position = result.x[:3]
        velocity = result.x[3:6]
        
        residuals = self.hybrid_residuals(result.x, sensors, measurements)
        rms_residual = np.sqrt(np.mean(residuals**2))
        
        # Compute GDOP using position only
        gdop = self.tdoa_solver.compute_gdop(position, sensors)
        
        return GeolocationResult(
            position=position,
            velocity=velocity,
            residual=rms_residual,
            gdop=gdop,
            num_measurements=num_tdoa + num_fdoa,
            converged=result.success
        )


def simulate_tdoa_measurements(
    emitter_pos: np.ndarray,
    sensors: List[SensorPosition],
    noise_std: float = 1e-9
) -> List[GeolocationMeasurement]:
    """
    Simulate TDOA measurements for testing
    
    Args:
        emitter_pos: True emitter position
        sensors: Sensor positions
        noise_std: TDOA noise standard deviation
        
    Returns:
        List of simulated measurements
    """
    measurements = []
    tdoa_calc = TDOAGeolocation()
    
    # Create measurements for all sensor pairs
    for i in range(len(sensors)):
        for j in range(i + 1, len(sensors)):
            true_tdoa = tdoa_calc.compute_tdoa(emitter_pos, sensors[i], sensors[j])
            noisy_tdoa = true_tdoa + np.random.randn() * noise_std
            
            measurements.append(GeolocationMeasurement(
                sensor_1_id=sensors[i].id,
                sensor_2_id=sensors[j].id,
                tdoa=noisy_tdoa,
                std_tdoa=noise_std
            ))
    
    return measurements


def simulate_fdoa_measurements(
    emitter_pos: np.ndarray,
    emitter_vel: np.ndarray,
    sensors: List[SensorPosition],
    carrier_freq: float = 1e9,
    noise_std: float = 10.0
) -> List[GeolocationMeasurement]:
    """
    Simulate FDOA measurements for testing
    
    Args:
        emitter_pos: True emitter position
        emitter_vel: True emitter velocity
        sensors: Sensor positions
        carrier_freq: Carrier frequency
        noise_std: FDOA noise standard deviation
        
    Returns:
        List of simulated measurements
    """
    measurements = []
    fdoa_calc = FDOAGeolocation(carrier_freq=carrier_freq)
    
    for i in range(len(sensors)):
        for j in range(i + 1, len(sensors)):
            true_fdoa = fdoa_calc.compute_fdoa(
                emitter_pos, emitter_vel, sensors[i], sensors[j]
            )
            noisy_fdoa = true_fdoa + np.random.randn() * noise_std
            
            measurements.append(GeolocationMeasurement(
                sensor_1_id=sensors[i].id,
                sensor_2_id=sensors[j].id,
                fdoa=noisy_fdoa,
                std_fdoa=noise_std
            ))
    
    return measurements