"""
Multilateration Algorithms
Solves for emitter position using range/range-difference measurements
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import least_squares, minimize
from scipy.linalg import svd


@dataclass
class RangeMeasurement:
    """Range measurement from a sensor"""
    sensor_id: int
    sensor_position: np.ndarray  # [x, y, z]
    range: float  # Distance in meters
    std: float = 10.0  # Standard deviation in meters


@dataclass
class RangeDifferenceMeasurement:
    """Range difference measurement between sensor pair"""
    sensor_1_id: int
    sensor_2_id: int
    sensor_1_position: np.ndarray
    sensor_2_position: np.ndarray
    range_difference: float  # r2 - r1 in meters
    std: float = 10.0


class SphericalMultilateration:
    """
    Spherical multilateration using range measurements
    Solves for position given ranges to multiple sensors
    """
    
    def __init__(self, method: str = 'nonlinear'):
        """
        Args:
            method: Solution method ('linear', 'nonlinear', or 'hybrid')
        """
        self.method = method
    
    def solve_linear(
        self,
        measurements: List[RangeMeasurement]
    ) -> Tuple[np.ndarray, bool]:
        """
        Linear least squares solution (closed-form)
        
        Args:
            measurements: List of range measurements
            
        Returns:
            Tuple of (position, success)
        """
        if len(measurements) < 4:
            return np.zeros(3), False
        
        # Use first sensor as reference
        ref = measurements[0]
        x0, y0, z0 = ref.sensor_position
        r0 = ref.range
        
        # Build linear system Ax = b
        A = []
        b = []
        
        for i in range(1, len(measurements)):
            xi, yi, zi = measurements[i].sensor_position
            ri = measurements[i].range
            
            # Linearize around reference sensor
            A.append([2*(xi - x0), 2*(yi - y0), 2*(zi - z0)])
            
            b_val = (xi**2 - x0**2 + yi**2 - y0**2 + zi**2 - z0**2 
                    + r0**2 - ri**2)
            b.append(b_val)
        
        A = np.array(A)
        b = np.array(b)
        
        # Solve using least squares
        try:
            position, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            success = True
        except:
            position = np.zeros(3)
            success = False
        
        return position, success
    
    def solve_nonlinear(
        self,
        measurements: List[RangeMeasurement],
        initial_guess: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Nonlinear least squares solution
        
        Args:
            measurements: List of range measurements
            initial_guess: Initial position estimate
            
        Returns:
            Tuple of (position, success, residual)
        """
        if len(measurements) < 3:
            return np.zeros(3), False, 0.0
        
        # Initial guess: centroid of sensors
        if initial_guess is None:
            initial_guess = np.mean(
                [m.sensor_position for m in measurements], axis=0
            )
        
        def residuals(pos):
            res = []
            for meas in measurements:
                predicted_range = np.linalg.norm(pos - meas.sensor_position)
                residual = (meas.range - predicted_range) / meas.std
                res.append(residual)
            return np.array(res)
        
        result = least_squares(residuals, initial_guess, max_nfev=100)
        
        rms_residual = np.sqrt(np.mean(residuals(result.x)**2))
        
        return result.x, result.success, rms_residual
    
    def solve(
        self,
        measurements: List[RangeMeasurement],
        initial_guess: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Solve multilateration problem
        
        Args:
            measurements: Range measurements
            initial_guess: Initial position guess
            
        Returns:
            Tuple of (position, success, residual)
        """
        if self.method == 'linear':
            pos, success = self.solve_linear(measurements)
            residual = 0.0
        elif self.method == 'nonlinear':
            pos, success, residual = self.solve_nonlinear(measurements, initial_guess)
        else:  # hybrid
            # Try linear first, refine with nonlinear
            pos_linear, success_linear = self.solve_linear(measurements)
            if success_linear:
                pos, success, residual = self.solve_nonlinear(
                    measurements, initial_guess=pos_linear
                )
            else:
                pos, success, residual = self.solve_nonlinear(measurements, initial_guess)
        
        return pos, success, residual


class HyperbolicMultilateration:
    """
    Hyperbolic multilateration using range differences
    Solves TDOA-style problems where only range differences are known
    """
    
    def __init__(self, max_iterations: int = 100):
        """
        Args:
            max_iterations: Maximum optimization iterations
        """
        self.max_iterations = max_iterations
    
    def solve_chan(
        self,
        measurements: List[RangeDifferenceMeasurement]
    ) -> Tuple[np.ndarray, bool]:
        """
        Chan's algorithm for hyperbolic positioning (closed-form)
        
        Args:
            measurements: Range difference measurements
            
        Returns:
            Tuple of (position, success)
        """
        if len(measurements) < 3:
            return np.zeros(3), False
        
        # Use first measurement's first sensor as reference
        ref_pos = measurements[0].sensor_1_position
        x0, y0, z0 = ref_pos
        
        # Build system
        A = []
        b = []
        
        for meas in measurements:
            x1, y1, z1 = meas.sensor_1_position
            x2, y2, z2 = meas.sensor_2_position
            rd = meas.range_difference
            
            # Build row of A matrix
            a_row = [
                2*(x2 - x1),
                2*(y2 - y1),
                2*(z2 - z1)
            ]
            A.append(a_row)
            
            # Build b vector
            K1 = x1**2 + y1**2 + z1**2
            K2 = x2**2 + y2**2 + z2**2
            b_val = K2 - K1 - rd**2 + 2*rd*np.linalg.norm(meas.sensor_1_position - ref_pos)
            b.append(b_val)
        
        A = np.array(A)
        b = np.array(b)
        
        try:
            position, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            success = True
        except:
            position = np.zeros(3)
            success = False
        
        return position, success
    
    def solve_nonlinear(
        self,
        measurements: List[RangeDifferenceMeasurement],
        initial_guess: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Nonlinear optimization for hyperbolic positioning
        
        Args:
            measurements: Range difference measurements
            initial_guess: Initial position estimate
            
        Returns:
            Tuple of (position, success, residual)
        """
        if len(measurements) < 3:
            return np.zeros(3), False, 0.0
        
        # Initial guess: average of sensor positions
        if initial_guess is None:
            all_positions = []
            for meas in measurements:
                all_positions.append(meas.sensor_1_position)
                all_positions.append(meas.sensor_2_position)
            initial_guess = np.mean(all_positions, axis=0)
        
        def residuals(pos):
            res = []
            for meas in measurements:
                r1 = np.linalg.norm(pos - meas.sensor_1_position)
                r2 = np.linalg.norm(pos - meas.sensor_2_position)
                predicted_rd = r2 - r1
                residual = (meas.range_difference - predicted_rd) / meas.std
                res.append(residual)
            return np.array(res)
        
        result = least_squares(
            residuals,
            initial_guess,
            max_nfev=self.max_iterations
        )
        
        rms_residual = np.sqrt(np.mean(residuals(result.x)**2))
        
        return result.x, result.success, rms_residual
    
    def solve(
        self,
        measurements: List[RangeDifferenceMeasurement],
        initial_guess: Optional[np.ndarray] = None,
        method: str = 'nonlinear'
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Solve hyperbolic multilateration
        
        Args:
            measurements: Range difference measurements
            initial_guess: Initial position guess
            method: 'chan' for closed-form, 'nonlinear' for optimization
            
        Returns:
            Tuple of (position, success, residual)
        """
        if method == 'chan':
            pos, success = self.solve_chan(measurements)
            residual = 0.0
        else:
            pos, success, residual = self.solve_nonlinear(measurements, initial_guess)
        
        return pos, success, residual


class WeightedLeastSquares:
    """
    Weighted least squares multilateration with covariance support
    """
    
    def __init__(self):
        self.position = None
        self.covariance = None
    
    def solve_wls(
        self,
        measurements: List[RangeMeasurement],
        initial_guess: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Weighted least squares with covariance estimation
        
        Args:
            measurements: Range measurements with uncertainties
            initial_guess: Initial position
            
        Returns:
            Tuple of (position, covariance, success)
        """
        if len(measurements) < 3:
            return np.zeros(3), np.eye(3) * 1e6, False
        
        if initial_guess is None:
            initial_guess = np.mean(
                [m.sensor_position for m in measurements], axis=0
            )
        
        # Build weight matrix
        W = np.diag([1.0 / m.std**2 for m in measurements])
        
        def residuals(pos):
            res = []
            for meas in measurements:
                predicted_range = np.linalg.norm(pos - meas.sensor_position)
                res.append(meas.range - predicted_range)
            return np.array(res)
        
        def jacobian(pos):
            J = []
            for meas in measurements:
                diff = pos - meas.sensor_position
                dist = np.linalg.norm(diff)
                if dist > 0:
                    J.append(-diff / dist)
                else:
                    J.append(np.zeros(3))
            return np.array(J)
        
        # Iterative weighted least squares
        pos = initial_guess.copy()
        max_iter = 20
        
        for iteration in range(max_iter):
            r = residuals(pos)
            J = jacobian(pos)
            
            # Weighted normal equations
            try:
                N = J.T @ W @ J
                delta = np.linalg.solve(N, J.T @ W @ r)
                pos = pos + delta
                
                if np.linalg.norm(delta) < 1e-6:
                    break
            except:
                return pos, np.eye(3) * 1e6, False
        
        # Estimate covariance
        try:
            J_final = jacobian(pos)
            N_final = J_final.T @ W @ J_final
            covariance = np.linalg.inv(N_final)
        except:
            covariance = np.eye(3) * 1e6
        
        self.position = pos
        self.covariance = covariance
        
        return pos, covariance, True
    
    def get_position_uncertainty(self) -> Tuple[float, float, float]:
        """
        Get position uncertainty (standard deviations)
        
        Returns:
            Tuple of (sigma_x, sigma_y, sigma_z)
        """
        if self.covariance is None:
            return 1e3, 1e3, 1e3
        
        sigma_x = np.sqrt(self.covariance[0, 0])
        sigma_y = np.sqrt(self.covariance[1, 1])
        sigma_z = np.sqrt(self.covariance[2, 2])
        
        return sigma_x, sigma_y, sigma_z
    
    def get_cep(self) -> float:
        """
        Get Circular Error Probable (CEP) - 50% confidence radius
        
        Returns:
            CEP in meters
        """
        if self.covariance is None:
            return 1e3
        
        # 2D CEP approximation
        sigma_x = np.sqrt(self.covariance[0, 0])
        sigma_y = np.sqrt(self.covariance[1, 1])
        
        # Approximate CEP for circular normal distribution
        cep = 0.5887 * (sigma_x + sigma_y)
        
        return cep


class GeometricDilution:
    """
    Compute geometric dilution of precision (GDOP) and related metrics
    """
    
    @staticmethod
    def compute_gdop(
        emitter_pos: np.ndarray,
        sensor_positions: List[np.ndarray]
    ) -> float:
        """
        Compute GDOP
        
        Args:
            emitter_pos: Estimated emitter position
            sensor_positions: List of sensor positions
            
        Returns:
            GDOP value (lower is better)
        """
        if len(sensor_positions) < 4:
            return 999.0
        
        # Compute geometry matrix
        H = []
        for sensor_pos in sensor_positions:
            diff = emitter_pos - sensor_pos
            dist = np.linalg.norm(diff)
            if dist > 0:
                unit_vec = diff / dist
                H.append(list(unit_vec) + [1])  # Add time component
        
        H = np.array(H)
        
        try:
            Q = np.linalg.inv(H.T @ H)
            gdop = np.sqrt(np.trace(Q))
        except:
            gdop = 999.0
        
        return gdop
    
    @staticmethod
    def compute_pdop(
        emitter_pos: np.ndarray,
        sensor_positions: List[np.ndarray]
    ) -> float:
        """
        Compute Position Dilution of Precision (PDOP)
        
        Args:
            emitter_pos: Estimated position
            sensor_positions: Sensor positions
            
        Returns:
            PDOP value
        """
        if len(sensor_positions) < 4:
            return 999.0
        
        H = []
        for sensor_pos in sensor_positions:
            diff = emitter_pos - sensor_pos
            dist = np.linalg.norm(diff)
            if dist > 0:
                unit_vec = diff / dist
                H.append(list(unit_vec) + [1])
        
        H = np.array(H)
        
        try:
            Q = np.linalg.inv(H.T @ H)
            pdop = np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])
        except:
            pdop = 999.0
        
        return pdop
    
    @staticmethod
    def compute_hdop(
        emitter_pos: np.ndarray,
        sensor_positions: List[np.ndarray]
    ) -> float:
        """
        Compute Horizontal Dilution of Precision (HDOP)
        
        Args:
            emitter_pos: Estimated position
            sensor_positions: Sensor positions
            
        Returns:
            HDOP value
        """
        if len(sensor_positions) < 3:
            return 999.0
        
        H = []
        for sensor_pos in sensor_positions:
            diff = emitter_pos - sensor_pos
            dist = np.linalg.norm(diff)
            if dist > 0:
                unit_vec = diff / dist
                H.append(list(unit_vec) + [1])
        
        H = np.array(H)
        
        try:
            Q = np.linalg.inv(H.T @ H)
            hdop = np.sqrt(Q[0, 0] + Q[1, 1])
        except:
            hdop = 999.0
        
        return hdop


def create_sensor_network(
    num_sensors: int = 4,
    area_size: float = 10000.0,
    altitude_range: Tuple[float, float] = (0.0, 1000.0)
) -> List[np.ndarray]:
    """
    Create a sensor network for testing
    
    Args:
        num_sensors: Number of sensors
        area_size: Size of deployment area (meters)
        altitude_range: Range of sensor altitudes
        
    Returns:
        List of sensor positions
    """
    sensors = []
    
    for i in range(num_sensors):
        x = np.random.uniform(-area_size/2, area_size/2)
        y = np.random.uniform(-area_size/2, area_size/2)
        z = np.random.uniform(altitude_range[0], altitude_range[1])
        sensors.append(np.array([x, y, z]))
    
    return sensors


def simulate_range_measurements(
    emitter_pos: np.ndarray,
    sensor_positions: List[np.ndarray],
    noise_std: float = 10.0
) -> List[RangeMeasurement]:
    """
    Simulate range measurements for testing
    
    Args:
        emitter_pos: True emitter position
        sensor_positions: Sensor positions
        noise_std: Measurement noise standard deviation
        
    Returns:
        List of noisy range measurements
    """
    measurements = []
    
    for i, sensor_pos in enumerate(sensor_positions):
        true_range = np.linalg.norm(emitter_pos - sensor_pos)
        noisy_range = true_range + np.random.randn() * noise_std
        
        measurements.append(RangeMeasurement(
            sensor_id=i,
            sensor_position=sensor_pos,
            range=noisy_range,
            std=noise_std
        ))
    
    return measurements