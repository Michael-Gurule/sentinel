"""
Kalman Filter for OPIR Event Tracking
Tracks event position and velocity over time with noise handling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TrackState:
    """State of a tracked object"""
    track_id: int
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    covariance: np.ndarray  # State covariance matrix
    last_update: float
    confidence: float
    event_type: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class KalmanFilter:
    """
    3D Kalman Filter for tracking objects in space
    State vector: [x, y, z, vx, vy, vz]
    """
    
    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
        dt: float = 1.0
    ):
        """
        Args:
            process_noise: Process noise magnitude
            measurement_noise: Measurement noise magnitude
            dt: Time step between updates
        """
        self.dt = dt
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])
        
        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = self._build_process_noise(process_noise)
        
        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise
        
        # Initial state covariance
        self.P_init = np.eye(6) * 10.0
    
    def _build_process_noise(self, noise: float) -> np.ndarray:
        """Build process noise covariance matrix"""
        dt = self.dt
        dt2 = dt * dt
        dt3 = dt2 * dt / 2
        dt4 = dt3 * dt / 3
        
        Q = np.array([
            [dt4, 0,   0,   dt3, 0,   0],
            [0,   dt4, 0,   0,   dt3, 0],
            [0,   0,   dt4, 0,   0,   dt3],
            [dt3, 0,   0,   dt2, 0,   0],
            [0,   dt3, 0,   0,   dt2, 0],
            [0,   0,   dt3, 0,   0,   dt2]
        ]) * noise
        
        return Q
    
    def initialize_track(
        self,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize track state
        
        Args:
            position: Initial position [x, y, z]
            velocity: Initial velocity [vx, vy, vz] (optional)
            
        Returns:
            Tuple of (state_vector, covariance_matrix)
        """
        if velocity is None:
            velocity = np.zeros(3)
        
        state = np.concatenate([position, velocity])
        covariance = self.P_init.copy()
        
        return state, covariance
    
    def predict(
        self,
        state: np.ndarray,
        covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step
        
        Args:
            state: Current state vector [6]
            covariance: Current covariance [6, 6]
            
        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        # Predict state
        state_pred = self.F @ state
        
        # Predict covariance
        P_pred = self.F @ covariance @ self.F.T + self.Q
        
        return state_pred, P_pred
    
    def update(
        self,
        state_pred: np.ndarray,
        covariance_pred: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step with new measurement
        
        Args:
            state_pred: Predicted state
            covariance_pred: Predicted covariance
            measurement: New position measurement [x, y, z]
            
        Returns:
            Tuple of (updated_state, updated_covariance)
        """
        # Innovation (measurement residual)
        y = measurement - self.H @ state_pred
        
        # Innovation covariance
        S = self.H @ covariance_pred @ self.H.T + self.R
        
        # Kalman gain
        K = covariance_pred @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        state_updated = state_pred + K @ y
        
        # Update covariance
        I = np.eye(6)
        P_updated = (I - K @ self.H) @ covariance_pred
        
        return state_updated, P_updated
    
    def compute_innovation(
        self,
        state_pred: np.ndarray,
        measurement: np.ndarray
    ) -> float:
        """
        Compute Mahalanobis distance for data association
        
        Args:
            state_pred: Predicted state
            measurement: New measurement
            
        Returns:
            Mahalanobis distance
        """
        innovation = measurement - self.H @ state_pred
        S = self.R  # Simplified for speed
        distance = np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation)
        return distance


class MultiTargetTracker:
    """
    Multi-target tracker using Kalman filters
    Manages multiple tracks with data association
    """
    
    def __init__(
        self,
        max_coast: int = 5,
        association_threshold: float = 10.0,
        min_confidence: float = 0.3
    ):
        """
        Args:
            max_coast: Maximum frames to coast without measurement
            association_threshold: Maximum distance for association
            min_confidence: Minimum confidence to maintain track
        """
        self.kalman = KalmanFilter()
        self.max_coast = max_coast
        self.association_threshold = association_threshold
        self.min_confidence = min_confidence
        
        self.tracks: List[TrackState] = []
        self.next_track_id = 0
        self.current_time = 0.0
    
    def predict_all(self):
        """Predict all tracks forward one time step"""
        for track in self.tracks:
            state = np.concatenate([track.position, track.velocity])
            state_pred, cov_pred = self.kalman.predict(state, track.covariance)
            
            track.position = state_pred[:3]
            track.velocity = state_pred[3:]
            track.covariance = cov_pred
            
            # Decrease confidence for coasting tracks
            track.confidence *= 0.9
    
    def associate_measurements(
        self,
        measurements: List[np.ndarray]
    ) -> Dict[int, int]:
        """
        Associate measurements to tracks using nearest neighbor
        
        Args:
            measurements: List of position measurements
            
        Returns:
            Dictionary mapping track_idx to measurement_idx
        """
        if not self.tracks or not measurements:
            return {}
        
        # Compute distance matrix
        n_tracks = len(self.tracks)
        n_meas = len(measurements)
        distance_matrix = np.zeros((n_tracks, n_meas))
        
        for i, track in enumerate(self.tracks):
            state = np.concatenate([track.position, track.velocity])
            for j, meas in enumerate(measurements):
                distance_matrix[i, j] = self.kalman.compute_innovation(state, meas)
        
        # Simple nearest neighbor assignment
        associations = {}
        used_measurements = set()
        
        # Sort by distance and assign
        track_meas_pairs = []
        for i in range(n_tracks):
            for j in range(n_meas):
                if distance_matrix[i, j] < self.association_threshold:
                    track_meas_pairs.append((i, j, distance_matrix[i, j]))
        
        track_meas_pairs.sort(key=lambda x: x[2])
        
        for track_idx, meas_idx, dist in track_meas_pairs:
            if track_idx not in associations and meas_idx not in used_measurements:
                associations[track_idx] = meas_idx
                used_measurements.add(meas_idx)
        
        return associations
    
    def update_tracks(
        self,
        measurements: List[np.ndarray],
        event_types: Optional[List[str]] = None
    ):
        """
        Update tracks with new measurements
        
        Args:
            measurements: List of position measurements [x, y, z]
            event_types: Optional list of event type strings
        """
        self.current_time += self.kalman.dt
        
        # Predict all tracks
        self.predict_all()
        
        # Associate measurements
        associations = self.associate_measurements(measurements)
        
        # Update associated tracks
        updated_tracks = set()
        for track_idx, meas_idx in associations.items():
            track = self.tracks[track_idx]
            measurement = measurements[meas_idx]
            
            state = np.concatenate([track.position, track.velocity])
            state_updated, cov_updated = self.kalman.update(
                state,
                track.covariance,
                measurement
            )
            
            track.position = state_updated[:3]
            track.velocity = state_updated[3:]
            track.covariance = cov_updated
            track.last_update = self.current_time
            track.confidence = min(track.confidence + 0.2, 1.0)
            
            if event_types and meas_idx < len(event_types):
                track.event_type = event_types[meas_idx]
            
            updated_tracks.add(track_idx)
        
        # Initialize new tracks for unassociated measurements
        associated_meas = set(associations.values())
        for meas_idx, measurement in enumerate(measurements):
            if meas_idx not in associated_meas:
                self._initialize_new_track(
                    measurement,
                    event_types[meas_idx] if event_types else None
                )
        
        # Remove low confidence tracks
        self._prune_tracks()
    
    def _initialize_new_track(
        self,
        position: np.ndarray,
        event_type: Optional[str] = None
    ):
        """Initialize a new track"""
        state, covariance = self.kalman.initialize_track(position)
        
        track = TrackState(
            track_id=self.next_track_id,
            position=state[:3].copy(),
            velocity=state[3:].copy(),
            covariance=covariance.copy(),
            last_update=self.current_time,
            confidence=0.5,
            event_type=event_type
        )
        
        self.tracks.append(track)
        self.next_track_id += 1
    
    def _prune_tracks(self):
        """Remove tracks that are coasting too long or have low confidence"""
        valid_tracks = []
        
        for track in self.tracks:
            coast_time = self.current_time - track.last_update
            
            if coast_time <= self.max_coast and track.confidence >= self.min_confidence:
                valid_tracks.append(track)
        
        self.tracks = valid_tracks
    
    def get_active_tracks(self) -> List[TrackState]:
        """Get all currently active tracks"""
        return self.tracks.copy()
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackState]:
        """Get track by ID"""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None