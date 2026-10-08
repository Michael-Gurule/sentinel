"""
Multi-Sensor Fusion Engine
Combines OPIR thermal detection with RF geolocation for enhanced tracking
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..tracking.kalman_tracker import KalmanFilter, TrackState
from ..detection.opir_detectors import DetectionResult
from ..models.cnn_classifier import ClassificationResult
from ..geolocation.tdoa_fdoa import GeolocationResult


class SensorType(Enum):
    """Types of sensors in the system"""
    OPIR = "opir"
    RF = "rf"
    FUSED = "fused"


@dataclass
class SensorMeasurement:
    """Generic sensor measurement"""
    sensor_type: SensorType
    timestamp: float
    position: Optional[np.ndarray] = None  # [x, y, z]
    velocity: Optional[np.ndarray] = None  # [vx, vy, vz]
    covariance: Optional[np.ndarray] = None
    confidence: float = 0.5
    event_type: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class FusedTrack:
    """Fused track combining multiple sensor modalities"""
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    last_update: float
    confidence: float
    event_type: Optional[str] = None
    
    # Track which sensors contributed
    opir_detections: int = 0
    rf_detections: int = 0
    last_opir_update: float = 0.0
    last_rf_update: float = 0.0
    
    # Quality metrics
    position_uncertainty: float = 0.0
    velocity_uncertainty: float = 0.0
    track_quality: float = 0.0
    
    metadata: Dict = field(default_factory=dict)


class DataAssociation:
    """
    Associates measurements from different sensors to existing tracks
    Uses gating and nearest neighbor assignment
    """
    
    def __init__(
        self,
        gate_threshold: float = 10.0,  # Chi-square gate (Mahalanobis distance)
        association_method: str = 'nearest_neighbor'
    ):
        """
        Args:
            gate_threshold: Maximum distance for association
            association_method: 'nearest_neighbor' or 'global_nearest'
        """
        self.gate_threshold = gate_threshold
        self.association_method = association_method
    
    def mahalanobis_distance(
        self,
        measurement_pos: np.ndarray,
        track_pos: np.ndarray,
        covariance: np.ndarray
    ) -> float:
        """
        Compute Mahalanobis distance between measurement and track
        
        Args:
            measurement_pos: Measurement position
            track_pos: Track position
            covariance: Position covariance matrix
            
        Returns:
            Mahalanobis distance
        """
        diff = measurement_pos - track_pos
        
        try:
            inv_cov = np.linalg.inv(covariance[:3, :3])  # Position only
            distance = np.sqrt(diff.T @ inv_cov @ diff)
        except:
            # Fallback to Euclidean if covariance is singular
            distance = np.linalg.norm(diff)
        
        return distance
    
    def gate_measurement(
        self,
        measurement: SensorMeasurement,
        track: FusedTrack
    ) -> bool:
        """
        Check if measurement is within association gate
        
        Args:
            measurement: Sensor measurement
            track: Existing track
            
        Returns:
            True if measurement is within gate
        """
        if measurement.position is None:
            return False
        
        distance = self.mahalanobis_distance(
            measurement.position,
            track.position,
            track.covariance
        )
        
        return distance <= self.gate_threshold
    
    def associate_measurements(
        self,
        measurements: List[SensorMeasurement],
        tracks: List[FusedTrack]
    ) -> Dict[int, int]:
        """
        Associate measurements to tracks
        
        Args:
            measurements: List of sensor measurements
            tracks: List of existing tracks
            
        Returns:
            Dictionary mapping track_idx -> measurement_idx
        """
        if not tracks or not measurements:
            return {}
        
        # Build distance matrix
        n_tracks = len(tracks)
        n_meas = len(measurements)
        distance_matrix = np.full((n_tracks, n_meas), np.inf)
        
        for i, track in enumerate(tracks):
            for j, meas in enumerate(measurements):
                if meas.position is not None and self.gate_measurement(meas, track):
                    distance_matrix[i, j] = self.mahalanobis_distance(
                        meas.position, track.position, track.covariance
                    )
        
        # Nearest neighbor assignment
        associations = {}
        used_measurements = set()
        
        # Sort all possible associations by distance
        pairs = []
        for i in range(n_tracks):
            for j in range(n_meas):
                if distance_matrix[i, j] < np.inf:
                    pairs.append((i, j, distance_matrix[i, j]))
        
        pairs.sort(key=lambda x: x[2])
        
        # Assign greedily
        for track_idx, meas_idx, dist in pairs:
            if track_idx not in associations and meas_idx not in used_measurements:
                associations[track_idx] = meas_idx
                used_measurements.add(meas_idx)
        
        return associations


class MeasurementFusion:
    """
    Fuses measurements from multiple sensors using optimal weighting
    """
    
    def __init__(self):
        self.kalman = KalmanFilter()
    
    def fuse_positions(
        self,
        measurements: List[SensorMeasurement]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse multiple position measurements with covariance weighting
        
        Args:
            measurements: List of measurements with positions and covariances
            
        Returns:
            Tuple of (fused_position, fused_covariance)
        """
        # Filter measurements with positions
        valid_meas = [m for m in measurements if m.position is not None]
        
        if not valid_meas:
            return np.zeros(3), np.eye(3) * 1e6
        
        if len(valid_meas) == 1:
            meas = valid_meas[0]
            pos = meas.position
            cov = meas.covariance[:3, :3] if meas.covariance is not None else np.eye(3) * 100
            return pos, cov
        
        # Covariance intersection / weighted average
        fused_pos = np.zeros(3)
        fused_info = np.zeros((3, 3))  # Information matrix
        
        for meas in valid_meas:
            pos = meas.position
            if meas.covariance is not None:
                cov = meas.covariance[:3, :3]
            else:
                cov = np.eye(3) * 100  # Default uncertainty
            
            try:
                info = np.linalg.inv(cov)
                fused_info += info
                fused_pos += info @ pos
            except:
                # Skip if covariance is singular
                continue
        
        try:
            fused_cov = np.linalg.inv(fused_info)
            fused_pos = fused_cov @ fused_pos
        except:
            # Fallback to simple average
            fused_pos = np.mean([m.position for m in valid_meas], axis=0)
            fused_cov = np.eye(3) * 100
        
        return fused_pos, fused_cov
    
    def fuse_velocities(
        self,
        measurements: List[SensorMeasurement]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse velocity measurements
        
        Args:
            measurements: Measurements with velocity data
            
        Returns:
            Tuple of (fused_velocity, velocity_covariance)
        """
        valid_vels = [m for m in measurements if m.velocity is not None]
        
        if not valid_vels:
            return np.zeros(3), np.eye(3) * 100
        
        if len(valid_vels) == 1:
            vel = valid_vels[0].velocity
            cov = np.eye(3) * 10  # Default velocity uncertainty
            return vel, cov
        
        # Simple weighted average
        velocities = np.array([m.velocity for m in valid_vels])
        fused_vel = np.mean(velocities, axis=0)
        fused_cov = np.eye(3) * 10
        
        return fused_vel, fused_cov
    
    def compute_fusion_confidence(
        self,
        measurements: List[SensorMeasurement]
    ) -> float:
        """
        Compute overall confidence from multiple measurements
        
        Args:
            measurements: List of measurements
            
        Returns:
            Fused confidence score [0, 1]
        """
        if not measurements:
            return 0.0
        
        # Combine confidences using product of complements
        # P(at least one correct) = 1 - P(all incorrect)
        combined = 1.0
        for meas in measurements:
            combined *= (1.0 - meas.confidence)
        
        fused_confidence = 1.0 - combined
        
        return min(fused_confidence, 1.0)


class SensorFusionEngine:
    """
    Main sensor fusion engine
    Manages tracks and fuses OPIR + RF measurements
    """
    
    def __init__(
        self,
        max_coast_time: float = 10.0,
        min_confidence: float = 0.2,
        gate_threshold: float = 10.0
    ):
        """
        Args:
            max_coast_time: Maximum time to maintain track without updates
            min_confidence: Minimum confidence to keep track
            gate_threshold: Association gate threshold
        """
        self.max_coast_time = max_coast_time
        self.min_confidence = min_confidence
        
        self.kalman = KalmanFilter()
        self.data_association = DataAssociation(gate_threshold=gate_threshold)
        self.measurement_fusion = MeasurementFusion()
        
        self.tracks: List[FusedTrack] = []
        self.next_track_id = 0
        self.current_time = 0.0
    
    def predict_tracks(self, dt: float):
        """
        Predict all tracks forward in time
        
        Args:
            dt: Time step
        """
        for track in self.tracks:
            # Build state vector
            state = np.concatenate([track.position, track.velocity])
            
            # Predict
            state_pred, cov_pred = self.kalman.predict(state, track.covariance)
            
            # Update track
            track.position = state_pred[:3]
            track.velocity = state_pred[3:]
            track.covariance = cov_pred
            
            # Decay confidence for coasting tracks
            track.confidence *= 0.95
    
    def update_track(
        self,
        track: FusedTrack,
        measurements: List[SensorMeasurement]
    ):
        """
        Update track with new measurements
        
        Args:
            track: Track to update
            measurements: Associated measurements
        """
        # Fuse measurements
        fused_pos, pos_cov = self.measurement_fusion.fuse_positions(measurements)
        fused_vel, vel_cov = self.measurement_fusion.fuse_velocities(measurements)
        
        # Build fused covariance
        fused_cov = np.zeros((6, 6))
        fused_cov[:3, :3] = pos_cov
        fused_cov[3:, 3:] = vel_cov
        
        # Kalman update with fused measurement
        state = np.concatenate([track.position, track.velocity])
        
        # Measurement is position only
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        
        # Innovation
        innovation = fused_pos - track.position
        S = H @ track.covariance @ H.T + pos_cov
        
        try:
            K = track.covariance @ H.T @ np.linalg.inv(S)
            state_updated = state + K @ innovation
            cov_updated = (np.eye(6) - K @ H) @ track.covariance
        except:
            # Fallback if update fails
            state_updated = np.concatenate([fused_pos, fused_vel])
            cov_updated = fused_cov
        
        # Update track
        track.position = state_updated[:3]
        track.velocity = state_updated[3:]
        track.covariance = cov_updated
        track.last_update = self.current_time
        
        # Update confidence
        meas_confidence = self.measurement_fusion.compute_fusion_confidence(measurements)
        track.confidence = min(track.confidence + meas_confidence * 0.2, 1.0)
        
        # Track sensor contributions
        for meas in measurements:
            if meas.sensor_type == SensorType.OPIR:
                track.opir_detections += 1
                track.last_opir_update = self.current_time
            elif meas.sensor_type == SensorType.RF:
                track.rf_detections += 1
                track.last_rf_update = self.current_time
        
        # Update event type if available
        for meas in measurements:
            if meas.event_type is not None:
                track.event_type = meas.event_type
                break
        
        # Update quality metrics
        track.position_uncertainty = np.sqrt(np.trace(track.covariance[:3, :3]))
        track.velocity_uncertainty = np.sqrt(np.trace(track.covariance[3:, 3:]))
        track.track_quality = self._compute_track_quality(track)
    
    def _compute_track_quality(self, track: FusedTrack) -> float:
        """
        Compute overall track quality score
        
        Args:
            track: Track to evaluate
            
        Returns:
            Quality score [0, 1]
        """
        # Factors: confidence, uncertainty, sensor diversity
        confidence_score = track.confidence
        
        # Uncertainty penalty (lower is better)
        max_uncertainty = 1000.0
        uncertainty_score = 1.0 - min(track.position_uncertainty / max_uncertainty, 1.0)
        
        # Sensor diversity bonus
        diversity_score = 0.5
        if track.opir_detections > 0 and track.rf_detections > 0:
            diversity_score = 1.0
        elif track.opir_detections > 0 or track.rf_detections > 0:
            diversity_score = 0.7
        
        # Combined quality
        quality = 0.5 * confidence_score + 0.3 * uncertainty_score + 0.2 * diversity_score
        
        return quality
    
    def initialize_track(self, measurements: List[SensorMeasurement]):
        """
        Initialize new track from measurements
        
        Args:
            measurements: Initial measurements
        """
        # Fuse measurements
        fused_pos, pos_cov = self.measurement_fusion.fuse_positions(measurements)
        fused_vel, vel_cov = self.measurement_fusion.fuse_velocities(measurements)
        
        # Build covariance
        fused_cov = np.zeros((6, 6))
        fused_cov[:3, :3] = pos_cov
        fused_cov[3:, 3:] = vel_cov
        
        # Get event type
        event_type = None
        for meas in measurements:
            if meas.event_type is not None:
                event_type = meas.event_type
                break
        
        # Create track
        track = FusedTrack(
            track_id=self.next_track_id,
            position=fused_pos,
            velocity=fused_vel,
            covariance=fused_cov,
            last_update=self.current_time,
            confidence=self.measurement_fusion.compute_fusion_confidence(measurements),
            event_type=event_type
        )
        
        # Count sensor types
        for meas in measurements:
            if meas.sensor_type == SensorType.OPIR:
                track.opir_detections = 1
                track.last_opir_update = self.current_time
            elif meas.sensor_type == SensorType.RF:
                track.rf_detections = 1
                track.last_rf_update = self.current_time
        
        # Quality metrics
        track.position_uncertainty = np.sqrt(np.trace(pos_cov))
        track.velocity_uncertainty = np.sqrt(np.trace(vel_cov))
        track.track_quality = self._compute_track_quality(track)
        
        self.tracks.append(track)
        self.next_track_id += 1
    
    def process_measurements(
        self,
        measurements: List[SensorMeasurement],
        timestamp: float
    ):
        """
        Process new measurements from all sensors
        
        Args:
            measurements: List of sensor measurements
            timestamp: Current timestamp
        """
        dt = timestamp - self.current_time
        self.current_time = timestamp
        
        if dt > 0:
            self.predict_tracks(dt)
        
        # Associate measurements to tracks
        associations = self.data_association.associate_measurements(
            measurements, self.tracks
        )
        
        # Update associated tracks
        updated_tracks = set()
        associated_measurements = set()
        
        for track_idx, meas_idx in associations.items():
            # Can associate multiple measurements to one track
            if track_idx not in updated_tracks:
                associated_meas = [measurements[meas_idx]]
                
                # Find other measurements close to this track
                for j, meas in enumerate(measurements):
                    if j != meas_idx and j not in associated_measurements:
                        if self.data_association.gate_measurement(meas, self.tracks[track_idx]):
                            associated_meas.append(meas)
                            associated_measurements.add(j)
                
                self.update_track(self.tracks[track_idx], associated_meas)
                updated_tracks.add(track_idx)
                associated_measurements.add(meas_idx)
        
        # Initialize new tracks from unassociated measurements
        unassociated = [m for i, m in enumerate(measurements) 
                       if i not in associated_measurements and m.position is not None]
        
        for meas in unassociated:
            self.initialize_track([meas])
        
        # Prune old/low-confidence tracks
        self._prune_tracks()
    
    def _prune_tracks(self):
        """Remove tracks that are coasting too long or have low confidence"""
        valid_tracks = []
        
        for track in self.tracks:
            coast_time = self.current_time - track.last_update
            
            keep_track = (
                coast_time <= self.max_coast_time and
                track.confidence >= self.min_confidence
            )
            
            if keep_track:
                valid_tracks.append(track)
        
        self.tracks = valid_tracks
    
    def get_tracks(self) -> List[FusedTrack]:
        """Get all active tracks"""
        return self.tracks.copy()
    
    def get_high_quality_tracks(self, min_quality: float = 0.6) -> List[FusedTrack]:
        """
        Get tracks above quality threshold
        
        Args:
            min_quality: Minimum track quality
            
        Returns:
            List of high-quality tracks
        """
        return [t for t in self.tracks if t.track_quality >= min_quality]


def convert_opir_to_measurement(
    detection: DetectionResult,
    classification: Optional[ClassificationResult],
    timestamp: float,
    position: Optional[np.ndarray] = None
) -> SensorMeasurement:
    """
    Convert OPIR detection to sensor measurement
    
    Args:
        detection: OPIR detection result
        classification: Optional classification result
        timestamp: Measurement timestamp
        position: Estimated position (if available)
        
    Returns:
        SensorMeasurement
    """
    event_type = None
    if classification is not None:
        event_type = classification.class_name
    
    return SensorMeasurement(
        sensor_type=SensorType.OPIR,
        timestamp=timestamp,
        position=position,
        confidence=detection.confidence if detection.detected else 0.0,
        event_type=event_type,
        metadata={
            'detection_method': detection.method,
            'peak_intensity': detection.peak_intensity,
            'detection_time': detection.detection_time
        }
    )


def convert_rf_to_measurement(
    geolocation: GeolocationResult,
    timestamp: float
) -> SensorMeasurement:
    """
    Convert RF geolocation to sensor measurement
    
    Args:
        geolocation: RF geolocation result
        timestamp: Measurement timestamp
        
    Returns:
        SensorMeasurement
    """
    # Build covariance from geolocation
    if geolocation.covariance is not None:
        cov = np.zeros((6, 6))
        cov[:3, :3] = geolocation.covariance
        if geolocation.velocity is not None:
            cov[3:, 3:] = np.eye(3) * 10  # Default velocity uncertainty
    else:
        cov = np.eye(6) * 100
    
    confidence = 0.8 if geolocation.converged else 0.3
    confidence *= (1.0 / (1.0 + geolocation.gdop / 10.0))  # GDOP penalty
    
    return SensorMeasurement(
        sensor_type=SensorType.RF,
        timestamp=timestamp,
        position=geolocation.position,
        velocity=geolocation.velocity,
        covariance=cov,
        confidence=confidence,
        metadata={
            'gdop': geolocation.gdop,
            'residual': geolocation.residual,
            'num_measurements': geolocation.num_measurements
        }
    )