"""
Phase 3 Integration Pipeline
Complete multi-sensor fusion system combining OPIR and RF intelligence
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from ..detection.opir_detectors import MultiMethodDetector, DetectionResult
from ..models.cnn_classifier import OPIRClassifier, ClassificationResult
from ..tracking.kalman_tracker import MultiTargetTracker
from ..geolocation.tdoa_fdoa import (
    TDOAGeolocation,
    FDOAGeolocation,
    HybridTDOAFDOA,
    SensorPosition,
    GeolocationMeasurement,
    GeolocationResult
)
from ..geolocation.multilateration import (
    SphericalMultilateration,
    HyperbolicMultilateration,
    WeightedLeastSquares
)
from ..fusion.sensor_fusion import (
    SensorFusionEngine,
    SensorMeasurement,
    SensorType,
    FusedTrack,
    convert_opir_to_measurement,
    convert_rf_to_measurement
)


class SENTINELPhase3Pipeline:
    """
    Complete SENTINEL Multi-INT Platform
    Integrates OPIR detection, RF geolocation, and multi-sensor fusion
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        carrier_freq: float = 1e9,
        device: str = 'cpu'
    ):
        """
        Args:
            model_path: Path to trained CNN model
            carrier_freq: RF carrier frequency for FDOA
            device: Device for CNN inference
        """
        # OPIR components
        self.opir_detector = MultiMethodDetector()
        self.opir_classifier = OPIRClassifier(model_path=model_path, device=device)
        
        # RF geolocation components
        self.tdoa_solver = TDOAGeolocation()
        self.fdoa_solver = FDOAGeolocation(carrier_freq=carrier_freq)
        self.hybrid_solver = HybridTDOAFDOA(carrier_freq=carrier_freq)
        self.multilateration = SphericalMultilateration(method='nonlinear')
        
        # Fusion engine
        self.fusion_engine = SensorFusionEngine(
            max_coast_time=10.0,    # 10 seconds max without updates
            min_confidence=0.15,    # Lower threshold for noisy environments
            gate_threshold=2000.0   # 2km association gate (typical for this scale)
        )
        
        # Sensor network (example configuration)
        self.sensors = self._initialize_sensor_network()
        
        # Processing history
        self.processing_history = []
        self.current_time = 0.0
    
    def _initialize_sensor_network(self) -> List[SensorPosition]:
        """
        Initialize realistic multi-altitude sensor network
    
        Simulates mixed deployment of ground-based, airborne, and 
        high-altitude sensors typical of actual ISR systems.
        Varying altitudes provide 3D geometric diversity for improved
        positioning accuracy (lower GDOP).

        Returns:
            List of sensor positions
        """
        
        sensors = [
            SensorPosition(
                id=0,
                position=np.array([0.0, 0.0, 500.0]),  # Ground-based SIGINT station
                velocity=np.array([0.0, 0.0, 0.0])
            ),
            SensorPosition(
                id=1,
                position=np.array([10000.0, 0.0, 1500.0]),  # Low-altitude ISR aircraft
                velocity=np.array([0.0, 0.0, 0.0])
            ),
            SensorPosition(
                id=2,
                position=np.array([10000.0, 10000.0, 1000.0]),  # Medium-altitude platform
                velocity=np.array([0.0, 0.0, 0.0])
            ),
            SensorPosition(
                id=3,
                position=np.array([0.0, 10000.0, 2000.0]),  # High-altitude ISR (Global Hawk class)
                velocity=np.array([0.0, 0.0, 0.0])
            )
        ]
        return sensors
    
    def process_opir_signal(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        timestamp: float
    ) -> Optional[SensorMeasurement]:
        """
        Process OPIR signal through detection and classification
        
        Args:
            signal: OPIR intensity time series
            sampling_rate: Sampling rate in Hz
            timestamp: Signal timestamp
            
        Returns:
            SensorMeasurement if event detected, None otherwise
        """
        # Step 1: Detection
        detection = self.opir_detector.detect(signal, sampling_rate)
        
        if not detection.detected:
            return None
        
        # Step 2: Classification
        classification = self.opir_classifier.classify(signal)
        
        # Step 3: Convert to measurement (no position yet, just detection)
        measurement = convert_opir_to_measurement(
            detection=detection,
            classification=classification,
            timestamp=timestamp,
            position=None  # OPIR alone doesn't provide position
        )
        
        return measurement
    
    def process_rf_measurements(
        self,
        rf_measurements: List[GeolocationMeasurement],
        timestamp: float,
        use_hybrid: bool = True
    ) -> Optional[SensorMeasurement]:
        """
        Process RF measurements to estimate emitter position
        
        Args:
            rf_measurements: TDOA/FDOA measurements from sensor pairs
            timestamp: Measurement timestamp
            use_hybrid: Use hybrid TDOA/FDOA if True
            
        Returns:
            SensorMeasurement with position estimate
        """
        if use_hybrid:
            # Use hybrid solver for best accuracy
            result = self.hybrid_solver.estimate(
                sensors=self.sensors,
                measurements=rf_measurements
            )
        else:
            # Use TDOA only
            result = self.tdoa_solver.estimate_position(
                sensors=self.sensors,
                measurements=rf_measurements
            )
        
        if not result.converged:
            return None
        
        # Convert to sensor measurement
        measurement = convert_rf_to_measurement(
            geolocation=result,
            timestamp=timestamp
        )
        
        return measurement
    
    def process_multi_sensor_frame(
        self,
        opir_signals: List[np.ndarray],
        rf_measurements: List[List[GeolocationMeasurement]],
        sampling_rate: float,
        timestamp: float
    ) -> Dict:
        """
        Process synchronized frame from multiple sensors
        
        Args:
            opir_signals: List of OPIR signals from different sensors
            rf_measurements: List of RF measurement sets
            sampling_rate: OPIR sampling rate
            timestamp: Frame timestamp
            
        Returns:
            Processing result dictionary
        """
        self.current_time = timestamp
        
        result = {
            'timestamp': timestamp,
            'opir_detections': 0,
            'rf_geolocations': 0,
            'fused_tracks': 0,
            'measurements': []
        }
        
        # Process OPIR signals
        for signal in opir_signals:
            opir_meas = self.process_opir_signal(signal, sampling_rate, timestamp)
            if opir_meas is not None:
                result['measurements'].append(opir_meas)
                result['opir_detections'] += 1
        
        # Process RF measurements
        for rf_meas_set in rf_measurements:
            rf_meas = self.process_rf_measurements(rf_meas_set, timestamp)
            if rf_meas is not None:
                result['measurements'].append(rf_meas)
                result['rf_geolocations'] += 1
        
        # Fuse all measurements
        if result['measurements']:
            self.fusion_engine.process_measurements(
                measurements=result['measurements'],
                timestamp=timestamp
            )
        
        # Get current tracks
        tracks = self.fusion_engine.get_tracks()
        result['fused_tracks'] = len(tracks)
        result['tracks'] = [self._track_to_dict(t) for t in tracks]
        
        # Store in history
        self.processing_history.append(result)
        
        return result
    
    def _track_to_dict(self, track: FusedTrack) -> Dict:
        """Convert FusedTrack to dictionary"""
        return {
            'track_id': track.track_id,
            'position': track.position.tolist(),
            'velocity': track.velocity.tolist(),
            'confidence': track.confidence,
            'event_type': track.event_type,
            'position_uncertainty': track.position_uncertainty,
            'velocity_uncertainty': track.velocity_uncertainty,
            'track_quality': track.track_quality,
            'opir_detections': track.opir_detections,
            'rf_detections': track.rf_detections
        }
    
    def get_situation_awareness(self) -> Dict:
        """
        Get overall situation awareness summary
        
        Returns:
            Situation summary dictionary
        """
        tracks = self.fusion_engine.get_tracks()
        high_quality = self.fusion_engine.get_high_quality_tracks(min_quality=0.6)
        
        # Count by event type
        event_counts = {}
        for track in tracks:
            event_type = track.event_type or 'unknown'
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Average track quality
        avg_quality = np.mean([t.track_quality for t in tracks]) if tracks else 0.0
        
        # Sensor health (simplified)
        opir_active = sum(1 for t in tracks if t.opir_detections > 0)
        rf_active = sum(1 for t in tracks if t.rf_detections > 0)
        
        return {
            'total_tracks': len(tracks),
            'high_quality_tracks': len(high_quality),
            'average_track_quality': avg_quality,
            'event_type_distribution': event_counts,
            'opir_active_tracks': opir_active,
            'rf_active_tracks': rf_active,
            'fusion_ratio': rf_active / max(len(tracks), 1),
            'current_time': self.current_time
        }
    
    def get_track_by_id(self, track_id: int) -> Optional[FusedTrack]:
        """Get specific track by ID"""
        for track in self.fusion_engine.get_tracks():
            if track.track_id == track_id:
                return track
        return None
    
    def export_tracks_to_file(self, filepath: str):
        """
        Export current tracks to file
        
        Args:
            filepath: Output file path
        """
        import json
        
        tracks = self.fusion_engine.get_tracks()
        data = {
            'timestamp': self.current_time,
            'tracks': [self._track_to_dict(t) for t in tracks],
            'situation_awareness': self.get_situation_awareness()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def demo_phase3_system():
    """Demonstration of complete Phase 3 system"""
    print("=" * 70)
    print("SENTINEL PHASE 3 MULTI-SENSOR FUSION DEMO")
    print("=" * 70)
    
    # Initialize system
    print("\n1. Initializing SENTINEL system...")
    pipeline = SENTINELPhase3Pipeline()
    print("   ✓ OPIR detector initialized")
    print("   ✓ RF geolocation initialized")
    print("   ✓ Sensor fusion engine initialized")
    print(f"   ✓ Sensor network: {len(pipeline.sensors)} sensors")
    
    # Generate synthetic scenario
    print("\n2. Generating synthetic multi-sensor scenario...")
    from ..models.signal_generator import OPIRSignalGenerator
    from ..geolocation.tdoa_fdoa import simulate_tdoa_measurements
    
    generator = OPIRSignalGenerator()
    
    # Simulate moving emitter
    emitter_positions = []
    emitter_velocity = np.array([100.0, 50.0, 0.0])  # m/s
    start_pos = np.array([5000.0, 5000.0, 500.0])
    
    num_frames = 10
    
    for frame in range(num_frames):
        timestamp = frame * 1.0  # 1 second per frame
        emitter_pos = start_pos + emitter_velocity * timestamp
        emitter_positions.append(emitter_pos)
        
        # Generate OPIR signals (multiple sensors)
        opir_signals = []
        for _ in range(2):  # 2 OPIR sensors
            signal = generator.generate_launch_signature(start_time=2.0)
            opir_signals.append(signal)
        
        # Generate RF measurements
        rf_measurements = [
            simulate_tdoa_measurements(
                emitter_pos=emitter_pos,
                sensors=pipeline.sensors,
                noise_std=1e-8
            )
        ]
        
        # Process frame
        result = pipeline.process_multi_sensor_frame(
            opir_signals=opir_signals,
            rf_measurements=rf_measurements,
            sampling_rate=generator.sampling_rate,
            timestamp=timestamp
        )
        
        print(f"\n   Frame {frame}: t={timestamp:.1f}s")
        print(f"      OPIR detections: {result['opir_detections']}")
        print(f"      RF geolocations: {result['rf_geolocations']}")
        print(f"      Fused tracks: {result['fused_tracks']}")
    
    # Display final situation awareness
    print("\n3. Final Situation Awareness:")
    print("-" * 70)
    
    sa = pipeline.get_situation_awareness()
    for key, value in sa.items():
        print(f"   {key}: {value}")
    
    # Display track details
    print("\n4. Track Details:")
    print("-" * 70)
    
    tracks = pipeline.fusion_engine.get_tracks()
    for track in tracks:
        print(f"\n   Track {track.track_id}:")
        print(f"      Position: [{track.position[0]:.1f}, {track.position[1]:.1f}, {track.position[2]:.1f}] m")
        print(f"      Velocity: [{track.velocity[0]:.1f}, {track.velocity[1]:.1f}, {track.velocity[2]:.1f}] m/s")
        print(f"      Confidence: {track.confidence:.3f}")
        print(f"      Quality: {track.track_quality:.3f}")
        print(f"      Event type: {track.event_type}")
        print(f"      OPIR detections: {track.opir_detections}")
        print(f"      RF detections: {track.rf_detections}")
    
    print("\n" + "=" * 70)
    print("PHASE 3 DEMO COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    demo_phase3_system()