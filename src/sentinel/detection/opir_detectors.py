"""
OPIR Event Detection Algorithms
Implements temporal difference, anomaly detection, and rise-time analysis
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Container for detection results"""
    detected: bool
    confidence: float
    detection_time: float
    peak_intensity: float
    method: str
    metadata: Dict


class TemporalDifferenceDetector:
    """
    Detects events by analyzing frame-to-frame differences
    Effective for sudden intensity changes
    """
    
    def __init__(
        self,
        threshold: float = 3.0,
        window_size: int = 5,
        min_duration: int = 3
    ):
        """
        Args:
            threshold: Standard deviations above baseline for detection
            window_size: Frames for background estimation
            min_duration: Minimum frames to confirm detection
        """
        self.threshold = threshold
        self.window_size = window_size
        self.min_duration = min_duration
    
    def detect(self, signal_data: np.ndarray, sampling_rate: float) -> DetectionResult:
        """
        Detect events using temporal differencing
        
        Args:
            signal_data: OPIR intensity values [time_steps]
            sampling_rate: Samples per second
            
        Returns:
            DetectionResult with detection information
        """
        # Compute frame-to-frame differences
        diff = np.diff(signal_data)
        
        # Estimate baseline noise using initial window
        baseline_window = diff[:self.window_size]
        baseline_std = np.std(baseline_window)
        baseline_mean = np.mean(baseline_window)
        
        # Detect anomalous differences
        z_scores = (diff - baseline_mean) / (baseline_std + 1e-10)
        detections = z_scores > self.threshold
        
        # Require minimum duration
        detected = self._check_duration(detections, self.min_duration)
        
        if detected:
            detection_idx = np.where(detections)[0][0]
            detection_time = detection_idx / sampling_rate
            peak_intensity = np.max(signal_data[detection_idx:])
            confidence = float(np.max(z_scores[detections]))
            
            return DetectionResult(
                detected=True,
                confidence=min(confidence / 10.0, 1.0),  # Normalize to [0,1]
                detection_time=detection_time,
                peak_intensity=peak_intensity,
                method='temporal_difference',
                metadata={
                    'max_z_score': float(np.max(z_scores)),
                    'detection_frame': int(detection_idx),
                    'baseline_std': float(baseline_std)
                }
            )
        else:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                detection_time=0.0,
                peak_intensity=0.0,
                method='temporal_difference',
                metadata={}
            )
    
    @staticmethod
    def _check_duration(detections: np.ndarray, min_duration: int) -> bool:
        """Check if detections meet minimum duration requirement"""
        if not np.any(detections):
            return False
        
        # Find consecutive detection sequences
        padded = np.concatenate(([0], detections.astype(int), [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        durations = ends - starts
        
        return np.any(durations >= min_duration)


class AnomalyDetector:
    """
    Statistical anomaly detection using moving statistics
    Effective for identifying outliers in intensity patterns
    """
    
    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 3.5,
        method: str = 'mad'  # 'mad' or 'zscore'
    ):
        """
        Args:
            window_size: Samples for rolling statistics
            threshold: Detection threshold (MAD or z-score units)
            method: 'mad' (Median Absolute Deviation) or 'zscore'
        """
        self.window_size = window_size
        self.threshold = threshold
        self.method = method
    
    def detect(self, signal_data: np.ndarray, sampling_rate: float) -> DetectionResult:
        """
        Detect anomalies using statistical methods
        
        Args:
            signal_data: OPIR intensity values
            sampling_rate: Samples per second
            
        Returns:
            DetectionResult with detection information
        """
        if self.method == 'mad':
            scores = self._compute_mad_scores(signal_data)
        else:
            scores = self._compute_z_scores(signal_data)
        
        # Detect anomalies
        anomalies = scores > self.threshold
        detected = np.any(anomalies)
        
        if detected:
            detection_idx = np.where(anomalies)[0][0]
            detection_time = detection_idx / sampling_rate
            peak_intensity = signal_data[detection_idx]
            confidence = float(scores[detection_idx] / (self.threshold * 2))
            confidence = min(confidence, 1.0)
            
            return DetectionResult(
                detected=True,
                confidence=confidence,
                detection_time=detection_time,
                peak_intensity=peak_intensity,
                method=f'anomaly_{self.method}',
                metadata={
                    'max_score': float(np.max(scores)),
                    'detection_frame': int(detection_idx),
                    'num_anomalies': int(np.sum(anomalies))
                }
            )
        else:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                detection_time=0.0,
                peak_intensity=0.0,
                method=f'anomaly_{self.method}',
                metadata={}
            )
    
    def _compute_mad_scores(self, data: np.ndarray) -> np.ndarray:
        """Compute Median Absolute Deviation scores"""
        scores = np.zeros_like(data)
        
        for i in range(len(data)):
            start_idx = max(0, i - self.window_size)
            window = data[start_idx:i+1]
            
            if len(window) < 3:
                continue
            
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            
            if mad > 0:
                scores[i] = np.abs(data[i] - median) / mad
        
        return scores
    
    def _compute_z_scores(self, data: np.ndarray) -> np.ndarray:
        """Compute rolling z-scores"""
        scores = np.zeros_like(data)
        
        for i in range(len(data)):
            start_idx = max(0, i - self.window_size)
            window = data[start_idx:i+1]
            
            if len(window) < 3:
                continue
            
            mean = np.mean(window)
            std = np.std(window)
            
            if std > 0:
                scores[i] = np.abs(data[i] - mean) / std
        
        return scores


class RiseTimeDetector:
    """
    Detects events by analyzing signal rise characteristics
    Effective for identifying specific event types by rise profile
    """
    
    def __init__(
        self,
        min_rise_rate: float = 0.1,
        rise_threshold: float = 0.3,
        smoothing_window: int = 3
    ):
        """
        Args:
            min_rise_rate: Minimum rate of intensity increase
            rise_threshold: Minimum total rise magnitude
            smoothing_window: Window for signal smoothing
        """
        self.min_rise_rate = min_rise_rate
        self.rise_threshold = rise_threshold
        self.smoothing_window = smoothing_window
    
    def detect(self, signal_data: np.ndarray, sampling_rate: float) -> DetectionResult:
        """
        Detect events by analyzing rise characteristics
        
        Args:
            signal_data: OPIR intensity values
            sampling_rate: Samples per second
            
        Returns:
            DetectionResult with rise-time analysis
        """
        # Smooth signal to reduce noise
        smoothed = self._smooth_signal(signal_data)
        
        # Compute derivative (rise rate)
        derivative = np.gradient(smoothed)
        
        # Find regions with significant rise
        rising = derivative > self.min_rise_rate
        
        if not np.any(rising):
            return DetectionResult(
                detected=False,
                confidence=0.0,
                detection_time=0.0,
                peak_intensity=0.0,
                method='rise_time',
                metadata={}
            )
        
        # Analyze rise segments
        rise_segments = self._find_rise_segments(smoothed, rising)
        
        if not rise_segments:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                detection_time=0.0,
                peak_intensity=0.0,
                method='rise_time',
                metadata={}
            )
        
        # Get most significant rise
        best_segment = max(rise_segments, key=lambda x: x['magnitude'])
        
        if best_segment['magnitude'] < self.rise_threshold:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                detection_time=0.0,
                peak_intensity=0.0,
                method='rise_time',
                metadata={}
            )
        
        detection_time = best_segment['start_idx'] / sampling_rate
        confidence = min(best_segment['magnitude'] / (self.rise_threshold * 2), 1.0)
        
        return DetectionResult(
            detected=True,
            confidence=confidence,
            detection_time=detection_time,
            peak_intensity=best_segment['peak_value'],
            method='rise_time',
            metadata={
                'rise_time': best_segment['duration'] / sampling_rate,
                'rise_rate': best_segment['avg_rate'],
                'magnitude': best_segment['magnitude']
            }
        )
    
    def _smooth_signal(self, data: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing"""
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        return np.convolve(data, kernel, mode='same')
    
    def _find_rise_segments(
        self,
        smoothed: np.ndarray,
        rising: np.ndarray
    ) -> List[Dict]:
        """Find and characterize rising segments"""
        segments = []
        
        # Find consecutive rising regions
        padded = np.concatenate(([0], rising.astype(int), [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        
        for start, end in zip(starts, ends):
            if end - start < 2:
                continue
            
            segment_data = smoothed[start:end]
            magnitude = segment_data[-1] - segment_data[0]
            duration = end - start
            avg_rate = magnitude / duration if duration > 0 else 0
            
            segments.append({
                'start_idx': start,
                'end_idx': end,
                'duration': duration,
                'magnitude': magnitude,
                'avg_rate': avg_rate,
                'peak_value': np.max(smoothed[start:end+10])  # Look ahead for peak
            })
        
        return segments


class MultiMethodDetector:
    """
    Ensemble detector combining multiple detection methods
    Provides robust detection with voting or consensus
    """
    
    def __init__(
        self,
        detectors: Optional[List] = None,
        voting_threshold: float = 0.5
    ):
        """
        Args:
            detectors: List of detector instances
            voting_threshold: Fraction of detectors that must agree
        """
        if detectors is None:
            self.detectors = [
                TemporalDifferenceDetector(),
                AnomalyDetector(method='mad'),
                RiseTimeDetector()
            ]
        else:
            self.detectors = detectors
        
        self.voting_threshold = voting_threshold
    
    def detect(self, signal_data: np.ndarray, sampling_rate: float) -> DetectionResult:
        """
        Run all detectors and combine results
        
        Args:
            signal_data: OPIR intensity values
            sampling_rate: Samples per second
            
        Returns:
            Combined DetectionResult
        """
        results = []
        for detector in self.detectors:
            result = detector.detect(signal_data, sampling_rate)
            results.append(result)
        
        # Count detections
        num_detected = sum(r.detected for r in results)
        detection_rate = num_detected / len(self.detectors)
        
        if detection_rate >= self.voting_threshold:
            # Combine detected results
            detected_results = [r for r in results if r.detected]
            
            avg_confidence = np.mean([r.confidence for r in detected_results])
            avg_time = np.mean([r.detection_time for r in detected_results])
            max_intensity = max([r.peak_intensity for r in detected_results])
            
            methods = [r.method for r in detected_results]
            
            return DetectionResult(
                detected=True,
                confidence=float(avg_confidence * detection_rate),
                detection_time=float(avg_time),
                peak_intensity=float(max_intensity),
                method='multi_method',
                metadata={
                    'num_detected': num_detected,
                    'detection_rate': detection_rate,
                    'methods': methods,
                    'individual_confidences': [r.confidence for r in detected_results]
                }
            )
        else:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                detection_time=0.0,
                peak_intensity=0.0,
                method='multi_method',
                metadata={
                    'num_detected': num_detected,
                    'detection_rate': detection_rate
                }
            )


def detect_event(
    signal_data: np.ndarray,
    sampling_rate: float,
    method: str = 'multi'
) -> DetectionResult:
    """
    Convenience function for event detection
    
    Args:
        signal_data: OPIR intensity values
        sampling_rate: Samples per second
        method: Detection method ('temporal', 'anomaly', 'rise', 'multi')
        
    Returns:
        DetectionResult
    """
    detectors = {
        'temporal': TemporalDifferenceDetector(),
        'anomaly': AnomalyDetector(),
        'rise': RiseTimeDetector(),
        'multi': MultiMethodDetector()
    }
    
    if method not in detectors:
        raise ValueError(f"Unknown method: {method}. Choose from {list(detectors.keys())}")
    
    detector = detectors[method]
    return detector.detect(signal_data, sampling_rate)