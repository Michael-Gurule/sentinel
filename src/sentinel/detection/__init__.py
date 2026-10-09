"""Window-level OPIR event detection with false-alarm-controlled thresholds."""

from sentinel.detection.base import DetectionScores, Detector, calibrate_threshold
from sentinel.detection.cfar import CFARDetector
from sentinel.detection.cusum import CUSUMDetector, siegmund_arl0
from sentinel.detection.glrt import StepGLRTDetector, noise_sigma_from_differences

__all__ = [
    "CFARDetector",
    "CUSUMDetector",
    "DetectionScores",
    "Detector",
    "StepGLRTDetector",
    "calibrate_threshold",
    "noise_sigma_from_differences",
    "siegmund_arl0",
]
