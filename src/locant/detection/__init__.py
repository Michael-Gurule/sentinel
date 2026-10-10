"""Window-level OPIR event detection with false-alarm-controlled thresholds."""

from locant.detection.base import DetectionScores, Detector, calibrate_threshold
from locant.detection.cfar import CFAR_THRESHOLD_PFA_1E2, CFARDetector
from locant.detection.cusum import CUSUMDetector, siegmund_arl0
from locant.detection.glrt import StepGLRTDetector, noise_sigma_from_differences

__all__ = [
    "CFAR_THRESHOLD_PFA_1E2",
    "CFARDetector",
    "CUSUMDetector",
    "DetectionScores",
    "Detector",
    "StepGLRTDetector",
    "calibrate_threshold",
    "noise_sigma_from_differences",
    "siegmund_arl0",
]
