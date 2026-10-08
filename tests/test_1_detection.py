"""
Test 1: Detection Algorithms Only
"""

from sentinel.detection.opir_detectors import (
    AnomalyDetector,
    MultiMethodDetector,
    RiseTimeDetector,
    TemporalDifferenceDetector,
)
from sentinel.models.signal_generator import OPIRSignalGenerator


def test_detection():
    print("=" * 60)
    print("TEST: Detection Algorithms")
    print("=" * 60)

    # Generate one test signal
    print("\n1. Generating test signal...")
    generator = OPIRSignalGenerator()
    signal = generator.generate_launch_signature(start_time=2.0)
    print(f"   Signal shape: {signal.shape}")
    print(f"   Sampling rate: {generator.sampling_rate} Hz")

    # Test temporal difference detector
    print("\n2. Testing Temporal Difference Detector...")
    detector = TemporalDifferenceDetector()
    result = detector.detect(signal, generator.sampling_rate)
    print(f"   Detected: {result.detected}")
    print(f"   Confidence: {result.confidence:.3f}")

    # Test anomaly detector
    print("\n3. Testing Anomaly Detector...")
    detector = AnomalyDetector()
    result = detector.detect(signal, generator.sampling_rate)
    print(f"   Detected: {result.detected}")
    print(f"   Confidence: {result.confidence:.3f}")

    # Test rise time detector
    print("\n4. Testing Rise Time Detector...")
    detector = RiseTimeDetector()
    result = detector.detect(signal, generator.sampling_rate)
    print(f"   Detected: {result.detected}")
    print(f"   Confidence: {result.confidence:.3f}")

    # Test multi-method detector
    print("\n5. Testing Multi-Method Detector...")
    detector = MultiMethodDetector()
    result = detector.detect(signal, generator.sampling_rate)
    print(f"   Detected: {result.detected}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Metadata: {result.metadata}")

    print("\n" + "=" * 60)
    print("Detection test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_detection()
