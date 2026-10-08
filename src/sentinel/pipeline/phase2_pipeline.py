"""
Phase 2 Integration Pipeline
Combines detection, classification, and tracking
"""

import numpy as np

from ..detection.opir_detectors import MultiMethodDetector
from ..models.cnn_classifier import OPIRClassifier
from ..tracking.kalman_tracker import MultiTargetTracker


class SENTINELPhase2Pipeline:
    """
    Complete Phase 2 pipeline integrating:
    - Event detection
    - Event classification
    - Multi-target tracking
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        """
        Args:
            model_path: Path to trained CNN model
            device: Device for inference
        """
        self.detector = MultiMethodDetector()
        self.classifier = OPIRClassifier(model_path=model_path, device=device)
        self.tracker = MultiTargetTracker()

        self.processing_history = []

    def process_signal(
        self, signal: np.ndarray, sampling_rate: float, timestamp: float
    ) -> dict:
        """
        Process single OPIR signal through full pipeline

        Args:
            signal: OPIR intensity time series
            sampling_rate: Samples per second
            timestamp: Signal timestamp

        Returns:
            Dictionary with detection, classification, and tracking results
        """
        result = {
            "timestamp": timestamp,
            "detected": False,
            "classified": False,
            "tracked": False,
        }

        # Step 1: Detection
        detection = self.detector.detect(signal, sampling_rate)
        result["detection"] = detection

        if not detection.detected:
            return result

        result["detected"] = True

        # Step 2: Classification
        classification = self.classifier.classify(signal)
        result["classification"] = classification.to_dict()
        result["classified"] = True

        # Step 3: Tracking (placeholder position for demo)
        # In real system, position would come from geolocation
        position = np.random.randn(3) * 100  # Simulated position

        self.tracker.update_tracks(
            measurements=[position], event_types=[classification.class_name]
        )

        active_tracks = self.tracker.get_active_tracks()
        result["tracked"] = len(active_tracks) > 0
        result["num_tracks"] = len(active_tracks)

        if active_tracks:
            result["tracks"] = [
                {
                    "track_id": t.track_id,
                    "position": t.position.tolist(),
                    "velocity": t.velocity.tolist(),
                    "confidence": t.confidence,
                    "event_type": t.event_type,
                }
                for t in active_tracks
            ]

        self.processing_history.append(result)

        return result

    def process_batch(
        self, signals: np.ndarray, sampling_rate: float, start_time: float = 0.0
    ) -> list[dict]:
        """
        Process batch of signals

        Args:
            signals: Array of signals [N, time_steps]
            sampling_rate: Samples per second
            start_time: Starting timestamp

        Returns:
            List of result dictionaries
        """
        results = []

        for i, signal in enumerate(signals):
            timestamp = start_time + i
            result = self.process_signal(signal, sampling_rate, timestamp)
            results.append(result)

        return results

    def get_summary_statistics(self) -> dict:
        """Get summary statistics from processing history"""
        if not self.processing_history:
            return {}

        total = len(self.processing_history)
        detected = sum(1 for r in self.processing_history if r["detected"])
        classified = sum(1 for r in self.processing_history if r["classified"])

        class_counts = {}
        for r in self.processing_history:
            if "classification" in r:
                class_name = r["classification"]["class_name"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return {
            "total_processed": total,
            "detection_rate": detected / total,
            "classification_rate": classified / total,
            "class_distribution": class_counts,
            "active_tracks": len(self.tracker.get_active_tracks()),
        }


def demo_phase2_pipeline():
    """Demonstration of Phase 2 pipeline"""
    print("=" * 60)
    print("SENTINEL Phase 2 Pipeline Demo")
    print("=" * 60)

    # Initialize pipeline
    pipeline = SENTINELPhase2Pipeline()

    # Generate test signals
    print("\nGenerating test signals...")
    from sentinel.models.signal_generator import OPIRSignalGenerator

    generator = OPIRSignalGenerator()
    test_signals = []

    for _ in range(10):
        event = generator.generate_event()
        test_signals.append(event["signal"])

    test_signals = np.array(test_signals)

    # Process signals
    print(f"Processing {len(test_signals)} signals...")
    results = pipeline.process_batch(test_signals, sampling_rate=100.0)

    # Display results
    print("\nProcessing Results:")
    print("-" * 60)

    for i, result in enumerate(results):
        print(f"\nSignal {i + 1}:")
        print(f"  Detected: {result['detected']}")

        if result["classified"]:
            classification = result["classification"]
            print(f"  Class: {classification['class_name']}")
            print(f"  Confidence: {classification['confidence']:.2%}")

        if result["tracked"]:
            print(f"  Active tracks: {result['num_tracks']}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    stats = pipeline.get_summary_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nPhase 2 Demo Complete!")


if __name__ == "__main__":
    demo_phase2_pipeline()
