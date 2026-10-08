"""
Demo: Single signal classification
Shows how to use trained model for individual predictions
"""

import numpy as np

from sentinel.inference.opir_inference import OPIRInference


def main():
    print("\n" + "=" * 60)
    print("SENTINEL OPIR Classifier - Single Prediction Demo")
    print("=" * 60 + "\n")

    # Initialize inference engine
    print("Loading trained model...")
    classifier = OPIRInference()

    # Display model info
    info = classifier.get_model_info()
    print("✓ Model loaded successfully")
    print(f"  Device: {info['device']}")
    print(f"  Training accuracy: {info['validation_accuracy']:.2f}%")
    print(f"  Classes: {', '.join(info['class_names'])}\n")

    # Test on each class
    test_samples = {
        "launch": "data/synthetic/opir/test/launch/launch_01700.npy",
        "explosion": "data/synthetic/opir/test/explosion/explosion_01700.npy",
        "fire": "data/synthetic/opir/test/fire/fire_01700.npy",
        "aircraft": "data/synthetic/opir/test/aircraft/aircraft_01700.npy",
        "background": "data/synthetic/opir/test/background/background_01700.npy",
    }

    print("Testing on sample signals from each class:\n")
    print("=" * 60)

    for true_label, file_path in test_samples.items():
        # Load signal
        signal = np.load(file_path)

        # Predict
        result = classifier.predict(signal)

        # Display result
        correct = "✓" if result["class_name"] == true_label else "✗"

        print(f"\nTrue Label: {true_label}")
        print(
            f"Predicted:  {result['class_name']} ({result['confidence'] * 100:.2f}%) {correct}"
        )
        print("Probabilities:")
        for class_name, prob in result["probabilities"].items():
            bar = "█" * int(prob * 50)
            print(f"  {class_name:12s}: {prob * 100:5.2f}% {bar}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
