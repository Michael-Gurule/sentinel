"""
Test 3: Classifier Wrapper Only
"""

from sentinel.models.cnn_classifier import OPIRClassifier
from sentinel.models.signal_generator import OPIRSignalGenerator


def test_classifier():
    print("=" * 60)
    print("TEST: OPIR Classifier Wrapper")
    print("=" * 60)

    # Initialize classifier
    print("\n1. Initializing classifier...")
    classifier = OPIRClassifier(device="cpu")
    print("   Classifier initialized")

    # Generate test signal
    print("\n2. Generating test signal...")
    generator = OPIRSignalGenerator()
    signal = generator.generate_launch_signature(start_time=2.0)
    print(f"   Signal shape: {signal.shape}")

    # Test preprocessing
    print("\n3. Testing preprocessing...")
    preprocessed = classifier.preprocess(signal)
    print(f"   Preprocessed shape: {preprocessed.shape}")

    # Test classification
    print("\n4. Testing classification...")
    result = classifier.classify(signal)
    print(f"   Predicted class: {result.class_name}")
    print(f"   Confidence: {result.confidence:.3f}")
    print("   All probabilities:")
    for name, prob in result.to_dict()["probabilities"].items():
        print(f"      {name}: {prob:.3f}")

    print("\n" + "=" * 60)
    print("Classifier test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_classifier()
