"""
Demo: Batch classification
Shows efficient processing of multiple signals
"""

from pathlib import Path
import numpy as np
import time

from sentinel.inference.opir_inference import OPIRInference


def load_test_batch(class_name: str, num_samples: int = 50):
    """Load batch of test samples"""
    test_dir = Path(f'data/synthetic/opir/test/{class_name}')
    files = sorted(test_dir.glob('*.npy'))[:num_samples]
    
    signals = []
    for file_path in files:
        signal = np.load(file_path)
        signals.append(signal)
    
    return np.array(signals)


def main():
    print("\n" + "="*60)
    print("SENTINEL OPIR Classifier - Batch Prediction Demo")
    print("="*60 + "\n")
    
    # Initialize inference engine
    print("Loading trained model...")
    classifier = OPIRInference()
    print("✓ Model loaded\n")
    
    # Load batch from each class
    classes = ['launch', 'explosion', 'fire', 'aircraft', 'background']
    batch_size = 50
    
    print(f"Processing {batch_size} samples from each class...\n")
    
    overall_start = time.time()
    
    for class_name in classes:
        # Load batch
        signals = load_test_batch(class_name, batch_size)
        
        # Time prediction
        start_time = time.time()
        results = classifier.predict_batch(signals, batch_size=32)
        elapsed = time.time() - start_time
        
        # Calculate accuracy
        correct = sum(1 for r in results if r['class_name'] == class_name)
        accuracy = correct / len(results) * 100
        
        # Average confidence
        avg_confidence = np.mean([r['confidence'] for r in results]) * 100
        
        print(f"{class_name.capitalize():12s}: {correct}/{len(results)} correct "
              f"({accuracy:.1f}% acc, {avg_confidence:.1f}% conf, {elapsed:.3f}s)")
    
    overall_elapsed = time.time() - overall_start
    total_samples = batch_size * len(classes)
    
    print(f"\n{'='*60}")
    print(f"Total: {total_samples} samples in {overall_elapsed:.2f}s")
    print(f"Throughput: {total_samples/overall_elapsed:.0f} samples/second")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()