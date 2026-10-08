"""
Generate comprehensive OPIR training dataset

Creates synthetic thermal events for training the CNN classifier.
"""

import os

from sentinel.models.signal_generator import OPIRSignalGenerator
import numpy as np
from tqdm import tqdm
import json

def generate_dataset(
    num_samples_per_class: int = 2000,  # samples per event type
    output_dir: str = 'data/synthetic/opir', # output directory
    sequence_length: int = 100 # length of each sequence in frames
):
    """
    Generate complete OPIR training dataset
    
    Args:
        num_samples_per_class: Number of samples per event type
        output_dir: Output directory for dataset
        sequence_length: Length of each sequence in frames
    """
    
    print("\n" + "="*70)
    print("GENERATING OPIR TRAINING DATASET")
    print("="*70)
    
    # Create directory structure
    splits = ['train', 'validation', 'test']
    classes = ['launch', 'explosion', 'fire', 'aircraft', 'background']
    
    for split in splits:
        for cls in classes:
            os.makedirs(f"{output_dir}/{split}/{cls}", exist_ok=True)
    
    # Split ratios
    split_ratios = {'train': 0.7, 'validation': 0.15, 'test': 0.15}
    
    generator = OPIRSignalGenerator(sample_rate_hz=1.0, duration_s=sequence_length)
    
    # Generate for each class
    for class_name in classes:
        print(f"\n📊 Generating {num_samples_per_class} samples for class: {class_name}")
        print("-" * 70)
        
        # Determine split counts
        split_counts = {
            split: int(num_samples_per_class * ratio)
            for split, ratio in split_ratios.items()
        }
        
        # Adjust for rounding
        split_counts['train'] = num_samples_per_class - split_counts['validation'] - split_counts['test']
        
        sample_idx = 0
        
        for split, count in split_counts.items():
            print(f"   {split:12s}: {count:4d} samples", end=' ')
            
            for i in tqdm(range(count), desc=f"   Generating", ncols=80):
                # Generate sample based on class
                if class_name == 'launch':
                    events = [{
                        'type': 'launch',
                        'start_time': np.random.uniform(10, 20),
                        'lat': 0, 'lon': 0,
                        'peak_temp': np.random.uniform(3000, 4500),
                        'rise_time': np.random.uniform(2, 5),
                        'sustain_duration': np.random.uniform(20, 60),
                        'decay_time': np.random.uniform(30, 50)
                    }]
                
                elif class_name == 'explosion':
                    events = [{
                        'type': 'explosion',
                        'start_time': np.random.uniform(10, 20),
                        'lat': 0, 'lon': 0,
                        'peak_temp': np.random.uniform(4000, 6000),
                        'flash_duration': np.random.uniform(1, 3),
                        'decay_time': np.random.uniform(5, 15)
                    }]
                
                elif class_name == 'fire':
                    events = [{
                        'type': 'fire',
                        'start_time': np.random.uniform(5, 15),
                        'lat': 0, 'lon': 0,
                        'peak_temp': np.random.uniform(800, 1800),
                        'growth_time': np.random.uniform(30, 80),
                        'sustain_duration': np.random.uniform(0, 30)
                    }]
                
                elif class_name == 'aircraft':
                    events = [{
                        'type': 'aircraft',
                        'start_time': np.random.uniform(10, 30),
                        'lat': 0, 'lon': 0,
                        'peak_temp': np.random.uniform(800, 1200),
                        'transit_duration': np.random.uniform(20, 50)
                    }]
                
                else:  # background
                    events = []  # No events, just background
                
                # Generate scenario
                scenario, _ = generator.generate_scenario(events)
                
                # Extract features for each frame
                features = extract_features_sequence(scenario)
                
                # Save
                filename = f"{output_dir}/{split}/{class_name}/{class_name}_{sample_idx:05d}.npy"
                np.save(filename, features)
                
                sample_idx += 1
    
    # Save metadata
    metadata = {
        'num_samples_per_class': num_samples_per_class,
        'sequence_length': sequence_length,
        'classes': classes,
        'splits': split_counts,
        'features': ['max_temp', 'mean_temp', 'std_temp', 'rise_rate']
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ DATASET GENERATION COMPLETE")
    print("="*70)
    print(f"\n📁 Dataset saved to: {output_dir}")
    print(f"📊 Total samples: {num_samples_per_class * len(classes):,}")
    print(f"💾 Disk space used: ~{estimate_size(num_samples_per_class, len(classes), sequence_length):.1f} MB")


def extract_features_sequence(thermal_sequence: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Extract feature sequence from thermal data
    
    Returns: (4, sequence_length) array
    Features: [max_temp, mean_temp, std_temp, rise_rate]
    """
    features = []
    
    for i in range(len(thermal_sequence)):
        # Get window
        start_idx = max(0, i - window_size)
        window = thermal_sequence[start_idx:i+1]
        
        # Extract features
        max_temp = np.max(window)
        mean_temp = np.mean(window)
        std_temp = np.std(window)
        
        # Rise rate (temperature change per second)
        if len(window) > 1:
            rise_rate = (window[-1] - window[0]) / len(window)
        else:
            rise_rate = 0
        
        features.append([max_temp, mean_temp, std_temp, rise_rate])
    
    return np.array(features).T  # Transpose to (features, time)


def estimate_size(num_samples, num_classes, seq_length):
    """Estimate dataset size in MB"""
    bytes_per_sample = 4 * seq_length * 8  # 4 features, float64
    total_bytes = bytes_per_sample * num_samples * num_classes
    return total_bytes / (1024 * 1024)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate OPIR training dataset')
    parser.add_argument('--samples', type=int, default=2000, help='Samples per class')
    parser.add_argument('--output', type=str, default='data/synthetic/opir', help='Output directory')
    parser.add_argument('--length', type=int, default=100, help='Sequence length')
    
    args = parser.parse_args()
    
    generate_dataset(
        num_samples_per_class=args.samples,
        output_dir=args.output,
        sequence_length=args.length
    )