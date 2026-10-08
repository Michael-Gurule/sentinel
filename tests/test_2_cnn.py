"""
Test 2: CNN Architecture Only
"""

import torch
import numpy as np
from sentinel.models.cnn_classifier import OPIREventCNN


def test_cnn():
    print("=" * 60)
    print("TEST: CNN Architecture")
    print("=" * 60)
    
    # Initialize model
    print("\n1. Initializing CNN model...")
    model = OPIREventCNN(input_length=256, num_classes=5)
    print(f"   Model initialized")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 256)
    print(f"   Input shape: {test_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output sample: {output[0]}")
    
    # Test prediction
    print("\n3. Testing prediction method...")
    preds, probs = model.predict(test_input)
    print(f"   Predictions: {preds}")
    print(f"   Probabilities shape: {probs.shape}")
    print(f"   Sample probs: {probs[0]}")
    
    print("\n" + "=" * 60)
    print("CNN test complete!")
    print("=" * 60)


if __name__ == '__main__':
    test_cnn()