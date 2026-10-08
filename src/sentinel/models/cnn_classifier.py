"""
CNN-based OPIR Event Classifier
Classifies detected events into: missile_launch, explosion, wildfire, industrial
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class OPIREventCNN(nn.Module):
    """
    1D CNN for classifying OPIR time-series events
    Architecture optimized for temporal patterns in thermal signatures
    """
    
    def __init__(
        self,
        input_length: int = 256,
        num_classes: int = 4,
        dropout_rate: float = 0.3
    ):
        """
        Args:
            input_length: Length of input time series
            num_classes: Number of event classes
            dropout_rate: Dropout probability for regularization
        """
        super(OPIREventCNN, self).__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        
        # Convolutional layers for feature extraction
        # Layer 1: Capture fast temporal changes
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=7,
            stride=1,
            padding=3
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 2: Capture medium-term patterns
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 3: Capture longer-term patterns
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate flattened size
        self.flat_size = 128 * (input_length // 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, 1, input_length]
            
        Returns:
            Class logits [batch_size, num_classes]
        """
        # Convolutional blocks with ReLU and pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, self.flat_size)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions with probabilities
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs
    
    def get_feature_maps(self, x: torch.Tensor, layer: int = 1) -> torch.Tensor:
        """
        Extract feature maps from specified layer for visualization
        
        Args:
            x: Input tensor
            layer: Layer number (1, 2, or 3)
            
        Returns:
            Feature maps tensor
        """
        self.eval()
        with torch.no_grad():
            if layer >= 1:
                x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            if layer >= 2:
                x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            if layer >= 3:
                x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x


class ClassificationResult:
    """Container for classification results"""
    
    def __init__(
        self,
        predicted_class: int,
        class_name: str,
        probabilities: np.ndarray,
        confidence: float
    ):
        self.predicted_class = predicted_class
        self.class_name = class_name
        self.probabilities = probabilities
        self.confidence = confidence
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'predicted_class': int(self.predicted_class),
            'class_name': self.class_name,
            'confidence': float(self.confidence),
            'probabilities': {
                'missile_launch': float(self.probabilities[0]),
                'explosion': float(self.probabilities[1]),
                'wildfire': float(self.probabilities[2]),
                'industrial': float(self.probabilities[3])
            }
        }


class OPIRClassifier:
    """
    Wrapper class for OPIR event classification
    Handles model loading, preprocessing, and inference
    """
    
    CLASS_NAMES = ['missile_launch', 'explosion', 'wildfire', 'industrial']
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            model_path: Path to trained model weights
            device: Device for inference ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model = OPIREventCNN().to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
    
    def load_model(self, path: str):
        """Load trained model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
    
    def preprocess(self, signal: np.ndarray) -> torch.Tensor:
        """
        Preprocess signal for model input
        
        Args:
            signal: Raw OPIR signal [time_steps]
            
        Returns:
            Preprocessed tensor [1, 1, 256]
        """
        # Normalize signal
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        # Resize to model input length
        if len(signal) != 256:
            from scipy import interpolate
            x_old = np.linspace(0, 1, len(signal))
            x_new = np.linspace(0, 1, 256)
            f = interpolate.interp1d(x_old, signal, kind='linear')
            signal = f(x_new)
        
        # Convert to tensor
        tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def classify(self, signal: np.ndarray) -> ClassificationResult:
        """
        Classify OPIR event
        
        Args:
            signal: OPIR intensity time series
            
        Returns:
            ClassificationResult with predictions
        """
        # Preprocess
        x = self.preprocess(signal)
        
        # Predict
        pred_class, probs = self.model.predict(x)
        
        pred_class = pred_class.cpu().numpy()[0]
        probs = probs.cpu().numpy()[0]
        
        return ClassificationResult(
            predicted_class=pred_class,
            class_name=self.CLASS_NAMES[pred_class],
            probabilities=probs,
            confidence=float(probs[pred_class])
        )
    
    def classify_batch(self, signals: np.ndarray) -> list:
        """
        Classify multiple signals
        
        Args:
            signals: Array of signals [batch_size, time_steps]
            
        Returns:
            List of ClassificationResult objects
        """
        results = []
        for signal in signals:
            result = self.classify(signal)
            results.append(result)
        return results