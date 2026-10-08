"""
OPIR Event Inference Module
Clean interface for using trained CNN classifier
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from ..models.cnn_classifier import OPIREventCNN


class OPIRInference:
    """
    Production-ready inference interface for OPIR event classification
    """
    
    CLASS_NAMES = ['launch', 'explosion', 'fire', 'aircraft', 'background']
    
    def __init__(
        self,
        model_path: str = 'outputs/training/best_model.pth',
        device: Optional[str] = None
    ):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device for inference (auto-detected if None)
        """
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        self.device = torch.device(device)
        
        # Load model
        self.model = OPIREventCNN(
            input_length=100,
            num_classes=5,
            dropout_rate=0.3
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'val_acc': checkpoint.get('val_acc', 'unknown')
        }
    
    def preprocess(self, signal: np.ndarray) -> torch.Tensor:
        """
        Preprocess signal for model input
        
        Args:
            signal: OPIR signal array [time_steps] or [1, time_steps]
            
        Returns:
            Preprocessed tensor [1, 1, time_steps]
        """
        # Handle 2D arrays
        if signal.ndim == 2:
            signal = signal[0]
        
        # Ensure correct length
        if len(signal) != 100:
            raise ValueError(f"Expected signal length 100, got {len(signal)}")
        
        # Normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        # Convert to tensor
        tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(
        self,
        signal: np.ndarray,
        return_probs: bool = True
    ) -> Dict:
        """
        Classify OPIR event
        
        Args:
            signal: OPIR signal array
            return_probs: Whether to return all class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        signal_tensor = self.preprocess(signal)
        
        # Inference
        with torch.no_grad():
            output = self.model(signal_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # Build result
        result = {
            'predicted_class': pred_class,
            'class_name': self.CLASS_NAMES[pred_class],
            'confidence': float(confidence)
        }
        
        if return_probs:
            result['probabilities'] = {
                name: float(probs[0, i].item())
                for i, name in enumerate(self.CLASS_NAMES)
            }
        
        return result
    
    def predict_batch(
        self,
        signals: np.ndarray,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Classify multiple signals
        
        Args:
            signals: Array of signals [N, time_steps]
            batch_size: Batch size for inference
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(signals), batch_size):
            batch = signals[i:i+batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for signal in batch:
                try:
                    tensor = self.preprocess(signal)
                    batch_tensors.append(tensor)
                except ValueError as e:
                    results.append({'error': str(e)})
                    continue
            
            if not batch_tensors:
                continue
            
            # Stack into batch
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_classes = torch.argmax(probs, dim=1).cpu().numpy()
                confidences = probs.max(dim=1)[0].cpu().numpy()
            
            # Build results
            for pred_class, confidence in zip(pred_classes, confidences):
                results.append({
                    'predicted_class': int(pred_class),
                    'class_name': self.CLASS_NAMES[pred_class],
                    'confidence': float(confidence)
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        return {
            'model_type': 'OPIREventCNN',
            'num_classes': len(self.CLASS_NAMES),
            'class_names': self.CLASS_NAMES,
            'input_length': 100,
            'device': str(self.device),
            'training_epoch': self.checkpoint_info['epoch'],
            'validation_accuracy': self.checkpoint_info['val_acc']
        }