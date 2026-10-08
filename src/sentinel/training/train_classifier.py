"""
Training Pipeline for OPIR Event Classifier
Handles data loading, training loop, validation, and model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from tqdm import tqdm

from ..models.cnn_classifier import OPIREventCNN


class OPIRDataset(Dataset):
    """PyTorch Dataset for OPIR signals"""
    
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        transform=None
    ):
        """
        Args:
            signals: Array of OPIR signals [N, time_steps]
            labels: Array of class labels [N]
            transform: Optional transform function
        """
        self.signals = signals
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.signals)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]
        label = self.labels[idx]
        
        # Normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        if self.transform:
            signal = self.transform(signal)
        
        # Convert to tensor [1, time_steps]
        signal_tensor = torch.FloatTensor(signal).unsqueeze(0)
        label_tensor = torch.LongTensor([label])
        
        return signal_tensor, label_tensor


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class ModelTrainer:
    """
    Trainer for OPIR Event CNN
    Manages training loop, validation, and checkpointing
    """
    
    def __init__(
        self,
        model: OPIREventCNN,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Args:
            model: CNN model instance
            device: Training device
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for signals, labels in dataloader:
            signals = signals.to(self.device)
            labels = labels.to(self.device).squeeze()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Validate model
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for signals, labels in dataloader:
                signals = signals.to(self.device)
                labels = labels.to(self.device).squeeze()
                
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[str] = None
    ) -> Dict:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')
        
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Training on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < best_val_loss and checkpoint_dir:
                best_val_loss = val_loss
                self.save_checkpoint(
                    checkpoint_path / 'best_model.pth',
                    epoch,
                    val_loss,
                    val_acc
                )
                print(f"Saved best model (Val Loss: {val_loss:.4f})")
            
            # Early stopping
            if early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save final model
        if checkpoint_dir:
            self.save_checkpoint(
                checkpoint_path / 'final_model.pth',
                epoch,
                val_loss,
                val_acc
            )
        
        return self.history
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
        val_acc: float
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)


def prepare_data_loaders(
    signals: np.ndarray,
    labels: np.ndarray,
    train_split: float = 0.8,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare train and validation data loaders
    
    Args:
        signals: All signal data [N, time_steps]
        labels: All labels [N]
        train_split: Fraction for training
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split data
    n_samples = len(signals)
    n_train = int(n_samples * train_split)
    
    if shuffle:
        indices = np.random.permutation(n_samples)
        signals = signals[indices]
        labels = labels[indices]
    
    train_signals = signals[:n_train]
    train_labels = labels[:n_train]
    val_signals = signals[n_train:]
    val_labels = labels[n_train:]
    
    # Create datasets
    train_dataset = OPIRDataset(train_signals, train_labels)
    val_dataset = OPIRDataset(val_signals, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


def train_model_from_data(
    data_path: str,
    output_dir: str,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = 'cpu'
) -> Dict:
    """
    Complete training pipeline from data file
    
    Args:
        data_path: Path to NPZ file with signals and labels
        output_dir: Directory for outputs
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Training device
        
    Returns:
        Training history
    """
    print("Loading data...")
    data = np.load(data_path)
    signals = data['signals']
    labels = data['labels']
    
    print(f"Data shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Prepare data loaders
    print("\nPreparing data loaders...")
    train_loader, val_loader = prepare_data_loaders(
        signals,
        labels,
        batch_size=batch_size
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = OPIREventCNN(input_length=signals.shape[1])
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate
    )
    
    # Train
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        checkpoint_dir=output_dir
    )
    
    # Save training history
    history_path = Path(output_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete. Outputs saved to {output_dir}")
    
    return history