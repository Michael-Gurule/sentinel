"""
Simple Training Script for OPIR CNN Classifier
Works with folder-based dataset structure
"""

import torch
from torch.utils.data import DataLoader

from sentinel.models.cnn_classifier import OPIREventCNN
from sentinel.training.datasets import FolderDataset
from sentinel.training.train_classifier import ModelTrainer


def main():
    """Main training function"""
    
    # ========== Configuration ==========
    DATA_DIR = 'data/synthetic/opir'
    OUTPUT_DIR = 'outputs/training'
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Auto-detect device (MPS for Apple Silicon, CUDA for NVIDIA, CPU otherwise)
    if torch.backends.mps.is_available():
        DEVICE = 'mps'
    elif torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    
    print("\n" + "="*60)
    print("SENTINEL OPIR CNN Classifier Training")
    print("="*60 + "\n")
    
    # ========== Load Data ==========
    print("Loading datasets...")
    train_dataset = FolderDataset(DATA_DIR, split='train')
    val_dataset = FolderDataset(DATA_DIR, split='validation')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Get input length from first sample
    sample_signal, _ = train_dataset[0]
    input_length = sample_signal.shape[1]
    
    print(f"\nDataset Configuration:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Input length: {input_length}")
    print(f"  Classes: {train_dataset.class_names}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Device: {DEVICE}")
    
    # ========== Initialize Model ==========
    print(f"\nInitializing CNN model...")
    model = OPIREventCNN(
        input_length=input_length,
        num_classes=5,  # 5 classes
        dropout_rate=0.3
    )
    
    # ========== Initialize Trainer ==========
    trainer = ModelTrainer(
        model=model,
        device=DEVICE,
        learning_rate=LEARNING_RATE
    )
    
    # ========== Train ==========
    print(f"\nStarting training...\n")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=EPOCHS,
        early_stopping_patience=10,
        checkpoint_dir=OUTPUT_DIR
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    return history


if __name__ == '__main__':
    main()