"""
Simple Training Script for OPIR CNN Classifier
Works with folder-based dataset structure
"""

from torch.utils.data import DataLoader

from sentinel.models.cnn_classifier import OPIREventCNN, select_device
from sentinel.models.taxonomy import INPUT_LENGTH
from sentinel.training.datasets import FolderDataset
from sentinel.training.train_classifier import ModelTrainer


def main():
    """Main training function"""

    # ========== Configuration ==========
    DATA_DIR = "data/synthetic/opir"
    OUTPUT_DIR = "outputs/training"
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # Auto-detect device (MPS for Apple Silicon, CUDA for NVIDIA, CPU otherwise)
    DEVICE = str(select_device())

    print("\n" + "=" * 60)
    print("SENTINEL OPIR CNN Classifier Training")
    print("=" * 60 + "\n")

    # ========== Load Data ==========
    print("Loading datasets...")
    train_dataset = FolderDataset(DATA_DIR, split="train")
    val_dataset = FolderDataset(DATA_DIR, split="validation")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print("\nDataset Configuration:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Input length: {INPUT_LENGTH}")
    print(f"  Classes: {train_dataset.class_names}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Device: {DEVICE}")

    # ========== Initialize Model ==========
    print("\nInitializing CNN model...")
    model = OPIREventCNN(dropout_rate=0.3)

    # ========== Initialize Trainer ==========
    trainer = ModelTrainer(model=model, device=DEVICE, learning_rate=LEARNING_RATE)

    # ========== Train ==========
    print("\nStarting training...\n")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=EPOCHS,
        early_stopping_patience=10,
        checkpoint_dir=OUTPUT_DIR,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    return history


if __name__ == "__main__":
    main()
