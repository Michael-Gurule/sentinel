"""
Evaluation Script for Trained OPIR CNN Classifier
Tests on held-out test set and generates performance metrics
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

from sentinel.models.cnn_classifier import OPIREventCNN
from sentinel.training.datasets import FolderDataset


def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model and return predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            labels = labels.to(device).squeeze()
            
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    
    return {
        'accuracy': accuracy,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Test Set', fontsize=14, weight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix to {save_path}")


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path):
    """Plot per-class accuracy"""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, per_class_acc, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{acc:.1f}%',
            ha='center',
            va='bottom',
            fontsize=11,
            weight='bold'
        )
    
    plt.ylim(0, 105)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy - Test Set', fontsize=14, weight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved per-class accuracy to {save_path}")


def plot_training_curves(history_path, save_path):
    """Plot training curves from history"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, weight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, weight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves to {save_path}")


def main():
    """Main evaluation function"""
    
    # Configuration
    DATA_DIR = 'data/synthetic/opir'
    MODEL_PATH = 'outputs/training/best_model.pth'
    OUTPUT_DIR = 'outputs/evaluation'
    
    # Auto-detect device
    if torch.backends.mps.is_available():
        DEVICE = 'mps'
    elif torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("SENTINEL CNN Classifier Evaluation")
    print("="*60 + "\n")
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = FolderDataset(DATA_DIR, split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    class_names = test_dataset.class_names
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Classes: {class_names}")
    
    # Load model
    print(f"\nLoading trained model from {MODEL_PATH}...")
    sample_signal, _ = test_dataset[0]
    input_length = sample_signal.shape[1]
    
    model = OPIREventCNN(
        input_length=input_length,
        num_classes=5,
        dropout_rate=0.3
    )
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    
    print(f"  Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Validation accuracy during training: {checkpoint['val_acc']:.2f}%")
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    results = evaluate_model(model, test_loader, DEVICE, class_names)
    
    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {results['accuracy']:.2f}%")
    print(f"{'='*60}\n")
    
    # Classification report
    print("Per-Class Metrics:")
    print(classification_report(
        results['labels'],
        results['predictions'],
        target_names=class_names,
        digits=4
    ))
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        class_names,
        output_path / 'confusion_matrix.png'
    )
    
    # Per-class accuracy
    plot_per_class_accuracy(
        results['labels'],
        results['predictions'],
        class_names,
        output_path / 'per_class_accuracy.png'
    )
    
    # Training curves
    history_path = Path('outputs/training/training_history.json')
    if history_path.exists():
        plot_training_curves(
            history_path,
            output_path / 'training_curves.png'
        )
    
    # Save detailed results
    detailed_results = {
        'test_accuracy': float(results['accuracy']),
        'num_samples': len(test_dataset),
        'class_names': class_names,
        'confusion_matrix': confusion_matrix(
            results['labels'],
            results['predictions']
        ).tolist(),
        'classification_report': classification_report(
            results['labels'],
            results['predictions'],
            target_names=class_names,
            output_dict=True
        )
    }
    
    with open(output_path / 'test_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()