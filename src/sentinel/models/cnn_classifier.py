"""
1-D CNN for OPIR event classification, and a wrapper that applies the shared
preprocessing and class taxonomy.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sentinel.core.linalg import FloatArray
from sentinel.models.taxonomy import EVENT_CLASSES, INPUT_LENGTH, preprocess_signal


def select_device(preferred: str | None = None) -> torch.device:
    """``preferred`` if given, else MPS, then CUDA, then CPU."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class OPIREventCNN(nn.Module):
    """
    1D CNN for classifying OPIR time-series events
    Architecture optimized for temporal patterns in thermal signatures
    """

    def __init__(
        self,
        input_length: int = INPUT_LENGTH,
        num_classes: int = len(EVENT_CLASSES),
        dropout_rate: float = 0.3,
    ) -> None:
        """
        Args:
            input_length: Length of input time series
            num_classes: Number of event classes
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()

        self.input_length = input_length
        self.num_classes = num_classes

        # Convolutional layers for feature extraction
        # Layer 1: Capture fast temporal changes
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 2: Capture medium-term patterns
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 3: Capture longer-term patterns
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
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

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


@dataclass(frozen=True, eq=False)
class ClassificationResult:
    """Predicted class with the full probability vector."""

    predicted_class: int
    class_name: str
    probabilities: FloatArray
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "predicted_class": self.predicted_class,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "probabilities": {
                name: float(p)
                for name, p in zip(EVENT_CLASSES, self.probabilities, strict=True)
            },
        }


def load_state_dict(
    model: nn.Module, path: str | Path, device: torch.device
) -> dict[str, object]:
    """Load weights from a checkpoint (or bare state dict) into ``model``.

    Returns the checkpoint metadata (everything except the weights).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    model.load_state_dict(checkpoint)
    return {}


class OPIRClassifier:
    """Preprocesses OPIR series and classifies them with :class:`OPIREventCNN`.

    Without ``model_path`` the network has random weights; the pipeline uses
    this until a model is trained on the Phase 2 dataset (audit defect C8).
    """

    CLASS_NAMES = EVENT_CLASSES

    def __init__(self, model_path: str | Path | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = OPIREventCNN().to(self.device)
        self.checkpoint_info: dict[str, object] = {}
        if model_path:
            self.load_model(model_path)
        self.model.eval()

    def load_model(self, path: str | Path) -> None:
        self.checkpoint_info = load_state_dict(self.model, path, self.device)
        self.model.eval()

    def preprocess(self, signals: FloatArray) -> torch.Tensor:
        """``[time]`` or ``[batch, time]`` array → ``[batch, 1, INPUT_LENGTH]``."""
        batch = np.atleast_2d(np.asarray(signals, dtype=np.float64))
        processed = np.stack([preprocess_signal(s) for s in batch])
        return torch.as_tensor(processed, dtype=torch.float32, device=self.device)[
            :, None, :
        ]

    def classify_batch(self, signals: FloatArray) -> list[ClassificationResult]:
        """Classify a ``[batch, time]`` array in one forward pass."""
        _, probabilities = self.model.predict(self.preprocess(signals))
        probs = probabilities.cpu().numpy().astype(np.float64)
        results = []
        for row in probs:
            index = int(np.argmax(row))
            results.append(
                ClassificationResult(
                    predicted_class=index,
                    class_name=EVENT_CLASSES[index],
                    probabilities=row,
                    confidence=float(row[index]),
                )
            )
        return results

    def classify(self, signal: FloatArray) -> ClassificationResult:
        """Classify a single ``[time]`` series."""
        return self.classify_batch(np.asarray(signal)[None, :])[0]
