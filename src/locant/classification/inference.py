"""Calibrated event classification with prediction sets, from a saved artifact."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from locant.classification.artifact import ModelArtifact, load_artifact
from locant.classification.calibration import energy_score, probabilities
from locant.classification.conformal import prediction_sets
from locant.classification.preprocess import preprocess
from locant.classification.train import predict_logits
from locant.core.linalg import FloatArray


@dataclass(frozen=True, eq=False)
class Prediction:
    """Per-window outputs (arrays over the batch)."""

    labels: tuple[str, ...]
    """Predicted class name per window."""
    probabilities: np.ndarray
    """Temperature-calibrated class probabilities (N, C)."""
    prediction_sets: np.ndarray
    """Conformal set membership (N, C); all-True columns if uncalibrated."""
    energy: np.ndarray
    """Energy score; low values flag inputs unlike the training data."""


class EventClassifier:
    """Loads an artifact once and classifies raw irradiance windows."""

    def __init__(self, artifact: ModelArtifact, model: nn.Module, device: str = "cpu"):
        self.artifact = artifact
        self.model = model
        self.device = torch.device(device)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "EventClassifier":
        artifact, model = load_artifact(path, device)
        return cls(artifact, model, device)

    @property
    def classes(self) -> tuple[str, ...]:
        return self.artifact.classes

    def predict(self, signals: FloatArray) -> Prediction:
        inputs = preprocess(signals, self.artifact.preprocess)
        logits = predict_logits(self.model, inputs, self.device)
        return prediction_from_logits(self.artifact, logits)


def prediction_from_logits(artifact: ModelArtifact, logits: np.ndarray) -> Prediction:
    """Calibrated probabilities, conformal sets, and energy from raw logits
    (shared by every inference backend)."""
    t = artifact.temperature
    probs = probabilities(logits, t)
    sets = (
        prediction_sets(probs, artifact.conformal.threshold, artifact.conformal.method)
        if artifact.conformal is not None
        else np.ones_like(probs, dtype=bool)
    )
    return Prediction(
        labels=tuple(artifact.classes[i] for i in probs.argmax(axis=1)),
        probabilities=probs,
        prediction_sets=sets,
        energy=energy_score(logits, t),
    )
