"""OPIR event classification: preprocessing, models, training, calibration, inference."""

from locant.classification.artifact import ModelArtifact, load_artifact, save_artifact
from locant.classification.baselines import FeatureClassifier
from locant.classification.inference import EventClassifier, Prediction
from locant.classification.preprocess import PreprocessSpec, preprocess
from locant.classification.train import TrainConfig, select_device, train_model

__all__ = [
    "EventClassifier",
    "FeatureClassifier",
    "ModelArtifact",
    "Prediction",
    "PreprocessSpec",
    "TrainConfig",
    "load_artifact",
    "preprocess",
    "save_artifact",
    "select_device",
    "train_model",
]
