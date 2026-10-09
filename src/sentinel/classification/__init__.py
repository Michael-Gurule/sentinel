"""OPIR event classification: preprocessing, models, training, calibration, inference."""

from sentinel.classification.artifact import ModelArtifact, load_artifact, save_artifact
from sentinel.classification.baselines import FeatureClassifier
from sentinel.classification.inference import EventClassifier, Prediction
from sentinel.classification.preprocess import PreprocessSpec, preprocess
from sentinel.classification.train import TrainConfig, select_device, train_model

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
