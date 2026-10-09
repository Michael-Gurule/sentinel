"""Classical baselines on handcrafted features."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from sentinel.classification.features import extract_features
from sentinel.core.linalg import FloatArray


@dataclass
class FeatureClassifier:
    """Features → scikit-learn classifier, with a ``predict_proba`` over raw windows."""

    kind: str = "logistic"
    seed: int = 0
    estimator: Pipeline | HistGradientBoostingClassifier = field(init=False)

    def __post_init__(self) -> None:
        if self.kind == "logistic":
            self.estimator = make_pipeline(
                StandardScaler(), LogisticRegression(C=1.0, max_iter=5_000)
            )
        elif self.kind == "gbm":
            self.estimator = HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self.seed,
            )
        else:
            raise ValueError(f"unknown baseline {self.kind!r}")

    def fit(self, signals: FloatArray, labels: np.ndarray) -> "FeatureClassifier":
        self.estimator.fit(extract_features(signals), labels)
        return self

    def predict_proba(self, signals: FloatArray) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(extract_features(signals)))
