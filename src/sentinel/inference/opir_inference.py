"""Inference interface for a trained OPIR event classifier."""

from pathlib import Path

from sentinel.core.linalg import FloatArray
from sentinel.models.cnn_classifier import OPIRClassifier, select_device
from sentinel.models.taxonomy import EVENT_CLASSES, INPUT_LENGTH


class OPIRInference:
    """Loads a trained checkpoint and returns JSON-friendly predictions.

    Shares preprocessing, class order, and input length with
    :class:`~sentinel.models.cnn_classifier.OPIRClassifier`.
    """

    CLASS_NAMES = EVENT_CLASSES

    def __init__(
        self,
        model_path: str | Path = "outputs/training/best_model.pth",
        device: str | None = None,
    ) -> None:
        self.classifier = OPIRClassifier(
            model_path=model_path, device=str(select_device(device))
        )

    def predict(
        self, signal: FloatArray, return_probs: bool = True
    ) -> dict[str, object]:
        """Classify one ``[time]`` series."""
        result = self.classifier.classify(signal).to_dict()
        if not return_probs:
            result.pop("probabilities")
        return result

    def predict_batch(
        self, signals: FloatArray, batch_size: int = 32
    ) -> list[dict[str, object]]:
        """Classify ``[n, time]`` series; output order matches input order."""
        results: list[dict[str, object]] = []
        for start in range(0, len(signals), batch_size):
            batch = self.classifier.classify_batch(signals[start : start + batch_size])
            results.extend(r.to_dict() for r in batch)
        return results

    def get_model_info(self) -> dict[str, object]:
        info = self.classifier.checkpoint_info
        return {
            "model_type": "OPIREventCNN",
            "num_classes": len(EVENT_CLASSES),
            "class_names": list(EVENT_CLASSES),
            "input_length": INPUT_LENGTH,
            "device": str(self.classifier.device),
            "training_epoch": info.get("epoch", "unknown"),
            "validation_accuracy": info.get("val_acc", "unknown"),
        }
