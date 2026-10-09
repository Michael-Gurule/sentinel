"""ONNX export and an ONNX Runtime inference backend.

The exported graph contains only the network (raw logits); preprocessing,
temperature scaling, and conformal sets stay in Python and are shared with
:class:`~locant.classification.inference.EventClassifier`, so both backends
apply identical calibration. ONNX Runtime needs no PyTorch at inference time
and is about 10x faster on CPU for this TCN (``make bench``).

Requires the ``onnx`` extra: ``pip install locant[onnx]``.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from locant.classification.artifact import METADATA, ModelArtifact, load_artifact
from locant.classification.inference import Prediction, prediction_from_logits
from locant.classification.preprocess import preprocess
from locant.core.linalg import FloatArray

ONNX_MODEL = "model.onnx"
_BATCH = 512


def export_onnx(artifact_dir: str | Path, out: str | Path | None = None) -> Path:
    """Export an artifact's network to ONNX (dynamic batch and length).

    Writes ``model.onnx`` into the artifact directory unless ``out`` is given.
    """
    root = Path(artifact_dir)
    _, model = load_artifact(root, "cpu")
    target = Path(out) if out is not None else root / ONNX_MODEL
    example = torch.zeros(2, 1, 640, dtype=torch.float32)
    torch.onnx.export(
        model,
        (example,),
        str(target),
        dynamo=True,
        dynamic_shapes=(
            {0: torch.export.Dim("batch"), 2: torch.export.Dim("length", min=16)},
        ),
        input_names=["signals"],
        output_names=["logits"],
        opset_version=18,
        verbose=False,
    )
    return target


class OnnxEventClassifier:
    """:class:`~locant.pipeline.stages.Classifier` backed by ONNX Runtime."""

    def __init__(self, artifact: ModelArtifact, session: Any) -> None:
        self.artifact = artifact
        self.session = session
        self._input = session.get_inputs()[0].name

    @classmethod
    def load(
        cls, path: str | Path, model_file: str = ONNX_MODEL
    ) -> "OnnxEventClassifier":
        """Load an artifact directory whose ``model.onnx`` was produced by
        :func:`export_onnx` (``locant export-onnx``).

        Raises:
            FileNotFoundError: the ONNX model has not been exported.
        """
        import onnxruntime

        root = Path(path)
        model = root / model_file
        if not model.exists():
            raise FileNotFoundError(
                f"{model} not found; export it with `locant export-onnx {root}`"
            )
        artifact = ModelArtifact.model_validate_json((root / METADATA).read_text())
        session = onnxruntime.InferenceSession(
            str(model), providers=["CPUExecutionProvider"]
        )
        return cls(artifact, session)

    @property
    def classes(self) -> tuple[str, ...]:
        return self.artifact.classes

    def predict(self, signals: FloatArray) -> Prediction:
        inputs = preprocess(signals, self.artifact.preprocess).astype(np.float32)
        logits = np.concatenate(
            [
                self.session.run(None, {self._input: inputs[i : i + _BATCH, None, :]})[
                    0
                ]
                for i in range(0, len(inputs), _BATCH)
            ]
        )
        return prediction_from_logits(self.artifact, logits.astype(np.float64))
