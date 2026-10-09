"""Model artifacts: weights plus everything needed to reproduce inference.

An artifact is a directory with ``weights.pt`` (a ``state_dict``; loaded with
``weights_only=True``, so no pickled code) and ``artifact.json`` describing the
architecture, preprocessing, class order, calibration, and provenance.
"""

import json
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from sentinel.classification.models import build_model
from sentinel.classification.preprocess import PreprocessSpec
from sentinel.taxonomy import EVENT_CLASSES

WEIGHTS = "weights.pt"
METADATA = "artifact.json"


class ConformalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    method: Literal["lac", "aps"] = "lac"
    alpha: float = Field(gt=0, lt=1)
    threshold: float


class ModelArtifact(BaseModel):
    """Metadata stored next to the weights."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    model: str
    model_kwargs: dict[str, Any] = {}
    classes: tuple[str, ...] = EVENT_CLASSES
    preprocess: PreprocessSpec = PreprocessSpec()
    temperature: float = Field(1.0, gt=0)
    conformal: ConformalSpec | None = None
    dataset: dict[str, Any] = {}
    """Dataset name, version, and config hash the model was trained on."""
    training: dict[str, Any] = {}
    metrics: dict[str, Any] = {}

    def build(self) -> nn.Module:
        return build_model(self.model, len(self.classes), **self.model_kwargs)


def save_artifact(path: str | Path, artifact: ModelArtifact, model: nn.Module) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, out / WEIGHTS)
    (out / METADATA).write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    return out


def load_artifact(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[ModelArtifact, nn.Module]:
    root = Path(path)
    artifact = ModelArtifact.model_validate_json((root / METADATA).read_text())
    if artifact.classes != EVENT_CLASSES and set(artifact.classes) - set(EVENT_CLASSES):
        raise ValueError(f"artifact classes {artifact.classes} are not in the taxonomy")
    model = artifact.build()
    model.load_state_dict(
        torch.load(root / WEIGHTS, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    return artifact, model
