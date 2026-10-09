"""Deterministic training loop for the neural classifiers."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from sentinel.classification.models import build_model
from sentinel.classification.preprocess import PreprocessSpec, preprocess


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Literal["cnn", "tcn"] = "tcn"
    model_kwargs: dict[str, Any] = {}
    preprocess: PreprocessSpec = PreprocessSpec()
    epochs: int = Field(30, ge=1)
    batch_size: int = Field(128, ge=1)
    learning_rate: float = Field(3e-3, gt=0)
    weight_decay: float = Field(1e-2, ge=0)
    patience: int = Field(6, ge=1)
    augment: bool = True
    max_shift_fraction: float = Field(0.1, ge=0, lt=0.5)
    seed: int = 0


@dataclass
class TrainResult:
    model: nn.Module
    history: dict[str, list[float]] = field(default_factory=dict)
    best_epoch: int = 0


def select_device(preferred: str | None = None) -> torch.device:
    """``preferred`` if given, else MPS, then CUDA, then CPU."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)  # noqa: NPY002 - torch/sklearn internals may read it
    torch.manual_seed(seed)


def _augment(
    batch: torch.Tensor, rng: np.random.Generator, max_shift: int
) -> torch.Tensor:
    """Random time shift (edge-padded) and ±10% gain, label-preserving."""
    n, _, length = batch.shape
    shifts = torch.as_tensor(rng.integers(-max_shift, max_shift + 1, n))
    source = (torch.arange(length)[None, :] - shifts[:, None]).clamp(0, length - 1)
    shifted = torch.gather(batch[:, 0, :], 1, source)[:, None, :]
    gains = torch.as_tensor(rng.uniform(0.9, 1.1, (n, 1, 1)), dtype=batch.dtype)
    return shifted * gains


@torch.no_grad()
def predict_logits(
    model: nn.Module, inputs: np.ndarray, device: torch.device, batch_size: int = 512
) -> np.ndarray:
    """Logits (N, C) for preprocessed inputs (N, L)."""
    model.eval()
    outputs = []
    for start in range(0, len(inputs), batch_size):
        chunk = torch.as_tensor(inputs[start : start + batch_size], device=device)[
            :, None, :
        ]
        outputs.append(model(chunk).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def train_model(
    config: TrainConfig,
    train_signals: np.ndarray,
    train_labels: np.ndarray,
    val_signals: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int,
    device: torch.device | None = None,
) -> TrainResult:
    """Train with AdamW + one-cycle LR; keep the weights with the lowest validation loss."""
    seed_everything(config.seed)
    device = device or select_device()
    rng = np.random.default_rng(config.seed)
    x_train = torch.as_tensor(preprocess(train_signals, config.preprocess))[:, None, :]
    y_train = torch.as_tensor(np.asarray(train_labels, dtype=np.int64))
    x_val = preprocess(val_signals, config.preprocess)
    y_val = np.asarray(val_labels, dtype=np.int64)

    model = build_model(config.model, num_classes, **config.model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    steps = int(np.ceil(len(x_train) / config.batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate, total_steps=config.epochs * steps
    )
    loss_fn = nn.CrossEntropyLoss()
    max_shift = int(config.max_shift_fraction * x_train.shape[-1])

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_loss, best_state, best_epoch, stale = np.inf, None, 0, 0
    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(len(x_train))
        total = 0.0
        for start in range(0, len(order), config.batch_size):
            idx = order[start : start + config.batch_size]
            xb = x_train[idx]
            if config.augment and max_shift > 0:
                xb = _augment(xb, rng, max_shift)
            xb, yb = xb.to(device), y_train[idx].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss.item()) * len(idx)

        logits = predict_logits(model, x_val, device)
        val_loss = float(
            nn.functional.cross_entropy(torch.as_tensor(logits), torch.as_tensor(y_val))
        )
        history["train_loss"].append(total / len(x_train))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(float(np.mean(logits.argmax(axis=1) == y_val)))
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= config.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return TrainResult(model=model, history=history, best_epoch=best_epoch)
