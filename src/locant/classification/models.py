"""Neural classifiers for (N, 1, T) pixel time series.

Both models end in global average + max pooling, so they accept any input
length; the v1 CNN's flatten layer tied the weights to one length.
"""

from typing import Any

import torch
from torch import nn


class CNN1D(nn.Module):
    """Three conv-BN-ReLU-pool blocks with growing receptive field."""

    def __init__(
        self,
        num_classes: int,
        channels: tuple[int, ...] = (32, 64, 128),
        kernel_sizes: tuple[int, ...] = (7, 5, 3),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if len(channels) != len(kernel_sizes):
            raise ValueError("channels and kernel_sizes must have the same length")
        blocks: list[nn.Module] = []
        c_in = 1
        for c_out, k in zip(channels, kernel_sizes, strict=True):
            blocks += [
                nn.Conv1d(c_in, c_out, k, padding=k // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c_in, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        out: torch.Tensor = self.head(pooled)
        return out


class _CausalConv(nn.Conv1d):
    """Conv1d padded on the left only, so output t sees inputs ≤ t."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int) -> None:
        super().__init__(c_in, c_out, kernel, dilation=dilation)
        self.left_pad = (kernel - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(nn.functional.pad(x, (self.left_pad, 0)))


class _TemporalBlock(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float
    ):
        super().__init__()
        self.body = nn.Sequential(
            _CausalConv(c_in, c_out, kernel, dilation),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            _CausalConv(c_out, c_out, kernel, dilation),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = torch.relu(self.body(x) + self.skip(x))
        return out


class TCN(nn.Module):
    """Temporal convolutional network (Bai, Kolter & Koltun 2018).

    Causal dilated residual blocks with dilations 1, 2, 4, …; with kernel 3
    and 7 levels the receptive field is 1 + 2·(3-1)·(2⁷-1) = 509 frames
    (51 s at 10 Hz), most of a 64 s window. Causality allows the same weights
    to run on streaming data.
    """

    def __init__(
        self,
        num_classes: int,
        channels: int = 32,
        levels: int = 7,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks = [
            _TemporalBlock(
                1 if i == 0 else channels, channels, kernel_size, 2**i, dropout
            )
            for i in range(levels)
        ]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Linear(2 * channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        out: torch.Tensor = self.head(
            torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        )
        return out


MODELS: dict[str, type[nn.Module]] = {"cnn": CNN1D, "tcn": TCN}


def build_model(name: str, num_classes: int, **kwargs: Any) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](num_classes=num_classes, **kwargs)
