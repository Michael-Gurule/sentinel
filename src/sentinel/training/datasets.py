"""
Folder-based OPIR dataset: <data_dir>/<split>/<class_name>/*.npy
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sentinel.models.taxonomy import EVENT_CLASSES, preprocess_signal


class FolderDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Samples stored as ``<split>/<class_name>/*.npy``, labeled by folder.

    Files may hold a 1-D series or a 2-D ``[features, time]`` array; for the
    latter the first feature row is used (v1 dataset layout).
    """

    def __init__(self, data_dir: str | Path, split: str = "train") -> None:
        self.data_dir = Path(data_dir) / split
        self.class_names = list(EVENT_CLASSES)
        self.samples: list[tuple[Path, int]] = [
            (path, label)
            for label, name in enumerate(self.class_names)
            for path in sorted((self.data_dir / name).glob("*.npy"))
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        signal = np.load(path)
        if signal.ndim == 2:
            signal = signal[0]
        x = torch.as_tensor(preprocess_signal(signal), dtype=torch.float32)[None, :]
        return x, torch.tensor([label], dtype=torch.long)
