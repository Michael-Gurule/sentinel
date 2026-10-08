"""
Folder-based OPIR dataset: <data_dir>/<split>/<class_name>/*.npy
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FolderDataset(Dataset):
    """Load data from folder structure: train/class_name/*.npy"""

    def __init__(self, data_dir, split="train"):
        self.data_dir = Path(data_dir) / split
        self.class_names = ["launch", "explosion", "fire", "aircraft", "background"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Collect all file paths
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            for file_path in sorted(class_dir.glob("*.npy")):
                self.samples.append((file_path, self.class_to_idx[class_name]))

        print(f"  Loaded {len(self.samples)} samples from {split} set")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        signal = np.load(file_path)

        # Handle 2D arrays
        if signal.ndim == 2:
            signal = signal[0]

        # Normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

        # Convert to tensor [1, time_steps]
        signal_tensor = torch.FloatTensor(signal).unsqueeze(0)
        label_tensor = torch.LongTensor([label])

        return signal_tensor, label_tensor
