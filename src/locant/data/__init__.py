"""Synthetic OPIR datasets: configuration, deterministic building, loading."""

from locant.data.build import build_dataset, content_hash
from locant.data.config import DatasetConfig, load_dataset_config
from locant.data.dataset import OPIRSplit, load_manifest, load_split

__all__ = [
    "DatasetConfig",
    "OPIRSplit",
    "build_dataset",
    "content_hash",
    "load_dataset_config",
    "load_manifest",
    "load_split",
]
