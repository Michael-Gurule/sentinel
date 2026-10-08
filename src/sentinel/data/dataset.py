"""Loading built dataset splits, with optional integrity verification."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sentinel.core.linalg import FloatArray
from sentinel.data.build import MANIFEST_NAME, content_hash
from sentinel.data.generate import METADATA_FIELDS


@dataclass(frozen=True, eq=False)
class OPIRSplit:
    """One dataset split held in memory.

    Attributes:
        signals: Measured pixel irradiance, (N, T) float32, pW/m².
        labels: Class indices into ``EVENT_CLASSES``, (N,).
        times: Frame times within the window, (T,) seconds.
        metadata: Numeric per-sample metadata, each (N,).
        params: Per-sample generator parameters as JSON strings.
    """

    name: str
    signals: np.ndarray
    labels: np.ndarray
    times: FloatArray
    metadata: dict[str, FloatArray]
    params: np.ndarray

    def __len__(self) -> int:
        return int(self.labels.size)

    def sample_params(self, index: int) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(str(self.params[index]))
        return result


def load_manifest(root: str | Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((Path(root) / MANIFEST_NAME).read_text())
    return manifest


def load_split(root: str | Path, name: str, verify: bool = True) -> OPIRSplit:
    """Load a split; with ``verify`` the content hash must match the manifest.

    Raises:
        ValueError: if verification fails.
    """
    manifest = load_manifest(root)
    entry = manifest["splits"][name]
    with np.load(Path(root) / entry["file"]) as data:
        arrays = {key: data[key] for key in data.files}
    if verify and content_hash(arrays) != entry["content_sha256"]:
        raise ValueError(f"split {name!r} does not match its manifest hash")
    return OPIRSplit(
        name=name,
        signals=arrays["signals"],
        labels=arrays["labels"],
        times=arrays["times"].astype(np.float64),
        metadata={f: arrays[f"meta_{f}"] for f in METADATA_FIELDS},
        params=arrays["params"],
    )
