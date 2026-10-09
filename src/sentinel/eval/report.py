"""Deterministic JSON reports."""

import json
from pathlib import Path
from typing import Any

import numpy as np


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float):
        return round(value, 6) if np.isfinite(value) else str(value)
    return value


def write_report(path: str | Path, content: dict[str, Any]) -> Path:
    """Write ``content`` as sorted, rounded JSON (stable diffs across reruns)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_plain(content), indent=2, sort_keys=True) + "\n")
    return out
