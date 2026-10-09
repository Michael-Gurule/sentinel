"""Shared paths, plotting style, and helpers for the Phase 3 experiments.

Experiments read the versioned dataset (``make data``) and write JSON reports
and figures under ``reports/phase3``. Trained models go to ``outputs/models``
(ignored); the selected classifier is exported to ``models/``.

Every experiment accepts ``--quick``: tiny sizes, CPU, and no figures, so the
test suite can exercise the full code path in seconds.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sentinel.data import load_manifest, load_split
from sentinel.data.dataset import OPIRSplit

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "opir_v2"
REPORT_DIR = ROOT / "reports" / "phase3"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_DIR = ROOT / "outputs" / "models"
EXPORT_DIR = ROOT / "models"

EVAL_SPLITS = ("test", "shift_low_snr", "shift_params", "shift_clutter")
SNR_BINS = (0.0, 2.0, 5.0, 20.0, 100.0, float("inf"))
SNR_BIN_LABELS = ("<2", "2-5", "5-20", "20-100", ">100")

# Reference palette (light mode), categorical slots in fixed order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
MUTED = "#a3a29c"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "legend.frameon": False,
            "legend.labelcolor": TEXT_SECONDARY,
        }
    )


@dataclass(frozen=True)
class RunOptions:
    quick: bool
    data_dir: Path
    report_dir: Path
    model_dir: Path
    device: str | None
    workers: int


def parse_options(description: str, report_dir: Path = REPORT_DIR) -> RunOptions:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--quick", action="store_true", help="Tiny smoke run (CPU, no figures)"
    )
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--reports", type=Path, default=report_dir)
    parser.add_argument("--models", type=Path, default=MODEL_DIR)
    parser.add_argument("--device", default=None, help="Torch device (default: auto)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    return RunOptions(
        quick=args.quick,
        data_dir=args.data,
        report_dir=args.reports,
        model_dir=args.models,
        device="cpu" if args.quick and args.device is None else args.device,
        workers=1 if args.quick else args.workers,
    )


def load(options: RunOptions, name: str, per_class: int | None = None) -> OPIRSplit:
    """Load a split; in quick mode keep only ``per_class`` samples per class."""
    split = load_split(options.data_dir, name)
    if not options.quick or per_class is None:
        return split
    keep = np.concatenate(
        [np.flatnonzero(split.labels == c)[:per_class] for c in np.unique(split.labels)]
    )
    return OPIRSplit(
        name=split.name,
        signals=split.signals[keep],
        labels=split.labels[keep],
        times=split.times,
        metadata={k: v[keep] for k, v in split.metadata.items()},
        params=split.params[keep],
    )


def dataset_info(options: RunOptions) -> dict[str, Any]:
    manifest = load_manifest(options.data_dir)
    return {
        "dataset": manifest["dataset"],
        "version": manifest["version"],
        "config_sha256": manifest["config_sha256"],
        "root_seed": manifest["root_seed"],
    }


def snr_bin(snr: np.ndarray) -> np.ndarray:
    """Index into SNR_BIN_LABELS for each value."""
    return np.clip(np.digitize(snr, SNR_BINS[1:-1]), 0, len(SNR_BIN_LABELS) - 1)
