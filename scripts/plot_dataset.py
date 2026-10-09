"""Figures for docs/data_card.md from a built dataset.

python scripts/plot_dataset.py --data data/opir_v2 --out docs/figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sentinel.data import load_split
from sentinel.taxonomy import EVENT_CLASSES

# Reference palette (light mode): one series hue, muted ink, recessive grid.
SERIES = "#2a78d6"
MUTED = "#a3a29c"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

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
    }
)


def _closest(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def plot_examples(data_dir: Path, out: Path) -> None:
    split = load_split(data_dir, "train")
    snr = split.metadata["peak_snr"]
    fig, axes = plt.subplots(len(EVENT_CLASSES), 1, figsize=(8, 9), sharex=True)
    for index, (ax, name) in enumerate(zip(axes, EVENT_CLASSES, strict=True)):
        members = np.flatnonzero(split.labels == index)
        class_snr = snr[members]
        typical = members[_closest(class_snr, float(np.median(class_snr)))]
        hard = members[_closest(class_snr, float(np.quantile(class_snr, 0.1)))]
        examples = [(typical, SERIES, 2.0)]
        if name != "background":  # SNR is zero for every background sample
            examples.insert(0, (hard, MUTED, 1.0))
        for sample, color, width in examples:
            series = split.signals[sample] - np.median(split.signals[sample])
            ax.plot(split.times, series, color=color, linewidth=width)
        label = (
            name
            if name == "background"
            else f"{name}  (median SNR {np.median(class_snr):.0f})"
        )
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_ylabel("pW/m²")
    handles = [
        plt.Line2D([], [], color=SERIES, linewidth=2.0, label="median-SNR example"),
        plt.Line2D(
            [], [], color=MUTED, linewidth=1.0, label="10th-percentile-SNR example"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.965),
        ncol=2,
        frameon=False,
        labelcolor=TEXT_SECONDARY,
    )
    axes[-1].set_xlabel("time in window (s)")
    fig.suptitle(
        "OPIR pixel irradiance, median-subtracted (opir_v2 train split)",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "dataset_examples.png", dpi=150)
    plt.close(fig)


def plot_snr(data_dir: Path, out: Path) -> None:
    splits = ["train", "shift_low_snr", "shift_clutter"]
    events = [c for c in EVENT_CLASSES if c != "background"]
    fig, axes = plt.subplots(1, len(splits), figsize=(10, 3.6), sharey=True)
    for ax, split_name in zip(axes, splits, strict=True):
        split = load_split(data_dir, split_name)
        values = [
            np.clip(
                split.metadata["peak_snr"][split.labels == EVENT_CLASSES.index(c)],
                1e-2,
                None,
            )
            for c in events
        ]
        parts = ax.boxplot(
            values,
            tick_labels=events,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            widths=0.5,
        )
        for box in parts["boxes"]:
            box.set(facecolor=SERIES, edgecolor=SERIES, alpha=0.85)
        for key in ("whiskers", "caps"):
            for line in parts[key]:
                line.set(color=TEXT_SECONDARY, linewidth=1.0)
        for line in parts["medians"]:
            line.set(color=SURFACE, linewidth=2.0)
        ax.set_yscale("log")
        ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--")
        ax.set_title(split_name, loc="left", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("peak SNR (log scale)")
    axes[0].annotate(
        "SNR = 1",
        xy=(0.55, 1.0),
        xytext=(0, 4),
        textcoords="offset points",
        ha="left",
        color=TEXT_SECONDARY,
        fontsize=8,
    )
    fig.suptitle(
        "Peak SNR by class: boxes span quartiles, whiskers 5th-95th percentile",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    fig.savefig(out / "dataset_snr.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/opir_v2")
    parser.add_argument("--out", default="docs/figures")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_examples(Path(args.data), out)
    plot_snr(Path(args.data), out)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
