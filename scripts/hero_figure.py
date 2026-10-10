"""Render the README hero figure (docs/figures/hero.png).

Left: one benchmark scenario seen from above, with the fused confirmed tracks
over the true paths. Right: the GOSPA tracking error over time for RF-only,
OPIR-only, and fused tracking, averaged over the 10 E6 benchmark seeds.

    python -m scripts.hero_figure      # or: make hero
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiments.common import GRID, MUTED, SERIES, TEXT_PRIMARY, TEXT_SECONDARY, style
from experiments.fusion_common import (
    CUTOFF,
    gospa_series,
    run_fusion,
    simulate,
    target_type,
)

from locant.eval import gospa

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "hero.png"
FUSED = SERIES[0]
SEEDS = range(10)
MAP_SEED = 0
LABELS = {"launch-0": (6, -4), "aircraft-0": (6, -2), "fire-0": (8, -3)}
"""One labeled target per type, at the end of its true path (offset in points)."""
START_S = 3
"""Skip the first scans, before any track can be confirmed (3-of-5)."""


def _map(ax: plt.Axes) -> None:
    scenario = simulate(MAP_SEED)
    for truth in scenario.truth.values():
        p = truth.position / 1_000.0
        ax.plot(p[:, 0], p[:, 1], color=MUTED, linewidth=1.0, zorder=1)
    output = run_fusion(scenario, "centralized")
    for snap in output.snapshots:
        ids = list(snap.truth)
        truth = np.array([snap.truth[t] for t in ids]).reshape(-1, 3)
        for _, c in gospa(truth, snap.positions, CUTOFF).assignments:
            x, y = snap.positions[c, :2] / 1_000.0
            ax.scatter(x, y, s=9, color=FUSED, linewidths=0, zorder=2)
    for event_id, offset in LABELS.items():
        end_x, end_y = scenario.truth[event_id].position[-1, :2] / 1_000.0
        ax.annotate(
            target_type(event_id),
            (end_x, end_y),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color=TEXT_SECONDARY,
        )
    assert scenario.config.rf is not None
    for receiver in scenario.config.rf.receivers:
        ax.scatter(
            receiver.position_m[0] / 1_000.0,
            receiver.position_m[1] / 1_000.0,
            marker="^",
            s=36,
            color=TEXT_SECONDARY,
            zorder=3,
        )
    ax.scatter([], [], s=12, color=FUSED, label="fused track estimates")
    ax.plot([], [], color=MUTED, linewidth=1.0, label="true paths")
    ax.scatter([], [], marker="^", s=36, color=TEXT_SECONDARY, label="RF receivers")
    ax.set_aspect("equal")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.legend(
        loc="upper left", bbox_to_anchor=(0.0, -0.13), fontsize=8, markerscale=1.2
    )
    ax.set_title("One scenario, seen from above", loc="left", fontsize=10)


def _gospa_over_time(ax: plt.Axes) -> None:
    lines = (
        ("rf", "RF only", MUTED, ":"),
        ("opir", "OPIR only", MUTED, "-"),
        ("centralized", "fused (centralized)", FUSED, "-"),
    )
    scenarios = [simulate(seed) for seed in SEEDS]
    for mode, label, color, linestyle in lines:
        series = np.mean(
            [gospa_series(run_fusion(s, mode).snapshots) for s in scenarios], axis=0
        )
        t = np.arange(series.size)[START_S:]
        series = series[START_S:]
        ax.plot(t, series, color=color, linestyle=linestyle, linewidth=2.0, label=label)
        ax.annotate(
            label,
            (t[-1], series[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT_PRIMARY if mode == "centralized" else TEXT_SECONDARY,
        )
    ax.set_xlim(0, 210)
    ax.set_ylim(0, None)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("GOSPA error (m), mean of 10 scenarios")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Tracking error over time (lower is better)", loc="left", fontsize=10)
    ax.grid(axis="x", visible=False)


def main() -> None:
    style()
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.0, 5.2), gridspec_kw={"width_ratios": [1.0, 1.35]}
    )
    _map(left)
    _gospa_over_time(right)
    for ax in (left, right):
        ax.tick_params(colors=TEXT_SECONDARY)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
