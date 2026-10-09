"""E5: multi-target tracking quality and track management.

Questions:
1. How do OPIR false reports (clutter) and the M-of-N confirmation rule trade
   false tracks against confirmation latency?
2. Two satellites see a target only up to its epipolar plane, so rays from
   different targets can pair into "ghosts". How many false tracks do ghosts
   cause, and what does deferring ambiguous pairs cost?
3. Does an IMM (quiet + maneuvering constant-velocity models) beat a single
   constant-velocity model across steady aircraft, ground fires, and boosting
   launches?

Every run is the centralized fusion architecture on the multi-target scenario
of ``experiments.fusion_common`` (2 launches, 3 aircraft with datalinks, 1
fire; GEO + Molniya OPIR; 5-receiver RF network). No dataset is needed.

    python -m experiments.e5_tracking [--quick]
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import (
    MUTED,
    ROOT,
    SERIES,
    RunOptions,
    run_experiment,
    style,
)
from experiments.fusion_common import (
    ANGLE_STD,
    CUTOFF,
    FusionOptions,
    aggregate,
    run_fusion,
    score,
    simulate,
)
from sentinel.eval import write_report
from sentinel.fusion import associate_stereo
from sentinel.sim.opir.reports import event_reports

REPORT_DIR = ROOT / "reports" / "phase5"
CLUTTER_RATES = (0.0, 0.5, 2.0, 5.0)
CONFIRM_RULES = ((1, 1), (2, 3), (3, 5))


def _scenarios(quick: bool) -> list[Any]:
    seeds = range(1 if quick else 10)
    return [simulate(seed, duration_s=60.0 if quick else 180.0) for seed in seeds]


def track_management(scenarios: list[Any], quick: bool) -> dict[str, Any]:
    rates = (0.0, 2.0) if quick else CLUTTER_RATES
    rules = ((1, 1), (3, 5)) if quick else CONFIRM_RULES
    out: dict[str, Any] = {}
    for rate in rates:
        for rule in rules:
            runs = [
                score(
                    run_fusion(
                        result,
                        "centralized",
                        FusionOptions(clutter_rate=rate, confirm=rule, clutter_seed=i),
                    ).snapshots
                )
                for i, result in enumerate(scenarios)
            ]
            out[f"clutter_{rate:g}/{rule[0]}of{rule[1]}"] = {
                "clutter_rate": rate,
                "confirm": rule,
                **aggregate(runs),
            }
    return out


def stereo_pairing(scenarios: list[Any]) -> dict[str, Any]:
    """Audit GEO-HEO pairings against truth: how many pairs join rays from
    different targets (ghosts), with and without deferring ambiguous pairs."""
    out: dict[str, Any] = {}
    for reject in (False, True):
        correct = ghost = 0
        for result in scenarios:
            frame = result.config.origin.frame()
            scans = np.arange(0.0, result.config.duration_s, 1.0)
            reports = event_reports(
                frame, result.opir_by_sensor, result.times, scans, ANGLE_STD
            )
            for t in scans:
                a = [r for r in reports if r.time == t and r.sensor_index == 0]
                b = [r for r in reports if r.time == t and r.sensor_index == 1]
                for i, j in associate_stereo(
                    [(r.sensor_position, r.line_of_sight, r.angle_std) for r in a],
                    [(r.sensor_position, r.line_of_sight, r.angle_std) for r in b],
                    reject_ambiguous=reject,
                ):
                    if a[i].event_id == b[j].event_id:
                        correct += 1
                    else:
                        ghost += 1
        out["defer" if reject else "gnn"] = {
            "correct_pairs": correct,
            "ghost_pairs": ghost,
            "ghost_fraction": ghost / max(correct + ghost, 1),
        }
    out["correct_pairs_kept"] = out["defer"]["correct_pairs"] / max(
        out["gnn"]["correct_pairs"], 1
    )
    return out


def stereo_ambiguity(scenarios: list[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in ("opir", "centralized"):
        for reject in (False, True):
            runs = [
                score(
                    run_fusion(
                        result, mode, FusionOptions(reject_ambiguous=reject)
                    ).snapshots
                )
                for result in scenarios
            ]
            out[f"{mode}/{'defer' if reject else 'gnn'}"] = aggregate(runs)
    return out


def motion_model(scenarios: list[Any]) -> dict[str, Any]:
    variants = {
        "cv_q25": FusionOptions(imm=False, process_noise=25.0),
        "cv_q400": FusionOptions(imm=False, process_noise=400.0),
        "imm_q25_q1600": FusionOptions(imm=True),
    }
    return {
        name: aggregate(
            [
                score(run_fusion(result, "centralized", opts).snapshots)
                for result in scenarios
            ]
        )
        for name, opts in variants.items()
    }


def run(options: RunOptions) -> dict[str, Any]:
    quick = options.quick
    scenarios = _scenarios(quick)
    report = {
        "experiment": "E5 tracking",
        "quick": quick,
        "seeds": len(scenarios),
        "gospa_cutoff_m": CUTOFF,
        "track_management": track_management(scenarios, quick),
        "stereo_pairing": stereo_pairing(scenarios),
        "stereo_ambiguity": stereo_ambiguity(scenarios),
        "motion_model": motion_model(scenarios),
    }
    write_report(options.report_dir / "e5_tracking.json", report)
    if not quick:
        _figures(options.report_dir / "figures", report)
    return report


def _value(entry: dict[str, Any], *keys: str) -> float:
    for key in keys:
        entry = entry[key]
    return float(entry["value"])


def _figures(out: Path, report: dict[str, Any]) -> None:
    style()
    out.mkdir(parents=True, exist_ok=True)
    grid = report["track_management"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    panels = (
        ("false_per_scan", "false tracks per scan", ()),
        ("latency_by_type", "median confirmation latency, launches (s)", ("launch",)),
        ("gospa", "mean GOSPA (m)", ()),
    )
    for ax, (key, label, sub) in zip(axes, panels, strict=True):
        for color, rule in zip(SERIES, CONFIRM_RULES, strict=False):
            name = f"{rule[0]}of{rule[1]}"
            points = [
                (v["clutter_rate"], _value(v, key, *sub))
                for k, v in grid.items()
                if k.endswith(f"/{name}")
            ]
            if not points:
                continue
            x, y = zip(*points, strict=True)
            ax.plot(
                x,
                y,
                color=color,
                marker="o",
                markersize=7,
                linewidth=2.0,
                label=f"{rule[0]}-of-{rule[1]}",
            )
        ax.set_xlabel("OPIR false reports per sensor per scan")
        ax.set_ylabel(label)
    axes[0].legend(loc="upper left", title="confirmation")
    axes[0].set_title("Track management under clutter", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "e5_track_management.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ambiguity = report["stereo_ambiguity"]
    labels = ["OPIR only", "centralized"]
    x = np.arange(len(labels))
    for ax, key, ylabel in (
        (axes[0], "false_per_scan", "false tracks per scan"),
        (axes[1], "gospa", "mean GOSPA (m)"),
    ):
        for offset, (variant, color, name) in zip(
            (-0.18, 0.18),
            (("gnn", MUTED, "pair by GNN"), ("defer", SERIES[0], "defer ambiguous")),
            strict=True,
        ):
            values = [ambiguity[f"{m}/{variant}"][key] for m in ("opir", "centralized")]
            ax.bar(
                x + offset,
                [v["value"] for v in values],
                width=0.34,
                color=color,
                label=name,
                yerr=[
                    [v["value"] - v["ci_low"] for v in values],
                    [v["ci_high"] - v["value"] for v in values],
                ],
                error_kw={"ecolor": "#52514e", "elinewidth": 1.0, "capsize": 3},
            )
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Stereo ghost handling", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "e5_stereo_ambiguity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_experiment(
        "e5_tracking",
        run,
        (__doc__ or "e5_tracking").splitlines()[0],
        report_dir=REPORT_DIR,
    )
