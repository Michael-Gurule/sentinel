"""E6: what multi-INT fusion buys, by architecture, and track classification.

Questions:
1. Against the same truth set (every target any sensor can see), how do
   RF-only, OPIR-only, centralized measurement fusion, and track-to-track
   fusion (naive independence vs covariance intersection) compare on GOSPA,
   per-type accuracy and coverage, false/missed targets, and NEES?
2. Track classification: OPIR reports carry calibrated class probabilities
   from the exported classifier (E3), pooled on each track with a tempered
   log-linear rule. How do the pooling weight and the classifier's domain of
   validity (which detection ages it is applied to) affect time to a
   confident, correct label and the calibration of the track posterior?

    python -m experiments.e6_fusion [--quick]
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import (
    EXPORT_DIR,
    MUTED,
    ROOT,
    SERIES,
    TEXT_SECONDARY,
    RunOptions,
    parse_options,
    style,
)
from experiments.fusion_common import (
    CUTOFF,
    MODES,
    TARGET_TYPES,
    FusionOptions,
    RunOutput,
    aggregate,
    run_fusion,
    score,
    simulate,
    target_type,
    true_class,
)
from sentinel.classification import EventClassifier
from sentinel.eval import expected_calibration_error, gospa, write_report
from sentinel.sim import ScenarioResult

REPORT_DIR = ROOT / "reports" / "phase5"
MODE_LABELS = {
    "rf": "RF only",
    "opir": "OPIR only",
    "centralized": "centralized",
    "t2t_naive": "T2T naive",
    "t2t_ci": "T2T CI",
}
CLASS_WEIGHTS = (1.0, 0.3, 0.1)
AGE_POLICIES = {
    "all_ages": (0.0, float("inf")),
    "onset_in_window": (0.0, 62.0),
    "trained_range": (24.0, 62.0),
}
CONFIDENT = 0.9


def _scenarios(quick: bool) -> list[ScenarioResult]:
    seeds = range(1 if quick else 10)
    return [simulate(seed, duration_s=60.0 if quick else 180.0) for seed in seeds]


def architectures(scenarios: list[ScenarioResult]) -> dict[str, Any]:
    return {
        mode: aggregate([score(run_fusion(r, mode).snapshots) for r in scenarios])
        for mode in MODES
    }


def class_metrics(output: RunOutput) -> dict[str, Any]:
    """Track-classification metrics of one run.

    Time to confident classification is measured from the first scan at which
    the target is in the truth set to the first scan at which its matched
    track's posterior gives the true class probability ≥ 0.9.
    """
    first_seen: dict[str, float] = {}
    for snap in output.snapshots:
        for tid in snap.truth:
            first_seen.setdefault(tid, snap.time)
    probs, labels = [], []
    time_to_confident: dict[str, list[float]] = {k: [] for k in TARGET_TYPES}
    final_correct = []
    for tid, trace in output.class_trace.items():
        label = true_class(tid)
        scored = [(t, p) for t, p in trace if p is not None]
        for _, p in scored:
            probs.append(p)
            labels.append(label)
        confident = next(
            (t for t, p in scored if p.argmax() == label and p.max() >= CONFIDENT),
            None,
        )
        time_to_confident[target_type(tid)].append(
            confident - first_seen[tid] if confident is not None else float("inf")
        )
        final_correct.append(bool(scored) and scored[-1][1].argmax() == label)
    p = np.array(probs).reshape(-1, len(probs[0]) if probs else 1)
    y = np.array(labels, dtype=int)
    ece, _ = expected_calibration_error(p, y)
    onehot = np.eye(p.shape[1])[y]
    all_times = [v for values in time_to_confident.values() for v in values]
    return {
        "final_accuracy": float(np.mean(final_correct)),
        "scan_accuracy": float(np.mean(p.argmax(axis=1) == y)),
        "ece": ece,
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "mean_confidence": float(np.mean(p.max(axis=1))),
        "time_to_confident": _median(all_times),
        "never_confident": int(sum(not np.isfinite(v) for v in all_times)),
        "time_to_confident_by_type": {
            k: _median(v) for k, v in time_to_confident.items()
        },
    }


def _median(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def classification(
    scenarios: list[ScenarioResult], classifier: EventClassifier, quick: bool
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    policies = (
        {"onset_in_window": AGE_POLICIES["onset_in_window"]} if quick else AGE_POLICIES
    )
    for policy, ages in policies.items():
        for weight in (0.3,) if quick else CLASS_WEIGHTS:
            runs = [
                class_metrics(
                    run_fusion(
                        result,
                        "centralized",
                        FusionOptions(
                            classifier=classifier, class_weight=weight, class_age=ages
                        ),
                    )
                )
                for result in scenarios
            ]
            out[f"{policy}/w{weight:g}"] = {
                "class_age_s": ages,
                "class_weight": weight,
                **aggregate(runs),
            }
    return out


def run(options: RunOptions) -> dict[str, Any]:
    quick = options.quick
    scenarios = _scenarios(quick)
    classifier = EventClassifier.load(
        EXPORT_DIR / "opir_event_classifier", device=options.device or "cpu"
    )
    report = {
        "experiment": "E6 fusion",
        "quick": quick,
        "seeds": len(scenarios),
        "gospa_cutoff_m": CUTOFF,
        "architectures": architectures(scenarios),
        "classification": classification(scenarios, classifier, quick),
    }
    write_report(options.report_dir / "e6_fusion.json", report)
    if not quick:
        _figures(options.report_dir / "figures", report, scenarios[0])
    return report


def _bars(
    ax: Any,
    entries: list[dict[str, Any]],
    labels: list[str],
    colors: list[str],
) -> None:
    values = [e["value"] for e in entries]
    errors = [
        [e["value"] - e.get("ci_low", e["value"]) for e in entries],
        [e.get("ci_high", e["value"]) - e["value"] for e in entries],
    ]
    x = np.arange(len(entries))
    ax.bar(
        x,
        values,
        color=colors,
        width=0.7,
        yerr=errors,
        error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1.0, "capsize": 3},
    )
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.grid(axis="x", visible=False)


def _figures(out: Path, report: dict[str, Any], scenario: ScenarioResult) -> None:
    style()
    out.mkdir(parents=True, exist_ok=True)
    arch = report["architectures"]
    labels = [MODE_LABELS[m] for m in MODES]
    colors = [MUTED, MUTED, SERIES[0], SERIES[1], SERIES[2]]

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.8))
    _bars(axes[0], [arch[m]["gospa"] for m in MODES], labels, colors)
    axes[0].set_ylabel("mean GOSPA (m)")
    axes[0].set_title("Overall (lower is better)", loc="left", fontsize=10)
    for ax, kind in zip(axes[1:], TARGET_TYPES, strict=True):
        coverage = [arch[m]["coverage_by_type"][kind]["value"] for m in MODES]
        rmse = [arch[m]["rmse_by_type"][kind]["value"] for m in MODES]
        x = np.arange(len(MODES))
        ax.bar(x, coverage, color=colors, width=0.7)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("fraction of scans tracked")
        ax.set_title(f"{kind}: coverage, RMSE above bar", loc="left", fontsize=10)
        for xi, (c, r) in enumerate(zip(coverage, rmse, strict=True)):
            if np.isfinite(r) and c > 0:
                ax.annotate(
                    f"{r:.0f} m",
                    (xi, c),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=TEXT_SECONDARY,
                )
    fig.tight_layout()
    fig.savefig(out / "e6_architectures.png", dpi=150)
    plt.close(fig)

    cls = report["classification"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ax, key, ylabel in (
        (axes[0], "final_accuracy", "final track label accuracy"),
        (axes[1], "time_to_confident", "median time to confident label (s)"),
        (axes[2], "ece", "posterior ECE"),
    ):
        for color, weight in zip(SERIES, CLASS_WEIGHTS, strict=False):
            x = np.arange(len(AGE_POLICIES))
            y = [cls[f"{p}/w{weight:g}"][key]["value"] for p in AGE_POLICIES]
            ax.plot(
                x,
                y,
                color=color,
                marker="o",
                markersize=7,
                linewidth=2.0,
                label=f"w = {weight:g}",
            )
        ax.set_xticks(
            np.arange(len(AGE_POLICIES)),
            ["all ages", "0–62 s", "24–62 s"],
        )
        ax.set_xlabel("detection ages the classifier is applied to")
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="lower right", title="pooling weight")
    axes[0].set_title("Track classification", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "e6_classification.png", dpi=150)
    plt.close(fig)

    _scenario_figure(out, scenario)


def _scenario_figure(out: Path, scenario: ScenarioResult) -> None:
    """Top view of one scenario: truth paths and centralized track estimates."""
    output = run_fusion(scenario, "centralized")
    fig, ax = plt.subplots(figsize=(5.0, 7.6))
    colors = dict(zip(TARGET_TYPES, SERIES, strict=False))
    for tid, truth in scenario.truth.items():
        if tid.split("-")[0] not in colors:
            continue
        p = truth.position / 1_000.0
        ax.plot(p[:, 0], p[:, 1], color=MUTED, linewidth=1.0)
    plotted: set[str] = set()
    for snap in output.snapshots:
        truth_ids = list(snap.truth)
        truth_pos = np.array([snap.truth[t] for t in truth_ids]).reshape(-1, 3)
        pairs = {
            c: truth_ids[r]
            for r, c in gospa(truth_pos, snap.positions, CUTOFF).assignments
        }
        for c, position in enumerate(snap.positions):
            kind = target_type(pairs[c]) if c in pairs else "false"
            label = kind if kind not in plotted else None
            plotted.add(kind)
            ax.scatter(
                position[0] / 1_000.0,
                position[1] / 1_000.0,
                s=6,
                color=colors.get(kind, "#0b0b0b"),
                label=label,
                linewidths=0,
            )
    for receiver in scenario.config.rf.receivers if scenario.config.rf else []:
        ax.scatter(
            receiver.position_m[0] / 1_000.0,
            receiver.position_m[1] / 1_000.0,
            marker="^",
            s=30,
            color=TEXT_SECONDARY,
            label="RF receiver" if "rx" not in plotted else None,
        )
        plotted.add("rx")
    ax.set_aspect("equal")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        "Confirmed fused tracks over truth (grey)",
        loc="left",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "e6_scenario.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run(parse_options(__doc__.splitlines()[0], report_dir=REPORT_DIR))
