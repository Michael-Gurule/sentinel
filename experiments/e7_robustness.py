"""E7: robustness of the fused picture to sensor loss, latency, and bias.

Questions:
1. HEO dropout: with one OPIR satellite down, stereo is lost. How does the
   track picture degrade, and does centralized fusion hold up better?
2. RF outage: do OPIR updates keep aircraft tracks alive while the RF network
   is down?
3. RF latency: RF fixes arrive late, after newer OPIR scans. Drop them (out
   of sequence), buffer every measurement and predict the picture forward
   (in sequence, but always late), or extrapolate each late fix to the
   current time with its own FDOA velocity?
4. Time-correlated RF bias: clock bias and survey error are fixed for a
   scenario. Fixes solved with the consider covariance are consistent one at
   a time (E4); is the *track* still consistent after it has averaged many
   fixes from the same biased network? Does adding the systematic part of the
   fix covariance back as a floor restore consistency?

    python -m experiments.e7_robustness [--quick]
"""

from itertools import pairwise
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
    CUTOFF,
    FusionOptions,
    Mode,
    aggregate,
    gospa_series,
    run_fusion,
    score,
    simulate,
    window,
)
from sentinel.core import mean_nees_bounds
from sentinel.eval import write_report
from sentinel.sim import ScenarioResult

REPORT_DIR = ROOT / "reports" / "phase5"
DELAYS_S = (0.0, 1.0, 2.0, 5.0)
BIAS_LEVELS = {"5ns_2m": (5.0, 2.0), "30ns_10m": (30.0, 10.0)}
AGE_BINS_S = (0.0, 10.0, 30.0, 60.0, 120.0, 180.0)


def _outage(duration: float) -> tuple[float, float]:
    return duration / 3.0, 2.0 * duration / 3.0


def _sweep(
    scenarios: list[ScenarioResult],
    modes: tuple[Mode, ...],
    options: dict[str, FusionOptions],
    interval: tuple[float, float],
) -> dict[str, Any]:
    """Score each mode × option set over the whole run and within ``interval``,
    and keep the seed-mean per-scan GOSPA for plotting."""
    out: dict[str, Any] = {}
    for mode in modes:
        for name, opts in options.items():
            outputs = [run_fusion(r, mode, opts).snapshots for r in scenarios]
            out[f"{mode}/{name}"] = {
                "overall": aggregate([score(s) for s in outputs]),
                "during": aggregate([score(window(s, *interval)) for s in outputs]),
                "after": aggregate(
                    [score(window(s, interval[1], np.inf)) for s in outputs]
                ),
                "gospa_series": np.mean([gospa_series(s) for s in outputs], axis=0),
            }
    return out


def opir_dropout(
    scenarios: list[ScenarioResult], interval: tuple[float, float]
) -> dict[str, Any]:
    return _sweep(
        scenarios,
        ("opir", "centralized"),
        {
            "nominal": FusionOptions(),
            "heo_down": FusionOptions(opir_outage=(1, *interval)),
        },
        interval,
    )


def rf_outage(
    scenarios: list[ScenarioResult], interval: tuple[float, float]
) -> dict[str, Any]:
    return _sweep(
        scenarios,
        ("rf", "centralized", "t2t_ci"),
        {"nominal": FusionOptions(), "rf_down": FusionOptions(rf_outage=interval)},
        interval,
    )


def rf_latency(scenarios: list[ScenarioResult], quick: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for delay in (0.0, 2.0) if quick else DELAYS_S:
        strategies = ("drop",) if delay == 0 else ("drop", "buffer", "extrapolate")
        for strategy in strategies:
            opts = FusionOptions(rf_delay_s=delay, delay_strategy=strategy)  # type: ignore[arg-type]
            runs = [
                score(run_fusion(r, "centralized", opts).snapshots) for r in scenarios
            ]
            out[f"{delay:g}s/{strategy}"] = {
                "delay_s": delay,
                "strategy": strategy,
                **aggregate(runs),
            }
    return out


def rf_bias(seeds: range, duration: float) -> dict[str, Any]:
    """NEES of RF-updated aircraft tracks vs track age (all records pooled),
    under the filter covariance and under the reported covariance (filter +
    the library's bias floor, present only for consider-covariance fixes)."""
    out: dict[str, Any] = {}
    low, high = mean_nees_bounds(3, 1_000)
    for level, (bias_ns, survey_m) in BIAS_LEVELS.items():
        scenarios = [
            simulate(
                seed, duration_s=duration, clock_bias_ns=bias_ns, survey_m=survey_m
            )
            for seed in seeds
        ]
        for consider in (False, True):
            records = np.array(
                [
                    row
                    for r in scenarios
                    for row in run_fusion(
                        r, "rf", FusionOptions(use_systematic=consider)
                    ).track_ages
                ]
            ).reshape(-1, 4)
            bins = []
            for lo, hi in pairwise(AGE_BINS_S):
                mask = (records[:, 1] >= lo) & (records[:, 1] < hi)
                if not mask.any():
                    continue
                bins.append(
                    {
                        "age_s": (lo, hi),
                        "records": int(mask.sum()),
                        "nees": float(np.mean(records[mask, 2])),
                        "nees_reported": float(np.mean(records[mask, 3])),
                    }
                )
            out[f"{level}/{'consider' if consider else 'naive'}"] = {
                "clock_bias_ns": bias_ns,
                "survey_m": survey_m,
                "consider_fixes": consider,
                "nees_mean": float(np.mean(records[:, 2])),
                "nees_reported_mean": float(np.mean(records[:, 3])),
                "by_age": bins,
            }
    out["nees_reference"] = {"dof": 3, "mean": 3.0, "single_record_95": (low, high)}
    return out


def run(options: RunOptions) -> dict[str, Any]:
    quick = options.quick
    seeds = range(1 if quick else 10)
    duration = 60.0 if quick else 180.0
    scenarios = [simulate(seed, duration_s=duration) for seed in seeds]
    interval = _outage(duration)
    report = {
        "experiment": "E7 robustness",
        "quick": quick,
        "seeds": len(scenarios),
        "gospa_cutoff_m": CUTOFF,
        "outage_interval_s": interval,
        "opir_dropout": opir_dropout(scenarios, interval),
        "rf_outage": rf_outage(scenarios, interval),
        "rf_latency": rf_latency(scenarios, quick),
        "rf_bias": rf_bias(seeds, duration),
    }
    write_report(options.report_dir / "e7_robustness.json", report)
    if not quick:
        _figures(options.report_dir / "figures", report)
    return report


def _series_panel(
    ax: Any,
    section: dict[str, Any],
    lines: tuple[tuple[str, str, str, str], ...],
    interval: tuple[float, float],
) -> None:
    ax.axvspan(*interval, color=MUTED, alpha=0.18, linewidth=0)
    for key, color, style_, label in lines:
        series = np.asarray(section[key]["gospa_series"])
        ax.plot(
            np.arange(series.size),
            series,
            color=color,
            linestyle=style_,
            linewidth=1.6,
            label=label,
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("GOSPA (m), mean over seeds")


def _figures(out: Path, report: dict[str, Any]) -> None:
    style()
    out.mkdir(parents=True, exist_ok=True)
    interval = tuple(report["outage_interval_s"])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), sharey=True)
    _series_panel(
        axes[0],
        report["opir_dropout"],
        (
            ("opir/nominal", SERIES[1], ":", "OPIR only, nominal"),
            ("opir/heo_down", SERIES[1], "-", "OPIR only, HEO down"),
            ("centralized/nominal", SERIES[0], ":", "centralized, nominal"),
            ("centralized/heo_down", SERIES[0], "-", "centralized, HEO down"),
        ),
        interval,  # type: ignore[arg-type]
    )
    axes[0].set_title("HEO satellite outage (shaded)", loc="left", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8)
    _series_panel(
        axes[1],
        report["rf_outage"],
        (
            ("rf/rf_down", MUTED, "-", "RF only, RF down"),
            ("centralized/nominal", SERIES[0], ":", "centralized, nominal"),
            ("centralized/rf_down", SERIES[0], "-", "centralized, RF down"),
            ("t2t_ci/rf_down", SERIES[2], "-", "T2T CI, RF down"),
        ),
        interval,  # type: ignore[arg-type]
    )
    axes[1].set_title("RF network outage (shaded)", loc="left", fontsize=10)
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "e7_outages.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    latency = report["rf_latency"]
    ax = axes[0]
    for color, strategy, label in (
        (SERIES[3], "drop", "drop late RF fixes"),
        (SERIES[1], "buffer", "buffer, predict forward"),
        (SERIES[0], "extrapolate", "extrapolate late fixes"),
    ):
        points = [
            (v["delay_s"], v["gospa"]["value"])
            for v in latency.values()
            if v["strategy"] == strategy or v["delay_s"] == 0
        ]
        x, y = zip(*sorted(points), strict=True)
        ax.plot(x, y, color=color, marker="o", markersize=7, linewidth=2.0, label=label)
    ax.set_xlabel("RF latency (s)")
    ax.set_ylabel("mean GOSPA (m)")
    ax.set_title("Late RF fixes", loc="left", fontsize=10)
    ax.legend(loc="upper left")

    ax = axes[1]
    bias = report["rf_bias"]
    ax.axhline(
        3.0, color=MUTED, linewidth=1.0, linestyle=":", label="consistent (E[NEES] = 3)"
    )
    for color, key, field, label in (
        (MUTED, "30ns_10m/naive", "nees", "naive fixes"),
        (SERIES[1], "30ns_10m/consider", "nees", "consider fixes"),
        (
            SERIES[0],
            "30ns_10m/consider",
            "nees_reported",
            "consider, reported (+ bias floor)",
        ),
    ):
        entry = bias[key]["by_age"]
        x = [0.5 * (b["age_s"][0] + b["age_s"][1]) for b in entry]
        ax.plot(
            x,
            [b[field] for b in entry],
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.0,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("track age (s)")
    ax.set_ylabel("mean position NEES")
    ax.set_title(
        "RF track consistency, 30 ns clock bias + 10 m survey error",
        loc="left",
        fontsize=10,
    )
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(out / "e7_latency_bias.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_experiment(
        "e7_robustness",
        run,
        (__doc__ or "e7_robustness").splitlines()[0],
        report_dir=REPORT_DIR,
    )
