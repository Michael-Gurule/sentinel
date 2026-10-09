"""E1: window-level event detection with controlled false-alarm rate.

Questions:
1. Do analytic (white-noise) thresholds deliver their design false-alarm rate
   on realistic backgrounds?
2. At empirically calibrated thresholds, how do CFAR, CUSUM, and the step
   GLRT compare in Pd, overall and as a function of SNR?
3. How do the v1 heuristics compare?

Background windows for false-alarm estimation come from dedicated seed
streams (independent of every dataset split): one set to calibrate thresholds,
another to measure the resulting false-alarm rate.

    python -m experiments.e1_detection [--quick]
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import (
    FIGURE_DIR,
    MUTED,
    SERIES,
    SNR_BIN_LABELS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    RunOptions,
    dataset_info,
    load,
    parse_options,
    snr_bin,
    style,
)
from experiments.legacy.v1_detectors import MultiMethodDetector
from sentinel.data.build import generate_samples
from sentinel.data.config import Priors
from sentinel.detection import (
    CFARDetector,
    CUSUMDetector,
    StepGLRTDetector,
    calibrate_threshold,
)
from sentinel.eval import binomial_interval, detection_roc, write_report
from sentinel.taxonomy import BACKGROUND, EVENT_CLASSES

PFAS = (1e-2, 1e-3)
FS = 10.0


def _background(
    options: RunOptions, stream: str, count: int
) -> tuple[np.ndarray, np.ndarray]:
    info = dataset_info(options)
    signals, meta = generate_samples(
        info["root_seed"], stream, "background", count, Priors(), 64.0, options.workers
    )
    return signals.astype(np.float64), meta["glint_count"] > 0


def _rate(flags: np.ndarray) -> dict[str, float]:
    return binomial_interval(int(flags.sum()), int(flags.size)).as_dict()


def run(options: RunOptions) -> dict[str, Any]:
    n_background = 400 if options.quick else 20_000
    cal, cal_glint = _background(options, "e1_background_calibration", n_background)
    ev, ev_glint = _background(options, "e1_background_evaluation", n_background)
    test = load(options, "test", per_class=40)
    is_event = test.labels != BACKGROUND
    events = test.signals[is_event].astype(np.float64)
    event_labels = test.labels[is_event]
    event_snr = test.metadata["peak_snr"][is_event]
    bins = snr_bin(event_snr)

    detectors = [CFARDetector(), CUSUMDetector(), StepGLRTDetector()]
    pfas = (1e-1,) if options.quick else PFAS
    results: dict[str, Any] = {}
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for detector in detectors:
        s_cal = detector.score(cal, FS).score
        s_ev = detector.score(ev, FS).score
        s_evt = detector.score(events, FS).score
        all_pfa, all_pd, _ = detection_roc(s_ev, s_evt)
        clean_pfa, clean_pd, _ = detection_roc(s_ev[~ev_glint], s_evt)
        curves[detector.name] = {
            "all background": (all_pfa, all_pd),
            "glint-free background": (clean_pfa, clean_pd),
        }
        entry: dict[str, Any] = {}
        for pfa in pfas:
            analytic = detector.analytic_threshold(pfa, events.shape[1], FS)
            calibrated = calibrate_threshold(s_cal, pfa)
            calibrated_clean = calibrate_threshold(s_cal[~cal_glint], pfa)
            per_threshold = {}
            for name, threshold in (
                ("analytic", analytic),
                ("calibrated_all_background", calibrated),
                ("calibrated_glint_free", calibrated_clean),
            ):
                alarms = s_ev > threshold
                detected = s_evt > threshold
                per_threshold[name] = {
                    "threshold": threshold,
                    "pfa_all_background": _rate(alarms),
                    "pfa_glint_free": _rate(alarms[~ev_glint]),
                    "pd": _rate(detected),
                    "pd_by_class": {
                        EVENT_CLASSES[c]: _rate(detected[event_labels == c])
                        for c in np.unique(event_labels)
                    },
                    "pd_by_snr": {
                        SNR_BIN_LABELS[b]: _rate(detected[bins == b])
                        for b in range(len(SNR_BIN_LABELS))
                        if np.any(bins == b)
                    },
                }
            entry[f"pfa_{pfa:g}"] = per_threshold
        results[detector.name] = entry

    v1 = MultiMethodDetector()
    v1_alarms = np.array([v1.detect(x, FS).detected for x in ev[: min(len(ev), 2_000)]])
    v1_detected = np.array([v1.detect(x, FS).detected for x in events])
    results["v1_multi_method"] = {
        "note": "binary output; thresholds fixed in v1 temperature units",
        "pfa_all_background": _rate(v1_alarms),
        "pd": _rate(v1_detected),
    }

    report = {
        "experiment": "E1 detection",
        "dataset": dataset_info(options),
        "quick": options.quick,
        "background_windows": {"calibration": len(cal), "evaluation": len(ev)},
        "glint_fraction_evaluation": float(ev_glint.mean()),
        "event_windows": len(events),
        "results": results,
    }
    write_report(options.report_dir / "e1_detection.json", report)
    if not options.quick:
        _figures(curves, results)
    return report


def _figures(
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], results: dict[str, Any]
) -> None:
    style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    names = list(curves)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for ax, population in zip(
        axes, ("all background", "glint-free background"), strict=True
    ):
        for color, name in zip(SERIES, names, strict=False):
            pfa, pd = curves[name][population]
            keep = pfa > 0
            ax.plot(pfa[keep], pd[keep], color=color, linewidth=2.0, label=name.upper())
        for pfa in PFAS:
            ax.axvline(pfa, color=MUTED, linewidth=1.0, linestyle="--")
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel(f"false-alarm probability per window ({population})")
        ax.set_title(population, loc="left", fontsize=10)
    v1 = results["v1_multi_method"]
    axes[0].plot(
        v1["pfa_all_background"]["value"],
        v1["pd"]["value"],
        "o",
        color=MUTED,
        markersize=8,
    )
    axes[0].annotate(
        "v1 ensemble (alarms on every window)",
        (v1["pfa_all_background"]["value"], v1["pd"]["value"]),
        xytext=(-10, -28),
        textcoords="offset points",
        ha="right",
        color=TEXT_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": MUTED},
    )
    axes[0].set_ylabel("detection probability (test events)")
    axes[1].legend(loc="lower right")
    fig.suptitle("E1 detection ROC", x=0.01, ha="left", color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "e1_roc.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(SNR_BIN_LABELS))
    for color, name in zip(SERIES, names, strict=False):
        table = results[name]["pfa_0.01"]["calibrated_glint_free"]["pd_by_snr"]
        values = [
            table[label]["value"] if label in table else np.nan
            for label in SNR_BIN_LABELS
        ]
        ax.plot(
            x,
            values,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=7,
            label=name.upper(),
        )
    ax.set_xticks(x, SNR_BIN_LABELS)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("peak SNR bin")
    ax.set_ylabel("detection probability")
    ax.legend(loc="lower right")
    ax.set_title(
        "E1 Pd vs SNR, threshold for Pfa = 1e-2 on glint-free background",
        loc="left",
        color=TEXT_PRIMARY,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "e1_pd_vs_snr.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run(parse_options(__doc__ or "E1"))
