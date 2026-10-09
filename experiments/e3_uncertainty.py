"""E3: calibration, conformal prediction sets, two-stage alarms, and OOD detection.

Uses the architecture E2 selected (best mean validation macro-F1 among the
neural models) and its per-seed artifacts.

1. Temperature scaling on validation; ECE before/after on every split.
2. Split-conformal sets (α = 0.1) calibrated on validation, LAC (shipped)
   and non-randomized APS (comparison); coverage and
   set size on test (exchangeable) and the shift splits (not exchangeable).
3. Two-stage system: CFAR at its glint-free calibrated threshold, then the
   classifier rejects windows it labels background. System false-alarm rate
   on E1's evaluation background windows and system Pd on test events.
4. Out-of-distribution detection by leave-one-event-class-out retraining:
   AUROC of energy and max-softmax scores, held-out class vs. known classes.
5. Export the selected seed with temperature and conformal threshold.

    python -m experiments.e3_uncertainty [--quick]
"""

import json
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import (
    EVAL_SPLITS,
    EXPORT_DIR,
    FIGURE_DIR,
    MUTED,
    SERIES,
    TEXT_PRIMARY,
    RunOptions,
    dataset_info,
    load,
    run_experiment,
    style,
)
from sentinel.classification import (
    ModelArtifact,
    TrainConfig,
    load_artifact,
    preprocess,
    save_artifact,
    select_device,
    train_model,
)
from sentinel.classification.artifact import ConformalSpec
from sentinel.classification.calibration import (
    energy_score,
    fit_temperature,
    max_softmax,
    nll,
    probabilities,
)
from sentinel.classification.conformal import calibrate, coverage, prediction_sets
from sentinel.classification.train import predict_logits
from sentinel.data.build import generate_samples
from sentinel.data.config import Priors
from sentinel.detection import CFARDetector
from sentinel.eval import (
    auroc,
    binomial_interval,
    expected_calibration_error,
    macro_f1,
    seed_summary,
    write_report,
)
from sentinel.taxonomy import BACKGROUND, EVENT_CLASSES

ALPHA = 0.1
MIN_BIN_COUNT = 20
"""Reliability-diagram bins with fewer windows are not drawn (too noisy)."""
FS = 10.0


def _selected(options: RunOptions) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    e2 = json.loads((options.report_dir / "e2_classification.json").read_text())
    deep = [m for m in ("cnn", "tcn") if m in e2["summary"]]
    arch = max(deep, key=lambda m: e2["summary"][m]["validation"]["macro_f1"]["value"])
    return arch, e2["runs"][arch], e2


def run(options: RunOptions) -> dict[str, Any]:
    per_class = 30
    device = select_device(options.device)
    arch, e2_runs, e2 = _selected(options)
    validation = load(options, "validation", per_class)
    splits = {name: load(options, name, per_class) for name in EVAL_SPLITS}
    num_classes = len(EVENT_CLASSES)

    seeds: list[dict[str, Any]] = []
    for record in e2_runs:
        seed = record["seed"]
        artifact, model = load_artifact(
            options.model_dir / "e2" / f"{arch}_seed{seed}", device
        )

        def logits_for(
            signals: np.ndarray, a: ModelArtifact = artifact, m: Any = model
        ) -> np.ndarray:
            return predict_logits(m, preprocess(signals, a.preprocess), device)

        val_logits = logits_for(validation.signals)
        temperature = fit_temperature(val_logits, validation.labels)
        val_probs = probabilities(val_logits, temperature)
        threshold = calibrate(val_probs, validation.labels, ALPHA, "lac")
        aps_threshold = calibrate(val_probs, validation.labels, ALPHA, "aps")
        entry: dict[str, Any] = {
            "seed": seed,
            "temperature": temperature,
            "lac_threshold": threshold,
            "aps_threshold": aps_threshold,
            "validation_nll": {
                "before": nll(val_logits, validation.labels),
                "after": nll(val_logits, validation.labels, temperature),
            },
        }
        for name, split in splits.items():
            logits = logits_for(split.signals)
            before, _ = expected_calibration_error(probabilities(logits), split.labels)
            probs = probabilities(logits, temperature)
            after, table = expected_calibration_error(probs, split.labels)
            sets = prediction_sets(probs, threshold, "lac")
            aps_sets = prediction_sets(probs, aps_threshold, "aps")
            entry[name] = {
                "ece_before": before,
                "ece_after": after,
                "reliability_after": table,
                "reliability_before": expected_calibration_error(
                    probabilities(logits), split.labels
                )[1],
                "coverage": coverage(sets, split.labels),
                "mean_set_size": float(sets.sum(axis=1).mean()),
                "singleton_fraction": float(np.mean(sets.sum(axis=1) == 1)),
                "aps_coverage": coverage(aps_sets, split.labels),
                "aps_mean_set_size": float(aps_sets.sum(axis=1).mean()),
                "class_coverage": {
                    EVENT_CLASSES[c]: coverage(
                        sets[split.labels == c], split.labels[split.labels == c]
                    )
                    for c in range(num_classes)
                    if np.any(split.labels == c)
                },
                "macro_f1": macro_f1(split.labels, probs.argmax(axis=1)),
            }
        seeds.append(entry)
        print(
            f"seed {seed}: T={temperature:.3f} test ECE {entry['test']['ece_before']:.3f}->{entry['test']['ece_after']:.3f} coverage {entry['test']['coverage']:.3f}",
            flush=True,
        )

    summary = {
        name: {
            metric: seed_summary([s[name][metric] for s in seeds]).as_dict()
            for metric in (
                "ece_before",
                "ece_after",
                "coverage",
                "mean_set_size",
                "singleton_fraction",
                "aps_coverage",
                "aps_mean_set_size",
            )
        }
        for name in EVAL_SPLITS
    }

    best = max(e2_runs, key=lambda r: r["validation"]["macro_f1"])
    best_entry = next(s for s in seeds if s["seed"] == best["seed"])
    artifact, model = load_artifact(
        options.model_dir / "e2" / f"{arch}_seed{best['seed']}", device
    )
    two_stage = _two_stage(
        options, artifact, model, best_entry["temperature"], splits["test"], device
    )
    ood = _ood(options, arch, e2, device)

    exported = artifact.model_copy(
        update={
            "name": "opir_event_classifier",
            "temperature": best_entry["temperature"],
            "conformal": ConformalSpec(
                method="lac", alpha=ALPHA, threshold=best_entry["lac_threshold"]
            ),
            "metrics": {
                "selected_seed": best["seed"],
                "validation_macro_f1": best["validation"]["macro_f1"],
                "test_macro_f1": best["test"]["macro_f1"],
                "test_ece_after_temperature": best_entry["test"]["ece_after"],
                "test_conformal_coverage": best_entry["test"]["coverage"],
                "shift_macro_f1": {
                    s: best[s]["macro_f1"] for s in EVAL_SPLITS if s != "test"
                },
            },
        }
    )
    export_dir = (
        options.model_dir / "export" if options.quick else EXPORT_DIR
    ) / "opir_event_classifier"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    save_artifact(export_dir, exported, model)

    report = {
        "experiment": "E3 uncertainty",
        "dataset": dataset_info(options),
        "quick": options.quick,
        "device": str(device),
        "architecture": arch,
        "alpha": ALPHA,
        "summary": summary,
        "seeds": seeds,
        "two_stage": two_stage,
        "ood": ood,
        "exported_seed": best["seed"],
    }
    write_report(options.report_dir / "e3_uncertainty.json", report)
    if not options.quick:
        _figures(best_entry, summary)
    return report


def _two_stage(
    options: RunOptions,
    artifact: ModelArtifact,
    model: Any,
    temperature: float,
    test: Any,
    device: Any,
) -> dict[str, Any]:
    e1 = json.loads((options.report_dir / "e1_detection.json").read_text())
    key = "pfa_0.1" if options.quick else "pfa_0.01"
    threshold = e1["results"]["cfar"][key]["calibrated_glint_free"]["threshold"]
    count = e1["background_windows"]["evaluation"]
    info = dataset_info(options)
    background, meta = generate_samples(
        info["root_seed"],
        "e1_background_evaluation",
        "background",
        count,
        Priors(),
        64.0,
        options.workers,
    )
    glint = meta["glint_count"] > 0
    detector = CFARDetector()

    def classify(signals: np.ndarray) -> np.ndarray:
        logits = predict_logits(model, preprocess(signals, artifact.preprocess), device)
        return np.asarray(probabilities(logits, temperature).argmax(axis=1))

    bg_alarm = detector.score(background, FS).score > threshold
    bg_system = bg_alarm.copy()
    bg_system[bg_alarm] = classify(background[bg_alarm]) != BACKGROUND
    events = test.labels != BACKGROUND
    ev_detect = detector.score(test.signals[events], FS).score > threshold
    ev_pred = classify(test.signals[events])
    ev_system = ev_detect & (ev_pred != BACKGROUND)
    ev_correct = ev_detect & (ev_pred == test.labels[events])

    def rate(flags: np.ndarray) -> dict[str, float]:
        return binomial_interval(int(flags.sum()), int(flags.size)).as_dict()

    return {
        "cfar_threshold": threshold,
        "cfar_threshold_source": f"E1 cfar {key} calibrated_glint_free",
        "detector_only": {
            "pfa_all_background": rate(bg_alarm),
            "pfa_glint_background": rate(bg_alarm[glint]),
            "pd": rate(ev_detect),
        },
        "detector_plus_classifier": {
            "pfa_all_background": rate(bg_system),
            "pfa_glint_background": rate(bg_system[glint]),
            "pd": rate(ev_system),
            "pd_and_correct_class": rate(ev_correct),
        },
    }


def _ood(
    options: RunOptions, arch: str, e2: dict[str, Any], device: Any
) -> dict[str, Any]:
    per_class = 30
    train = load(options, "train", per_class)
    validation = load(options, "validation", per_class)
    test = load(options, "test", per_class)
    base = e2["grid"][arch]["config"]
    held_out_classes = (
        ("launch",) if options.quick else ("launch", "explosion", "fire", "aircraft")
    )
    results: dict[str, Any] = {}
    for held in held_out_classes:
        h = EVENT_CLASSES.index(held)
        known = [c for c in range(len(EVENT_CLASSES)) if c != h]
        remap = np.full(len(EVENT_CLASSES), -1)
        remap[known] = np.arange(len(known))
        keep_tr, keep_va = train.labels != h, validation.labels != h
        config = TrainConfig.model_validate({**base, "seed": 0})
        result = train_model(
            config,
            train.signals[keep_tr],
            remap[train.labels[keep_tr]],
            validation.signals[keep_va],
            remap[validation.labels[keep_va]],
            len(known),
            device,
        )
        val_logits = predict_logits(
            result.model,
            preprocess(validation.signals[keep_va], config.preprocess),
            device,
        )
        temperature = fit_temperature(val_logits, remap[validation.labels[keep_va]])
        logits = predict_logits(
            result.model, preprocess(test.signals, config.preprocess), device
        )
        is_ood = test.labels == h
        energy = energy_score(logits, temperature)
        msp = max_softmax(logits, temperature)
        results[held] = {
            "auroc_energy": auroc(energy[~is_ood], energy[is_ood]),
            "auroc_max_softmax": auroc(msp[~is_ood], msp[is_ood]),
            "known_class_macro_f1": macro_f1(
                remap[test.labels[~is_ood]], logits[~is_ood].argmax(axis=1)
            ),
        }
        print(
            f"OOD held-out {held}: AUROC energy {results[held]['auroc_energy']:.3f} MSP {results[held]['auroc_max_softmax']:.3f}",
            flush=True,
        )
    return results


def _figures(entry: dict[str, Any], summary: dict[str, Any]) -> None:
    style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, linestyle="--")
    for color, key, label in (
        (SERIES[1], "reliability_before", "before scaling"),
        (SERIES[0], "reliability_after", "after temperature scaling"),
    ):
        table = entry["test"][key]
        keep = np.asarray(table["bin_count"]) >= MIN_BIN_COUNT
        ax.plot(
            np.asarray(table["bin_confidence"])[keep],
            np.asarray(table["bin_accuracy"])[keep],
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.0,
            label=label,
        )
    ax.set_xlabel(f"confidence (bins with ≥ {MIN_BIN_COUNT} windows)")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="upper left")
    ax.set_title(
        f"E3 reliability, test (ECE {entry['test']['ece_before']:.3f} → {entry['test']['ece_after']:.3f})",
        loc="left",
        color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "e3_reliability.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    x = np.arange(len(EVAL_SPLITS))
    for ax, metric, ylabel in (
        (axes[0], "coverage", "coverage (target 0.90)"),
        (axes[1], "mean_set_size", "mean set size (of 5 classes)"),
    ):
        for offset, (color, prefix, label) in zip(
            (-0.08, 0.08),
            ((SERIES[0], "", "LAC (shipped)"), (SERIES[1], "aps_", "APS")),
            strict=True,
        ):
            values = [summary[s][prefix + metric] for s in EVAL_SPLITS]
            ax.errorbar(
                x + offset,
                [v["value"] for v in values],
                yerr=[
                    [v["value"] - v["ci_low"] for v in values],
                    [v["ci_high"] - v["value"] for v in values],
                ],
                fmt="o",
                color=color,
                markersize=8,
                capsize=3,
                label=label,
            )
        if metric == "coverage":
            ax.axhline(1.0 - ALPHA, color=MUTED, linestyle="--", linewidth=1.0)
            ax.set_ylim(0.6, 1.02)
        else:
            ax.set_ylim(0.0, 5.0)
        ax.set_xticks(x, EVAL_SPLITS, rotation=20)
        ax.set_ylabel(ylabel)
    axes[1].legend(loc="lower right")
    fig.suptitle(
        "E3 conformal prediction sets, α = 0.1", x=0.01, ha="left", color=TEXT_PRIMARY
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "e3_conformal.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_experiment("e3_uncertainty", run, (__doc__ or "e3_uncertainty").splitlines()[0])
