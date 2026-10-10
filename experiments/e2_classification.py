"""E2: event classification, baselines vs. neural models, with robustness.

Models: handcrafted features + logistic regression, gradient-boosted trees,
1D CNN, and TCN. Deep models and GBM run over 5 seeds; ablations over 3.
Every run is evaluated on the in-distribution test split and the three
domain-shift splits. Model selection uses validation macro-F1 only.

    python -m experiments.e2_classification [--quick]
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import (
    EVAL_SPLITS,
    FIGURE_DIR,
    SERIES,
    SNR_BIN_LABELS,
    TEXT_PRIMARY,
    RunOptions,
    dataset_info,
    load,
    run_experiment,
    snr_bin,
    style,
)
from locant.classification import (
    FeatureClassifier,
    ModelArtifact,
    TrainConfig,
    preprocess,
    save_artifact,
    select_device,
    train_model,
)
from locant.classification.calibration import probabilities
from locant.classification.train import predict_logits
from locant.data.dataset import OPIRSplit
from locant.eval import (
    accuracy,
    bootstrap,
    confusion,
    macro_f1,
    per_class_f1,
    seed_summary,
    write_report,
)
from locant.taxonomy import BACKGROUND, EVENT_CLASSES

NUM_CLASSES = len(EVENT_CLASSES)
MAIN_MODELS = ("logistic", "gbm", "cnn", "tcn")


def experiment_grid(quick: bool) -> dict[str, dict[str, Any]]:
    """Name → {kind, seeds, train config (deep models)}."""
    epochs, seeds5, seeds3 = (
        (1, [0], [0]) if quick else (40, [0, 1, 2, 3, 4], [0, 1, 2])
    )
    deep = {"epochs": epochs, "patience": 8}
    grid: dict[str, dict[str, Any]] = {
        "logistic": {"kind": "logistic", "seeds": [0]},
        "gbm": {"kind": "gbm", "seeds": seeds5},
        "cnn": {"kind": "deep", "seeds": seeds5, "config": {"model": "cnn", **deep}},
        "tcn": {"kind": "deep", "seeds": seeds5, "config": {"model": "tcn", **deep}},
        "cnn_length100": {
            "kind": "deep",
            "seeds": seeds3,
            "config": {"model": "cnn", "preprocess": {"length": 100}, **deep},
        },
        "cnn_zscore": {
            "kind": "deep",
            "seeds": seeds3,
            "config": {
                "model": "cnn",
                "preprocess": {"normalization": "zscore"},
                **deep,
            },
        },
        "cnn_v1_preprocessing": {
            "kind": "deep",
            "seeds": seeds3,
            "config": {
                "model": "cnn",
                "preprocess": {"normalization": "zscore", "length": 100},
                **deep,
            },
        },
        "cnn_no_augment": {
            "kind": "deep",
            "seeds": seeds3,
            "config": {"model": "cnn", "augment": False, **deep},
        },
    }
    if quick:
        return {k: grid[k] for k in ("logistic", "gbm", "cnn", "tcn", "cnn_zscore")}
    return grid


def evaluate(
    probs: np.ndarray, split: OPIRSplit, rng: np.random.Generator
) -> dict[str, Any]:
    pred = probs.argmax(axis=1)
    labels = split.labels.astype(np.int64)
    event = labels != BACKGROUND
    bins = snr_bin(split.metadata["peak_snr"])
    return {
        "macro_f1": macro_f1(labels, pred),
        "accuracy": accuracy(labels, pred),
        "per_class_f1": dict(
            zip(EVENT_CLASSES, per_class_f1(labels, pred, NUM_CLASSES), strict=True)
        ),
        "event_recall_by_snr": {
            SNR_BIN_LABELS[b]: accuracy(
                labels[event & (bins == b)], pred[event & (bins == b)]
            )
            for b in range(len(SNR_BIN_LABELS))
            if np.any(event & (bins == b))
        },
        "macro_f1_bootstrap": bootstrap(
            macro_f1, labels, pred, rng, resamples=500
        ).as_dict(),
        "confusion": confusion(labels, pred, NUM_CLASSES),
    }


def run(options: RunOptions) -> dict[str, Any]:
    per_class = 30
    train = load(options, "train", per_class)
    validation = load(options, "validation", per_class)
    splits = {name: load(options, name, per_class) for name in EVAL_SPLITS}
    device = select_device(options.device)
    rng = np.random.default_rng(0)
    info = dataset_info(options)

    runs: dict[str, list[dict[str, Any]]] = {}
    for name, spec in experiment_grid(options.quick).items():
        runs[name] = []
        for seed in spec["seeds"]:
            if spec["kind"] == "deep":
                config = TrainConfig.model_validate({**spec["config"], "seed": seed})
                result = train_model(
                    config,
                    train.signals,
                    train.labels,
                    validation.signals,
                    validation.labels,
                    NUM_CLASSES,
                    device,
                )

                def predict(
                    s: OPIRSplit, cfg: TrainConfig = config, model: Any = result.model
                ) -> np.ndarray:
                    return probabilities(
                        predict_logits(
                            model, preprocess(s.signals, cfg.preprocess), device
                        )
                    )

                extra = {
                    "best_epoch": result.best_epoch,
                    "epochs_run": len(result.history["val_loss"]),
                }
                if name in ("cnn", "tcn"):
                    save_artifact(
                        options.model_dir / "e2" / f"{name}_seed{seed}",
                        ModelArtifact(
                            name=f"{name}_seed{seed}",
                            model=config.model,
                            model_kwargs=config.model_kwargs,
                            preprocess=config.preprocess,
                            dataset=info,
                            training=config.model_dump(mode="json"),
                        ),
                        result.model,
                    )
            else:
                baseline = FeatureClassifier(spec["kind"], seed).fit(
                    train.signals, train.labels
                )

                def predict(
                    s: OPIRSplit, model: FeatureClassifier = baseline
                ) -> np.ndarray:
                    return model.predict_proba(s.signals)

                extra = {}
            record: dict[str, Any] = {"seed": seed, **extra}
            record["validation"] = evaluate(predict(validation), validation, rng)
            for split_name, split in splits.items():
                record[split_name] = evaluate(predict(split), split, rng)
            runs[name].append(record)
            print(
                f"{name:22s} seed {seed}  val {record['validation']['macro_f1']:.3f}  "
                f"test {record['test']['macro_f1']:.3f}",
                flush=True,
            )

    summary = {
        name: {
            split: {
                metric: seed_summary([r[split][metric] for r in records]).as_dict()
                for metric in ("macro_f1", "accuracy")
            }
            for split in ("validation", *EVAL_SPLITS)
        }
        for name, records in runs.items()
    }
    candidates = [m for m in MAIN_MODELS if m in summary]
    selected = max(
        candidates, key=lambda m: summary[m]["validation"]["macro_f1"]["value"]
    )
    report = {
        "experiment": "E2 classification",
        "dataset": info,
        "quick": options.quick,
        "device": str(device),
        "grid": experiment_grid(options.quick),
        "selected_model": selected,
        "selection_rule": "highest mean validation macro-F1 among the main models",
        "summary": summary,
        "runs": runs,
    }
    write_report(options.report_dir / "e2_classification.json", report)
    if not options.quick:
        _figures(summary, runs, selected)
    return report


def _figures(summary: dict[str, Any], runs: dict[str, Any], selected: str) -> None:
    style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    models = [m for m in MAIN_MODELS if m in summary]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(EVAL_SPLITS))
    width = 0.18
    for i, (color, model) in enumerate(zip(SERIES, models, strict=False)):
        values = [summary[model][s]["macro_f1"] for s in EVAL_SPLITS]
        centers = x + (i - (len(models) - 1) / 2) * width
        ax.errorbar(
            centers,
            [v["value"] for v in values],
            yerr=[
                [v["value"] - v["ci_low"] for v in values],
                [v["ci_high"] - v["value"] for v in values],
            ],
            fmt="o",
            color=color,
            markersize=7,
            capsize=3,
            linewidth=1.5,
            label=model.upper() if model in ("cnn", "tcn", "gbm") else "Logistic",
        )
    ax.set_xticks(x, EVAL_SPLITS)
    ax.set_ylabel("macro-F1 (mean over seeds, 95% CI)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower left", ncol=4)
    ax.set_title("E2 macro-F1 by split", loc="left", color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "e2_macro_f1.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xb = np.arange(len(SNR_BIN_LABELS))
    for color, model in zip(SERIES, models, strict=False):
        means = [
            np.nanmean(
                [r["test"]["event_recall_by_snr"].get(b, np.nan) for r in runs[model]]
            )
            for b in SNR_BIN_LABELS
        ]
        ax.plot(
            xb,
            means,
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.0,
            label=model.upper(),
        )
    ax.set_xticks(xb, SNR_BIN_LABELS)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("peak SNR bin")
    ax.set_ylabel("event recall (test)")
    ax.legend(loc="lower right")
    ax.set_title("E2 recall of event classes vs SNR", loc="left", color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "e2_recall_vs_snr.png", dpi=150)
    plt.close(fig)

    matrix = np.asarray(runs[selected][0]["test"]["confusion"], dtype=np.float64)
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.grid(False)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j,
                i,
                f"{normalized[i, j]:.2f}",
                ha="center",
                va="center",
                color="#ffffff" if normalized[i, j] > 0.6 else TEXT_PRIMARY,
                fontsize=9,
            )
    ax.set_xticks(range(NUM_CLASSES), EVENT_CLASSES, rotation=30)
    ax.set_yticks(range(NUM_CLASSES), EVENT_CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(
        f"E2 confusion, {selected.upper()} seed 0, test", loc="left", color=TEXT_PRIMARY
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "e2_confusion.png", dpi=150)
    plt.close(fig)

    ablations = [
        "cnn",
        "cnn_no_augment",
        "cnn_zscore",
        "cnn_length100",
        "cnn_v1_preprocessing",
    ]
    ablations = [a for a in ablations if a in summary]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for row, name in enumerate(ablations):
        for color, split in zip(
            (SERIES[0], SERIES[1]), ("test", "shift_low_snr"), strict=True
        ):
            v = summary[name][split]["macro_f1"]
            ax.errorbar(
                v["value"],
                row,
                xerr=[[v["value"] - v["ci_low"]], [v["ci_high"] - v["value"]]],
                fmt="o",
                color=color,
                markersize=7,
                capsize=3,
                label=split if row == 0 else None,
            )
    ax.set_yticks(range(len(ablations)), ablations)
    ax.invert_yaxis()
    ax.set_xlabel("macro-F1 (mean over seeds, 95% CI)")
    ax.legend(loc="center left", bbox_to_anchor=(0.44, 0.5))
    ax.set_title("E2 CNN ablations", loc="left", color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "e2_ablations.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_experiment(
        "e2_classification", run, (__doc__ or "e2_classification").splitlines()[0]
    )
