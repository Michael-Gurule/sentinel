"""One source for every number quoted in the docs.

``collect`` reads the experiment reports (E1–E7, under ``reports/``) and
gathers the headline metrics, with their confidence intervals, into
``reports/metrics.json``. Markdown tables are rendered from that file into
the README and the docs, between markers::

    <!-- metrics:fusion_architectures -->
    ...generated table...
    <!-- /metrics:fusion_architectures -->

    python -m experiments.metrics          # regenerate metrics.json and every table
    python -m experiments.metrics --check  # exit 1 if anything is stale (CI, tests)

A test runs the check, so a rerun experiment that changes a number cannot
leave a stale table behind.
"""

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METRICS = Path("reports/metrics.json")
REPORTS = {
    "e1": "reports/phase3/e1_detection.json",
    "e2": "reports/phase3/e2_classification.json",
    "e3": "reports/phase3/e3_uncertainty.json",
    "e4": "reports/phase4/e4_geolocation.json",
    "e5": "reports/phase5/e5_tracking.json",
    "e6": "reports/phase5/e6_fusion.json",
    "e7": "reports/phase5/e7_robustness.json",
}
DOCUMENTS = (
    "README.md",
    "docs/technical_report.md",
    "docs/fusion.md",
    "docs/geolocation.md",
)
SHIFT_SPLITS = ("shift_low_snr", "shift_params", "shift_clutter")
MODELS = {
    "logistic": "Features + logistic regression",
    "gbm": "Features + gradient-boosted trees",
    "cnn": "1D CNN",
    "tcn": "TCN",
}
ARCHITECTURES = {
    "rf": "RF only",
    "opir": "OPIR only",
    "centralized": "Centralized",
    "t2t_naive": "T2T naive",
    "t2t_ci": "T2T CI",
}

Metric = dict[str, Any]


# -- collection ---------------------------------------------------------------


def _estimate(entry: Any) -> Metric:
    """``{"value", "ci_low", "ci_high"}`` from a report entry (CI optional)."""
    if isinstance(entry, dict):
        out = {"value": _number(entry["value"])}
        if "ci_low" in entry:
            out["ci_low"] = _number(entry["ci_low"])
            out["ci_high"] = _number(entry["ci_high"])
        return out
    return {"value": _number(entry)}


def _number(value: Any) -> float | None:
    if isinstance(value, str):  # reports write NaN/inf as strings
        return None if value.lower() == "nan" else float(value)
    return float(value)


def _detection(e1: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for detector in ("cfar", "cusum", "glrt"):
        at_1pct = e1["results"][detector]["pfa_0.01"]
        calibrated = at_1pct["calibrated_glint_free"]
        out[detector] = {
            "pd_at_pfa_1pct": _estimate(calibrated["pd"]),
            "pd_by_class": {
                k: _estimate(v) for k, v in calibrated["pd_by_class"].items()
            },
            "analytic_threshold_pfa_realistic": _estimate(
                at_1pct["analytic"]["pfa_all_background"]
            ),
        }
    v1 = e1["results"]["v1_multi_method"]
    out["v1_ensemble"] = {
        "pd": _estimate(v1["pd"]),
        "pfa_realistic": _estimate(v1["pfa_all_background"]),
    }
    out["background_windows"] = e1["background_windows"]
    return out


def _classification(e2: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected": e2["selected_model"],
        "macro_f1": {
            model: {
                split: _estimate(e2["summary"][model][split]["macro_f1"])
                for split in ("test", *SHIFT_SPLITS)
            }
            for model in MODELS
        },
    }


def _uncertainty(e3: dict[str, Any]) -> dict[str, Any]:
    summary = e3["summary"]
    stages = e3["two_stage"]
    return {
        "architecture": e3["architecture"],
        "alpha": e3["alpha"],
        "conformal": {
            split: {
                "coverage": _estimate(summary[split]["coverage"]),
                "mean_set_size": _estimate(summary[split]["mean_set_size"]),
                "ece_before": _estimate(summary[split]["ece_before"]),
                "ece_after": _estimate(summary[split]["ece_after"]),
            }
            for split in ("test", *SHIFT_SPLITS)
        },
        "two_stage": {
            stage: {k: _estimate(v) for k, v in stages[stage].items()}
            for stage in ("detector_only", "detector_plus_classifier")
        },
        "ood_auroc_energy": {
            k: _estimate(v["auroc_energy"]) for k, v in e3["ood"].items()
        },
    }


def _geolocation(e4: dict[str, Any]) -> dict[str, Any]:
    sweep = sorted(e4["noise_sweep"].values(), key=lambda v: v["toa_std_ns"])
    return {
        "noise_sweep": [
            {
                "toa_std_ns": v["toa_std_ns"],
                "crlb_rms_m": _number(v["crlb_rms"]),
                "ml_rmse_m": _number(v["ml"]["rmse"]),
                "ml_nees_mean": _number(v["ml"]["nees_mean"]),
                "chan_ho_rmse_m": _number(v["chan_ho"]["rmse"]),
                "nees_95_bounds": v["ml"]["nees_95_bounds"],
            }
            for v in sweep
        ],
        "systematics": [
            {
                "clock_bias_ns": _number(row["clock_bias_ns"]),
                "survey_error_m": _number(row["survey_error_m"]),
                "rmse_m": _number(row["rmse"]),
                "reported_rms_naive_m": _number(row["reported_rms_naive"]),
                "nees_naive_mean": _number(row["nees_naive_mean"]),
                "nees_consider_mean": _number(row["nees_consider_mean"]),
            }
            for row in e4["systematics"]["mixed_altitude"]
        ],
        "hybrid_1hz": {
            "velocity_rmse_mps": _number(
                e4["hybrid_tdoa_fdoa"]["1Hz"]["velocity_rmse"]
            ),
            "velocity_crlb_rms_mps": _number(
                e4["hybrid_tdoa_fdoa"]["1Hz"]["velocity_crlb_rms"]
            ),
            "nees6_mean": _number(e4["hybrid_tdoa_fdoa"]["1Hz"]["nees6_mean"]),
        },
        "median_bound_inside_network_m": {
            name: _number(m["median_inside_10km_box_m"])
            for name, m in e4["dop_maps"].items()
        },
    }


def _score(entry: dict[str, Any]) -> dict[str, Any]:
    """The tracking metrics shared by E5–E7 entries."""
    types = ("aircraft", "launch", "fire")
    return {
        "gospa_m": _estimate(entry["gospa"]),
        "false_per_scan": _estimate(entry["false_per_scan"]),
        "missed_per_scan": _estimate(entry["missed_per_scan"]),
        "nees": _estimate(entry["nees"]),
        "id_switches": _estimate(entry["id_switches"]),
        "fragmentations": _estimate(entry["fragmentations"]),
        "rmse_m": {t: _estimate(entry["rmse_by_type"][t]) for t in types},
        "coverage": {t: _estimate(entry["coverage_by_type"][t]) for t in types},
        "latency_s": {t: _estimate(entry["latency_by_type"][t]) for t in types},
    }


def _tracking(e5: dict[str, Any]) -> dict[str, Any]:
    pairing = e5["stereo_pairing"]
    return {
        "seeds": e5["seeds"],
        "track_management": {
            key: _score(entry) for key, entry in e5["track_management"].items()
        },
        "stereo_pairing": {
            "ghost_pairs_gnn": pairing["gnn"]["ghost_pairs"],
            "pairs_gnn": pairing["gnn"]["ghost_pairs"]
            + pairing["gnn"]["correct_pairs"],
            "ghost_pairs_deferred": pairing["defer"]["ghost_pairs"],
            "correct_pairs_kept": pairing["correct_pairs_kept"],
        },
        "stereo_ambiguity": {
            key: _score(entry) for key, entry in e5["stereo_ambiguity"].items()
        },
        "motion_model": {
            key: _score(entry) for key, entry in e5["motion_model"].items()
        },
    }


def _fusion(e6: dict[str, Any]) -> dict[str, Any]:
    return {
        "seeds": e6["seeds"],
        "architectures": {
            mode: _score(entry) for mode, entry in e6["architectures"].items()
        },
        "track_classification": {
            key: {
                "final_accuracy": _estimate(v["final_accuracy"]),
                "ece": _estimate(v["ece"]),
                "time_to_confident_s": _estimate(v["time_to_confident"]),
            }
            for key, v in e6["classification"].items()
        },
    }


def _robustness(e7: dict[str, Any]) -> dict[str, Any]:
    return {
        "outage_interval_s": e7["outage_interval_s"],
        "outages": {
            key: _score(entry["during"])
            | {"fragmentations_per_run": _estimate(entry["overall"]["fragmentations"])}
            for section in ("opir_dropout", "rf_outage")
            for key, entry in e7[section].items()
        },
        "latency": {key: _score(entry) for key, entry in e7["rf_latency"].items()},
        "rf_bias": {
            key: {
                "by_age": [
                    {
                        "age_s": b["age_s"],
                        "nees_filter": _number(b["nees"]),
                        "nees_reported": _number(b["nees_reported"]),
                    }
                    for b in entry["by_age"]
                ]
            }
            for key, entry in e7["rf_bias"].items()
            if key != "nees_reference"
        },
    }


def collect(root: Path = ROOT) -> dict[str, Any]:
    """Headline metrics from the experiment reports under ``root``."""
    reports = {k: json.loads((root / v).read_text()) for k, v in REPORTS.items()}
    quick = [k for k, r in reports.items() if r.get("quick")]
    if quick:
        raise ValueError(f"reports {quick} are quick-mode runs, not results")
    return {
        "sources": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in REPORTS.values()
        },
        "detection": _detection(reports["e1"]),
        "classification": _classification(reports["e2"]),
        "uncertainty": _uncertainty(reports["e3"]),
        "geolocation": _geolocation(reports["e4"]),
        "tracking": _tracking(reports["e5"]),
        "fusion": _fusion(reports["e6"]),
        "robustness": _robustness(reports["e7"]),
    }


# -- formatting ----------------------------------------------------------------


def _v(metric: Metric) -> float:
    value = metric["value"]
    return float("nan") if value is None else float(value)


def pct(metric: Metric, digits: int = 0) -> str:
    return f"{100 * _v(metric):.{digits}f}%"


def num(metric: Metric, digits: int = 0, unit: str = "") -> str:
    value = _v(metric)
    if value != value:
        return "–"
    return f"{value:,.{digits}f}{unit}"


def with_ci(metric: Metric, digits: int = 0, unit: str = "", scale: float = 1.0) -> str:
    """``value [low, high]``, or just the value when there is no interval."""
    fmt = f"{{:,.{digits}f}}"
    text = fmt.format(scale * _v(metric)) + unit
    if "ci_low" in metric and metric["ci_low"] != metric["ci_high"]:
        text += f" [{fmt.format(scale * metric['ci_low'])}, {fmt.format(scale * metric['ci_high'])}]"
    return text


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# -- tables ----------------------------------------------------------------------


def table_headline(m: dict[str, Any]) -> str:
    det, cls, unc = m["detection"], m["classification"], m["uncertainty"]
    geo, fus, rob = m["geolocation"], m["fusion"], m["robustness"]
    ten_ns = next(r for r in geo["noise_sweep"] if r["toa_std_ns"] == 10)
    arch = fus["architectures"]
    rf_down = rob["outages"]
    bias = rob["rf_bias"]["30ns_10m/consider"]["by_age"]
    shifts = [_v(cls["macro_f1"]["tcn"][s]) for s in SHIFT_SPLITS]
    shift_low, shift_high = min(shifts), max(shifts)
    rows = [
        [
            "Detection (E1)",
            f"CFAR detects {pct(det['cfar']['pd_at_pfa_1pct'])} of events at a calibrated 1% "
            f"false-alarm rate; the v1 detector alarmed on {pct(det['v1_ensemble']['pfa_realistic'])} "
            "of pure-background windows",
        ],
        [
            "Classification (E2)",
            f"TCN macro-F1 {num(cls['macro_f1']['tcn']['test'], 3)} on test and "
            f"{shift_low:.3f}–{shift_high:.3f} under domain shift "
            f"(feature baseline {num(cls['macro_f1']['gbm']['test'], 3)})",
        ],
        [
            "Uncertainty (E3)",
            f"90% conformal sets cover {pct(unc['conformal']['test']['coverage'], 1)} on test "
            f"with {num(unc['conformal']['test']['mean_set_size'], 2)} labels on average; "
            "two-stage alarms cut background false alarms from "
            f"{pct(unc['two_stage']['detector_only']['pfa_all_background'])} to "
            f"{pct(unc['two_stage']['detector_plus_classifier']['pfa_all_background'], 1)}",
        ],
        [
            "Geolocation (E4)",
            f"ML TDOA error {ten_ns['ml_rmse_m']:.1f} m against a Cramér-Rao bound of "
            f"{ten_ns['crlb_rms_m']:.1f} m at 10 ns, NEES {ten_ns['ml_nees_mean']:.2f} (3 = consistent)",
        ],
        [
            "Fusion (E6)",
            f"GOSPA error {num(arch['centralized']['gospa_m'])} m fused vs "
            f"{num(arch['opir']['gospa_m'])} m OPIR-only and {num(arch['rf']['gospa_m'])} m RF-only; "
            f"NEES {num(arch['centralized']['nees'], 1)}",
        ],
        [
            "Robustness (E7)",
            f"RF outage: fused tracking keeps {pct(rf_down['centralized/rf_down']['coverage']['aircraft'])} "
            f"of aircraft (RF alone {pct(rf_down['rf/rf_down']['coverage']['aircraft'])}); "
            f"RF bias: reported NEES {min(b['nees_reported'] for b in bias):.1f}–"
            f"{max(b['nees_reported'] for b in bias):.1f} at every track age",
        ],
    ]
    return _table(["Area", "Result"], rows)


def table_detection(m: dict[str, Any]) -> str:
    det = m["detection"]
    names = {"cfar": "CFAR", "cusum": "CUSUM", "glrt": "Step GLRT"}
    rows = []
    for key, name in names.items():
        d = det[key]
        by_class = d["pd_by_class"]
        rows.append(
            [
                name,
                f"{pct(d['pd_at_pfa_1pct'])} (explosion {pct(by_class['explosion'])}, "
                f"launch {pct(by_class['launch'])})",
                pct(d["analytic_threshold_pfa_realistic"]),
            ]
        )
    rows.append(
        [
            "v1 ensemble",
            "n/a",
            f"{pct(det['v1_ensemble']['pfa_realistic'])} (alarms on every window)",
        ]
    )
    return _table(
        [
            "Detector",
            "Pd at Pfa = 1% (calibrated, glint-free background)",
            "Pfa of the analytic 1% threshold on realistic background",
        ],
        rows,
    )


def table_classification(m: dict[str, Any]) -> str:
    cls = m["classification"]
    rows = []
    for key, name in MODELS.items():
        f1 = cls["macro_f1"][key]
        label = f"**{name} (shipped)**" if key == cls["selected"] else name
        test = with_ci(f1["test"], 3)
        rows.append(
            [label, f"**{test}**" if key == cls["selected"] else test]
            + [num(f1[s], 3) for s in SHIFT_SPLITS]
        )
    return _table(
        ["Model", "Test", "Low SNR", "Out-of-range physics", "Heavy clutter"], rows
    )


def table_conformal(m: dict[str, Any]) -> str:
    conformal = m["uncertainty"]["conformal"]
    names = {
        "test": "Test",
        "shift_low_snr": "Low SNR",
        "shift_params": "Out-of-range physics",
        "shift_clutter": "Heavy clutter",
    }
    rows = [
        [
            name,
            with_ci(conformal[s]["coverage"], 1, "%", scale=100.0),
            num(conformal[s]["mean_set_size"], 2),
            f"{num(conformal[s]['ece_before'], 3)} → {num(conformal[s]['ece_after'], 3)}",
        ]
        for s, name in names.items()
    ]
    return _table(
        [
            "Split",
            "Coverage (target 90%)",
            "Mean set size",
            "ECE before → after scaling",
        ],
        rows,
    )


def table_two_stage(m: dict[str, Any]) -> str:
    stages = m["uncertainty"]["two_stage"]
    rows = []
    for key, name in (
        ("detector_only", "CFAR only"),
        ("detector_plus_classifier", "CFAR + classifier"),
    ):
        s = stages[key]
        rows.append(
            [
                name,
                pct(s["pd"], 1),
                pct(s["pfa_all_background"], 1),
                pct(s["pfa_glint_background"], 1),
            ]
        )
    return _table(
        ["Alarm rule", "Pd", "Pfa, all background", "Pfa, glint background"], rows
    )


def table_ood(m: dict[str, Any]) -> str:
    ood = m["uncertainty"]["ood_auroc_energy"]
    return _table(
        ["Held-out class", "Energy-score AUROC"],
        [[k, num(v, 2)] for k, v in sorted(ood.items())],
    )


def meters(value: float) -> str:
    """Two decimals below 10 m, one below 100 m, none above."""
    digits = 2 if value < 10 else 1 if value < 100 else 0
    return f"{value:,.{digits}f} m"


def table_geolocation_crlb(m: dict[str, Any]) -> str:
    rows = [
        [
            f"{r['toa_std_ns']:g} ns",
            meters(r["crlb_rms_m"]),
            meters(r["ml_rmse_m"]),
            f"{r['ml_nees_mean']:.2f}",
            meters(r["chan_ho_rmse_m"]),
        ]
        for r in m["geolocation"]["noise_sweep"]
    ]
    return _table(
        ["Timing noise σ", "CRLB RMS", "ML RMSE", "ML mean NEES", "Chan-Ho RMSE"], rows
    )


def table_geolocation_systematics(m: dict[str, Any]) -> str:
    rows = []
    for r in m["geolocation"]["systematics"]:
        if r["clock_bias_ns"]:
            label = f"clock bias {r['clock_bias_ns']:g} ns"
        elif r["survey_error_m"]:
            label = f"survey error {r['survey_error_m']:g} m"
        else:
            label = "none"
        rows.append(
            [
                label,
                meters(r["rmse_m"]),
                meters(r["reported_rms_naive_m"]),
                f"{r['nees_naive_mean']:.1f}"
                if r["nees_naive_mean"] < 100
                else f"{r['nees_naive_mean']:.0f}",
                f"**{r['nees_consider_mean']:.2f}**",
            ]
        )
    return _table(
        [
            "Systematic error",
            "RMSE",
            "Reported RMS (naive)",
            "NEES naive",
            "NEES consider",
        ],
        rows,
    )


def table_fusion_architectures(m: dict[str, Any]) -> str:
    arch = m["fusion"]["architectures"]
    rows = []
    for key, name in ARCHITECTURES.items():
        a = arch[key]

        def rmse(kind: str, a: dict[str, Any] = a) -> str:
            value = a["rmse_m"][kind]
            return "not seen" if value["value"] is None else num(value, 0, " m")

        rows.append(
            [
                f"**{name}**" if key == "centralized" else name,
                with_ci(a["gospa_m"]),
                num(a["false_per_scan"], 2),
                rmse("aircraft"),
                rmse("launch"),
                rmse("fire"),
                num(a["nees"], 1),
                num(a["id_switches"], 1),
            ]
        )
    return _table(
        [
            "Architecture",
            "GOSPA (m)",
            "False tracks / scan",
            "Aircraft RMSE",
            "Launch RMSE",
            "Fire RMSE",
            "NEES",
            "Identity switches per run",
        ],
        rows,
    )


def table_track_management(m: dict[str, Any]) -> str:
    grid = m["tracking"]["track_management"]
    rows = []
    for rate in ("0", "2", "5"):
        cells = [rate]
        for rule in ("1of1", "2of3", "3of5"):
            g = grid[f"clutter_{rate}/{rule}"]
            cells.append(f"{num(g['false_per_scan'], 2)}, {num(g['gospa_m'], 0, ' m')}")
        rows.append(cells)
    return _table(
        ["Clutter rate", "1-of-1: false tracks/scan, GOSPA", "2-of-3", "3-of-5"], rows
    )


def table_motion_model(m: dict[str, Any]) -> str:
    models = m["tracking"]["motion_model"]
    names = {
        "cv_q25": "Constant velocity, q = 25",
        "cv_q400": "Constant velocity, q = 400",
        "imm_q25_q1600": "**IMM (q = 25 / 1600)**",
    }
    rows = [
        [
            name,
            num(models[k]["gospa_m"], 0, " m"),
            num(models[k]["rmse_m"]["launch"], 0, " m"),
            pct(models[k]["coverage"]["launch"]),
            num(models[k]["id_switches"], 1),
        ]
        for k, name in names.items()
    ]
    return _table(
        ["Model", "GOSPA", "Launch RMSE", "Launch coverage", "Identity switches"], rows
    )


def table_track_classification(m: dict[str, Any]) -> str:
    classification = m["fusion"]["track_classification"]
    policies = {
        "all_ages": "all ages",
        "onset_in_window": "0–62 s",
        "trained_range": "**24–62 s (trained range)**",
    }
    rows = []
    for policy, label in policies.items():
        for weight in ("1", "0.3"):
            c = classification[f"{policy}/w{weight}"]
            rows.append(
                [
                    label,
                    weight if weight == "1" else "0.3",
                    pct(c["final_accuracy"]),
                    num(c["ece"], 2),
                    num(c["time_to_confident_s"], 0, " s"),
                ]
            )
    return _table(
        [
            "Windows used (detection age)",
            "w",
            "Final label accuracy",
            "Posterior ECE",
            "Median time to confident label",
        ],
        rows,
    )


def table_outages(m: dict[str, Any]) -> str:
    outages = m["robustness"]["outages"]
    rows_spec = (
        ("none", "centralized", "centralized/nominal"),
        ("HEO satellite", "OPIR only", "opir/heo_down"),
        ("HEO satellite", "centralized", "centralized/heo_down"),
        ("RF network", "RF only", "rf/rf_down"),
        ("RF network", "centralized", "centralized/rf_down"),
        ("RF network", "T2T CI", "t2t_ci/rf_down"),
    )
    rows = []
    for outage, arch, key in rows_spec:
        o = outages[key]
        launch = o["coverage"]["launch"]
        rows.append(
            [
                outage,
                arch,
                num(o["gospa_m"], 0, " m"),
                pct(o["coverage"]["aircraft"]),
                "–"
                if launch["value"] is None or key.startswith("rf/")
                else pct(launch),
                num(o["fragmentations_per_run"], 1),
            ]
        )
    return _table(
        [
            "Outage",
            "Architecture",
            "GOSPA during",
            "Aircraft coverage",
            "Launch coverage",
            "Fragmentations per run",
        ],
        rows,
    )


def table_latency(m: dict[str, Any]) -> str:
    latency = m["robustness"]["latency"]
    rows = [["0 s", num(latency["0s/drop"]["gospa_m"], 0, " m"), "–", "–"]]
    for delay in ("1", "2", "5"):
        rows.append(
            [f"{delay} s"]
            + [
                num(latency[f"{delay}s/{s}"]["gospa_m"], 0, " m")
                for s in ("drop", "buffer", "extrapolate")
            ]
        )
    return _table(
        [
            "Latency",
            "Drop late fixes",
            "Buffer and predict forward",
            "Extrapolate late fixes",
        ],
        rows,
    )


def table_rf_bias(m: dict[str, Any]) -> str:
    bias = m["robustness"]["rf_bias"]
    ages = ((0.0, 10.0), (30.0, 60.0), (120.0, 180.0))

    def at(key: str, field: str, age: tuple[float, float]) -> str:
        row = next(b for b in bias[key]["by_age"] if tuple(b["age_s"]) == age)
        value = row[field]
        return f"{value:.1f}" if value < 10 else f"{value:.0f}"

    specs = (
        ("naive fixes", "30ns_10m/naive", "nees_filter"),
        ("consider fixes, filter covariance", "30ns_10m/consider", "nees_filter"),
        (
            "**consider fixes, reported covariance (with bias floor)**",
            "30ns_10m/consider",
            "nees_reported",
        ),
    )
    rows = [[label] + [at(key, field, a) for a in ages] for label, key, field in specs]
    return _table(
        [
            "30 ns clock bias + 10 m survey",
            "NEES, track age 0–10 s",
            "30–60 s",
            "120–180 s",
        ],
        rows,
    )


TABLES: dict[str, Callable[[dict[str, Any]], str]] = {
    "headline": table_headline,
    "detection": table_detection,
    "classification": table_classification,
    "conformal": table_conformal,
    "two_stage": table_two_stage,
    "ood": table_ood,
    "geolocation_crlb": table_geolocation_crlb,
    "geolocation_systematics": table_geolocation_systematics,
    "fusion_architectures": table_fusion_architectures,
    "track_management": table_track_management,
    "motion_model": table_motion_model,
    "track_classification": table_track_classification,
    "outages": table_outages,
    "latency": table_latency,
    "rf_bias": table_rf_bias,
}

_BLOCK_NAMES = re.compile(r"<!-- metrics:([a-z_]+) -->")
_BLOCK = re.compile(
    r"<!-- metrics:(?P<name>[a-z_]+) -->(?P<body>.*?)<!-- /metrics:(?P=name) -->",
    re.S,
)


def render(text: str, metrics: dict[str, Any]) -> str:
    """``text`` with every metrics block regenerated.

    Raises:
        KeyError: a block names an unknown table.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in TABLES:
            raise KeyError(f"unknown metrics table {name!r}")
        return f"<!-- metrics:{name} -->\n{TABLES[name](metrics)}\n<!-- /metrics:{name} -->"

    return _BLOCK.sub(replace, text)


def serialize(metrics: dict[str, Any]) -> str:
    return json.dumps(metrics, indent=2, sort_keys=True) + "\n"


def stale(root: Path = ROOT) -> list[str]:
    """Files whose contents differ from what the reports generate."""
    metrics = collect(root)
    out = []
    path = root / METRICS
    if not path.exists() or path.read_text() != serialize(metrics):
        out.append(str(METRICS))
    for name in DOCUMENTS:
        doc = root / name
        if doc.exists() and doc.read_text() != render(doc.read_text(), metrics):
            out.append(name)
    return out


def update(root: Path = ROOT) -> list[str]:
    """Regenerate ``reports/metrics.json`` and every table; return changed files."""
    changed = stale(root)
    metrics = collect(root)
    (root / METRICS).write_text(serialize(metrics))
    for name in DOCUMENTS:
        doc = root / name
        if doc.exists():
            doc.write_text(render(doc.read_text(), metrics))
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="Fail if anything is stale"
    )
    args = parser.parse_args(argv)
    if args.check:
        problems = stale()
        for name in problems:
            print(f"stale: {name} (run `make metrics`)", file=sys.stderr)
        return 1 if problems else 0
    for name in update():
        print(f"updated {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
