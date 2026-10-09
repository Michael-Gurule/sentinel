"""Evaluation metrics and report writing."""

from sentinel.eval.metrics import (
    Estimate,
    accuracy,
    auroc,
    binomial_interval,
    bootstrap,
    confusion,
    detection_roc,
    expected_calibration_error,
    macro_f1,
    per_class_f1,
    seed_summary,
)
from sentinel.eval.report import write_report
from sentinel.eval.tracking import (
    GOSPA,
    Snapshot,
    TrackingEvaluation,
    evaluate_tracking,
    gospa,
    ospa,
)

__all__ = [
    "GOSPA",
    "Estimate",
    "Snapshot",
    "TrackingEvaluation",
    "accuracy",
    "auroc",
    "binomial_interval",
    "bootstrap",
    "confusion",
    "detection_roc",
    "evaluate_tracking",
    "expected_calibration_error",
    "gospa",
    "macro_f1",
    "ospa",
    "per_class_f1",
    "seed_summary",
    "write_report",
]
