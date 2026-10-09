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

__all__ = [
    "Estimate",
    "accuracy",
    "auroc",
    "binomial_interval",
    "bootstrap",
    "confusion",
    "detection_roc",
    "expected_calibration_error",
    "macro_f1",
    "per_class_f1",
    "seed_summary",
    "write_report",
]
