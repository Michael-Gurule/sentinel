import json

import numpy as np
import pytest

from locant.eval import (
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
    write_report,
)


def test_classification_metrics():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 1, 2, 0])
    assert accuracy(y, p) == pytest.approx(4 / 6)
    assert per_class_f1(y, p, 3) == pytest.approx([0.5, 0.8, 2 / 3])
    assert macro_f1(y, p) == pytest.approx(np.mean([0.5, 0.8, 2 / 3]))
    np.testing.assert_array_equal(confusion(y, p, 3), [[1, 1, 0], [0, 2, 0], [1, 0, 1]])


def test_bootstrap_interval_contains_estimate(rng):
    y = rng.integers(0, 3, 500)
    p = np.where(rng.random(500) < 0.8, y, rng.integers(0, 3, 500))
    est = bootstrap(accuracy, y, p, rng, resamples=300)
    assert est.low < est.value < est.high
    assert est.high - est.low < 0.15


def test_seed_summary():
    single = seed_summary([0.7])
    assert single.low == single.value == single.high == 0.7
    est = seed_summary([0.70, 0.72, 0.71, 0.69, 0.73])
    assert est.value == pytest.approx(0.71)
    assert est.low < 0.71 < est.high


def test_ece_is_zero_for_perfectly_calibrated_bins():
    probs = np.tile([0.8, 0.2], (1_000, 1))
    labels = np.array([0] * 800 + [1] * 200)
    ece, table = expected_calibration_error(probs, labels)
    assert ece == pytest.approx(0.0, abs=1e-12)
    assert table["bin_count"] == [1_000.0]
    overconfident = np.tile([0.99, 0.01], (1_000, 1))
    assert expected_calibration_error(overconfident, labels)[0] == pytest.approx(0.19)


def test_auroc_and_detection_roc():
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    pfa, pd, _ = detection_roc(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
    assert pfa[0] == 0.0
    assert pd[-1] == 1.0


def test_binomial_interval_edges():
    zero = binomial_interval(0, 100)
    assert zero.low == 0.0
    assert 0.0 < zero.high < 0.05
    full = binomial_interval(10, 10)
    assert full.high == 1.0


def test_write_report_is_deterministic(tmp_path):
    content = {
        "b": np.float64(1 / 3),
        "a": [np.int64(2), np.array([0.5])],
        "c": float("inf"),
    }
    path = write_report(tmp_path / "r.json", content)
    first = path.read_text()
    write_report(path, content)
    assert path.read_text() == first
    assert json.loads(first) == {"a": [2, [0.5]], "b": 0.333333, "c": "inf"}
