"""Behavioral tests for the v1 detectors (rewritten in Phase 3).

False-alarm defects (C5) are captured in tests/characterization.
"""

import numpy as np
import pytest

from sentinel.detection.opir_detectors import (
    AnomalyDetector,
    MultiMethodDetector,
    RiseTimeDetector,
    TemporalDifferenceDetector,
    detect_event,
)
from sentinel.models.signal_generator import OPIRSignalGenerator

FS = 100.0


def signature(kind: str) -> np.ndarray:
    gen = OPIRSignalGenerator(rng=np.random.default_rng(0))
    if kind == "launch":
        return gen.generate_launch_signature(start_time=2.0)
    return gen.generate_explosion_signature(start_time=2.0)


@pytest.mark.parametrize(
    ("detector", "kind", "window"),
    [
        (AnomalyDetector(method="zscore"), "launch", (2.0, 5.0)),
        (RiseTimeDetector(), "launch", (2.0, 5.0)),
    ],
)
def test_detectors_fire_during_the_event(detector, kind, window):
    result = detector.detect(signature(kind), FS)
    assert result.detected
    assert 0.0 <= result.confidence <= 1.0
    assert window[0] <= result.detection_time <= window[1]


def test_ensemble_detects_launch():
    result = MultiMethodDetector().detect(signature("launch"), FS)
    assert result.detected
    assert result.metadata["num_detected"] >= 2


def test_temporal_difference_fires_on_sustained_ramp():
    # Fires only on >= 3 consecutive rising frames; the events it should catch
    # (launch, explosion) are captured as defect H11 in tests/characterization.
    rng = np.random.default_rng(0)
    signal = np.concatenate([np.zeros(50), np.arange(1, 21) * 50.0, np.full(30, 1e3)])
    signal += rng.normal(0.0, 5.0, signal.size)
    # Detection time is not asserted: v1 reports the first above-threshold
    # frame anywhere, not the start of the qualifying run (H11, Phase 3).
    assert TemporalDifferenceDetector().detect(signal, FS).detected


def test_detect_event_dispatch():
    assert detect_event(signature("launch"), FS, method="rise").method == "rise_time"
    with pytest.raises(ValueError, match="Unknown method"):
        detect_event(signature("launch"), FS, method="bogus")
