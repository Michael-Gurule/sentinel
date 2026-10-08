"""Smoke tests for the legacy RF signal generator (rewritten in Phase 2)."""

import numpy as np
import pytest

from sentinel.models.rf_generator import RFScenarioGenerator


@pytest.mark.parametrize(
    "method",
    [
        "generate_early_warning_radar",
        "generate_fire_control_radar",
        "generate_tactical_radio",
        "generate_satellite_uplink",
    ],
)
def test_emitters_produce_aligned_signal_and_time(method):
    np.random.seed(0)  # noqa: NPY002 - legacy module uses the global RNG
    signal, t, emitter = getattr(RFScenarioGenerator(), method)(0.0, 1e-3, 40.0, -100)
    assert len(signal) == len(t) > 0
    assert np.all(np.isfinite(signal))
    assert emitter.carrier_freq_hz > 0
