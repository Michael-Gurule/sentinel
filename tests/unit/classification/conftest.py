import numpy as np
import pytest

from locant.taxonomy import EVENT_CLASSES


@pytest.fixture
def toy_windows():
    """Separable toy classes: one shape per class plus noise, (N, 200)."""
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 1.0, 200)
    shapes = [
        1.0 / (1.0 + np.exp(-30 * (t - 0.3))),  # launch-like step
        np.where(t >= 0.4, np.exp(-(t - 0.4) * 40), 0.0),  # explosion-like spike
        np.clip(t - 0.2, 0, None),  # fire-like ramp
        0.3 * np.sin(2 * np.pi * 3 * t) + 0.3,  # aircraft-like modulation
        np.zeros_like(t),  # background
    ]
    signals, labels = [], []
    for label, shape in enumerate(shapes):
        signals.append(100.0 + 20.0 * shape + rng.normal(0.0, 1.0, (40, t.size)))
        labels.append(np.full(40, label))
    assert len(shapes) == len(EVENT_CLASSES)
    return np.concatenate(signals), np.concatenate(labels)
