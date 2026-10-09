"""Shared fixtures and Hypothesis configuration."""

import numpy as np
import pytest
from hypothesis import settings

from locant.geolocation import Receiver

# Deterministic property tests: the same examples on every run and machine.
settings.register_profile("default", max_examples=50, deadline=None, derandomize=True)
settings.load_profile("default")

# Non-coplanar receivers (m, local ENU) and an emitter inside their hull.
RECEIVER_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [10_000.0, 0.0, 200.0],
        [10_000.0, 10_000.0, 50.0],
        [0.0, 10_000.0, 400.0],
        [5_000.0, -3_000.0, 1_000.0],
    ]
)
RECEIVER_VELOCITIES = np.array(
    [
        [0.0, 0.0, 0.0],
        [30.0, 0.0, 0.0],
        [0.0, -40.0, 0.0],
        [20.0, 20.0, 0.0],
        [-30.0, 10.0, 0.0],
    ]
)
EMITTER_POSITION = np.array([3_000.0, 4_000.0, 800.0])
EMITTER_VELOCITY = np.array([100.0, 50.0, 0.0])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261008)


@pytest.fixture
def receivers() -> list[Receiver]:
    return [Receiver(i, p) for i, p in enumerate(RECEIVER_POSITIONS)]


@pytest.fixture
def moving_receivers() -> list[Receiver]:
    return [
        Receiver(i, p, v)
        for i, (p, v) in enumerate(
            zip(RECEIVER_POSITIONS, RECEIVER_VELOCITIES, strict=True)
        )
    ]


@pytest.fixture
def emitter() -> np.ndarray:
    return EMITTER_POSITION.copy()


@pytest.fixture
def emitter_velocity() -> np.ndarray:
    return EMITTER_VELOCITY.copy()
