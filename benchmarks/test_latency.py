"""Per-stage latency budgets (``make bench``).

Each benchmark times one stage on a representative load and fails if its
median exceeds a budget. Budgets are about 5x the medians measured on the
development machine (Apple M-series, CPU), which leaves room for slower CI
runners while still catching an accidental return to per-element Python
loops. Medians before/after Phase 6 vectorization: tracker scan 149 → 11 ms,
stereo association 38 → 0.1 ms.

These are not run by the default ``pytest`` (testpaths is ``tests/``).
"""

import copy
from pathlib import Path

import numpy as np
import pytest

from locant.detection import CFARDetector
from locant.fusion import FusionEngine, LineOfSightMeasurement, associate_stereo
from locant.fusion.opir_geoloc import perpendicular_basis
from locant.geolocation import (
    Receiver,
    SystematicErrors,
    simulate_fdoa,
    simulate_tdoa,
    solve_tdoa_fdoa,
)
from locant.tracking import LinearMeasurement

ROOT = Path(__file__).resolve().parents[1]
GEO = np.array([0.0, -30e6, 25e6])
HEO = np.array([10e6, 15e6, 35e6])
ANGLE_STD = 10e-6
TARGETS = 30


def check_budget(benchmark, seconds: float) -> None:
    stats = benchmark.stats
    if stats is None:  # --benchmark-disable: timing not collected
        return
    median = stats.stats.median
    assert median <= seconds, (
        f"median {median * 1e3:.2f} ms > budget {seconds * 1e3:.1f} ms"
    )


def _ray(sensor: np.ndarray, target: np.ndarray, rng: np.random.Generator):
    u = (target - sensor) / np.linalg.norm(target - sensor)
    u = u + perpendicular_basis(u).T @ rng.normal(0.0, ANGLE_STD, 2)
    return sensor, u / np.linalg.norm(u), ANGLE_STD


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(0)
    positions = np.column_stack(
        [rng.uniform(-50e3, 50e3, (TARGETS, 2)), rng.uniform(0, 12e3, TARGETS)]
    )
    velocities = np.column_stack([rng.normal(0, 150, (TARGETS, 2)), np.zeros(TARGETS)])
    return rng, positions, velocities


@pytest.fixture(scope="module")
def warm_engine(scene):
    """An IMM engine holding 30 confirmed tracks."""
    rng, positions, velocities = scene
    engine = FusionEngine()
    for t in range(5):
        engine.process(
            [
                LinearMeasurement.position(
                    p + v * t + rng.normal(0, 20, 3), np.eye(3) * 400.0, "rf"
                )
                for p, v in zip(positions, velocities, strict=True)
            ],
            float(t),
        )
    assert len(engine.confirmed_tracks) == TARGETS
    return engine


def test_tracker_scan(benchmark, scene, warm_engine):
    """One scan: 30 IMM tracks, 30 RF fixes and 60 line-of-sight rays."""
    rng, positions, velocities = scene
    now = positions + velocities * 5.0
    scan = [
        LinearMeasurement.position(p + rng.normal(0, 20, 3), np.eye(3) * 400.0, "rf")
        for p in now
    ]
    for index, sensor in enumerate((GEO, HEO)):
        scan += [
            LineOfSightMeasurement(*_ray(sensor, p, rng), source=f"opir_los/{index}")
            for p in now
        ]
    benchmark.pedantic(
        lambda engine: engine.process(scan, 5.0),
        setup=lambda: ((copy.deepcopy(warm_engine),), {}),
        rounds=30,
    )
    check_budget(benchmark, 0.060)


def test_stereo_association(benchmark, scene):
    """Pair 30 rays from each of two satellites."""
    rng, positions, _ = scene
    rays_a = [_ray(GEO, p, rng) for p in positions]
    rays_b = [_ray(HEO, p, rng) for p in positions[::-1]]
    pairs = benchmark(associate_stereo, rays_a, rays_b)
    assert len(pairs) > 0
    check_budget(benchmark, 0.0005)


def test_rf_fix(benchmark):
    """One joint TDOA/FDOA fix with consider covariance (5 moving receivers)."""
    rng = np.random.default_rng(1)
    receivers = [
        Receiver(i, p, v)
        for i, (p, v) in enumerate(
            zip(
                np.array(
                    [
                        [0, 0, 500],
                        [10e3, 0, 1500],
                        [10e3, 10e3, 1000],
                        [0, 10e3, 2000],
                        [5e3, -4e3, 6000],
                    ],
                    dtype=float,
                ),
                np.array(
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 150, 0], [120, 0, 0]],
                    dtype=float,
                ),
                strict=True,
            )
        )
    ]
    emitter, velocity = np.array([3e3, 4e3, 800.0]), np.array([100.0, 50.0, 0.0])
    tdoa = simulate_tdoa(emitter, receivers, 10e-9, rng)
    fdoa = simulate_fdoa(emitter, velocity, receivers, 1e9, 1.0, rng)
    levels = SystematicErrors(2.0, 5e-9)
    fix = benchmark(solve_tdoa_fdoa, receivers, tdoa, fdoa, systematic=levels)
    assert fix.converged
    check_budget(benchmark, 0.003)


def test_cfar_detection(benchmark):
    """CFAR scores for 64 pixel windows of 64 s at 10 Hz."""
    rng = np.random.default_rng(2)
    windows = 200.0 + rng.normal(0, 1, (64, 640))
    benchmark(CFARDetector().score, windows, 10.0)
    check_budget(benchmark, 0.0025)


def test_classifier_batch(benchmark):
    """Calibrated TCN classification of 16 windows on CPU."""
    from locant.classification import EventClassifier

    path = ROOT / "models" / "opir_event_classifier"
    if not path.exists():
        pytest.skip("exported classifier not present")
    classifier = EventClassifier.load(path)
    rng = np.random.default_rng(3)
    windows = 200.0 + rng.normal(0, 1, (16, 640))
    benchmark(classifier.predict, windows)
    check_budget(benchmark, 0.150)


def test_classifier_batch_onnx(benchmark, tmp_path):
    """The same 16 windows through ONNX Runtime (no PyTorch at inference)."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    import shutil

    from locant.classification.onnx_backend import OnnxEventClassifier, export_onnx

    source = ROOT / "models" / "opir_event_classifier"
    if not source.exists():
        pytest.skip("exported classifier not present")
    artifact = tmp_path / "classifier"
    shutil.copytree(source, artifact)
    export_onnx(artifact)
    classifier = OnnxEventClassifier.load(artifact)
    rng = np.random.default_rng(3)
    windows = 200.0 + rng.normal(0, 1, (16, 640))
    benchmark(classifier.predict, windows)
    check_budget(benchmark, 0.015)
