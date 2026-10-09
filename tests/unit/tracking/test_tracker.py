import numpy as np
import pytest

from sentinel.core import mean_nees_bounds, nees
from sentinel.tracking import ConstantVelocity, LinearMeasurement, MultiTargetTracker

R = np.eye(3) * 25.0
MODEL = ConstantVelocity(noise_intensity=4.0)


def simulate_track(rng, model, steps, x0, dt_range=(0.5, 2.0)):
    """Yield (time, true_state) along a CWNA trajectory."""
    x, t = np.asarray(x0, dtype=float), 0.0
    yield t, x
    for _ in range(steps - 1):
        dt = rng.uniform(*dt_range)
        t += dt
        x = model.transition(dt) @ x + rng.multivariate_normal(
            np.zeros(6), model.process_noise(dt)
        )
        yield t, x


def measure(rng, x, source="rf", label=None):
    z = x[:3] + rng.multivariate_normal(np.zeros(3), R)
    return LinearMeasurement.position(z, R, source, label)


def test_filter_is_consistent_across_independent_runs(rng):
    """NEES at the final scan, over independent runs, is χ²(6)-consistent."""
    runs, values = 150, []
    for _ in range(runs):
        tracker = MultiTargetTracker(MODEL, gate_probability=1 - 1e-9)
        for t, x in simulate_track(rng, MODEL, 30, [0, 0, 1e3, 100, -50, 0]):
            tracker.step([measure(rng, x)], t)
        (track,) = tracker.tracks
        values.append(nees(track.state.mean - x, track.state.covariance))
    low, high = mean_nees_bounds(6, runs, confidence=0.99)
    assert low <= np.mean(values) <= high


def test_two_separated_targets_keep_their_ids(rng):
    # Wide gate: this checks association, not the gate's miss rate.
    tracker = MultiTargetTracker(MODEL, gate_probability=1 - 1e-9)
    a = simulate_track(rng, MODEL, 20, [0, 0, 0, 50, 0, 0], (1.0, 1.0))
    b = simulate_track(rng, MODEL, 20, [0, 5e3, 0, -50, 0, 0], (1.0, 1.0))
    for (t, xa), (_, xb) in zip(a, b, strict=True):
        tracker.step([measure(rng, xb), measure(rng, xa)], t)
    tracks = sorted(tracker.tracks, key=lambda tr: tr.id)
    assert [tr.id for tr in tracks] == [0, 1]
    assert all(tr.hits == 20 for tr in tracks)
    np.testing.assert_allclose(tracks[0].position, xb[:3], atol=30)
    np.testing.assert_allclose(tracks[1].position, xa[:3], atol=30)


def test_coasting_track_is_deleted_after_max_coast_time():
    tracker = MultiTargetTracker(MODEL, max_coast_time=3.0)
    tracker.step([LinearMeasurement.position(np.zeros(3), R, "rf")], 0.0)
    tracker.step([], 3.0)
    assert len(tracker.tracks) == 1
    tracker.step([], 3.5)
    assert tracker.tracks == []


def test_distant_measurement_starts_a_new_track():
    tracker = MultiTargetTracker(MODEL)
    tracker.step([LinearMeasurement.position(np.zeros(3), R, "rf")], 0.0)
    far = LinearMeasurement.position(np.array([1e4, 0, 0]), R, "rf")
    assert not tracker.in_gate(tracker.tracks[0], far)
    tracker.step([far], 1.0)
    assert len(tracker.tracks) == 2


def test_sources_are_fused_sequentially_and_labels_propagate(rng):
    tracker = MultiTargetTracker(MODEL)
    x = np.array([0, 0, 0, 10, 0, 0.0])
    tracker.step([measure(rng, x, "rf")], 0.0)
    rf_only_cov = tracker.tracks[0].position_covariance.copy()
    x1 = MODEL.transition(1.0) @ x
    tracker.step([measure(rng, x1, "rf"), measure(rng, x1, "opir", "launch")], 1.0)
    (track,) = tracker.tracks
    assert track.hits_by_source == {"rf": 2, "opir": 1}
    assert track.label == "launch"
    assert np.trace(track.position_covariance) < np.trace(rf_only_cov)


def test_initiates_from_position_velocity_measurement():
    tracker = MultiTargetTracker(MODEL)
    state = np.array([1.0, 2, 3, 4, 5, 6])
    cov = np.eye(6) * 2.0
    track = tracker.initiate(LinearMeasurement.position_velocity(state, cov, "rf"), 0)
    np.testing.assert_allclose(track.state.mean, state)
    np.testing.assert_allclose(track.state.covariance, cov)


def test_position_only_birth_uses_velocity_prior():
    tracker = MultiTargetTracker(MODEL, initial_velocity_std=50.0)
    track = tracker.initiate(LinearMeasurement.position(np.zeros(3), R, "rf"), 0.0)
    np.testing.assert_allclose(track.velocity_covariance, np.eye(3) * 2500.0)
    assert track.position_rms_uncertainty == pytest.approx(np.sqrt(75.0))


def test_rejects_unsupported_measurement_for_birth():
    tracker = MultiTargetTracker(MODEL)
    velocity_only = LinearMeasurement(
        np.zeros(3), R, np.hstack([np.zeros((3, 3)), np.eye(3)]), "rf"
    )
    with pytest.raises(ValueError, match="position"):
        tracker.initiate(velocity_only, 0.0)


def test_out_of_sequence_is_rejected():
    tracker = MultiTargetTracker(MODEL)
    tracker.step([], 5.0)
    with pytest.raises(ValueError, match="out-of-sequence"):
        tracker.step([], 4.0)


def test_mixed_dimensions_from_one_source_are_gated_separately():
    """RF fixes with and without FDOA velocity (3-D and 6-D) in one scan."""
    tracker = MultiTargetTracker(MODEL, confirm_hits=1)
    far = np.array([50_000.0, 0.0, 0.0])
    tracker.step(
        [
            LinearMeasurement.position(np.zeros(3), R, "rf"),
            LinearMeasurement.position_velocity(
                np.concatenate([far, np.zeros(3)]), np.eye(6), "rf"
            ),
        ],
        0.0,
    )
    assert len(tracker.tracks) == 2
    tracker.step(
        [
            LinearMeasurement.position_velocity(np.zeros(6), np.eye(6) * 100, "rf"),
            LinearMeasurement.position(far, R, "rf"),
        ],
        1.0,
    )
    assert len(tracker.tracks) == 2
    assert all(t.hits_by_source["rf"] == 2 for t in tracker.tracks)


def test_invalid_configuration_and_measurement_shapes():
    with pytest.raises(ValueError, match="max_coast_time"):
        MultiTargetTracker(MODEL, max_coast_time=-1.0)
    with pytest.raises(ValueError, match="disagree"):
        LinearMeasurement(np.zeros(3), np.eye(2), np.eye(3, 6), "rf")
