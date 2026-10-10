import numpy as np
import pytest

from locant.core import is_psd
from locant.tracking import (
    ConstantVelocity,
    Gaussian,
    IMMState,
    LinearMeasurement,
    MultiTargetTracker,
    imm_predict,
    imm_update,
    transition_matrix,
)

R = np.eye(3) * 100.0
QUIET, AGILE = ConstantVelocity(1.0), ConstantVelocity(1_600.0)


def test_transition_matrix_rows_are_distributions():
    pi = transition_matrix((60.0, 10.0), 1.0)
    np.testing.assert_allclose(pi.sum(axis=1), 1.0)
    assert pi[0, 0] == pytest.approx(np.exp(-1 / 60))
    assert transition_matrix((5.0,), 1.0)[0, 0] == 1.0


def test_combined_state_is_moment_matched():
    a = Gaussian(np.zeros(6), np.eye(6))
    b = Gaussian(np.full(6, 2.0), np.eye(6))
    combined = IMMState([a, b], np.array([0.5, 0.5])).combined()
    np.testing.assert_allclose(combined.mean, 1.0)
    np.testing.assert_allclose(np.diag(combined.covariance), 2.0)  # 1 + spread of means
    assert is_psd(combined.covariance)


def test_mode_probability_follows_a_maneuver(rng):
    imm = IMMState([Gaussian(np.zeros(6), np.eye(6) * 100.0)] * 2, np.array([0.5, 0.5]))
    h = np.hstack([np.eye(3), np.zeros((3, 3))])

    def linearize(mean):
        return h @ mean, h

    position, velocity = np.zeros(3), np.zeros(3)
    quiet_probability = []
    for k in range(60):
        accel = np.array([30.0, 0, 0]) if 30 <= k < 45 else np.zeros(3)
        velocity = velocity + accel
        position = position + velocity
        imm = imm_predict(imm, [QUIET, AGILE], (60.0, 10.0), 1.0)
        imm = imm_update(imm, position + rng.normal(0, 10, 3), linearize, R)
        quiet_probability.append(imm.probabilities[0])
    assert np.mean(quiet_probability[15:30]) > 0.7  # cruising
    assert np.mean(quiet_probability[33:45]) < 0.3  # accelerating
    assert imm_predict(imm, [QUIET, AGILE], (60.0, 10.0), 0.0) is imm


def run(tracker):
    """Error of the original track (id 0) through a 30 m/s² turn; inf if lost."""
    rng = np.random.default_rng(3)
    position, velocity, errors = np.zeros(3), np.array([100.0, 0, 0]), []
    for k in range(60):
        if 30 <= k < 45:
            velocity = velocity + np.array([0.0, 30.0, 0.0])
        position = position + velocity
        z = position + rng.normal(0, 10, 3)
        tracker.step([LinearMeasurement.position(z, R, "s")], float(k))
        if k >= 30:
            original = [t for t in tracker.tracks if t.id == 0]
            errors.append(
                np.linalg.norm(original[0].position - position) if original else np.inf
            )
    return float(np.sqrt(np.mean(np.square(errors)))), len(tracker.tracks)


def test_imm_holds_a_maneuvering_target_that_a_quiet_model_loses():
    single_error, _ = run(MultiTargetTracker(QUIET, gate_probability=0.999))
    imm_error, imm_tracks = run(
        MultiTargetTracker(QUIET, gate_probability=0.999, imm_models=[QUIET, AGILE])
    )
    assert single_error == np.inf  # the quiet model drops the original track
    assert imm_error < 30.0
    assert imm_tracks == 1


def test_imm_configuration_must_align():
    with pytest.raises(ValueError, match="align"):
        MultiTargetTracker(QUIET, imm_models=[QUIET, AGILE], imm_sojourn_s=(10.0,))


def test_duplicate_tracks_are_merged():
    tracker = MultiTargetTracker(QUIET, gate_probability=0.5)
    tracker.step([LinearMeasurement.position(np.zeros(3), R, "s")], 0.0)
    tracker.initiate(LinearMeasurement.position(np.array([5.0, 0, 0]), R, "s"), 0.0)
    assert len(tracker.tracks) == 2
    tracker.step([], 1.0)
    (survivor,) = tracker.tracks
    assert survivor.id == 0  # created first (equal hits): kept
    unmerged = MultiTargetTracker(QUIET, merge_probability=None)
    unmerged.step([LinearMeasurement.position(np.zeros(3), R, "s")], 0.0)
    unmerged.initiate(LinearMeasurement.position(np.array([5.0, 0, 0]), R, "s"), 0.0)
    unmerged.step([], 1.0)
    assert len(unmerged.tracks) == 2
