import numpy as np
import pytest

from sentinel.tracking import (
    ConstantVelocity,
    LinearMeasurement,
    MultiTargetTracker,
    TrackStatus,
    pool_class_evidence,
)

R = np.eye(3) * 25.0


def at(position, source="rf", probs=None):
    return LinearMeasurement.position(
        np.asarray(position, float), R, source, class_probabilities=probs
    )


def test_m_of_n_confirmation_and_tentative_deletion():
    tracker = MultiTargetTracker(
        ConstantVelocity(1.0), confirm_hits=3, confirm_window=5
    )
    tracker.step([at([0, 0, 0]), at([5e3, 0, 0])], 0.0)
    assert {t.status for t in tracker.tracks} == {TrackStatus.TENTATIVE}
    for t in (1.0, 2.0):  # only the first target keeps reporting
        tracker.step([at([0, 0, 0])], t)
    (confirmed,) = tracker.confirmed_tracks
    assert confirmed.confirmed_at == 2.0
    assert len(tracker.tracks) == 2  # second still tentative, 1 hit in 3 scans
    for t in (3.0, 4.0):
        tracker.step([at([0, 0, 0])], t)
    assert [t.id for t in tracker.tracks] == [confirmed.id]  # 1 of 5: dropped


def test_single_hit_confirmation_and_validation():
    tracker = MultiTargetTracker(
        ConstantVelocity(1.0), confirm_hits=1, confirm_window=1
    )
    tracker.step([at([0, 0, 0])], 0.0)
    assert tracker.confirmed_tracks
    with pytest.raises(ValueError, match="confirm_hits"):
        MultiTargetTracker(ConstantVelocity(1.0), confirm_hits=4, confirm_window=3)


def test_class_posterior_accumulates_tempered_evidence():
    probs = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
    naive = tempered = None
    for _ in range(5):
        naive = pool_class_evidence(naive, probs, 1.0)
        tempered = pool_class_evidence(tempered, probs, 0.3)
    assert naive[0] > tempered[0] > probs[0] * 0.9
    np.testing.assert_allclose(naive.sum(), 1.0)
    # Five naive updates equal one update with probs^5 (independence).
    expected = probs**5 / np.sum(probs**5)
    np.testing.assert_allclose(naive, expected, rtol=1e-6)
    with pytest.raises(ValueError, match="weight"):
        pool_class_evidence(None, probs, 0.0)


def test_non_uniform_prior_is_divided_out():
    prior = np.array([0.7, 0.3])
    posterior = pool_class_evidence(None, np.array([0.7, 0.3]), 1.0, prior=prior)
    np.testing.assert_allclose(posterior, prior)  # output equal to prior = no evidence


def test_tracker_pools_class_probabilities():
    tracker = MultiTargetTracker(ConstantVelocity(1.0), class_weight=0.5)
    probs = np.array([0.1, 0.7, 0.1, 0.05, 0.05])
    for t in range(4):
        tracker.step([at([0, 0, 0], probs=probs)], float(t))
    (track,) = tracker.tracks
    assert track.class_posterior is not None
    assert int(np.argmax(track.class_posterior)) == 1
    assert track.class_posterior[1] > probs[1]
