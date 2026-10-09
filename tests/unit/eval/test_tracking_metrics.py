import numpy as np
import pytest

from sentinel.eval import Snapshot, evaluate_tracking, gospa, ospa

C = 100.0


def test_gospa_cases():
    t = np.array([[0.0, 0, 0], [1_000.0, 0, 0]])
    assert gospa(t, t, C).distance == 0.0
    one_missed = gospa(t, t[:1], C)
    assert (one_missed.missed, one_missed.false) == (1, 0)
    assert one_missed.distance == pytest.approx(C / np.sqrt(2))
    shifted = gospa(t, t + np.array([3.0, 4.0, 0.0]), C)
    assert shifted.localization == pytest.approx(50.0)
    far = gospa(t[:1], t[:1] + 500.0, C)  # beyond cutoff: one missed + one false
    assert (far.missed, far.false) == (1, 1)
    assert gospa(np.zeros((0, 3)), np.zeros((0, 3)), C).distance == 0.0


def test_ospa_cases():
    t = np.array([[0.0, 0, 0]])
    assert ospa(t, t, C) == 0.0
    assert ospa(t, np.zeros((0, 3)), C) == C
    assert ospa(np.zeros((0, 3)), np.zeros((0, 3)), C) == 0.0


def snap(time, truth, ids, positions):
    pos = np.array(positions, float).reshape(-1, 3)
    return Snapshot(
        time, truth, tuple(ids), pos, np.tile(np.eye(3) * 4.0, (len(ids), 1, 1))
    )


def test_track_level_measures():
    target = {"a": np.zeros(3)}
    snapshots = [
        snap(0.0, target, [], []),  # appears, not yet tracked
        snap(1.0, target, [7], [[2.0, 0, 0]]),  # tracked by 7
        snap(2.0, target, [], []),  # gap
        snap(3.0, target, [9], [[0, 2.0, 0]]),  # resumes under a new id
        snap(4.0, target, [9, 5], [[0, 2.0, 0], [5_000.0, 0, 0]]),  # plus a false track
    ]
    result = evaluate_tracking(snapshots, C)
    assert result.latency == {"a": 1.0}
    assert result.fragmentations == 1
    assert result.id_switches == 1
    assert result.purity == 1.0
    assert result.localization_rmse == pytest.approx(2.0)
    assert result.nees_mean == pytest.approx(1.0)
    assert result.false_mean == pytest.approx(0.2)
    assert result.missed_mean == pytest.approx(0.4)
    assert result.as_dict()["scans"] == 5


def test_never_tracked_target_has_infinite_latency():
    result = evaluate_tracking([snap(0.0, {"x": np.zeros(3)}, [], [])], C)
    assert result.latency["x"] == float("inf")
