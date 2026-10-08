import numpy as np

from sentinel.tracking import assign_gnn


def test_gnn_beats_greedy_on_classic_counterexample():
    # Greedy takes (0,0)=1 then (1,1)=100 (total 101); GNN finds 2 + 2 = 4.
    cost = np.array([[1.0, 2.0], [2.0, 100.0]])
    assert assign_gnn(cost, unassigned_cost=1e9) == [(0, 1), (1, 0)]


def test_infeasible_pairs_are_never_assigned():
    cost = np.array([[1.0, np.inf], [np.inf, np.inf]])
    assert assign_gnn(cost, unassigned_cost=9.0) == [(0, 0)]


def test_track_stays_unassigned_when_cheaper():
    cost = np.array([[10.0]])
    assert assign_gnn(cost, unassigned_cost=5.0) == []
    assert assign_gnn(cost, unassigned_cost=15.0) == [(0, 0)]


def test_rectangular_and_empty_inputs():
    assert assign_gnn(np.zeros((0, 3)), 1.0) == []
    assert assign_gnn(np.zeros((2, 0)), 1.0) == []
    pairs = assign_gnn(np.array([[3.0, 1.0, 2.0]]), unassigned_cost=10.0)
    assert pairs == [(0, 1)]


def test_per_track_unassigned_costs():
    cost = np.array([[4.0], [4.0]])
    pairs = assign_gnn(cost, unassigned_cost=np.array([1.0, 9.0]))
    assert pairs == [(1, 0)]
