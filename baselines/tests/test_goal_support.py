import numpy as np
import pytest

from copuzzle_baselines.study.goal_support import observed_graph_context, goal_support_scores


def test_observed_reachability_changes_when_a_bridge_is_removed():
    states = np.asarray([[0], [1], [1], [2], [3]], dtype=np.int32)
    targets = np.asarray([[1], [0], [2], [1], [3]], dtype=np.int32)
    legal = np.asarray([1, 1, 1, 1, 0], dtype=np.uint8)
    starts, goals = np.asarray([[0], [0], [3], [8]], np.int32), np.asarray([[2], [3], [3], [2]], np.int32)
    full = observed_graph_context(states, targets, legal, starts, goals)
    partial = observed_graph_context(states[[0, 1, 4]], targets[[0, 1, 4]], legal[[0, 1, 4]], starts, goals)
    assert full["legal_edges_inverse_closed"] and full["case_observed_path_exists"] == [True, False, True, False]
    assert partial["case_observed_path_exists"] == [False, False, True, False]
    assert full["weak_components"] == 2 and full["largest_weak_component"] == 3
    assert partial["case_goal_observed"] == [False, True, True, False]


def test_weak_components_do_not_imply_a_directed_path():
    result = observed_graph_context(np.asarray([[0], [1]]), np.asarray([[1], [2]]), np.asarray([1, 1]),
                                    np.asarray([[0], [2]]), np.asarray([[2], [0]]))
    assert not result["legal_edges_inverse_closed"] and result["weak_components"] == 1
    assert result["case_observed_path_exists"] == [True, False]


def test_unavailable_evaluations_keep_support_strata_and_zero_success():
    context = {"case_observed_path_exists": [True, False, False]}
    rows = [{"case_index": i, "success": i == 1} for i in range(3)]
    assert goal_support_scores(context, rows) == {"observed_path": {"cases": 1, "success_rate": 0}, "no_observed_path": {"cases": 2, "success_rate": .5}}
    assert goal_support_scores(context, None)["no_observed_path"] == {"cases": 2, "success_rate": 0}
    with pytest.raises(ValueError, match="unaligned"):
        goal_support_scores(context, rows[:2])
