"""Exact permutation tasks, artifact integrity and policy information limits."""

from collections import deque
from copy import deepcopy
import hashlib
import itertools
import json

import numpy as np
import pytest

from copuzzle.games.permutation.evaluation import (
    OraclePolicy, RandomPolicy, audit_overlap, evaluate_policy, generate_task_suite,
    load_task_suite, save_task_suite, validate_task_suite, write_json,
)
from copuzzle.games.permutation.rules import get_rule


def _resign(suite):
    payload = {key: value for key, value in suite.items() if key != "task_suite_sha256"}
    suite["task_suite_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return suite


def _pancake_bfs(start):
    """Independent list reversal reference, without production rule/oracle calls."""
    start = tuple(start)
    distances = {start: 0}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        for width in range(2, len(state) + 1):
            successor = state[:width][::-1] + state[width:]
            if successor not in distances:
                distances[successor] = distances[state] + 1
                queue.append(successor)
    return distances


def _topspin_bfs(start, window):
    start = tuple(start)
    distances = {start: 0}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        successors = (state[1:] + state[:1], state[-1:] + state[:-1],
                      state[:window][::-1] + state[window:])
        for successor in successors:
            if successor not in distances:
                distances[successor] = distances[state] + 1
                queue.append(successor)
    return distances


def _suite(**kwargs):
    defaults = dict(depths=(1, 2, 3, 4), tasks_per_depth=3, seed=4, max_episode_steps=8)
    defaults.update(kwargs)
    return generate_task_suite(get_rule("pancake-4"), **defaults)


def _pairs(suite):
    return {(tuple(case["state"]), tuple(case["goal"])) for case in suite["cases"]}


def test_strata_have_independent_exact_distances_and_diverse_starts():
    suite = _suite()
    for case in suite["cases"]:
        reference = _pancake_bfs(case["state"])
        assert reference[tuple(case["goal"])] == case["optimal_steps"]
    assert len(_pairs(suite)) == len(suite["cases"])
    assert len({tuple(case["state"]) for case in suite["cases"]}) > 1
    assert suite == _suite()
    assert suite != _suite(seed=5)


def test_oracle_completes_at_exact_minimum_distance():
    suite = _suite()
    report = evaluate_policy(OraclePolicy(get_rule("pancake-4")), suite)
    assert report["summary"]["success_rate"] == 1.0
    assert report["summary"]["timeout_rate"] == 0.0
    assert report["summary"]["mean_excess_steps_on_success"] == 0.0
    assert report["summary"]["policy_calls"] == sum(c["optimal_steps"] for c in suite["cases"])
    assert report["task_suite_sha256"] == suite["task_suite_sha256"]
    assert report["privileged_rule_access"] is True
    assert report["method_declarations"]["uses_solver"] is True
    assert set(report["by_distance"]) == {"1", "2", "3", "4"}
    assert all(row["steps"] == row["optimal_steps"] for row in report["cases"])


def test_topspin_strata_and_oracle_respect_disconnected_components():
    # Full-window reversal and rotation generate only 8 of the 24 states.
    # Sampling arbitrary starts must use each start's component, not reject
    # states outside the identity component or assume all pairs reachable.
    assert len(_topspin_bfs((1, 2, 3, 4), 4)) == 8
    rule = get_rule("topspin-4-4")
    suite = generate_task_suite(rule, depths=(1, 2, 3), tasks_per_depth=8, seed=2)
    for case in suite["cases"]:
        assert _topspin_bfs(case["state"], 4)[tuple(case["goal"])] == case["optimal_steps"]
    assert any(tuple(case["state"]) not in _topspin_bfs((1, 2, 3, 4), 4) for case in suite["cases"])
    report = evaluate_policy(OraclePolicy(rule), suite)
    assert report["summary"]["success_rate"] == 1.0
    assert report["summary"]["mean_excess_steps_on_success"] == 0.0


def test_policy_gets_only_copied_observations_and_reset_per_case():
    suite = _suite()

    class Policy:
        def __init__(self):
            self.oracle = OraclePolicy(get_rule("pancake-4"))
            self.resets = 0

        def reset(self):
            self.resets += 1

        def __call__(self, obs):
            assert set(obs) == {"board", "goal", "previous_event", "action_mask"}
            action = self.oracle(obs)
            for array in obs.values():
                array[:] = 0
            return action

    policy = Policy()
    report = evaluate_policy(policy, suite)
    assert report["summary"]["success_rate"] == 1.0
    assert policy.resets == len(suite["cases"])
    assert report["method_declarations"]["uses_solver"] is None
    assert report["parameter_freezing_enforced"] is False
    assert validate_task_suite(suite).name == "pancake-4"


@pytest.mark.parametrize("action", [True, -1, 3, 0.5, np.nan, None, [0]])
def test_malformed_policy_actions_fail(action):
    with pytest.raises((ValueError, TypeError)):
        evaluate_policy(lambda obs: action, _suite(depths=(1,), tasks_per_depth=1))


def test_distinct_suites_exclude_directed_pairs():
    development = _suite(tasks_per_depth=8)
    final = _suite(tasks_per_depth=8, exclude_pairs=development["cases"])
    assert not _pairs(development) & _pairs(final)
    assert final["excluded_pairs_count"] == len(development["cases"])
    equivalent = _suite(tasks_per_depth=8, exclude_pairs=sorted(_pairs(development)))
    assert equivalent == final


def test_small_stratum_is_exhausted_exactly_without_retries():
    rule = get_rule("pancake-2")
    full = generate_task_suite(rule, depths=(1,), tasks_per_depth=2)
    assert _pairs(full) == {((1, 2), (2, 1)), ((2, 1), (1, 2))}
    with pytest.raises(ValueError, match="Only 2"):
        generate_task_suite(rule, depths=(1,), tasks_per_depth=3)
    with pytest.raises(ValueError, match="Only 0"):
        generate_task_suite(rule, depths=(1,), tasks_per_depth=1, exclude_pairs=full["cases"])
    remaining = generate_task_suite(rule, depths=(1,), tasks_per_depth=1, exclude_pairs=full["cases"][:1])
    assert len(_pairs(remaining) | {next(iter(_pairs({"cases": full["cases"][:1]})))}) == 2


def test_all_small_pairs_and_nearly_exhausted_exclusions():
    rule = get_rule("pancake-3")
    permutations = list(itertools.permutations(range(1, 4)))
    all_depth_one = [(state, goal) for state in permutations
                     for goal, distance in _pancake_bfs(state).items() if distance == 1]
    suite = generate_task_suite(rule, depths=(1,), tasks_per_depth=len(all_depth_one))
    assert _pairs(suite) == set(all_depth_one)
    for omitted in range(len(all_depth_one)):
        excluded = all_depth_one[:omitted] + all_depth_one[omitted + 1:]
        remaining = generate_task_suite(rule, depths=(1,), tasks_per_depth=1, exclude_pairs=excluded)
        assert _pairs(remaining) == {all_depth_one[omitted]}


@pytest.mark.parametrize("changes", [
    {"depths": ()}, {"depths": (1, 1)}, {"depths": (0,)}, {"depths": (True,)},
    {"depths": (5,)}, {"depths": (4,), "max_episode_steps": 3},
    {"tasks_per_depth": 0}, {"tasks_per_depth": True}, {"seed": -1},
    {"exclude_pairs": [(1, 2)]},
])
def test_invalid_generation_requests(changes):
    with pytest.raises((ValueError, TypeError)):
        _suite(**changes)


def test_artifact_round_trip_and_overwrite_protection(tmp_path):
    suite = _suite()
    path = tmp_path / "suite.json"
    save_task_suite(suite, path)
    assert load_task_suite(path) == suite
    with pytest.raises(FileExistsError):
        save_task_suite(suite, path)
    with pytest.raises(FileExistsError):
        write_json(path, {})
    suite["cases"][0]["optimal_steps"] += 1
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="checksum"):
        load_task_suite(path)


@pytest.mark.parametrize("mutation", [
    lambda suite: suite.update(rule_id="0" * 64),
    lambda suite: suite.update(task_name="pancake-3"),
    lambda suite: suite.update(max_episode_steps=1),
    lambda suite: suite.update(n_actions=True),
    lambda suite: suite.update(observation_profile="with_oracle"),
    lambda suite: suite["cases"][0].update(optimal_steps=2),
    lambda suite: suite["cases"][0].update(case_id="forged"),
    lambda suite: suite["cases"].__setitem__(0, deepcopy(suite["cases"][1])),
    lambda suite: suite["cases"][0].update(state=[1, 1, 2, 3]),
])
def test_rehashed_but_inconsistent_suites_are_rejected(mutation):
    suite = _suite()
    mutation(suite)
    with pytest.raises(ValueError):
        validate_task_suite(_resign(suite))


def test_wrong_oracle_and_supplied_rule_are_rejected():
    suite = _suite()
    with pytest.raises(ValueError):
        validate_task_suite(suite, get_rule("pancake-3"))
    with pytest.raises(ValueError):
        evaluate_policy(OraclePolicy(get_rule("pancake-3")), suite)


def test_random_is_repeatable_and_has_honest_declarations():
    suite = _suite(max_episode_steps=4)
    first = evaluate_policy(RandomPolicy(3, seed=9), suite)
    second = evaluate_policy(RandomPolicy(3, seed=9), suite)
    assert [row["steps"] for row in first["cases"]] == [row["steps"] for row in second["cases"]]
    assert [row["success"] for row in first["cases"]] == [row["success"] for row in second["cases"]]
    assert first["privileged_rule_access"] is False
    assert first["summary"]["timeout_rate"] > 0
    assert all(row["excess_steps_on_success"] is None for row in first["cases"] if not row["success"])


def test_overlap_requires_strict_order_in_one_trajectory():
    from copuzzle.games.permutation.data import generate_dataset

    rule = get_rule("pancake-2")
    dataset = generate_dataset(rule, num_episodes=1, horizon=1, seed=4)
    suite = generate_task_suite(rule, depths=(1,), tasks_per_depth=2)
    report = audit_overlap(suite, dataset)
    start, goal = (tuple(int(x) for x in state) for state in dataset.states[0])
    seen = {row["case_id"]: row["forward_pair_seen_in_one_trajectory"] for row in report["cases"]}
    for case in suite["cases"]:
        assert seen[case["case_id"]] == ((tuple(case["state"]), tuple(case["goal"])) == (start, goal))
    assert report["start_seen_fraction"] == 1.0
    assert report["goal_seen_fraction"] == 1.0
    assert report["forward_pair_seen_fraction"] == 0.5
    other = _suite()
    with pytest.raises(ValueError, match="matching"):
        audit_overlap(other, dataset)
def test_task_loader_rejects_duplicate_json_fields(tmp_path):
    from copuzzle.games.permutation.rules import get_rule
    from copuzzle.games.permutation.evaluation import generate_task_suite, save_task_suite, load_task_suite
    import pytest

    suite = generate_task_suite(get_rule("pancake-4"), depths=(1,), tasks_per_depth=1, seed=100)
    path = tmp_path / "tasks.json"
    save_task_suite(suite, path)
    path.write_text(path.read_text().replace('"seed": 100', '"seed": 999, "seed": 100'))
    with pytest.raises(ValueError, match="Duplicate"):
        load_task_suite(path)
