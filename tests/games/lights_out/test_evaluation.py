"""Behavioral checks for fixed goal tasks, policy access, and overlap claims."""

from collections import Counter, deque
from copy import deepcopy
import json

import numpy as np
import pytest

from copuzzle.data import generate_dataset
from copuzzle.evaluation import (
    OraclePolicy,
    RandomPolicy,
    audit_overlap,
    evaluate_policy,
    generate_task_suite,
    load_task_suite,
    validate_task_suite,
    write_json,
)
from copuzzle.oracle import solve
from copuzzle.rules import get_rule


def _mask(board):
    return sum(int(bit) << index for index, bit in enumerate(board))


def _bfs_distances(rule):
    """Graph search checks distance labels without invoking the GF(2) solver."""
    effects = [_mask(rule.toggle_matrix[:, index]) for index in range(rule.n_buttons)]
    distances = {0: 0}
    queue = deque([0])
    while queue:
        state = queue.popleft()
        for effect in effects:
            successor = state ^ effect
            if successor not in distances:
                distances[successor] = distances[state] + 1
                queue.append(successor)
    return distances


def _suite_for_pairs(rule, pairs, *, horizon=8):
    suite = generate_task_suite(rule, per_depth=1, depths=(1,), seed=901, max_episode_steps=horizon)
    suite["cases"] = [
        {"case_id": f"test-case-{index}", "state": np.asarray(start).tolist(),
         "goal": np.asarray(goal).tolist(), "optimal_steps": len(solve(rule, start, goal))}
        for index, (start, goal) in enumerate(pairs)
    ]
    return suite


@pytest.mark.parametrize("task_id", ["grid-3x3", "grid-4x4", "graph-cycle9"])
def test_exact_distance_strata_against_bfs(task_id):
    rule = get_rule(task_id)
    suite = generate_task_suite(rule, per_depth=5, depths=(1, 2, 3, 4), seed=19)
    distances = _bfs_distances(rule)
    assert Counter(case["optimal_steps"] for case in suite["cases"]) == {1: 5, 2: 5, 3: 5, 4: 5}
    for case in suite["cases"]:
        delta = _mask(case["state"]) ^ _mask(case["goal"])
        assert distances[delta] == case["optimal_steps"]
    assert validate_task_suite(suite, rule) is rule
    assert len({case["case_id"] for case in suite["cases"]}) == 20
    assert len({(tuple(case["state"]), tuple(case["goal"])) for case in suite["cases"]}) == 20


def test_fixed_seed_generates_identical_suite_and_json(tmp_path):
    rule = get_rule("grid-3x3")
    first = generate_task_suite(rule, per_depth=3, seed=302)
    second = generate_task_suite(rule, per_depth=3, seed=302)
    assert first == second
    assert first["cases"] != generate_task_suite(rule, per_depth=3, seed=303)["cases"]
    first_path, second_path = tmp_path / "first.json", tmp_path / "second.json"
    write_json(first_path, first)
    write_json(second_path, second)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert load_task_suite(first_path) == first
    with pytest.raises(FileExistsError):
        write_json(first_path, first)
    assert json.loads(first_path.read_text()) == first


@pytest.mark.parametrize("duplication", ["whole_case", "pair", "id"])
def test_duplicate_case_rejected(duplication):
    suite = generate_task_suite(get_rule("grid-3x3"), per_depth=2, depths=(1,), seed=5)
    if duplication == "whole_case":
        suite["cases"][1] = deepcopy(suite["cases"][0])
    elif duplication == "pair":
        second_id = suite["cases"][1]["case_id"]
        suite["cases"][1] = deepcopy(suite["cases"][0])
        suite["cases"][1]["case_id"] = second_id
    else:
        suite["cases"][1]["case_id"] = suite["cases"][0]["case_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_task_suite(suite)


def test_incorrect_distance_and_unreachable_goal_rejected():
    rule = get_rule("grid-4x4")
    suite = generate_task_suite(rule, per_depth=1, depths=(1,), seed=3)
    wrong_distance = deepcopy(suite)
    wrong_distance["cases"][0]["optimal_steps"] = 2
    with pytest.raises(ValueError, match="incorrect shortest distance"):
        validate_task_suite(wrong_distance)

    reachable_deltas = _bfs_distances(rule)
    unreachable_delta = next(mask for mask in range(1 << rule.n_buttons) if mask not in reachable_deltas)
    unreachable = deepcopy(suite)
    unreachable["cases"][0]["state"] = [0] * rule.n_buttons
    unreachable["cases"][0]["goal"] = [(unreachable_delta >> index) & 1 for index in range(rule.n_buttons)]
    with pytest.raises(ValueError, match="Unreachable"):
        validate_task_suite(unreachable)


@pytest.mark.parametrize("distance", [0, -1, 1.0, True, 31])
def test_invalid_distance_metadata_rejected(distance):
    suite = generate_task_suite(get_rule("grid-3x3"), per_depth=1, depths=(1,))
    suite["cases"][0]["optimal_steps"] = distance
    with pytest.raises(ValueError, match="optimal_steps"):
        validate_task_suite(suite)


@pytest.mark.parametrize("task_id", ["grid-3x3", "grid-4x4", "graph-cycle9"])
def test_oracle_solves_every_case_in_exactly_optimal_presses(task_id):
    rule = get_rule(task_id)
    suite = generate_task_suite(rule, per_depth=3, depths=(1, 2, 3, 4), seed=54)
    result = evaluate_policy(OraclePolicy(rule), suite)
    assert result["privileged_rule_access"] is True
    assert result["policy"] == "exact_rule_oracle"
    assert result["summary"]["success_rate"] == 1.0
    assert result["summary"]["timeout_rate"] == 0.0
    assert result["summary"]["mean_excess_presses_on_success"] == 0.0
    for case in result["cases"]:
        assert case["success"] and not case["timeout"]
        assert case["presses"] == case["optimal_steps"] == case["policy_calls"]
        assert case["excess_presses_on_success"] == 0
        assert case["return"] == -(case["presses"] - 1)
        assert case["inference_seconds"] >= 0
    assert result["summary"]["policy_calls"] == sum(c["presses"] for c in result["cases"])
    assert set(result["by_optimal_steps"]) == {"1", "2", "3", "4"}


def test_policy_receives_only_public_observation_and_past_event():
    rule = get_rule("grid-3x3")
    suite = generate_task_suite(rule, per_depth=2, depths=(2,), seed=73, max_episode_steps=4)

    class ObservationSpy:
        def __init__(self):
            self.episodes = []

        def reset(self):
            self.episodes.append([])

        def __call__(self, observation):
            assert set(observation) == {"board", "goal", "previous_event"}
            assert all(isinstance(value, np.ndarray) for value in observation.values())
            assert all(value.shape == (9,) for value in observation.values())
            self.episodes[-1].append(deepcopy(observation))
            return 0

    policy = ObservationSpy()
    result = evaluate_policy(policy, suite)
    assert result["privileged_rule_access"] is False
    assert len(policy.episodes) == len(suite["cases"])
    for history, case in zip(policy.episodes, suite["cases"]):
        assert len(history) == 4  # Repeating one button cannot achieve a distance-two goal.
        np.testing.assert_array_equal(history[0]["board"], case["state"])
        np.testing.assert_array_equal(history[0]["previous_event"], np.zeros(9))
        for observation in history:
            np.testing.assert_array_equal(observation["goal"], case["goal"])
        for observation in history[1:]:
            np.testing.assert_array_equal(observation["previous_event"], [1, 0, 0, 0, 0, 0, 0, 0, 0])


def test_stateful_policy_resets_between_cases_without_task_arguments():
    rule = get_rule("grid-3x3")
    starts = [np.zeros(9, dtype=np.uint8), np.ones(9, dtype=np.uint8)]
    pairs = [(start, rule.transition(rule.transition(start, 0), 1)) for start in starts]
    suite = _suite_for_pairs(rule, pairs, horizon=2)

    class TwoPressPolicy:
        def __init__(self):
            self.resets = 0
            self.index = 99

        def reset(self):
            self.resets += 1
            self.index = 0

        def __call__(self, observation):
            action = [0, 1][self.index]
            self.index += 1
            return action

    policy = TwoPressPolicy()
    result = evaluate_policy(policy, suite)
    assert policy.resets == 2
    assert result["summary"]["success_rate"] == 1
    assert all(case["presses"] == 2 and not case["timeout"] for case in result["cases"])


def test_horizon_timeout_and_success_on_last_step_are_distinct():
    rule = get_rule("grid-3x3")
    start = np.zeros(9, dtype=np.uint8)
    goal = rule.transition(start, 4)
    suite = _suite_for_pairs(rule, [(start, goal)], horizon=1)
    failed = evaluate_policy(lambda observation: 0, suite)
    failure = failed["cases"][0]
    assert not failure["success"] and failure["timeout"]
    assert failure["presses"] == failure["policy_calls"] == 1
    assert failure["return"] == -1
    assert failure["excess_presses_on_success"] is None
    assert failed["summary"]["mean_successful_presses"] is None

    succeeded = evaluate_policy(OraclePolicy(rule), suite)["cases"][0]
    assert succeeded["success"] and not succeeded["timeout"]
    assert succeeded["presses"] == 1
    assert succeeded["return"] == 0


def test_random_policy_reproducibility_excluding_wall_clock():
    rule = get_rule("grid-3x3")
    suite = generate_task_suite(rule, per_depth=2, depths=(1, 2), seed=19, max_episode_steps=5)
    first = evaluate_policy(RandomPolicy(seed=111), suite)
    second = evaluate_policy(RandomPolicy(seed=111), suite)
    for result in (first, second):
        for section in [result["summary"], *result["by_optimal_steps"].values(), *result["cases"]]:
            section.pop("inference_seconds")
    assert first == second


def _dataset_with_explicit_episodes(rule, starts, actions):
    actions = np.asarray(actions, dtype=np.int64)
    dataset = generate_dataset(rule, num_episodes=len(starts), horizon=actions.shape[1], seed=3)
    dataset.actions[:] = actions
    dataset.events[:] = np.eye(rule.n_buttons, dtype=np.uint8)[actions]
    for episode, start in enumerate(starts):
        dataset.states[episode, 0] = start
        for step, action in enumerate(actions[episode]):
            dataset.states[episode, step + 1] = rule.transition(dataset.states[episode, step], int(action))
    dataset.validate(rule)
    return dataset


def test_overlap_checks_state_membership_order_and_episode_boundary():
    rule = get_rule("grid-3x3")
    dataset = _dataset_with_explicit_episodes(
        rule, [np.zeros(9, dtype=np.uint8), np.ones(9, dtype=np.uint8)], [[0, 1], [2, 3]]
    )
    a, b, c = dataset.states[0]
    d, e, f = dataset.states[1]
    all_states = {state.tobytes() for episode in dataset.states for state in episode}
    unseen = next(
        np.array([(mask >> index) & 1 for index in range(9)], dtype=np.uint8)
        for mask in range(512)
        if np.array([(mask >> index) & 1 for index in range(9)], dtype=np.uint8).tobytes() not in all_states
    )
    suite = _suite_for_pairs(rule, [(a, c), (c, a), (a, d), (d, f), (unseen, b)])
    report = audit_overlap(dataset, suite)
    rows = report["cases"]
    assert [row["forward_pair_seen_in_one_trajectory"] for row in rows] == [True, False, False, True, False]
    assert [row["start_seen"] for row in rows] == [True, True, True, True, False]
    assert all(row["goal_seen"] for row in rows)
    assert report["training_unique_states"] == 6
    assert report["start_seen_fraction"] == 4 / 5
    assert report["goal_seen_fraction"] == 1
    assert report["forward_pair_seen_fraction"] == 2 / 5


def test_overlap_uses_later_recurrence_of_goal():
    rule = get_rule("grid-3x3")
    dataset = _dataset_with_explicit_episodes(rule, [np.zeros(9, dtype=np.uint8)], [[0, 0]])
    a, b, repeated_a = dataset.states[0]
    np.testing.assert_array_equal(a, repeated_a)
    suite = _suite_for_pairs(rule, [(b, a)])
    assert audit_overlap(dataset, suite)["cases"][0]["forward_pair_seen_in_one_trajectory"]


def test_independent_seed_does_not_establish_state_or_pair_novelty():
    rule = get_rule("grid-1x1")
    dataset = generate_dataset(rule, num_episodes=1, horizon=2, seed=3, split="train")
    suite = generate_task_suite(rule, per_depth=2, depths=(1,), seed=887)
    assert suite["seed"] != dataset.manifest["seed"]
    report = audit_overlap(dataset, suite)
    assert report["training_unique_states"] == 2
    assert report["start_seen_fraction"] == report["goal_seen_fraction"] == 1
    assert report["forward_pair_seen_fraction"] == 1


def test_overlap_rejects_different_rule_instance():
    dataset = generate_dataset(get_rule("grid-3x3"), num_episodes=1, horizon=2)
    suite = generate_task_suite(get_rule("graph-cycle9"), per_depth=1, depths=(1,))
    with pytest.raises(ValueError, match="same rule"):
        audit_overlap(dataset, suite)


@pytest.mark.parametrize("malformation", ["empty", "missing_name", "missing_state", "missing_goal", "distance", "nonbinary"])
def test_overlap_validates_task_suite_before_reporting(malformation):
    rule = get_rule("grid-3x3")
    dataset = generate_dataset(rule, num_episodes=1, horizon=2)
    suite = generate_task_suite(rule, per_depth=1, depths=(1,))
    if malformation == "empty":
        suite["cases"] = []
    elif malformation == "missing_name":
        del suite["task_name"]
    elif malformation in {"missing_state", "missing_goal"}:
        del suite["cases"][0][malformation.removeprefix("missing_")]
    elif malformation == "distance":
        suite["cases"][0]["optimal_steps"] = 2
    else:
        suite["cases"][0]["state"][0] = 2
    with pytest.raises(ValueError):
        audit_overlap(dataset, suite)


def test_overlap_normalizes_valid_binary_numeric_states():
    rule = get_rule("grid-3x3")
    dataset = generate_dataset(rule, num_episodes=1, horizon=2)
    suite = _suite_for_pairs(rule, [(dataset.states[0, 0], dataset.states[0, 1])])
    expected = audit_overlap(dataset, suite)
    for case in suite["cases"]:
        case["state"] = [float(bit) for bit in case["state"]]
        case["goal"] = [float(bit) for bit in case["goal"]]
    assert validate_task_suite(suite, rule) is rule
    assert audit_overlap(dataset, suite) == expected


@pytest.mark.parametrize("options", [
    {"depths": ()}, {"depths": (1, 1)}, {"depths": (0,)}, {"depths": (True,)},
    {"depths": (1.0,)}, {"depths": (10,)}, {"per_depth": 0}, {"seed": -1},
    {"max_episode_steps": True}, {"max_attempts_per_depth": 0},
])
def test_invalid_generation_settings_rejected(options):
    with pytest.raises(ValueError):
        generate_task_suite(get_rule("grid-3x3"), **options)


def test_unattainable_distance_stops_at_attempt_budget():
    # cycle9 has binary rank 7 but graph diameter 4: rank is only an upper bound.
    rule = get_rule("graph-cycle9")
    with pytest.raises(ValueError, match="unattainable|attempt budget"):
        generate_task_suite(rule, depths=(5,), per_depth=1, max_attempts_per_depth=20)
