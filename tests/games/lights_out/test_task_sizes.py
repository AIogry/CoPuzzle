"""Small integration checks for parameterized task sizes and legacy identity."""

import numpy as np
import pytest

from copuzzle.games.lights_out import make_env
from copuzzle.games.lights_out.data import (
    GoalSampler,
    generate_dataset,
    load_dataset,
    relabel_transitions,
)
from copuzzle.games.lights_out.evaluation import (
    evaluate_policy,
    generate_task_suite,
    load_task_suite,
    write_json,
)
from copuzzle.games.lights_out.rules import get_rule


EXPANDED_TASKS = (
    "grid-4x5", "grid-4x6",
    "graph-path16", "graph-path20", "graph-path24",
    "graph-cycle16", "graph-cycle20", "graph-cycle24",
)


@pytest.mark.parametrize("family,size", [
    *(('path', size) for size in (1, 2, 9, 16, 20, 24)),
    *(('cycle', size) for size in (3, 9, 16, 20, 24)),
])
def test_parameterized_graph_neighbors(family, size):
    rule = get_rule(f"graph-{family}{size}")
    assert rule.name == f"graph-{family}{size}"
    assert rule.n_buttons == size
    # Check each labeled neighbor set, including path endpoints and cycle wrap.
    for node in range(size):
        if family == 'cycle':
            expected = {(node - 1) % size, (node + 1) % size}
        else:
            expected = {other for other in (node - 1, node + 1) if 0 <= other < size}
        assert set(np.flatnonzero(rule.adjacency[node])) == expected
        pressed = rule.transition(np.zeros(size, dtype=np.uint8), node)
        assert set(np.flatnonzero(pressed)) == expected | {node}


@pytest.mark.parametrize("task_id", [
    "graph-path0", "graph-path-1", "graph-cycle0", "graph-cycle1",
    "graph-cycle2", "graph-cycle-1",
])
def test_graph_size_lower_bounds(task_id):
    with pytest.raises(ValueError):
        get_rule(task_id)


@pytest.mark.parametrize("cols", [5, 6])
def test_large_grid_corners_row_boundaries_and_interior(cols):
    rule = get_rule(f"grid-4x{cols}")
    assert rule.n_buttons == 4 * cols
    zero = np.zeros(rule.n_buttons, dtype=np.uint8)
    expected_effects = {
        0: {0, 1, cols},
        cols - 1: {cols - 2, cols - 1, 2 * cols - 1},
        cols: {0, cols, cols + 1, 2 * cols},
        cols + 1: {1, cols, cols + 1, cols + 2, 2 * cols + 1},
        3 * cols: {2 * cols, 3 * cols, 3 * cols + 1},
        4 * cols - 1: {3 * cols - 1, 4 * cols - 2, 4 * cols - 1},
    }
    for action, expected in expected_effects.items():
        assert set(np.flatnonzero(rule.transition(zero, action))) == expected
    # Adjacent row-major indices across a row boundary are not grid neighbors.
    assert rule.adjacency[cols - 1, cols] == 0


@pytest.mark.parametrize("task_id,expected_id", [
    ("graph-path9", "f0240a42255c1a20d9f80c30c6f90b71203db4b00aa9176e6564529d5c35cc72"),
    ("graph-cycle9", "8df9fec29562ebc1d35ce00b6e63b0e84df0cc55f4a967c2f09aa674366f7d8f"),
])
def test_legacy_graph_rule_ids_are_stable(task_id, expected_id):
    # Snapshots from the original fixed graph9 implementations, before expansion.
    assert get_rule(task_id).rule_id == expected_id


@pytest.mark.parametrize("task_id", EXPANDED_TASKS)
def test_expanded_task_last_action_and_hidden_observation(task_id):
    rule = get_rule(task_id)
    size = rule.n_buttons
    start = np.zeros(size, dtype=np.uint8)
    goal = rule.transition(start, size - 1)
    env = make_env(task_id, max_episode_steps=1)
    try:
        observation, info = env.reset(options={"state": start, "goal": goal})
        assert env.action_space.n == size
        assert env.action_space.contains(size - 1)
        assert not env.action_space.contains(size)
        assert set(observation) == {"board", "goal", "previous_event"}
        assert all(value.shape == (size,) for value in observation.values())
        assert env.observation_space.contains(observation)
        assert set(info) == {"is_success", "steps", "rule_id"}
        with pytest.raises(ValueError):
            env.step(size)
        after, reward, terminated, truncated, info = env.step(size - 1)
        np.testing.assert_array_equal(after["board"], goal)
        assert set(np.flatnonzero(after["previous_event"])) == {size - 1}
        assert set(after) == set(observation)
        assert env.observation_space.contains(after)
        assert (reward, terminated, truncated) == (0.0, True, False)
        assert info["steps"] == 1
    finally:
        env.close()


@pytest.mark.parametrize("task_id", EXPANDED_TASKS)
def test_expanded_task_data_goals_and_evaluation(tmp_path, task_id):
    rule = get_rule(task_id)
    size = rule.n_buttons
    dataset = generate_dataset(rule, num_episodes=2, horizon=3, seed=17)
    archive = tmp_path / "train.npz"
    dataset.save(archive)
    loaded = load_dataset(archive)
    loaded.validate(rule)
    assert loaded.states.shape == (2, 4, size)
    assert loaded.actions.shape == (2, 3)
    assert loaded.events.shape == (2, 3, size)
    assert loaded.manifest["task_name"] == task_id
    assert loaded.manifest["rule_id"] == rule.rule_id
    assert not {"adjacency", "toggle_matrix", "coordinates", "M"} & loaded.manifest.keys()

    batch = GoalSampler(loaded, seed=19).sample(5)
    for key in ("observations", "next_observations", "goals", "events", "previous_events"):
        assert batch[key].shape == (5, size)
    assert not {"adjacency", "toggle_matrix", "coordinates", "M"} & batch.keys()
    assert np.all((batch["actions"] >= 0) & (batch["actions"] < size))
    assert np.all(np.any(batch["observations"] != batch["goals"], axis=1))

    # One known offline final transition supplies a goal and a replay action.
    # This is interface verification; the policy is not a learned/oracle baseline.
    start, goal = loaded.states[0, -2], loaded.states[0, -1]
    action = int(loaded.actions[0, -1])
    labeled = relabel_transitions(loaded, [0], [2], goal[None])
    assert labeled["rewards"].tolist() == [0.0]
    assert labeled["terminated"].tolist() == [True]
    assert labeled["truncated"].tolist() == [False]
    assert labeled["bootstrap_masks"].tolist() == [0.0]

    suite = generate_task_suite(rule, per_depth=1, depths=(1,), seed=23, max_episode_steps=1)
    suite["cases"] = [{
        "case_id": "offline-one-press", "state": start.tolist(),
        "goal": goal.tolist(), "optimal_steps": 1,
    }]
    suite_path = tmp_path / "tasks.json"
    write_json(suite_path, suite)
    restored_suite = load_task_suite(suite_path)

    def replay(observation):
        assert set(observation) == {"board", "goal", "previous_event"}
        np.testing.assert_array_equal(observation["board"], start)
        np.testing.assert_array_equal(observation["goal"], goal)
        assert not observation["previous_event"].any()
        return action

    # Resolve the rule from the new task name rather than passing a rule override.
    result = evaluate_policy(replay, restored_suite, policy_label="offline-replay-test")
    assert result["rule_id"] == rule.rule_id
    assert result["summary"]["success_rate"] == 1.0
    assert result["summary"]["policy_calls"] == 1
    assert result["cases"][0]["return"] == labeled["rewards"][0]
    assert result["cases"][0]["presses"] == 1
