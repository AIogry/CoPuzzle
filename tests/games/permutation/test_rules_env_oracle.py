"""Independent transition/search checks and the Gymnasium lifecycle."""

from collections import deque
from dataclasses import FrozenInstanceError
from math import factorial

from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from copuzzle.games.pancake import make_env as make_pancake
from copuzzle.games.permutation import ExactOracle, PermutationEnv, get_rule, make_env, rule_from_spec
from copuzzle.games.permutation.rules import PermutationRule, validate_rule_spec
from copuzzle.games.topspin import make_env as make_topspin


def independent_step(state, family, action, window=None):
    """Tuple-only implementation, deliberately separate from rule transitions."""
    state = tuple(state)
    if family == "pancake":
        length = action + 2
        return state[:length][::-1] + state[length:]
    if action == 0:
        return state[1:] + state[:1]
    if action == 1:
        return state[-1:] + state[:-1]
    return state[:window][::-1] + state[window:]


def independent_bfs(start, family, window=None):
    start = tuple(start)
    distances = {start: 0}
    frontier = deque([start])
    n_actions = len(start) - 1 if family == "pancake" else 3
    while frontier:
        state = frontier.popleft()
        for action in range(n_actions):
            successor = independent_step(state, family, action, window)
            if successor not in distances:
                distances[successor] = distances[state] + 1
                frontier.append(successor)
    return distances


@pytest.mark.parametrize("task_id,n_actions", [("pancake-4", 3), ("pancake-8", 7), ("topspin-8-4", 3)])
def test_rule_identity_roundtrip_and_immutability(task_id, n_actions):
    rule = get_rule(task_id)
    assert rule.name == task_id
    assert rule.n_actions == n_actions
    assert rule_from_spec(rule.spec()) == rule
    assert validate_rule_spec(rule.spec()) == rule.spec()
    assert get_rule(task_id).rule_id == rule.rule_id
    with pytest.raises(FrozenInstanceError):
        rule.n_tokens = 5
    spec = rule.spec()
    spec["name"] = "corrupted"
    assert rule.spec()["name"] == task_id
    with pytest.raises(ValueError):
        rule_from_spec(spec)


@pytest.mark.parametrize("field,value", [
    ("rule_id", "forged"), ("n_actions", 12), ("n_tokens", 4.0),
    ("action_encoding", "rotated-window-macro"), ("goal_equivalence", "cyclic"),
    ("schema", "v2"), ("extra", True), ("window", 3),
])
def test_rule_spec_tampering_rejected(field, value):
    spec = get_rule("pancake-4").spec()
    spec[field] = value
    with pytest.raises(ValueError):
        rule_from_spec(spec)


@pytest.mark.parametrize("task_id", ["pancake-1", "pancake-9", "pancake-04", "Pancake-4", "topspin-8-1", "topspin-8-9", "topspin-9-4", "grid-2x2", None])
def test_invalid_task_rejected(task_id):
    with pytest.raises(ValueError):
        get_rule(task_id)


@pytest.mark.parametrize("state", [[1, 2, 2, 4], [0, 1, 2, 3], [1, 2, 3, 5], [1, 2, 3], [[1, 2, 3, 4]], [1.0, 2.0, 3.0, 4.0], [True, 2, 3, 4], [1, 2, 3, 2**40], [1, 2, 3, np.nan]])
def test_invalid_state_rejected_before_casting(state):
    with pytest.raises(ValueError):
        get_rule("pancake-4").validate_state(state)


@pytest.mark.parametrize("action", [True, np.bool_(False), -1, 3, 0.0, [0], None])
def test_invalid_action_rejected(action):
    rule = get_rule("pancake-4")
    with pytest.raises(ValueError):
        rule.transition([1, 2, 3, 4], action)
    with pytest.raises(ValueError):
        rule.event(action)


def test_pancake_prefix_examples_inverse_and_order():
    rule = get_rule("pancake-4")
    start = np.array([3, 1, 4, 2], dtype=np.int64)
    assert rule.transition(start, 1).tolist() == [4, 1, 3, 2]
    assert rule.transition(rule.transition(start, 0), 1).tolist() == [4, 3, 1, 2]
    assert rule.transition(rule.transition(start, 1), 0).tolist() == [1, 4, 3, 2]
    for action in range(rule.n_actions):
        assert np.array_equal(rule.transition(rule.transition(start, action), action), start)
        assert rule.transition(start, action).dtype == np.int32
        assert np.array_equal(rule.event(action), np.eye(rule.n_actions, dtype=np.uint8)[action])
    assert start.tolist() == [3, 1, 4, 2]
    validated = rule.validate_state(start)
    validated[:] = 0
    assert start.tolist() == [3, 1, 4, 2]


def test_topspin_examples_inverses_and_noncommutativity():
    rule = get_rule("topspin-8-4")
    start = np.arange(1, 9, dtype=np.int32)
    assert rule.transition(start, 0).tolist() == [2, 3, 4, 5, 6, 7, 8, 1]
    assert rule.transition(start, 1).tolist() == [8, 1, 2, 3, 4, 5, 6, 7]
    assert rule.transition(start, 2).tolist() == [4, 3, 2, 1, 5, 6, 7, 8]
    assert np.array_equal(rule.transition(rule.transition(start, 0), 1), start)
    assert np.array_equal(rule.transition(rule.transition(start, 2), 2), start)
    assert not np.array_equal(rule.transition(rule.transition(start, 0), 2), rule.transition(rule.transition(start, 2), 0))


@pytest.mark.parametrize("n_tokens", [2, 3, 4, 5])
def test_pancake_bfs_matches_independent_graph(n_tokens):
    rule = get_rule(f"pancake-{n_tokens}")
    oracle = ExactOracle(rule)
    identity = np.arange(1, n_tokens + 1, dtype=np.int32)
    expected = independent_bfs(identity, "pancake")
    assert len(expected) == len(oracle.reachable_states) == factorial(n_tokens)
    assert oracle.diameter == max(expected.values())
    for state, distance in expected.items():
        assert oracle.distance(identity, state) == distance
        actions = oracle.solve(identity, state)
        assert len(actions) == distance
        reached = tuple(identity)
        for action in actions:
            reached = independent_step(reached, "pancake", action)
        assert reached == state


def test_topspin_8_4_full_graph_matches_independent_bfs():
    rule = get_rule("topspin-8-4")
    oracle = ExactOracle(rule)
    identity = np.arange(1, 9, dtype=np.int32)
    expected = independent_bfs(identity, "topspin", 4)
    assert len(expected) == len(oracle.reachable_states) == 40_320
    assert max(expected.values()) == oracle.diameter == 22
    for state, distance in zip(oracle.reachable_states, oracle.distances):
        assert expected[tuple(state)] == distance
    rng = np.random.default_rng(49)
    for _ in range(30):
        start = rng.permutation(identity)
        goal = rng.permutation(identity)
        actions = oracle.solve(start, goal)
        assert len(actions) == oracle.distance(start, goal)
        reached = tuple(start)
        for action in actions:
            reached = independent_step(reached, "topspin", action, 4)
        assert reached == tuple(goal)
    assert oracle.distance(identity, np.roll(identity, -1)) == 1


def test_disconnected_topspin_and_other_component_pairs():
    rule = get_rule("topspin-5-5")
    oracle = ExactOracle(rule)
    identity = [1, 2, 3, 4, 5]
    other = [2, 1, 3, 4, 5]
    assert len(oracle.reachable_states) == 10
    assert oracle.distance(identity, other) is None
    assert oracle.solve(identity, other) is None
    expected = independent_bfs(other, "topspin", 5)
    for goal, distance in expected.items():
        assert oracle.distance(other, goal) == distance
        actions = oracle.solve(other, goal)
        assert len(actions) == distance
        reached = tuple(other)
        for action in actions:
            reached = independent_step(reached, "topspin", action, 5)
        assert reached == goal
    assert oracle.solve(other, other) == []


def test_exact_search_budget_and_cache_cannot_be_mutated():
    rule = get_rule("pancake-4")
    with pytest.raises(ValueError, match="exceeds max_states"):
        ExactOracle(rule, max_states=23)
    first = ExactOracle(rule)
    second = ExactOracle(get_rule("pancake-4"))
    assert first.reachable_states is second.reachable_states
    assert first.reachable_states.dtype == np.int32
    for array in (first.reachable_states, first.distances):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert ExactOracle(get_rule("topspin-5-5"), max_states=10).reachable_states.shape == (10, 5)
    for limit in (0, -1, True, 4.2, 40_321):
        with pytest.raises(ValueError):
            ExactOracle(rule, max_states=limit)


@pytest.mark.parametrize("task_id", ["pancake-2", "pancake-6", "topspin-8-4", "topspin-5-5"])
def test_gym_checker_and_seeded_reachable_resets(task_id):
    env = make_env(task_id)
    check_env(env, skip_render_check=True)
    first, info = env.reset(seed=73)
    second, second_info = env.reset(seed=73)
    assert info == second_info
    assert set(first) == {"board", "goal", "previous_event", "action_mask"}
    assert set(info) == {"is_success", "steps", "rule_id"}
    for key in first:
        assert np.array_equal(first[key], second[key])
    assert first["board"].dtype == first["goal"].dtype == np.int32
    assert first["previous_event"].dtype == first["action_mask"].dtype == np.uint8
    assert first["previous_event"].shape == first["action_mask"].shape == (env.rule.n_actions,)
    assert not first["previous_event"].any()
    assert first["action_mask"].all()
    assert not info["is_success"]
    assert ExactOracle(env.rule).distance(first["board"], first["goal"]) is not None


def test_episode_boundaries_success_precedence_and_observation_copies():
    env = make_env("pancake-4", max_episode_steps=1, render_mode="ansi")
    with pytest.raises(ResetNeeded):
        env.step(0)
    with pytest.raises(ResetNeeded):
        env.render()
    state = np.array([1, 2, 3, 4], dtype=np.int32)
    goal = np.array([2, 1, 3, 4], dtype=np.int32)
    observation, _ = env.reset(options={"state": state, "goal": goal})
    state[:] = 0
    goal[:] = 0
    for array in observation.values():
        array[:] = 0
    assert "board: 1 2 3 4" in env.render()
    next_observation, reward, terminated, truncated, info = env.step(np.array(0))
    assert next_observation["board"].tolist() == [2, 1, 3, 4]
    assert next_observation["previous_event"].tolist() == [1, 0, 0]
    assert reward == 0.0 and terminated and not truncated and info["is_success"]
    with pytest.raises(ResetNeeded):
        env.step(0)
    env.reset(options={"state": [1, 2, 3, 4], "goal": [2, 1, 3, 4]})
    _, reward, terminated, truncated, info = env.step(1)
    assert reward == -1.0 and not terminated and truncated and not info["is_success"]
    with pytest.raises(ResetNeeded):
        env.step(0)
    _, info = env.reset(options={"state": [1, 2, 3, 4], "goal": [1, 2, 3, 4]})
    assert info["is_success"] and info["steps"] == 0
    with pytest.raises(ResetNeeded):
        env.step(0)


def test_reset_rejects_unreachable_and_unknown_options():
    env = make_env("topspin-5-5")
    with pytest.raises(ValueError, match="unreachable"):
        env.reset(options={"state": [1, 2, 3, 4, 5], "goal": [2, 1, 3, 4, 5]})
    with pytest.raises(ValueError, match="Unknown"):
        env.reset(options={"scramble": 5})
    with pytest.raises(TypeError):
        env.reset(options=[])
    for seed in range(30):
        observation, _ = env.reset(seed=seed)
        assert ExactOracle(env.rule).distance(observation["board"], observation["goal"]) is not None


def test_constructor_and_family_wrapper_validation():
    for horizon in (True, 0, -1, 0.5, None):
        with pytest.raises(ValueError):
            make_env("pancake-4", max_episode_steps=horizon)
    with pytest.raises(ValueError):
        make_env("pancake-4", render_mode="rgb_array")
    with pytest.raises(TypeError):
        PermutationEnv("pancake-4")
    with pytest.raises(ValueError):
        PermutationRule("pancake", 4, 2)
    with pytest.raises(ValueError):
        PermutationRule("unknown", 4)
    assert make_pancake().rule.name == "pancake-6"
    assert make_topspin().rule.name == "topspin-8-4"
    with pytest.raises(ValueError):
        make_pancake("topspin-8-4")
    with pytest.raises(ValueError):
        make_topspin("pancake-4")
