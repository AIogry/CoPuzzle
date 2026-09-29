"""Behavioral checks for the abstract environment contract."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env

from copuzzle.env import PuzzleEnv
from copuzzle.oracle import is_reachable
from copuzzle.rules import get_rule


@pytest.fixture
def rule():
    return get_rule("grid-3x3")


def apply_actions(rule, state, *actions):
    result = state.copy()
    for action in actions:
        result = rule.transition(result, action)
    return result


def test_gymnasium_contract_with_default_observation(rule):
    env = PuzzleEnv(rule)
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=5)
    assert set(observation) == {"board", "goal", "previous_event"}
    assert env.observation_space.contains(observation)
    for name, value in observation.items():
        assert isinstance(env.observation_space[name], gym.spaces.MultiBinary)
        assert value.dtype == np.uint8
        assert env.observation_space[name].dtype == np.dtype(np.uint8)
    assert info == {"is_success": False, "steps": 0, "rule_id": rule.rule_id}


def test_seed_reproduces_state_goal_and_transition(rule):
    first = PuzzleEnv(rule)
    second = PuzzleEnv(rule)
    left, left_info = first.reset(seed=71)
    np.random.default_rng(987).integers(0, 100, size=100)
    right, right_info = second.reset(seed=71)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])
    assert left_info == right_info
    left_step = first.step(0)
    right_step = second.step(0)
    for key in left_step[0]:
        np.testing.assert_array_equal(left_step[0][key], right_step[0][key])
    assert left_step[1:] == right_step[1:]
    repeated, _ = first.reset(seed=71)
    for key in left:
        np.testing.assert_array_equal(left[key], repeated[key])


def test_default_goals_are_reachable_and_nontrivial():
    rule = get_rule("grid-4x4")
    env = PuzzleEnv(rule)
    seen = set()
    for seed in range(16):
        observation, info = env.reset(seed=seed)
        assert not np.array_equal(observation["board"], observation["goal"])
        assert is_reachable(rule, observation["board"], observation["goal"])
        assert not info["is_success"]
        seen.add(observation["goal"].tobytes())
    assert len(seen) > 1


def test_previous_event_describes_only_last_completed_action(rule):
    state = np.zeros(rule.n_buttons, dtype=np.uint8)
    goal = apply_actions(rule, state, 0, 1, 2)
    env = PuzzleEnv(rule)
    initial, _ = env.reset(options={"state": state, "goal": goal})
    assert not initial["previous_event"].any()

    first, reward, terminated, truncated, info = env.step(0)
    np.testing.assert_array_equal(first["board"], rule.transition(state, 0))
    np.testing.assert_array_equal(first["previous_event"], np.eye(rule.n_buttons, dtype=np.uint8)[0])
    assert (reward, terminated, truncated, info["steps"]) == (-1.0, False, False, 1)

    second, *_ = env.step(np.int64(1))
    np.testing.assert_array_equal(second["previous_event"], np.eye(rule.n_buttons, dtype=np.uint8)[1])
    assert not initial["previous_event"].any()
    np.testing.assert_array_equal(first["previous_event"], np.eye(rule.n_buttons, dtype=np.uint8)[0])
    restarted, _ = env.reset(seed=3)
    assert not restarted["previous_event"].any()


def test_success_at_horizon_only_terminates(rule):
    state = np.zeros(rule.n_buttons, dtype=np.uint8)
    goal = rule.transition(state, 0)
    env = PuzzleEnv(rule, max_episode_steps=1)
    env.reset(options={"state": state, "goal": goal})
    observation, reward, terminated, truncated, info = env.step(np.array(0, dtype=np.int32))
    np.testing.assert_array_equal(observation["board"], goal)
    assert (reward, terminated, truncated) == (0.0, True, False)
    assert info["is_success"] and info["steps"] == 1
    with pytest.raises(ResetNeeded):
        env.step(0)


def test_timeout_without_success_requires_reset(rule):
    state = np.zeros(rule.n_buttons, dtype=np.uint8)
    goal = rule.transition(state, 0)
    env = PuzzleEnv(rule, max_episode_steps=1)
    env.reset(options={"state": state, "goal": goal})
    _, reward, terminated, truncated, info = env.step(1)
    assert (reward, terminated, truncated) == (-1.0, False, True)
    assert not info["is_success"] and info["steps"] == 1
    with pytest.raises(ResetNeeded):
        env.step(0)
    env.reset(seed=0)
    env.step(0)


def test_initial_success_is_allowed_but_cannot_be_stepped(rule):
    state = np.zeros(rule.n_buttons, dtype=np.uint8)
    env = PuzzleEnv(rule)
    with pytest.raises(ResetNeeded):
        env.step(0)
    observation, info = env.reset(options={"state": state, "goal": state})
    assert info["is_success"] and info["steps"] == 0
    assert not observation["previous_event"].any()
    with pytest.raises(ResetNeeded):
        env.step(0)
    env.reset(seed=7)
    env.step(0)


def test_info_does_not_leak_graph_or_oracle_results(rule):
    for reveal_graph in (False, True):
        env = PuzzleEnv(rule, reveal_graph=reveal_graph)
        observation, info = env.reset(seed=2)
        assert ("adjacency" in observation) is reveal_graph
        assert set(info) == {"is_success", "steps", "rule_id"}
        observation, _, _, _, info = env.step(0)
        assert set(info) == {"is_success", "steps", "rule_id"}
        assert env.observation_space.contains(observation)
        if reveal_graph:
            np.testing.assert_array_equal(observation["adjacency"], rule.adjacency)


def test_unreachable_explicit_goal_is_rejected():
    rule = get_rule("grid-4x4")
    state = np.zeros(rule.n_buttons, dtype=np.uint8)
    goal = next(
        candidate
        for candidate in np.eye(rule.n_buttons, dtype=np.uint8)
        if not is_reachable(rule, state, candidate)
    )
    with pytest.raises(ValueError, match="unreachable"):
        PuzzleEnv(rule).reset(options={"state": state, "goal": goal})


@pytest.mark.parametrize("field", ["state", "goal"])
@pytest.mark.parametrize(
    "invalid",
    [np.zeros(8), np.zeros((3, 3)), [2] * 9, [-1] * 9, [0.5] * 9, [np.nan] * 9],
)
def test_reset_rejects_nonbinary_or_wrong_shape(rule, field, invalid):
    options = {"state": np.zeros(9, dtype=np.uint8), "goal": np.zeros(9, dtype=np.uint8)}
    options[field] = invalid
    with pytest.raises((ValueError, TypeError)):
        PuzzleEnv(rule).reset(options=options)


@pytest.mark.parametrize(
    "invalid", [-1, 9, 0.0, 0.5, True, np.bool_(False), np.array(True), np.array([0]), "0", None]
)
def test_actions_are_strictly_validated(rule, invalid):
    env = PuzzleEnv(rule)
    env.reset(seed=12)
    with pytest.raises(ValueError):
        env.step(invalid)
    _, _, _, _, info = env.step(0)
    assert info["steps"] == 1


def test_outputs_and_reset_inputs_do_not_alias_environment(rule):
    state = np.zeros(rule.n_buttons, dtype=np.uint8)
    goal = apply_actions(rule, state, 0, 1, 2)
    expected_goal = goal.copy()
    expected_board = rule.transition(state, 0)
    env = PuzzleEnv(rule, reveal_graph=True)
    observation, info = env.reset(options={"state": state, "goal": goal})
    state.fill(1)
    goal.fill(0)
    for value in observation.values():
        value.fill(1)
    info["steps"] = 999

    after, _, _, _, after_info = env.step(0)
    np.testing.assert_array_equal(after["board"], expected_board)
    np.testing.assert_array_equal(after["goal"], expected_goal)
    np.testing.assert_array_equal(after["adjacency"], rule.adjacency)
    assert after_info["steps"] == 1
    assert not np.shares_memory(after["board"], after["goal"])
    saved = {key: value.copy() for key, value in after.items()}
    env.step(1)
    for key in saved:
        np.testing.assert_array_equal(after[key], saved[key])


def test_ansi_render_is_text_and_has_no_side_effect(rule):
    env = PuzzleEnv(rule, render_mode="ansi")
    with pytest.raises(ResetNeeded):
        env.render()
    observation, _ = env.reset(seed=22)
    rendered = env.render()
    assert isinstance(rendered, str)
    assert "board:" in rendered and "goal:" in rendered
    assert env.render() == rendered
    after, *_ = env.step(0)
    np.testing.assert_array_equal(after["board"], rule.transition(observation["board"], 0))


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True, None])
def test_invalid_horizon_is_rejected(rule, horizon):
    with pytest.raises(ValueError):
        PuzzleEnv(rule, max_episode_steps=horizon)


def test_invalid_render_mode_is_rejected(rule):
    with pytest.raises(ValueError):
        PuzzleEnv(rule, render_mode="rgb_array")
