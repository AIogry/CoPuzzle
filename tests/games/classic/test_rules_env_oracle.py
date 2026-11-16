"""Independent small-graph checks, arbitrary goals and masked episode semantics."""

from collections import deque
from dataclasses import FrozenInstanceError
from itertools import product

from gymnasium.error import ResetNeeded
import numpy as np
import pytest

from copuzzle.games.classic import ClassicEnv, ClassicRule, ExactOracle, get_rule, make_env, rule_from_spec, validate_rule_spec
from copuzzle.games.hanoi import make_env as make_hanoi
from copuzzle.games.sliding import make_env as make_sliding


def independent_moves(state, family, rows=2, cols=3):
    """Tuple graph unrelated to the production successor implementation."""
    if family == "sliding":
        blank = state.index(0)
        grid = [list(state[i * cols:(i + 1) * cols]) for i in range(rows)]
        y, x = blank // cols, blank % cols
        for action, (dy, dx) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                successor = [row[:] for row in grid]
                successor[y][x], successor[ny][nx] = successor[ny][nx], 0
                yield action, tuple(tile for row in successor for tile in row)
    else:
        actions = {(0, 1): 0, (0, 2): 1, (1, 0): 2, (1, 2): 3, (2, 0): 4, (2, 1): 5}
        for disk, source in enumerate(state):
            if source in state[:disk]:
                continue
            for target in range(3):
                if target == source or target in state[:disk]:
                    continue
                successor = list(state)
                successor[disk] = target
                yield actions[source, target], tuple(successor)


def independent_bfs(start, family, rows=2, cols=3, max_depth=None):
    start = tuple(start)
    distances = {start: 0}
    frontier = deque([start])
    while frontier:
        state = frontier.popleft()
        if max_depth is not None and distances[state] >= max_depth:
            continue
        for _, successor in independent_moves(state, family, rows, cols):
            if successor not in distances:
                distances[successor] = distances[state] + 1
                frontier.append(successor)
    return distances


@pytest.mark.parametrize("task_id,n_actions,size", [("sliding-2x3", 4, 6), ("sliding-3x3", 4, 9), ("hanoi-2", 6, 2), ("hanoi-6", 6, 6)])
def test_rule_identity_roundtrip_and_immutability(task_id, n_actions, size):
    rule = get_rule(task_id)
    assert (rule.name, rule.n_actions, rule.state_size) == (task_id, n_actions, size)
    assert rule_from_spec(rule.spec()) == rule
    assert validate_rule_spec(rule.spec()) == rule.spec()
    with pytest.raises(FrozenInstanceError):
        rule.state_size = 1
    spec = rule.spec()
    spec["name"] = "corrupt"
    assert rule.spec()["name"] == task_id
    with pytest.raises(ValueError):
        rule_from_spec(spec)
    goal = rule.canonical_goal
    goal[:] = -1
    assert np.all(rule.canonical_goal >= 0)


@pytest.mark.parametrize("field,value", [
    ("rule_id", "forged"), ("n_actions", 5), ("state_size", 6.0),
    ("invalid_action", "no-op"), ("schema", "v2"), ("extra", True),
    ("action_encoding", "tile-up-down-left-right"), ("rows", 2.0),
])
def test_tampered_spec_rejected(field, value):
    spec = get_rule("sliding-2x3").spec()
    spec[field] = value
    with pytest.raises(ValueError):
        rule_from_spec(spec)


def test_metadata_validation_never_runs_transitions(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("static spec check called transition")
    monkeypatch.setattr(ClassicRule, "transition", forbidden)
    monkeypatch.setattr(ClassicRule, "action_mask", forbidden)
    rule = get_rule("hanoi-3")
    assert rule_from_spec(rule.spec()) == rule


@pytest.mark.parametrize("task_id", ["sliding-2x2", "sliding-3x2", "sliding-4x4", "sliding-02x3", "hanoi-1", "hanoi-7", "hanoi-03", "Hanoi-3", "pancake-4", None])
def test_invalid_task_ids(task_id):
    with pytest.raises(ValueError):
        get_rule(task_id)


@pytest.mark.parametrize("state", [[1, 2, 3, 4, 5, 5], [1, 2, 3, 4, 5, 6], [1, 2, 3], [[0, 1, 2, 3, 4, 5]], [0., 1., 2., 3., 4., 5.], [False, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 2**40]])
def test_invalid_sliding_states_before_cast(state):
    with pytest.raises(ValueError):
        get_rule("sliding-2x3").validate_state(state)


@pytest.mark.parametrize("state", [[0, 1], [0, -1, 2], [0, 1, 3], [0, 1, 2**40], [True, 1, 2], [0., 1., 2.], [0, 1, np.nan]])
def test_invalid_hanoi_states_before_cast(state):
    with pytest.raises(ValueError):
        get_rule("hanoi-3").validate_state(state)


@pytest.mark.parametrize("task_id", ["sliding-2x3", "hanoi-3"])
@pytest.mark.parametrize("action", [True, np.bool_(False), -1, 6, 0., [0], None])
def test_invalid_action_types_and_range(task_id, action):
    rule = get_rule(task_id)
    with pytest.raises(ValueError):
        rule.transition(rule.canonical_goal, action)
    with pytest.raises(ValueError):
        rule.event(action)


def test_explicit_sliding_and_hanoi_semantics():
    rule = get_rule("sliding-2x3")
    board = np.array([1, 2, 3, 4, 0, 5], dtype=np.int64)
    assert rule.action_mask(board).tolist() == [1, 0, 1, 1]
    assert rule.transition(board, 0).tolist() == [1, 0, 3, 4, 2, 5]
    assert rule.transition(board, 3).tolist() == [1, 2, 3, 4, 5, 0]
    assert board.tolist() == [1, 2, 3, 4, 0, 5]
    with pytest.raises(ValueError, match="illegal"):
        rule.transition(board, 1)
    rule = get_rule("hanoi-3")
    assert rule.action_mask([0, 0, 0]).tolist() == [1, 1, 0, 0, 0, 0]
    assert rule.transition([0, 0, 0], 1).tolist() == [2, 0, 0]
    assert rule.action_mask([2, 0, 0]).tolist() == [1, 0, 0, 0, 1, 1]
    with pytest.raises(ValueError, match="illegal"):
        rule.transition([2, 0, 0], 1)
    with pytest.raises(ValueError, match="illegal"):
        rule.transition([0, 0, 0], 2)


@pytest.mark.parametrize("task_id", ["sliding-2x3", "hanoi-3", "hanoi-4"])
def test_complete_small_graph_masks_and_solutions(task_id):
    rule = get_rule(task_id)
    oracle = ExactOracle(rule)
    expected = independent_bfs(rule.canonical_goal, rule.family)
    assert len(expected) == len(oracle.reachable_states) == (360 if rule.family == "sliding" else 3**rule.state_size)
    assert oracle.max_distance == max(expected.values())
    for state, distance in expected.items():
        edges = dict(independent_moves(state, rule.family))
        assert np.flatnonzero(rule.action_mask(state)).tolist() == sorted(edges)
        assert oracle.distance(state, rule.canonical_goal) == distance
        actions = oracle.solve(state, rule.canonical_goal)
        assert len(actions) == distance
        reached = state
        for action in actions:
            reached = dict(independent_moves(reached, rule.family))[action]
        assert reached == tuple(rule.canonical_goal)
        for action in range(rule.n_actions):
            if action not in edges:
                with pytest.raises(ValueError, match="illegal"):
                    rule.transition(state, action)
            else:
                successor = rule.transition(state, action)
                assert tuple(successor) == edges[action]
                assert successor.dtype == np.int32
                assert tuple(rule.transition(successor, rule.inverse_action(action))) == state
                assert np.array_equal(rule.event(action), np.eye(rule.n_actions, dtype=np.uint8)[action])


def test_arbitrary_sliding_goals_blank_locations_and_other_component():
    rule = get_rule("sliding-2x3")
    oracle = ExactOracle(rule)
    # Includes goals in both parity components, with the blank in every cell.
    goals = [np.roll(rule.canonical_goal, shift) for shift in range(6)]
    goals.append(np.array([2, 1, 3, 4, 5, 0]))
    for goal in goals:
        expected = independent_bfs(goal, "sliding")
        for state, distance in expected.items():
            assert oracle.distance(state, goal) == distance
        for depth in range(max(expected.values()) + 2):
            found = oracle.states_at_distance(goal, depth)
            assert found.dtype == np.int32 and found.shape[1] == 6
            assert {tuple(row) for row in found} == {state for state, distance in expected.items() if distance == depth}
        start = next(state for state, distance in expected.items() if distance == max(expected.values()))
        path = oracle.solve(start, goal)
        for action in path:
            start = dict(independent_moves(start, "sliding"))[action]
        assert start == tuple(goal)
    assert oracle.distance([2, 1, 3, 4, 5, 0], rule.canonical_goal) is None
    assert oracle.solve([2, 1, 3, 4, 5, 0], rule.canonical_goal) is None
    assert not rule.is_reachable([2, 1, 3, 4, 5, 0], rule.canonical_goal)


def test_all_hanoi_three_disk_goal_distances():
    rule = get_rule("hanoi-3")
    oracle = ExactOracle(rule)
    for goal in product(range(3), repeat=3):
        expected = independent_bfs(goal, "hanoi")
        assert len(expected) == 27
        for state, distance in expected.items():
            assert oracle.distance(state, goal) == distance
        for depth in range(9):
            assert {tuple(row) for row in oracle.states_at_distance(goal, depth)} == {state for state, distance in expected.items() if distance == depth}
    for n in (2, 4, 5, 6):
        oracle = ExactOracle(get_rule(f"hanoi-{n}"))
        assert oracle.distance([0] * n, [2] * n) == 2**n - 1


def test_three_by_three_bounded_exact_search_and_arbitrary_blank_goal():
    rule = get_rule("sliding-3x3")
    oracle = ExactOracle(rule)
    assert len(oracle.reachable_states) == 181_440
    assert oracle.max_distance == 31
    start = [8, 6, 7, 2, 5, 4, 3, 0, 1]
    assert oracle.distance(start, rule.canonical_goal) == 31
    path = oracle.solve(start, rule.canonical_goal)
    for action in path:
        start = rule.transition(start, action)
    assert np.array_equal(start, rule.canonical_goal)
    # Central blank must not be relabeled into a corner blank.
    goal = [8, 7, 6, 5, 0, 4, 3, 2, 1]
    expected = independent_bfs(goal, "sliding", 3, 3, max_depth=6)
    for state, distance in expected.items():
        assert oracle.distance(state, goal) == distance
    assert {tuple(row) for row in oracle.states_at_distance(goal, 6)} == {state for state, distance in expected.items() if distance == 6}


def test_exact_budget_cache_and_immutable_arrays():
    rule = get_rule("sliding-2x3")
    with pytest.raises(ValueError, match="exceeds max_states"):
        ExactOracle(rule, max_states=359)
    oracle = ExactOracle(rule, max_states=360)
    other = ExactOracle(rule, max_states=360)
    assert oracle.reachable_states is other.reachable_states
    for array in (oracle.reachable_states, oracle.distances):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    for invalid in (0, -1, True, 3.5, 181_441):
        with pytest.raises(ValueError):
            ExactOracle(rule, max_states=invalid)
    for invalid in (-1, True, 2.5):
        with pytest.raises(ValueError):
            oracle.states_at_distance(rule.canonical_goal, invalid)
    states = oracle.states_at_distance(rule.canonical_goal, 1)
    states[:] = -1
    assert np.all(oracle.states_at_distance(rule.canonical_goal, 1) >= 0)


@pytest.mark.parametrize("task_id", ["sliding-2x3", "sliding-3x3", "hanoi-2", "hanoi-6"])
def test_seeded_reset_spaces_and_masks_without_search(task_id, monkeypatch):
    import copuzzle.games.classic.oracle as oracle_module
    def forbidden(*args, **kwargs):
        raise AssertionError("reset must not search")
    monkeypatch.setattr(oracle_module, "_search", forbidden)
    env = make_env(task_id)
    first, info = env.reset(seed=73)
    second, second_info = env.reset(seed=73)
    assert info == second_info
    assert set(first) == {"board", "goal", "previous_event", "action_mask"}
    assert set(info) == {"is_success", "steps", "rule_id"}
    for key in first:
        assert np.array_equal(first[key], second[key])
    assert first["board"].dtype == first["goal"].dtype == np.int32
    assert first["previous_event"].dtype == first["action_mask"].dtype == np.uint8
    assert not first["previous_event"].any()
    assert env.observation_space.contains(first)
    for seed in range(30):
        observation, info = env.reset(seed=seed)
        assert env.rule.is_reachable(observation["board"], observation["goal"])
        assert not info["is_success"]
        legal = np.flatnonzero(observation["action_mask"])
        observation, reward, terminated, truncated, info = env.step(legal[0])
        assert env.observation_space.contains(observation)
        assert np.array_equal(observation["action_mask"], env.rule.action_mask(observation["board"]))


def test_episode_lifecycle_success_precedence_and_invalid_action_atomicity():
    env = make_env("sliding-2x3", max_episode_steps=1, render_mode="ansi")
    with pytest.raises(ResetNeeded):
        env.step(0)
    with pytest.raises(ResetNeeded):
        env.render()
    state, goal = np.array([1, 2, 3, 4, 0, 5]), np.array([1, 2, 3, 4, 5, 0])
    observation, _ = env.reset(options={"state": state, "goal": goal})
    state[:] = goal[:] = -1
    for array in observation.values():
        array[:] = 0
    assert "board: 1 2 3 4 0 5" in env.render()
    with pytest.raises(ValueError, match="illegal"):
        env.step(1)
    observation, reward, terminated, truncated, info = env.step(np.array(3))
    assert observation["board"].tolist() == [1, 2, 3, 4, 5, 0]
    assert observation["previous_event"].tolist() == [0, 0, 0, 1]
    assert reward == 0. and terminated and not truncated and info["steps"] == 1
    with pytest.raises(ResetNeeded):
        env.step(0)
    env.reset(options={"state": [1, 2, 3, 4, 0, 5], "goal": [1, 2, 3, 4, 5, 0]})
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == -1. and not terminated and truncated and not info["is_success"]
    with pytest.raises(ResetNeeded):
        env.step(0)
    observation, info = env.reset(options={"state": [1, 2, 3, 4, 5, 0], "goal": [1, 2, 3, 4, 5, 0]})
    assert info["is_success"] and info["steps"] == 0
    assert not observation["previous_event"].any()
    with pytest.raises(ResetNeeded):
        env.step(0)


def test_reset_options_and_configuration_validation():
    env = make_env("sliding-2x3")
    with pytest.raises(ValueError, match="unreachable"):
        env.reset(options={"state": [1, 2, 3, 4, 5, 0], "goal": [2, 1, 3, 4, 5, 0]})
    with pytest.raises(ValueError, match="Unknown"):
        env.reset(options={"scramble": 5})
    with pytest.raises(TypeError):
        env.reset(options=[])
    for seed in range(10):
        observation, info = env.reset(seed=seed, options={"goal": [2, 1, 3, 4, 5, 0]})
        assert env.rule.is_reachable(observation["board"], observation["goal"])
        assert not info["is_success"]
    for horizon in (True, 0, -1, .5, None):
        with pytest.raises(ValueError):
            make_env("hanoi-3", max_episode_steps=horizon)
    with pytest.raises(ValueError):
        make_env("hanoi-3", render_mode="rgb_array")
    with pytest.raises(TypeError):
        ClassicEnv("hanoi-3")
    with pytest.raises(TypeError):
        ExactOracle("hanoi-3")
    for args in (("invalid", 3), ("hanoi", 3, 1, 3), ("sliding", 5, 2, 3)):
        with pytest.raises(ValueError):
            ClassicRule(*args)
    assert make_hanoi().rule.name == "hanoi-3"
    assert make_sliding().rule.name == "sliding-2x3"
    with pytest.raises(ValueError):
        make_hanoi("sliding-2x3")
    with pytest.raises(ValueError):
        make_sliding("hanoi-3")
