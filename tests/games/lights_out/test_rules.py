"""Independent checks of rule semantics and privileged solver correctness."""

from collections import deque

import numpy as np
import pytest

from copuzzle.oracle import is_reachable, rank, solve
from copuzzle.rules import PuzzleRule, get_rule, graph_rule, grid_rule


def test_grid_press_boundary_and_center():
    rule = grid_rule(3, 3)
    empty = np.zeros(9, dtype=np.uint8)
    np.testing.assert_array_equal(rule.transition(empty, 0), [1, 1, 0, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(rule.transition(empty, 4), [0, 1, 0, 1, 1, 1, 0, 1, 0])
    np.testing.assert_array_equal(empty, np.zeros(9))


def test_repeated_press_cancels_and_operations_commute():
    rule = grid_rule(4, 4)
    board = np.random.default_rng(10).integers(0, 2, rule.n_buttons, dtype=np.uint8)
    for first in range(rule.n_buttons):
        np.testing.assert_array_equal(rule.transition(rule.transition(board, first), first), board)
        for second in range(rule.n_buttons):
            np.testing.assert_array_equal(
                rule.transition(rule.transition(board, first), second),
                rule.transition(rule.transition(board, second), first),
            )


def test_event_is_press_parity_and_effect_is_state_change():
    rule = grid_rule(3, 3)
    board = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
    event = np.zeros(9, dtype=np.uint8)
    event[[0, 4]] = 1
    expected = rule.transition(rule.transition(board, 0), 4)
    np.testing.assert_array_equal(rule.apply_event(board, event), expected)
    np.testing.assert_array_equal(rule.effect(event), board ^ expected)
    np.testing.assert_array_equal(rule.effect(np.zeros(9, dtype=np.uint8)), np.zeros(9))
    assert not np.array_equal(event, rule.effect(event))


def test_fixed_graph_presets_and_small_grid():
    path = get_rule("graph-path9")
    cycle = get_rule("graph-cycle9")
    assert path.n_buttons == cycle.n_buttons == 9
    np.testing.assert_array_equal(path.adjacency.sum(axis=0), [1, 2, 2, 2, 2, 2, 2, 2, 1])
    np.testing.assert_array_equal(cycle.adjacency.sum(axis=0), np.full(9, 2))
    assert path.rule_id != cycle.rule_id
    np.testing.assert_array_equal(get_rule("grid-1x1").transition([0], 0), [1])


def test_rule_arrays_immutable_and_identifier_depends_on_logic():
    adjacency = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    coordinates = np.array([[0, 0], [1, 0]], dtype=float)
    rule = graph_rule(adjacency, name="one", coordinates=coordinates)
    renamed = graph_rule(adjacency.astype(float), name="two", coordinates=coordinates + 20)
    assert rule.rule_id == renamed.rule_id
    assert len(rule.rule_id) == 64
    adjacency[:] = 0
    coordinates[:] = 10
    assert rule.adjacency[0, 1] == 1
    assert rule.coordinates[0, 0] == 0
    for array in (rule.adjacency, rule.toggle_matrix, rule.coordinates):
        with pytest.raises(ValueError):
            array.flat[0] = 1
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("value", [
    [0] * 8, np.zeros((3, 3)), [0] * 8 + [2], [0] * 8 + [-1],
    [0] * 8 + [0.5], [0] * 8 + [256], [0] * 8 + [np.nan],
    [0] * 8 + [np.inf], ["0"] * 9, [0j] * 9,
])
def test_state_and_event_validation(value):
    rule = grid_rule(3, 3)
    with pytest.raises(ValueError):
        rule.validate_state(value)
    with pytest.raises(ValueError):
        rule.effect(value)


def test_validation_returns_copy_and_accepts_binary_floats():
    rule = grid_rule(1, 2)
    original = np.array([0.0, 1.0])
    validated = rule.validate_state(original)
    assert validated.dtype == np.uint8
    validated[:] = 0
    np.testing.assert_array_equal(original, [0, 1])


@pytest.mark.parametrize("action", [-1, 9, True, np.bool_(False), 0.0, "0", [0]])
def test_invalid_action(action):
    with pytest.raises(ValueError):
        grid_rule(3, 3).transition(np.zeros(9), action)


@pytest.mark.parametrize("adjacency", [
    [], [[0, 1]], [[1]], [[0, 1], [0, 0]], [[0, 2], [2, 0]],
    [[0, 0.5], [0.5, 0]], [[0, np.nan], [np.nan, 0]],
])
def test_invalid_graph(adjacency):
    with pytest.raises(ValueError):
        graph_rule(adjacency)


@pytest.mark.parametrize("coordinates", [[], [[0, 0]], [[0], [1]], [[0, np.nan], [1, 0]]])
def test_invalid_coordinates(coordinates):
    with pytest.raises(ValueError):
        graph_rule([[0, 1], [1, 0]], coordinates=coordinates)


@pytest.mark.parametrize("rows,cols", [(0, 3), (-1, 3), (3, 0), (True, 2), (2.5, 3), ("3", 3)])
def test_invalid_grid_dimensions(rows, cols):
    with pytest.raises(ValueError):
        grid_rule(rows, cols)


@pytest.mark.parametrize("task_id", ["", "grid-0x3", "grid-3X3", "graph-path8", 3])
def test_unknown_task(task_id):
    with pytest.raises(ValueError):
        get_rule(task_id)


@pytest.mark.parametrize("shape,expected_rank", [((3, 3), 9), ((4, 4), 12), ((4, 5), 20), ((4, 6), 24)])
def test_known_grid_ranks(shape, expected_rank):
    rule = grid_rule(*shape)
    assert rank(rule.toggle_matrix) == expected_rank


def test_rank_is_binary_not_real_and_supports_rectangles():
    matrix = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    assert np.linalg.matrix_rank(matrix) == 3
    assert rank(matrix) == 2
    assert rank(matrix[:2]) == 2
    assert rank(np.empty((0, 3), dtype=np.uint8)) == 0
    assert rank(np.empty((3, 0), dtype=np.uint8)) == 0
    with pytest.raises(ValueError):
        rank([1, 0])
    with pytest.raises(ValueError):
        rank([[1, 0.2]])


def _grid3_bfs_distances():
    # Build effects from grid geometry independently of rule.toggle_matrix.
    effects = []
    for row in range(3):
        for col in range(3):
            mask = 0
            for rr, cc in ((row, col), (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= rr < 3 and 0 <= cc < 3:
                    mask |= 1 << (3 * rr + cc)
            effects.append(mask)
    distances = {0: 0}
    queue = deque([0])
    while queue:
        board = queue.popleft()
        for effect in effects:
            successor = board ^ effect
            if successor not in distances:
                distances[successor] = distances[board] + 1
                queue.append(successor)
    return distances


def test_all_grid3_deltas_match_independent_bfs_shortest_paths():
    rule = grid_rule(3, 3)
    distances = _grid3_bfs_distances()
    assert len(distances) == 512
    start = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
    for delta, distance in distances.items():
        goal = start ^ np.array([(delta >> index) & 1 for index in range(9)], dtype=np.uint8)
        plan = solve(rule, start, goal)
        assert plan is not None and len(plan) == distance
        assert is_reachable(rule, start, goal)
        replay = start.copy()
        for action in plan:
            replay = rule.transition(replay, action)
        np.testing.assert_array_equal(replay, goal)


def test_grid4_unreachable_and_solver_minimizes_over_nullspace():
    rule = grid_rule(4, 4)
    start = np.zeros(16, dtype=np.uint8)
    # Exhaustive parity enumeration gives an independent shortest-effect map.
    minimum = {}
    for parity in range(1 << 16):
        event = np.array([(parity >> index) & 1 for index in range(16)], dtype=np.uint8)
        effect = rule.effect(event).tobytes()
        minimum[effect] = min(minimum.get(effect, 17), parity.bit_count())
    assert len(minimum) == 1 << 12
    unreachable = next(
        np.array([(bits >> index) & 1 for index in range(16)], dtype=np.uint8)
        for bits in range(1 << 16)
        if np.array([(bits >> index) & 1 for index in range(16)], dtype=np.uint8).tobytes() not in minimum
    )
    assert not is_reachable(rule, start, unreachable)
    assert solve(rule, start, unreachable) is None
    # Cover every reachable effect, including nonunique parity solutions.
    for effect, shortest in minimum.items():
        goal = np.frombuffer(effect, dtype=np.uint8)
        plan = solve(rule, start, goal)
        assert plan is not None and len(plan) == shortest
        event = np.zeros(16, dtype=np.uint8)
        event[plan] = 1
        np.testing.assert_array_equal(rule.apply_event(start, event), goal)


def test_nullity_budget_does_not_limit_reachability():
    adjacency = np.ones((23, 23), dtype=np.uint8) ^ np.eye(23, dtype=np.uint8)
    rule = graph_rule(adjacency)
    start, goal = np.zeros(23), np.ones(23)
    assert is_reachable(rule, start, goal)
    with pytest.raises(ValueError, match="nullity 22 exceeds max_nullity=20"):
        solve(rule, start, goal)
    small = graph_rule([[0, 1], [1, 0]])
    assert len(solve(small, [0, 0], [1, 1])) == 1
    with pytest.raises(ValueError, match="max_nullity"):
        solve(small, [0, 0], [1, 1], max_nullity=0)


@pytest.mark.parametrize("budget", [-1, 1.5, True, "20"])
def test_invalid_solver_budget(budget):
    rule = grid_rule(1, 1)
    with pytest.raises(ValueError):
        solve(rule, [0], [1], max_nullity=budget)


def test_solver_validates_inputs_and_does_not_mutate():
    rule = grid_rule(3, 3)
    start, goal = np.zeros(9, dtype=np.uint8), np.ones(9, dtype=np.uint8)
    start_copy, goal_copy = start.copy(), goal.copy()
    solve(rule, start, goal)
    np.testing.assert_array_equal(start, start_copy)
    np.testing.assert_array_equal(goal, goal_copy)
    for function in (solve, is_reachable):
        with pytest.raises(ValueError):
            function(rule, np.zeros(8), goal)
        with pytest.raises(ValueError):
            function(rule, start, [2] * 9)
        with pytest.raises(TypeError):
            function(np.eye(9), start, goal)
