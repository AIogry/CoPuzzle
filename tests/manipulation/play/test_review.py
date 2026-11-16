"""Independent evidence checks for continuing play and hindsight labels.

These checks use native model observations and integration-state snapshots.
Synthetic observation edits represent corrupt or stale recorded evidence;
they never move a live puzzle or stand in for successful contact execution.
"""
from copy import deepcopy

import numpy as np
import pytest

from copuzzle.manipulation.play import semantics
from copuzzle.manipulation.play.bridge import PlayBridge
from copuzzle.manipulation.play.collector import forward


TASKS = (
    ("grid-3x3", 9, 0, 8),
    ("pancake-4", 4, 2, 3),
    ("topspin-20-4", 20, "reverse", "shift_right"),
)


def _initial(task_id, size):
    if task_id == "grid-3x3":
        return np.zeros(size, dtype=np.uint8)
    return np.arange(1, size + 1, dtype=np.int32)


def _history(frames):
    return {key: np.stack([item[key] for item in frames]) for key in frames[0]}


@pytest.fixture(scope="module")
def native_snapshots():
    pytest.importorskip("mujoco")
    result = {}
    for task_id, size, operation, _ in TASKS:
        bridge = PlayBridge(task_id, 100)
        try:
            state = _initial(task_id, size)
            obs, _ = bridge.reset(seed=42, state=state, goal=forward(task_id, state, operation))
            assert semantics.safe_boundary(task_id, obs, bridge.configuration())
            result[task_id] = (deepcopy(obs), bridge.configuration())
        finally:
            bridge.close()
    return result


@pytest.mark.parametrize("task_id,size,operation,next_operation", TASKS)
def test_goal_change_preserves_full_integration_state_and_control_history(
    task_id, size, operation, next_operation,
):
    mujoco = pytest.importorskip("mujoco")
    bridge = PlayBridge(task_id, 100)
    try:
        state = _initial(task_id, size)
        bridge.reset(seed=4, state=state, goal=forward(task_id, state, operation))
        for _ in range(4):
            obs, *_ = bridge.step(np.zeros(bridge.env.action_space.shape, dtype=np.float32))
        assert semantics.safe_boundary(task_id, obs, bridge.configuration())
        before = semantics.public_observation(obs)
        state_type = mujoco.mjtState.mjSTATE_INTEGRATION
        integration = np.empty(mujoco.mj_stateSize(bridge.env.model, state_type))
        mujoco.mj_getState(bridge.env.model, bridge.env.data, integration, state_type)
        step_count = bridge.env._steps

        changed = bridge.set_goal(forward(task_id, state, next_operation))

        after = np.empty_like(integration)
        mujoco.mj_getState(bridge.env.model, bridge.env.data, after, state_type)
        np.testing.assert_array_equal(after, integration)
        for name, value in before.items():
            np.testing.assert_array_equal(changed[name], value, err_msg=name)
        assert bridge.env._steps == step_count == 4
        np.testing.assert_array_equal(changed["goal"], forward(task_id, state, next_operation))
        with pytest.raises(RuntimeError, match="exactly one"):
            bridge.reset(seed=4, state=state, goal=forward(task_id, state, operation))
    finally:
        bridge.close()


def test_lamp_success_is_not_a_safe_button_release_boundary(native_snapshots):
    obs, config = deepcopy(native_snapshots["grid-3x3"])
    # A correctly illuminated board may still have a held button and low tool.
    obs["button_armed"][0] = 0
    obs["button_displacements"][0] = .010
    obs["end_effector_position"] = obs["button_rest_positions"][0].copy()
    observations = _history([obs, obs])
    assert semantics.goal_matches("grid-3x3", observations, obs["board"], config).tolist() == [True, True]
    assert not semantics.safe_boundary("grid-3x3", obs, config)


def _reordered_topspin(snapshot, board):
    result = deepcopy(snapshot)
    original = snapshot["board"]
    for slot, token in enumerate(board):
        result["token_positions"][token - 1] = snapshot["token_positions"][original[slot] - 1]
    result["board"] = np.asarray(board, dtype=np.int32).copy()
    return result


def test_topspin_recomputes_three_stable_frames_for_each_physical_board(native_snapshots):
    obs, config = deepcopy(native_snapshots["topspin-20-4"])
    changed = _reordered_topspin(obs, np.roll(obs["board"], 1))
    observations = _history([obs] * 4 + [changed] * 3)
    # Neither a falsely saturated old monitor nor a zeroed new one changes the
    # goal-independent three-observation evidence required for the new board.
    observations["success_streak"][:] = 3
    assert semantics.settled_board_runs("topspin-20-4", observations, config).tolist() == [0, 1, 2, 3, 1, 2, 3]
    expected = [False, False, False, False, False, False, True]
    assert semantics.goal_matches("topspin-20-4", observations, changed["board"], config).tolist() == expected
    observations["success_streak"][:] = 0
    assert semantics.goal_matches("topspin-20-4", observations, changed["board"], config).tolist() == expected


def test_topspin_committed_board_cannot_override_measured_token_identity(native_snapshots):
    obs, config = deepcopy(native_snapshots["topspin-20-4"])
    obs["board"] = np.roll(obs["board"], 1)
    obs["board_valid"][:] = 1
    obs["success_streak"][:] = 3
    observations = _history([obs] * 5)
    assert not semantics.physical_ready("topspin-20-4", obs, config)
    assert not semantics.goal_matches("topspin-20-4", observations, obs["board"], config).any()


@pytest.mark.parametrize("corruption", ["stale_board", "airborne", "held", "not_withdrawn"])
def test_pancake_hindsight_requires_supported_released_physical_order(native_snapshots, corruption):
    obs, config = deepcopy(native_snapshots["pancake-4"])
    assert semantics.physical_ready("pancake-4", obs, config)
    if corruption == "stale_board":
        obs["board"] = obs["board"][::-1].copy()
    elif corruption == "airborne":
        obs["token_positions"][:, 2] += .020
    elif corruption == "held":
        obs["tool_contacts"][0] = 1
        obs["tool_contact_forces"][0] = .2
    else:
        obs["end_effector_position"][:2] = config["scene"]["stack_center_xy"]
    # Even a superficially valid cached board is insufficient evidence.
    obs["board_valid"][:] = 1
    observations = _history([obs, obs])
    assert not semantics.goal_matches("pancake-4", observations, obs["board"], config).any()


def test_public_physical_state_excludes_goal_monitors_and_does_not_alias(native_snapshots):
    obs, _ = deepcopy(native_snapshots["topspin-20-4"])
    copied = semantics.public_observation(obs)
    assert "goal" not in copied and "success_streak" not in copied
    copied["board"][:] = 0
    assert np.array_equal(obs["board"], np.arange(1, 21))


@pytest.fixture(scope="module")
def button_play():
    pytest.importorskip("mujoco")
    from copuzzle.manipulation.play.collector import collect_play
    return collect_play("grid-3x3", seed=27, num_segments=3, operations=[0, 0, 1],
                        max_episode_steps=700)


def test_boundary_transition_keeps_old_goal_and_next_action_gets_new_goal(button_play):
    trace = button_play
    first, second, third = trace.segments
    boundary = second["start_step"]
    assert first["end_step"] == boundary
    assert first["start_step"] == 0 and third["end_step"] == len(trace.actions)
    assert second["end_step"] == third["start_step"]
    assert len(trace.observations["board"]) == len(trace.actions) + 1
    assert len(trace.infos) == len(trace.actions) + 1
    np.testing.assert_array_equal(trace.behavior_goals[boundary - 1], first["goal"])
    np.testing.assert_array_equal(trace.behavior_goals[boundary], second["goal"])
    np.testing.assert_array_equal(trace.observations["board"][boundary], first["goal"])
    assert trace.segment_ids[boundary - 1] == 0 and trace.segment_ids[boundary] == 1
    assert trace.goal_reached[boundary - 1] and not trace.goal_reached[boundary]
    # infos[b] describes action b-1 under its old behavior goal. Replacing this
    # with set_goal diagnostics would silently shift the stored transition.
    assert trace.infos[boundary]["is_success"]
    assert not trace.infos[boundary + 1]["is_success"]
    assert [info["steps"] for info in trace.infos] == list(range(len(trace.actions) + 1))
    assert trace.metadata["reset_count"] == 1
    assert "goal" not in trace.observations and "success_streak" not in trace.observations


def test_goal_view_stops_at_first_hit_and_omits_release_actions(button_play):
    from copuzzle.manipulation.play.data import goal_episode
    trace = button_play
    first = trace.segments[0]
    view = goal_episode(trace, 0, first["goal"])
    stop = first["goal_first_step"] + 1
    assert view["source_end"] == stop < first["end_step"]
    np.testing.assert_array_equal(view["actions"], trace.actions[:stop])
    assert view["terminated"].tolist() == [False] * (stop - 1) + [True]
    assert view["rewards"].tolist() == [-1.] * (stop - 1) + [0.]
    assert view["bootstrap_masks"][-1] == 0
    assert not view["truncated"].any() and not view["collection_cutoff"].any()
    assert view["shortest_distance"] is None
    # The collector must continue the real release to obtain a safe boundary,
    # but those post-success actions are not positive actor supervision.
    already = goal_episode(trace, stop, first["goal"])
    assert already["initially_satisfied"] and len(already["actions"]) == 0
    assert already["source_start"] == already["source_end"] == stop
    assert len(already["initial_observation"]) == view["observations"].shape[1]
    names = {item["name"] for item in view["feature_spec"]}
    assert not names & {"goal", "success_streak", "behavior_goals", "segment_ids",
                        "requested_operation", "controller_phase", "goal_reached"}


def test_future_goal_can_cross_behavior_segments_but_not_a_physical_episode(button_play):
    from copuzzle.manipulation.play.data import future_episode
    trace = button_play
    target = len(trace.actions)
    with pytest.raises(ValueError, match="segment"):
        future_episode(trace, 0, target)
    view = future_episode(trace, 0, target, within_segment=False)
    assert view["crosses_behavior_boundary"]
    assert set(view["source_segment_ids"]) == {0, 1, 2}
    assert view["terminated"][-1]
    assert view["source_end"] <= target
    np.testing.assert_array_equal(view["actions"], trace.actions[:view["source_end"]])
    # A later frame with an already satisfied board is a zero-action state
    # example, not an expert action label merely because its index is future.
    initial_future = future_episode(trace, 0, trace.segments[1]["end_step"], within_segment=False)
    assert initial_future["initially_satisfied"] and len(initial_future["actions"]) == 0
    with pytest.raises(ValueError, match="target_index"):
        future_episode(trace, 0, target + 1, within_segment=False)
    multiple_resets = deepcopy(trace)
    multiple_resets.metadata["reset_count"] = 2
    with pytest.raises(ValueError):
        future_episode(multiple_resets, 0, target, within_segment=False)


def test_controller_completed_without_measured_goal_is_not_success():
    pytest.importorskip("mujoco")
    from copuzzle.manipulation.play.collector import collect_play

    class EmptyController:
        completed = False

        def __call__(self, observation):
            self.completed = True
            return np.zeros(3, dtype=np.float32)

    trace = collect_play("grid-3x3", seed=27, num_segments=2, operations=[0, 8],
                         max_episode_steps=50, policy_factory=lambda *_: EmptyController())
    assert len(trace.actions) == 1  # Its final returned action was still executed.
    assert len(trace.segments) == 1
    assert trace.segments[0]["controller_completed"]
    assert trace.segments[0]["outcome"] == trace.metadata["stop_reason"] == "mismatch"
    assert not trace.goal_reached.any() and not trace.timeouts.any()
    assert trace.collection_cutoff.tolist() == [True]


@pytest.mark.parametrize("completion", [False, None])
def test_archive_cannot_claim_a_completed_segment_without_controller_completion(button_play, completion):
    forged = deepcopy(button_play)
    if completion is None:
        del forged.segments[0]["controller_completed"]
    else:
        forged.segments[0]["controller_completed"] = completion
    with pytest.raises(ValueError):
        forged.validate()


def test_native_success_continues_until_release_but_last_budget_success_times_out(button_play):
    from copuzzle.manipulation.play.collector import collect_play
    trace = button_play
    first_hit = int(np.flatnonzero(trace.goal_reached)[0])
    assert first_hit + 1 < trace.segments[0]["end_step"]
    assert np.any(trace.goal_reached[first_hit + 1:trace.segments[0]["end_step"]])
    assert not trace.timeouts.any()
    assert semantics.safe_boundary("grid-3x3",
        semantics.frame(trace.observations, trace.segments[0]["end_step"]), trace.configuration)
    assert not semantics.safe_boundary("grid-3x3",
        semantics.frame(trace.observations, first_hit + 1), trace.configuration)

    # Repeating exactly the deterministic prefix with that first success as
    # the global last action must stop play, even though native success wins
    # over native truncation in the unchanged episodic environment.
    short = collect_play("grid-3x3", seed=27, num_segments=2, operations=[0, 8],
                         max_episode_steps=first_hit + 1)
    np.testing.assert_array_equal(short.actions, trace.actions[:first_hit + 1])
    assert short.goal_reached[-1] and short.timeouts[-1]
    assert not short.collection_cutoff.any()
    assert short.metadata["stop_reason"] == short.segments[-1]["outcome"] == "timeout"
    assert len(short.segments) == 1
    from copuzzle.manipulation.play.data import goal_episode
    view = goal_episode(short, 0, short.segments[0]["goal"])
    # Goal-task views intentionally prioritize success, while preserving the
    # separate fact that the underlying continuing run exhausted its budget.
    assert view["terminated"][-1] and not view["truncated"][-1]
    assert view["source_timeouts"][-1]


def _idle_factory(task_id, operation):
    size = 7 if task_id == "pancake-4" else 3
    return lambda observation: np.zeros(size, dtype=np.float32)


def test_topspin_future_goal_uses_explicit_stability_feature_not_native_monitor():
    pytest.importorskip("mujoco")
    from copuzzle.manipulation.play.collector import collect_play
    from copuzzle.manipulation.play.data import future_episode
    trace = collect_play("topspin-20-4", seed=19, num_segments=1, operations=["reverse"],
                         max_episode_steps=20, max_steps=4, policy_factory=_idle_factory)
    with pytest.raises(ValueError, match="physically achieved"):
        future_episode(trace, 0, 1)
    measured = semantics.goal_matches(trace.task_id, trace.observations,
                                      trace.observations["board"][4], trace.configuration)
    assert measured.tolist() == [False, False, False, True, True]
    # The goal board is already physically present at each of these starts.
    # Waiting for the monitor is neither confirmed success nor an example of
    # directed manipulation. Keep the raw frames, but produce no actor labels.
    for start in range(3):
        view = future_episode(trace, start, 4)
        assert view["source_start"] == view["source_end"] == start
        assert view["initial_goal_confirmation_pending"]
        assert not view["initially_satisfied"] and not view["actor_supervision_valid"]
        assert len(view["actions"]) == len(view["rewards"]) == len(view["terminated"]) == 0
        assert len(view["observations"]) == len(view["next_observations"]) == 0
        assert view["shortest_distance"] is None
        spec = {item["name"]: item for item in view["feature_spec"]}
        assert "success_streak" not in spec and "goal" not in spec
        field = spec["settled_board_steps"]
        assert field["stop"] - field["start"] == 1
        assert view["initial_observation"][field["start"]] == start
    confirmed = future_episode(trace, 3, 4)
    assert confirmed["initially_satisfied"]
    assert not confirmed["initial_goal_confirmation_pending"]
    assert not confirmed["actor_supervision_valid"] and len(confirmed["actions"]) == 0
    assert confirmed["source_start"] == confirmed["source_end"] == 3
    assert confirmed["initial_observation"][field["start"]] == 3
    forged = deepcopy(trace)
    forged.observations["success_streak"] = np.full((5, 1), 3, dtype=np.int32)
    with pytest.raises(ValueError, match="observation fields"):
        future_episode(forged, 0, 1)


def test_future_target_rejects_pancake_airborne_cached_board():
    pytest.importorskip("mujoco")
    from copuzzle.manipulation.play.collector import collect_play
    from copuzzle.manipulation.play.data import future_episode
    trace = collect_play("pancake-4", seed=19, num_segments=1, operations=[2],
                         max_episode_steps=20, max_steps=1, policy_factory=_idle_factory)
    # Retain a typed, non-success transition but make its future observation
    # explicitly unsupported. The cached permutation alone must not be used.
    unsupported = deepcopy(trace)
    unsupported.observations["token_positions"][-1, :, 2] += .030
    unsupported.observations["board_valid"][-1, 0] = 0
    unsupported.segments[-1]["switch_safe"] = False
    unsupported.validate()
    with pytest.raises(ValueError, match="physically achieved"):
        future_episode(unsupported, 0, 1)
