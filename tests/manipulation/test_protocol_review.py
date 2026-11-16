"""Independent cross-layer checks of information, outcomes and measured cost."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from copuzzle.manipulation.data import load_transitions
from copuzzle.manipulation.runner import load_native, measure_operations, run_collection
from copuzzle.manipulation.suite import generate_suite, split_overlap, suite_sha256, validate_suite


def _suite(task_id="grid-3x3", depths=(1,), *, horizon=None):
    value = generate_suite(seed=37, depths=depths, task_ids=(task_id,))
    if horizon is not None:
        for task in value["tasks"]:
            task["max_control_steps"] = horizon
        value["sha256"] = suite_sha256(value)
    return value


def test_reference_length_is_a_witness_not_a_claimed_shortest_distance():
    # Lights Out A,B,A visits four distinct states and reaches the same goal
    # as B alone. It is a valid length-three witness, not a distance of three.
    from copuzzle.games.lights_out.rules import grid_rule
    suite = _suite(depths=(3,))
    case = suite["tasks"][0]
    case["state"] = [0] * 9
    case["reference_operations"] = [0, 1, 0]
    case["goal"] = grid_rule(3, 3).transition(case["state"], 1).tolist()
    suite["sha256"] = suite_sha256(suite)
    validate_suite(suite)
    assert case["reference_length"] == 3
    assert "not a shortest distance" in suite["reference_semantics"]
    assert not {"optimal_steps", "exact_distance"} & set(case)


def test_reverse_endpoint_pair_cannot_cross_splits_and_state_novelty_is_not_claimed():
    suite = _suite("topspin-20-4")
    overlap = split_overlap(suite)
    assert overlap["unordered_pairs_disjoint"]
    assert overlap["unseen_state_split"] is False
    assert all(row["unordered_pair_overlap"] == 0 for row in overlap["comparisons"])
    forged = deepcopy(suite)
    source, other = forged["tasks"][:2]
    other["state"], other["goal"] = source["goal"][:], source["state"][:]
    inverse = {"reverse": "reverse", "shift_left": "shift_right", "shift_right": "shift_left"}
    other["reference_operations"] = [inverse[source["reference_operations"][0]]]
    with pytest.raises(ValueError, match="pair"):
        suite_sha256(forged)


def test_topspin20_suite_does_not_consult_the_small_permutation_oracle(monkeypatch):
    from copuzzle.games.permutation.oracle import ExactOracle
    def forbidden(*args, **kwargs):
        raise AssertionError("The 8-token exact oracle cannot define TopSpin-20 task difficulty")
    monkeypatch.setattr(ExactOracle, "__init__", forbidden)
    suite = _suite("topspin-20-4", depths=(1, 2, 3))
    validate_suite(suite)
    assert all(len(case["state"]) == len(case["goal"]) == 20 for case in suite["tasks"])


@pytest.fixture(scope="module")
def external_collection(tmp_path_factory):
    pytest.importorskip("mujoco")
    suite = _suite(depths=(1, 2))
    root = tmp_path_factory.mktemp("protocol_external") / "collection"
    calls, seen, resets = [], [], []
    command = np.array([.125, -.25, .075], dtype=np.float32)
    class Spy:
        def reset(self):
            resets.append(True)
        def __call__(self, observation):
            seen.append({key: np.array(value, copy=True) for key, value in observation.items()})
            # Mutation of the supplied observation must not edit environment
            # goals or the native record captured by the collector.
            observation["goal"][:] = observation["board"]
            return command.copy()
    def factory(task_id):
        assert isinstance(task_id, str)
        calls.append(task_id)
        return Spy()
    result = run_collection(suite, root, splits=("train",), policy="reference",
                            policy_factory=factory, max_steps=2, replay=True)
    return root, suite, result, calls, seen, resets, command


def test_external_policy_gets_only_task_type_and_copied_public_observations(external_collection):
    root, suite, result, calls, seen, resets, command = external_collection
    assert calls == ["grid-3x3", "grid-3x3"] and len(resets) == 2
    assert result["policy_type"] == "external"
    assert result["policy_configuration"]["reference_witness_access"] is False
    forbidden = {"reference_operations", "reference_length", "controller_steps", "phase", "selected_token_id",
                 "reference_final_board", "task", "task_id", "id", "reset_seed", "infos", "metadata"}
    assert seen and all(not forbidden & set(observation) for observation in seen)
    assert all(isinstance(value, np.ndarray) for observation in seen for value in observation.values())
    cases = {case["id"]: case for case in suite["tasks"]}
    for row in result["episodes"]:
        episode = load_native(row["task_id"], root / row["trajectory"])
        assert np.array_equal(episode.actions, np.tile(command, (2, 1)))
        assert np.all(episode.observations["goal"] == np.asarray(cases[row["id"]]["goal"]))
        assert episode.metadata["user"]["reference_witness_access"] is False


def test_collection_cap_preserves_native_horizon_and_batch_episode_boundaries(external_collection):
    root, _, result, *_ = external_collection
    assert result["summary"]["attempts"] == result["summary"]["failures"] == 2
    assert result["summary"]["success_rate"] == 0.
    assert result["summary"]["collection_cutoffs"] == 2
    data = load_transitions(root, task_id="grid-3x3")
    assert data.configuration["max_episode_steps"] == 1500
    assert data.episode_offsets.tolist() == [0, 2, 4]
    assert data.collection_cutoff.tolist() == [False, True, False, True]
    assert not data.terminated.any() and not data.truncated.any()
    assert data.actions.dtype == np.float32 and data.actions.shape == (4, 3)
    batch = data.sample(64, rng=np.random.default_rng(3))
    indices = data.episode_offsets[batch["episode_index"]] + batch["step_index"]
    assert np.array_equal(batch["next_observations"], data.next_observations[indices])
    assert np.array_equal(batch["episode_end"], batch["step_index"] == 1)
    assert not {"reference_operations", "controller_steps", "phase", "metadata", "infos"} & set(batch)


def test_continuous_loading_does_not_import_or_construct_a_simulator(external_collection):
    root = external_collection[0]
    script = """
import sys
class ForbidPhysics:
    def find_spec(self, fullname, *args):
        if fullname == 'mujoco' or fullname.startswith('mujoco.'):
            raise AssertionError('Offline reader imported MuJoCo')
sys.meta_path.insert(0, ForbidPhysics())
from copuzzle.manipulation.data import load_transitions
dataset = load_transitions(sys.argv[1], task_id='grid-3x3')
assert dataset.actions.shape == (4, 3)
assert 'mujoco' not in sys.modules
"""
    run = subprocess.run([sys.executable, "-c", script, str(root)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def test_deleting_an_entire_failed_attempt_cannot_create_a_hidden_subset(external_collection, tmp_path):
    root = tmp_path / "omitted_attempt"
    shutil.copytree(external_collection[0], root)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["episodes"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="attempt|selection|selected|coverage|missing|episode"):
        load_transitions(root, task_id="grid-3x3")


def test_policy_factory_errors_remain_in_the_denominator_and_are_not_dropped(tmp_path):
    pytest.importorskip("mujoco")
    count = 0
    def factory(task_id):
        nonlocal count
        count += 1
        if count == 1:
            raise RuntimeError("deliberate factory failure")
        return lambda observation: np.zeros(3, dtype=np.float32)
    root = tmp_path / "mixed_failure"
    report = run_collection(_suite(depths=(1, 2)), root, splits=("train",),
                            policy_factory=factory, max_steps=2, replay=False)
    summary = report["summary"]
    assert summary["attempts"] == summary["failures"] == 2
    assert summary["errors"] == 1 and summary["trajectories"] == 1
    assert summary["successes"] == 0 and summary["success_rate"] == 0.
    assert report["episodes"][0]["trajectory"] is None
    assert report["episodes"][0]["error"]["stage"] == "policy_factory"
    with pytest.raises(ValueError, match="failure|error|trajectory|drop"):
        load_transitions(root, task_id="grid-3x3")


def test_native_policy_failure_is_saved_as_cutoff_not_goal_termination(tmp_path):
    pytest.importorskip("mujoco")
    class FailureAfterTwoControls:
        completed = False
        failure = "deliberate controller failure"
        def reset(self):
            self.calls = 0
        @property
        def finished(self):
            return self.calls >= 3
        def __call__(self, observation):
            self.calls += 1
            return np.zeros(3, dtype=np.float32)
    root = tmp_path / "saved_failure"
    report = run_collection(_suite("topspin-20-4"), root, splits=("train",),
                            policy_factory=lambda task_id: FailureAfterTwoControls(), max_steps=10, replay=True)
    row = report["episodes"][0]
    assert row["outcome"] == "policy_failed" and row["error"] is None
    assert row["trajectory"] is not None and row["control_steps"] == 2
    assert row["goal_success"] is False and row["terminated"] is False and row["truncated"] is False
    assert report["summary"]["controller_failures"] == 1 and report["summary"]["failures"] == 1
    for mode in ("terminated_only", "finite_horizon"):
        data = load_transitions(root, task_id="topspin-20-4", bootstrap_mode=mode)
        assert data.collection_cutoff.tolist() == [False, True]
        assert data.bootstrap_masks.tolist() == [1., 1.]


def test_true_horizon_timeout_preserves_both_explicit_bootstrap_conventions(tmp_path):
    pytest.importorskip("mujoco")
    root = tmp_path / "timeout"
    report = run_collection(_suite(horizon=2), root, splits=("train",), policy="zero", replay=False)
    assert report["summary"]["timeouts"] == 1 and report["summary"]["collection_cutoffs"] == 0
    continuing = load_transitions(root, task_id="grid-3x3")
    finite = load_transitions(root, task_id="grid-3x3", bootstrap_mode="finite_horizon")
    assert continuing.truncated.tolist() == finite.truncated.tolist() == [False, True]
    assert not finite.collection_cutoff.any()
    assert continuing.bootstrap_masks.tolist() == [1., 1.]
    assert finite.bootstrap_masks.tolist() == [1., 0.]
    assert continuing.provenance["bootstrap_mode"] == "terminated_only"
    assert finite.provenance["bootstrap_mode"] == "finite_horizon"


def _pancake_evidence(boards):
    count = len(boards)
    positions = np.zeros((count, 4, 3))
    for frame, board in enumerate(boards):
        for rank, token in enumerate(board):
            positions[frame, token - 1] = [.52, 0., .28 + .007 + .014 * (3 - rank)]
    return SimpleNamespace(
        configuration={"scene": {"stack_center_xy": [.52, 0.], "support_top_m": .28,
                                  "symmetric_boss_height_m": .014, "stack_pitch_m": .014,
                                  "withdraw_distance_m": .069}},
        observations={"board": np.asarray(boards, np.int32), "board_valid": np.ones((count, 1), np.uint8),
                      "token_positions": positions,
                      "token_orientations": np.tile([1., 0., 0., 0.], (count, 4, 1)),
                      "token_velocities": np.zeros((count, 4, 6)), "tool_contacts": np.zeros((count, 4), np.uint8),
                      "tool_contact_forces": np.zeros((count, 4)),
                      "end_effector_position": np.tile([.40, 0., .336], (count, 1))})


def test_pancake_cost_requires_supported_released_withdrawn_physical_evidence():
    start, flipped = [1, 2, 3, 4], [2, 1, 3, 4]
    episode = _pancake_evidence([start, flipped, flipped, flipped, flipped])
    episode.observations["token_positions"][1, :, 2] += .20  # Board claim while airborne.
    episode.observations["tool_contacts"][2, 0] = 1  # Restacked but still clamped.
    episode.observations["tool_contact_forces"][2, 0] = .3
    episode.observations["end_effector_position"][3, :2] = [.52, 0.]  # Released but not withdrawn.
    report = measure_operations("pancake-4", episode)
    assert report["measured_operations"] == [2]
    assert report["measured_atomic_cost"] == 1
    assert report["operation_events"][0]["control_step"] == 3


def test_unknown_measured_changes_are_not_zero_cost_and_three_shifts_cost_three():
    infos = [{"event_log": [{"operation": "shift_right", "atomic_cost": 1}]} for _ in range(3)]
    report = measure_operations("topspin-20-4", SimpleNamespace(infos=infos))
    assert report["measured_atomic_cost"] == 3
    infos.append({"event_log": [{"operation": "unclassified_measured_rearrangement", "atomic_cost": None}]})
    report = measure_operations("topspin-20-4", SimpleNamespace(infos=infos))
    assert report["measured_atomic_cost"] is None and report["known_atomic_cost"] == 3
    assert report["unknown_atomic_cost_events"] == 1
    pancake = measure_operations("pancake-4", _pancake_evidence([[1, 2, 3, 4], [2, 3, 1, 4]]))
    assert pancake["measured_atomic_cost"] is None and pancake["unknown_atomic_cost_events"] == 1
