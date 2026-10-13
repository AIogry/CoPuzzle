"""Physics data contracts tested without a simulator or an executable policy learner."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from copuzzle.games.lights_out.physics import data as data_module
from copuzzle.games.lights_out.physics.data import (
    PHYSICS_TRAJECTORY_SCHEMA, PhysicsGoalSampler, load_physics_trajectory,
    record_episode, relabel_physics_transitions,
)
from copuzzle.games.lights_out.rules import get_rule, graph_rule
from copuzzle.games.lights_out.structure import export_structure


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class FakePhysicalEnv:
    """Predetermined measured events; does not simulate physical contact."""
    def __init__(self, *, rule=None, horizon=5, goal=None, schedule=None):
        self.rule = rule or graph_rule(np.zeros((3, 3), dtype=np.uint8), name='three-independent-buttons')
        self.public_structure = export_structure(self.rule)
        self.max_episode_steps = horizon
        self.n = self.rule.n_buttons
        self.goal = np.ones(self.n, dtype=np.uint8) if goal is None else np.asarray(goal, dtype=np.uint8)
        self.schedule = schedule if schedule is not None else [[], [(3, 0), (3, 1)], [(1, 2), (9, 2)], [(4, 0)], []]
        self.rest = np.column_stack((np.arange(self.n) * .07, np.zeros(self.n), np.full(self.n, .03))).astype(np.float64)
        self.model_sha256 = 'a' * 64
        self._configuration = dict(
            schema_version='copuzzle.physics_config.v1',
            **{key: getattr(self.public_structure, key) for key in
               ('task_name', 'family', 'n_buttons', 'rule_id', 'structure_sha256')},
            model_sha256=self.model_sha256, physics_timestep=.002, control_timestep=.05,
            substeps=25, max_episode_steps=horizon, finite_horizon_observation='steps_remaining',
            button_rest_positions=self.rest.tolist(),
        )
        self.configuration_sha256 = hashlib.sha256(_canonical(self._configuration)).hexdigest()
        self.submitted_actions = []

    def configuration(self):
        return deepcopy(self._configuration)

    def _observation(self, counts):
        return dict(
            board=self.board.copy(), goal=self.goal.copy(),
            joint_positions=np.full(6, self.step_index * .01, dtype=np.float64),
            joint_velocities=np.full(6, .2, dtype=np.float64),
            end_effector_position=np.array([.1, .2, .1 - self.step_index * .001], dtype=np.float64),
            end_effector_orientation=np.array([0., 1., 0., 0.], dtype=np.float64),
            button_positions=self.rest.copy(), button_rest_positions=self.rest.copy(),
            button_displacements=np.zeros(self.n, dtype=np.float64),
            button_velocities=np.zeros(self.n, dtype=np.float64),
            button_armed=np.ones(self.n, dtype=np.uint8), previous_event_counts=counts.copy(),
            steps_remaining=np.array([self.max_episode_steps - self.step_index], dtype=np.int32),
            **self.public_structure.observation_fields(),
        )

    def reset(self, *, seed, options=None):
        self.seed, self.options = seed, deepcopy(options)
        self.step_index = 0
        self.board = np.zeros(self.n, dtype=np.uint8)
        self.submitted_actions.clear()
        return self._observation(np.zeros(self.n, dtype=np.int32)), {'is_success': bool(np.array_equal(self.board, self.goal))}

    def step(self, action):
        self.submitted_actions.append(action.copy())
        events = self.schedule[self.step_index] if self.step_index < len(self.schedule) else []
        counts = np.zeros(self.n, dtype=np.int32)
        for _, button in events:
            counts[button] += 1
            self.board = self.rule.transition(self.board, button)
        self.step_index += 1
        success = bool(np.array_equal(self.board, self.goal))
        truncated = self.step_index == self.max_episode_steps and not success
        return self._observation(counts), 0. if success else -1., success, truncated, {
            'is_success': success, 'event_log': [{'substep': substep, 'button': button} for substep, button in events],
            'steps': self.step_index, 'elapsed_time': self.step_index * .05, 'rule_id': self.rule.rule_id,
        }


def _policy(observation):
    # Extra float64 precision must be quantized before submission to the env.
    return np.array([.123456789, observation['steps_remaining'][0] * .01, -.2], dtype=np.float64)


@pytest.fixture
def episode():
    return record_episode(FakePhysicalEnv(), _policy, seed=17, metadata={'controller': 'test-only-script'})


def _assert_same(left, right):
    assert left.configuration == right.configuration
    assert left.metadata == right.metadata
    assert left.structure.structure_sha256 == right.structure.structure_sha256
    for name, array in left._arrays().items():
        np.testing.assert_array_equal(array, right._arrays()[name])


def test_record_preserves_all_successors_measured_events_and_exact_actions():
    env = FakePhysicalEnv()
    trajectory = record_episode(env, _policy, seed=12, options={'example': np.array([1, 2])})
    assert trajectory.num_steps == 5
    assert all(len(value) == 6 for value in trajectory.observations.values())
    assert trajectory.actions.dtype == np.float32
    np.testing.assert_array_equal(trajectory.actions, env.submitted_actions)
    np.testing.assert_array_equal(trajectory.event_counts, [[0, 0, 0], [1, 1, 0], [0, 0, 2], [1, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(trajectory.event_steps, [1, 1, 2, 2, 3])
    np.testing.assert_array_equal(trajectory.event_substeps, [3, 3, 1, 9, 4])
    np.testing.assert_array_equal(trajectory.observations['board'][2], trajectory.observations['board'][3])
    np.testing.assert_array_equal(trajectory.observations['steps_remaining'][:, 0], [5, 4, 3, 2, 1, 0])
    assert trajectory.button_rest_positions.shape == (3, 3)
    assert 'button_rest_positions' not in trajectory.observations
    assert 'adjacency' not in trajectory.observations
    assert trajectory.metadata['reset_seed'] == 12
    assert trajectory.metadata['reset_options'] == {'example': [1, 2]}
    assert trajectory.truncated[-1] and not trajectory.terminated.any() and not trajectory.collection_cutoff.any()


@pytest.mark.parametrize('ending', ['terminated', 'truncated', 'collection_cutoff'])
def test_distinct_episode_ending_conditions(ending):
    env = FakePhysicalEnv(goal=[0, 1, 0] if ending == 'terminated' else None)
    trajectory = record_episode(env, _policy, seed=0, max_steps=3 if ending == 'collection_cutoff' else None)
    assert getattr(trajectory, ending)[-1]
    assert trajectory.num_steps == {'terminated': 4, 'truncated': 5, 'collection_cutoff': 3}[ending]
    for other in {'terminated', 'truncated', 'collection_cutoff'} - {ending}:
        assert not getattr(trajectory, other).any()


def test_success_has_priority_at_environment_horizon():
    trajectory = record_episode(FakePhysicalEnv(goal=[0, 1, 0], horizon=4), _policy, seed=0)
    assert trajectory.terminated[-1] and not trajectory.truncated[-1]
    assert trajectory.rewards[-1] == 0
    assert trajectory.observations['steps_remaining'][-1, 0] == 0


def test_relabel_keeps_physical_fields_and_only_board_success(episode):
    batch = relabel_physics_transitions(episode, [0, 1, 4], [[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(batch['rewards'], [-1, 0, -1])
    np.testing.assert_array_equal(batch['terminated'], [False, True, False])
    np.testing.assert_array_equal(batch['truncated'], [False, False, True])
    np.testing.assert_array_equal(batch['bootstrap_masks'], [1, 0, 1])
    for name in episode.observations.keys() - {'goal'}:
        np.testing.assert_array_equal(batch['observations'][name], episode.observations[name][[0, 1, 4]])
        np.testing.assert_array_equal(batch['next_observations'][name], episode.observations[name][[1, 2, 5]])
    np.testing.assert_array_equal(batch['observations']['goal'], batch['goals'])
    np.testing.assert_array_equal(batch['next_observations']['goal'], batch['goals'])
    assert batch['structure']['adjacency'].shape == (3, 3)
    assert batch['button_rest_positions'].shape == (3, 3)
    assert not {'configuration', 'metadata', 'event_log'} & batch.keys()


def test_original_success_at_other_goal_becomes_cutoff_with_bootstrap():
    episode = record_episode(FakePhysicalEnv(goal=[0, 1, 0]), _policy, seed=0)
    batch = relabel_physics_transitions(episode, [3], [[0, 0, 1]])
    assert batch['collection_cutoff'][0]
    assert not batch['terminated'][0] and not batch['truncated'][0]
    assert batch['bootstrap_masks'][0] == 1
    success = relabel_physics_transitions(episode, [3], [[0, 1, 0]])
    assert success['terminated'][0] and success['bootstrap_masks'][0] == 0
    assert not success['collection_cutoff'][0]


def test_sampler_persistent_rng_snapshot_and_no_event_steps(episode):
    first, second = PhysicsGoalSampler(episode, seed=4), PhysicsGoalSampler(episode, seed=4)
    batch1, batch2 = first.sample(80), first.sample(80)
    for actual, expected in ((batch1, second.sample(80)), (batch2, second.sample(80))):
        for key in ('goals', 'actions', 'event_counts', 'rewards'):
            np.testing.assert_array_equal(actual[key], expected[key])
    assert not np.array_equal(batch1['actions'], batch2['actions'])
    assert np.any(np.all(batch1['event_counts'] == 0, axis=1))
    assert np.all(np.any(batch1['observations']['board'] != batch1['goals'], axis=1))
    # The final no-event tail has no different future board and cannot be sampled.
    assert not np.any(batch1['observations']['steps_remaining'][:, 0] == 1)
    pristine = PhysicsGoalSampler(episode, seed=8)
    snapshot = PhysicsGoalSampler(episode, seed=8)
    episode.observations['board'][:] = 1
    episode.actions[:] = 1
    episode.button_rest_positions[:] = 999
    batch1['structure']['adjacency'][:] = 1
    batch1['button_rest_positions'][:] = 777
    expected, actual = pristine.sample(40), snapshot.sample(40)
    for key in ('goals', 'actions', 'button_rest_positions'):
        np.testing.assert_array_equal(actual[key], expected[key])
    np.testing.assert_array_equal(actual['structure']['adjacency'], np.zeros((3, 3)))


def test_no_press_episode_requires_explicit_goals():
    episode = record_episode(FakePhysicalEnv(schedule=[]), _policy, seed=0)
    with pytest.raises(ValueError, match='No transition'):
        PhysicsGoalSampler(episode)
    batch = relabel_physics_transitions(episode, [4], [[1, 0, 0]])
    assert batch['truncated'][0] and batch['bootstrap_masks'][0] == 1


@pytest.mark.parametrize('task', ['graph', 'grid'])
def test_npz_roundtrip_strict_static_storage_and_no_physics_import(tmp_path, task):
    rule = get_rule('grid-2x2') if task == 'grid' else None
    trajectory = record_episode(FakePhysicalEnv(rule=rule), _policy, seed=7, max_steps=2)
    path = trajectory.save(tmp_path / f'{task}.npz')
    _assert_same(trajectory, load_physics_trajectory(path))
    manifest = json.loads(path.with_suffix('.json').read_text())
    assert manifest['schema_version'] == PHYSICS_TRAJECTORY_SCHEMA
    assert manifest['artifact']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as archive:
        assert archive['button_rest_positions'].shape == (trajectory.n_buttons, 3)
        assert all(archive[name].dtype != object for name in archive.files)
        assert not {'configuration', 'metadata', 'model', 'rule'} & set(archive.files)
    code = '''
import importlib.abc, sys
class NoPhysics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mujoco' or fullname.startswith('mujoco.'):
            raise RuntimeError('Loader imported MuJoCo')
sys.meta_path.insert(0, NoPhysics())
from copuzzle.games.lights_out.physics.data import load_physics_trajectory
x = load_physics_trajectory(sys.argv[1])
assert x.num_steps == 2
assert not any(k == 'mujoco' or k.startswith('mujoco.') for k in sys.modules)
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_save_refuses_overwrite_and_failed_publish_cleans_only_own_files(tmp_path, episode, monkeypatch):
    path = episode.save(tmp_path / 'saved.npz')
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        episode.save(path)
    assert path.read_bytes() == before
    partial = tmp_path / 'raced.npz'
    original_link = data_module.os.link

    def racing_link(source, target):
        if Path(target).suffix == '.json':
            Path(target).write_text('other writer')
        original_link(source, target)

    monkeypatch.setattr(data_module.os, 'link', racing_link)
    with pytest.raises(FileExistsError):
        episode.save(partial)
    assert not partial.exists()
    assert partial.with_suffix('.json').read_text() == 'other writer'
    assert not list(tmp_path.glob('*.tmp'))


def test_checksum_and_duplicate_manifest_keys_rejected(tmp_path, episode):
    path = episode.save(tmp_path / 'checksum.npz')
    path.write_bytes(path.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='checksum'):
        load_physics_trajectory(path)
    path = episode.save(tmp_path / 'duplicate.npz')
    sidecar = path.with_suffix('.json')
    sidecar.write_text(sidecar.read_text().replace('{', '{"num_steps": 5,', 1))
    with pytest.raises(ValueError, match='Duplicate JSON'):
        load_physics_trajectory(path)


def _edit_archive(path, change):
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    change(arrays)
    np.savez_compressed(path, **arrays)
    sidecar = path.with_suffix('.json')
    manifest = json.loads(sidecar.read_text())
    manifest['artifact']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['arrays'] = {key: {'dtype': value.dtype.name, 'shape': list(value.shape)} for key, value in arrays.items()}
    sidecar.write_text(json.dumps(manifest))


@pytest.mark.parametrize('mutation', ['dtype', 'extra', 'missing', 'object'])
def test_rehashed_invalid_archive_still_fails_schema(tmp_path, episode, mutation):
    path = episode.save(tmp_path / f'{mutation}.npz')
    def change(arrays):
        if mutation == 'dtype':
            arrays['actions'] = arrays['actions'].astype(np.float64)
        elif mutation == 'extra':
            arrays['hidden_target'] = np.ones(3)
        elif mutation == 'missing':
            arrays.pop('observation__steps_remaining')
        else:
            arrays['actions'] = np.array([{'object': 1}], dtype=object)
    _edit_archive(path, change)
    with pytest.raises(ValueError):
        load_physics_trajectory(path)


@pytest.mark.parametrize('mutation', ['counts', 'logs', 'board', 'previous', 'remaining', 'goal', 'reward', 'ending', 'layout', 'config'])
def test_validation_rejects_temporal_or_provenance_inconsistency(episode, mutation):
    if mutation == 'counts':
        episode.event_counts[0, 0] = -1
    elif mutation == 'logs':
        episode.event_substeps[0] = 25
    elif mutation == 'board':
        episode.observations['board'][1, 2] ^= 1
    elif mutation == 'previous':
        episode.observations['previous_event_counts'][2, 0] = 0
    elif mutation == 'remaining':
        episode.observations['steps_remaining'][2, 0] = 4
    elif mutation == 'goal':
        episode.observations['goal'][-1, 0] = 0
    elif mutation == 'reward':
        episode.rewards[0] = 0
    elif mutation == 'ending':
        episode.collection_cutoff[-1] = True
    elif mutation == 'layout':
        episode.button_rest_positions[0, 0] = 100
    else:
        episode.configuration['rule_id'] = 'b' * 64
        episode.metadata['configuration_sha256'] = hashlib.sha256(_canonical(episode.configuration)).hexdigest()
    with pytest.raises(ValueError):
        episode.validate()


@pytest.mark.parametrize('indices,goals', [([-1], [[1, 0, 0]]), ([5], [[1, 0, 0]]), ([0], [[0, 0, 0]]), ([0], [[2, 0, 0]]), ([True], [[1, 0, 0]])])
def test_invalid_relabels_rejected(episode, indices, goals):
    with pytest.raises(ValueError):
        relabel_physics_transitions(episode, indices, goals)


@pytest.mark.parametrize('action', [[0, 0], [0, 0, np.nan], [2, 0, 0]])
def test_invalid_policy_actions_fail_before_step(action):
    env = FakePhysicalEnv()
    with pytest.raises(ValueError, match='continuous actions'):
        record_episode(env, lambda _: action, seed=0)
    assert not env.submitted_actions


def test_solved_reset_and_invalid_limits_are_rejected():
    with pytest.raises(ValueError, match='successful reset'):
        record_episode(FakePhysicalEnv(goal=[0, 0, 0]), _policy, seed=0)
    for seed, limit in ((True, None), (-1, None), (0, 0), (0, True)):
        with pytest.raises(ValueError):
            record_episode(FakePhysicalEnv(), _policy, seed=seed, max_steps=limit)


def test_relabel_original_success_at_horizon_recomputes_timeout():
    episode = record_episode(FakePhysicalEnv(goal=[0, 1, 0], horizon=4), _policy, seed=0)
    assert episode.terminated[-1] and not episode.truncated[-1]
    batch = relabel_physics_transitions(episode, [3], [[0, 0, 1]])
    assert batch['truncated'][0]
    assert not batch['terminated'][0] and not batch['collection_cutoff'][0]
    assert batch['bootstrap_masks'][0] == 1
