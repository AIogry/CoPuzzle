"""Permutation fixture data contracts, with synthetic and real replay checks."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from copuzzle.games.permutation.physics import data as data_module
from copuzzle.games.permutation.physics.data import (
    PHYSICS_TRAJECTORY_SCHEMA, PhysicsGoalSampler, load_physics_trajectory,
    record_trajectory, relabel_physics_transitions, replay_trajectory,
)
from copuzzle.games.permutation.rules import get_rule


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class FakeFixtureEnv:
    """Predetermined observations; this is a schema fixture, not physics data."""
    def __init__(self, *, horizon=6, goal=None, schedule=None):
        self.rule = get_rule('pancake-4')
        self.horizon = horizon
        self.goal = np.array(goal if goal is not None else [4, 3, 2, 1], dtype=np.int32)
        self.slots = np.array([[.5, 0, .3 - i * .04] for i in range(4)], dtype=np.float64)
        self.racks = np.array([[.3, i * .06, .1] for i in range(3)], dtype=np.float64)
        self.schedule = {1: [(5, 0, 0, 2)], 4: [(10, 1, 3, 4)]} if schedule is None else schedule
        self.configuration_value = {
            'schema_version': 'copuzzle.permutation_physics_config.v1',
            'task_name': self.rule.name, 'rule_spec': self.rule.spec(),
            'model_sha256': 'a' * 64, 'physics_timestep': .002,
            'control_timestep': .05, 'substeps': 25, 'max_episode_steps': horizon,
        }
        self.configuration_sha256 = hashlib.sha256(canonical(self.configuration_value)).hexdigest()
        self.submitted_actions = []

    def configuration(self):
        return deepcopy(self.configuration_value)

    def _obs(self, counts):
        phase = [0, 1, 2, 0, 1, 2, 0][min(self.index, 6)]
        active = -1 if phase == 0 else (0 if self.index < 3 else 1)
        positions = np.zeros((4, 3), dtype=np.float64)
        for slot, token in enumerate(self.board):
            positions[token - 1] = self.slots[slot]
        if phase == 1:
            positions[:, 0] += .02
        return {
            'board': self.board.copy(), 'goal': self.current_goal.copy(),
            'board_valid': np.array([phase != 1], dtype=np.uint8),
            'mechanism_phase': np.array([phase], dtype=np.int32),
            'active_operation': np.array([active], dtype=np.int32),
            'rack_start': np.array([0. if phase == 0 else .0005], dtype=np.float64),
            'joint_positions': np.full(6, self.index * .001, dtype=np.float64),
            'joint_velocities': np.zeros(6, dtype=np.float64),
            'end_effector_position': np.array([.2, .3, .2 - self.index * .001], dtype=np.float64),
            'end_effector_orientation': np.array([1., 0, 0, 0], dtype=np.float64),
            'token_positions': positions,
            'token_orientations': np.tile(np.array([1., 0, 0, 0], dtype=np.float64), (4, 1)),
            'token_velocities': np.full((4, 6), 0. if phase == 0 else .02, dtype=np.float64),
            'slot_positions': self.slots.copy(), 'rack_positions': self.racks.copy(),
            'rack_rest_positions': self.racks.copy(),
            'rack_displacements': np.zeros(3, dtype=np.float64),
            'rack_velocities': np.zeros(3, dtype=np.float64),
            'carrier_angles': np.zeros(3, dtype=np.float64),
            'carrier_velocities': np.zeros(3, dtype=np.float64),
            'previous_event_counts': counts.copy(),
            'steps_remaining': np.array([self.horizon - self.index], dtype=np.int32),
        }

    def _info(self, observation, events):
        success = bool(data_module._ready(observation) and np.array_equal(self.board, self.current_goal))
        return {'is_success': success, 'steps': self.index, 'elapsed_time': self.index * .05,
                'rule_id': self.rule.rule_id, 'event_log': events,
                'attempt_log': ([{'substep': 2, 'operation': 2, 'result': 'aborted_and_returned'}]
                                if self.index == 3 else []), 'failure_reason': None}

    def reset(self, *, seed, options=None):
        self.seed, self.options = seed, deepcopy(options)
        options = options or {}
        self.index = 0
        self.board = np.array(options.get('state', [1, 2, 3, 4]), dtype=np.int32)
        self.current_goal = np.array(options.get('goal', self.goal), dtype=np.int32)
        self.submitted_actions = []
        observation = self._obs(np.zeros(3, dtype=np.int32))
        return observation, self._info(observation, [])

    def step(self, action):
        self.submitted_actions.append(action.copy())
        events, counts = [], np.zeros(3, dtype=np.int32)
        for substep, operation, started_step, started_substep in self.schedule.get(self.index, []):
            self.board = self.rule.transition(self.board, operation)
            counts[operation] += 1
            events.append({'substep': substep, 'operation': operation,
                           'started_step': started_step, 'started_substep': started_substep,
                           'duration_seconds': (self.index - started_step) * .05 + (substep - started_substep) * .002})
        self.index += 1
        observation = self._obs(counts)
        info = self._info(observation, events)
        success = info['is_success']
        return observation, 0. if success else -1., success, self.index == self.horizon and not success, info


def policy(observation):
    return np.array([.123456789, .01 * observation['steps_remaining'][0], -.2], dtype=np.float64)


@pytest.fixture
def episode():
    return record_trajectory(FakeFixtureEnv(), policy, seed=17, metadata={'controller': 'synthetic-schema-test'})


def same_episode(left, right):
    assert left.configuration == right.configuration
    assert left.metadata == right.metadata
    assert left.infos == right.infos
    for name, array in left._arrays().items():
        np.testing.assert_array_equal(array, right._arrays()[name])


def test_recorder_retains_complete_observations_events_and_float32_actions():
    env = FakeFixtureEnv()
    result = record_trajectory(env, policy, seed=4, options={'state': np.arange(1, 5, dtype=np.int32)})
    assert result.num_steps == 6
    assert result.n_tokens == 4 and result.n_actions == 3
    assert all(len(value) == 7 for value in result.observations.values())
    assert len(result.infos) == 7 and result.infos[3]['attempt_log'][0]['result'] == 'aborted_and_returned'
    np.testing.assert_array_equal(result.actions, env.submitted_actions)
    assert result.actions.dtype == np.float32
    np.testing.assert_array_equal(result.event_steps, [1, 4])
    np.testing.assert_array_equal(result.event_substeps, [5, 10])
    np.testing.assert_array_equal(result.event_operations, [0, 1])
    np.testing.assert_array_equal(result.event_started_steps, [0, 3])
    np.testing.assert_array_equal(result.event_started_substeps, [2, 4])
    np.testing.assert_allclose(result.event_durations, [.056, .062], atol=1e-15)
    np.testing.assert_array_equal(result.event_counts.sum(0), [1, 1, 0])
    assert result.metadata['reset_options'] == {'state': [1, 2, 3, 4]}
    assert result.metadata['sources']['data'] == hashlib.sha256(Path(data_module.__file__).read_bytes()).hexdigest()
    assert result.metadata['collector']['stop_reason'] == 'environment_truncated'


@pytest.mark.parametrize('ending', ['terminated', 'truncated', 'collection_cutoff'])
def test_success_timeout_and_collection_cutoff_are_distinct(ending):
    env = FakeFixtureEnv(goal=[3, 1, 2, 4] if ending == 'terminated' else None)
    result = record_trajectory(env, policy, seed=0, max_steps=4 if ending == 'collection_cutoff' else None)
    assert getattr(result, ending)[-1]
    for other in {'terminated', 'truncated', 'collection_cutoff'} - {ending}:
        assert not getattr(result, other).any()
    if ending == 'terminated':
        assert result.num_steps == 6
        assert np.array_equal(result.observations['board'][5], result.observations['goal'][5])
        assert result.observations['mechanism_phase'][5, 0] == 2
        assert result.rewards[4] == -1 and not result.terminated[4]
        assert result.rewards[5] == 0


def test_policy_completion_preserves_prefix_without_extra_action():
    class CompletingPolicy:
        def reset(self):
            self.completed = False

        def __call__(self, observation):
            if observation['steps_remaining'][0] == 3:
                self.completed = True
            return np.zeros(3, dtype=np.float32)

    env = FakeFixtureEnv()
    result = record_trajectory(env, CompletingPolicy(), seed=0)
    assert result.num_steps == 3 and len(env.submitted_actions) == 3
    assert result.collection_cutoff[-1]
    assert result.metadata['collector']['stop_reason'] == 'policy_completed'
    assert result.observations['mechanism_phase'][-1, 0] == 0


def test_wrapped_controller_error_can_preserve_partial_episode():
    class TimeoutWrapper:
        def reset(self):
            self.completed, self.error, self.calls = False, None, 0

        def __call__(self, observation):
            self.calls += 1
            try:
                if self.calls == 3:
                    raise RuntimeError('test controller timeout')
                return policy(observation)
            except RuntimeError as error:
                self.error, self.completed = str(error), True
                return np.zeros(3, dtype=np.float32)

    wrapper = TimeoutWrapper()
    result = record_trajectory(FakeFixtureEnv(), wrapper, seed=0)
    result.metadata['user']['controller_failure'] = wrapper.error
    result.validate()
    assert result.num_steps == 2 and result.collection_cutoff[-1]
    assert result.observations['mechanism_phase'][-1, 0] == 2


def test_noncommuting_events_are_applied_in_recorded_order():
    env = FakeFixtureEnv(schedule={1: [(5, 1, 0, 0), (6, 0, 0, 1)]})
    result = record_trajectory(env, policy, seed=0)
    np.testing.assert_array_equal(result.event_operations, [1, 0])
    np.testing.assert_array_equal(result.observations['board'][2], [2, 3, 1, 4])
    corrupt = deepcopy(result)
    corrupt.event_operations[:] = [0, 1]
    corrupt.infos[2]['event_log'] = corrupt.events_at(1)
    with pytest.raises(ValueError, match='ordered measured operations'):
        corrupt.validate()


@pytest.mark.parametrize('mutation', [
    lambda e: e.observations.pop('token_velocities'),
    lambda e: e.observations.update(extra=np.zeros(1)),
    lambda e: e.observations.__setitem__('board', e.observations['board'].astype(np.int64)),
    lambda e: e.observations.__setitem__('token_positions', e.observations['token_positions'][:-1]),
    lambda e: e.observations['joint_positions'].__setitem__((0, 0), np.nan),
    lambda e: e.observations['board'].__setitem__((0, 0), 2),
    lambda e: e.observations['goal'].__setitem__(2, [2, 1, 3, 4]),
    lambda e: e.observations['board_valid'].__setitem__((1, 0), 1),
    lambda e: e.observations['active_operation'].__setitem__((0, 0), 0),
    lambda e: e.observations['mechanism_phase'].__setitem__((1, 0), 4),
    lambda e: e.observations['slot_positions'].__setitem__((1, 0, 0), 99.),
    lambda e: e.observations['token_orientations'].__setitem__((1, 0, 0), 2.),
    lambda e: e.observations['token_positions'].__setitem__((0, 0, 0), 99.),
    lambda e: e.actions.__setitem__((0, 0), 1.1),
    lambda e: e.event_counts.__setitem__((0, 0), -1),
    lambda e: e.observations['previous_event_counts'].__setitem__((1, 0), 1),
    lambda e: e.terminated.__setitem__(0, True),
    lambda e: e.collection_cutoff.__setitem__(-1, True),
    lambda e: e.rewards.__setitem__(0, 0),
    lambda e: e.observations['steps_remaining'].__setitem__((0, 0), 99),
    lambda e: e.event_started_substeps.__setitem__(0, 25),
    lambda e: e.event_substeps.__setitem__(0, 25),
    lambda e: e.event_operations.__setitem__(0, 3),
    lambda e: e.event_started_steps.__setitem__(0, 2),
    lambda e: e.event_durations.__setitem__(0, .1),
    lambda e: e.metadata.__setitem__('configuration_sha256', 'b' * 64),
    lambda e: e.metadata['sources'].__setitem__('data', 'bad'),
    lambda e: e.metadata['collector'].__setitem__('stop_reason', 'max_steps'),
    lambda e: e.infos[1].__setitem__('steps', 9),
    lambda e: e.infos[1].__setitem__('is_success', True),
    lambda e: e.infos[1].__setitem__('elapsed_time', 999.),
    lambda e: e.infos[2]['event_log'][0].__setitem__('operation', 2),
    lambda e: e.infos[2]['event_log'][0].__setitem__('operation', False),
    lambda e: e.infos[3]['attempt_log'][0].__setitem__('substep', 25),
])
def test_reject_corrupt_episode_fields(episode, mutation):
    mutation(episode)
    with pytest.raises(ValueError):
        episode.validate()


def test_stability_is_required_for_relabel_success(episode):
    goal = [3, 1, 2, 4]
    batch = relabel_physics_transitions(episode, [4, 5], [goal, goal])
    np.testing.assert_array_equal(batch['rewards'], [-1, 0])
    np.testing.assert_array_equal(batch['terminated'], [False, True])
    np.testing.assert_array_equal(batch['bootstrap_masks'], [1, 0])
    assert not batch['truncated'].any()
    np.testing.assert_array_equal(batch['raw_truncated'], [False, True])
    np.testing.assert_array_equal(batch['step_indices'], [4, 5])
    np.testing.assert_array_equal(batch['goal_observation_indices'], [-1, -1])
    for key in episode.observations.keys() - {'goal'}:
        np.testing.assert_array_equal(batch['observations'][key], episode.observations[key][[4, 5]])
        np.testing.assert_array_equal(batch['next_observations'][key], episode.observations[key][[5, 6]])


@pytest.mark.parametrize('field,limit', [('token_velocities', .01), ('carrier_velocities', .1), ('rack_velocities', .01)])
def test_ready_phase_with_moving_fixture_is_not_success(field, limit):
    observation, _ = FakeFixtureEnv().reset(seed=0)
    assert data_module._ready(observation)
    observation[field].flat[0] = limit
    assert not data_module._ready(observation)


def test_reset_option_boolean_cannot_impersonate_token_one(episode):
    episode.metadata['reset_options']['state'] = [True, 2, 3, 4]
    with pytest.raises(ValueError, match='Reset state'):
        episode.validate()


def test_same_board_during_motion_is_a_legitimate_future_goal(episode):
    batch = relabel_physics_transitions(episode, [2], [[2, 1, 3, 4]])
    assert batch['terminated'][0] and batch['bootstrap_masks'][0] == 0
    with pytest.raises(ValueError, match='already achieved'):
        relabel_physics_transitions(episode, [3], [[2, 1, 3, 4]])


def test_original_goal_ending_becomes_cutoff_for_other_goal():
    result = record_trajectory(FakeFixtureEnv(goal=[2, 1, 3, 4]), policy, seed=0)
    assert result.num_steps == 3
    batch = relabel_physics_transitions(result, [2], [[3, 1, 2, 4]])
    assert batch['collection_cutoff'][0] and batch['raw_terminated'][0]
    assert not batch['raw_collection_cutoff'][0] and not batch['terminated'][0]
    assert batch['bootstrap_masks'][0] == 1


def test_sampler_uses_stable_future_states_and_keeps_return_steps(episode):
    first, second = PhysicsGoalSampler(episode, 9), PhysicsGoalSampler(episode, 9)
    for _ in range(2):
        actual, expected = first.sample(100), second.sample(100)
        for field in ('step_indices', 'goal_observation_indices', 'actions', 'goals', 'rewards'):
            np.testing.assert_array_equal(actual[field], expected[field])
        assert set(actual['goal_observation_indices']).issubset({3, 6})
        assert np.all(actual['goal_observation_indices'] > actual['step_indices'])
        assert 2 in actual['step_indices'] and 5 in actual['step_indices']
        assert np.any(actual['observations']['board_valid'] == 0)
    snapshot = PhysicsGoalSampler(episode, 4)
    expected_sampler = PhysicsGoalSampler(deepcopy(episode), 4)
    episode.observations['board'][:] = 9
    episode.actions[:] = 1
    np.testing.assert_array_equal(snapshot.sample(50)['goals'], expected_sampler.sample(50)['goals'])


def test_no_stable_future_goal_is_rejected():
    result = record_trajectory(FakeFixtureEnv(), policy, seed=0, max_steps=2)
    with pytest.raises(ValueError, match='stable future'):
        PhysicsGoalSampler(result)


@pytest.mark.parametrize('indices,goals', [
    ([], []), ([True], [[3, 1, 2, 4]]), ([6], [[3, 1, 2, 4]]),
    ([1.0], [[3, 1, 2, 4]]), ([1], [[1, 1, 3, 4]]),
    ([1], [[1., 2., 3., 4.]]), ([1], [[True, False, True, False]]),
])
def test_invalid_relabel_inputs(episode, indices, goals):
    with pytest.raises(ValueError):
        relabel_physics_transitions(episode, indices, goals)


def test_save_load_hash_nooverwrite_and_exact_roundtrip(episode, tmp_path):
    path = episode.save(tmp_path / 'episode.npz')
    same_episode(episode, load_physics_trajectory(path))
    manifest = json.loads(path.with_suffix('.json').read_text())
    assert manifest['schema_version'] == PHYSICS_TRAJECTORY_SCHEMA
    assert manifest['artifact']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        episode.save(path)
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checksum'):
        load_physics_trajectory(path)


@pytest.mark.parametrize('change', [
    lambda m: m.update(extra=1),
    lambda m: m.__setitem__('schema_version', 'copuzzle.physics_trajectory.v1'),
    lambda m: m.__setitem__('num_steps', 7),
    lambda m: m['artifact'].__setitem__('filename', 'other.npz'),
    lambda m: m['arrays']['actions'].__setitem__('dtype', 'float64'),
    lambda m: m['arrays']['actions'].__setitem__('shape', [6.0, 3]),
    lambda m: m['metadata'].__setitem__('source_sha256', 'b' * 64),
])
def test_corrupt_manifest_rejected(episode, tmp_path, change):
    path = episode.save(tmp_path / 'episode.npz')
    sidecar = path.with_suffix('.json')
    manifest = json.loads(sidecar.read_text())
    change(manifest)
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_physics_trajectory(path)


def test_duplicate_json_key_rejected(episode, tmp_path):
    path = episode.save(tmp_path / 'episode.npz')
    sidecar = path.with_suffix('.json')
    text = sidecar.read_text()
    sidecar.write_text(text.replace('"num_steps": 6', '"num_steps": 6, "num_steps": 6'))
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        load_physics_trajectory(path)


def test_save_rollback_does_not_remove_unrelated_file(episode, tmp_path, monkeypatch):
    original_link = os.link
    calls = 0

    def failing_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            Path(target).write_text('other writer')
            raise FileExistsError('concurrent manifest')
        return original_link(source, target)

    monkeypatch.setattr(data_module.os, 'link', failing_link)
    with pytest.raises(FileExistsError):
        episode.save(tmp_path / 'episode.npz')
    assert not (tmp_path / 'episode.npz').exists()
    assert (tmp_path / 'episode.json').read_text() == 'other writer'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['episode.json']


def test_load_sample_use_no_mujoco_or_executable_rule(episode, tmp_path, monkeypatch):
    path = episode.save(tmp_path / 'episode.npz')
    from copuzzle.games.permutation.rules import PermutationRule
    from copuzzle.games.permutation.oracle import ExactOracle

    def forbidden(*args, **kwargs):
        raise AssertionError('Loader/sampler executed forbidden dynamics or search')

    monkeypatch.setattr(PermutationRule, 'transition', forbidden)
    monkeypatch.setattr(ExactOracle, '__init__', forbidden)
    loaded = load_physics_trajectory(path)
    assert PhysicsGoalSampler(loaded).sample(4)['actions'].shape == (4, 3)
    program = '''
import builtins, sys
real_import = builtins.__import__
def checked(name, *args, **kwargs):
    if name == 'mujoco' or name.startswith('mujoco.'):
        raise AssertionError('mujoco imported by data loader')
    return real_import(name, *args, **kwargs)
builtins.__import__ = checked
from copuzzle.games.permutation.physics.data import load_physics_trajectory, PhysicsGoalSampler
episode = load_physics_trajectory(sys.argv[1])
assert PhysicsGoalSampler(episode).sample(8)['actions'].shape == (8, 3)
assert 'mujoco' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', program, str(path)], check=True, capture_output=True, text=True)


def test_replay_uses_saved_actions_not_policy(episode):
    env = FakeFixtureEnv()
    report = replay_trajectory(env, episode)
    assert report['control_steps'] == 6 and report['operation_events'] == 2
    assert report['max_absolute_observation_error'] == 0
    np.testing.assert_array_equal(env.submitted_actions, episode.actions)


def test_replay_rejects_configuration_and_observation_drift(episode):
    with pytest.raises(ValueError, match='configuration'):
        replay_trajectory(FakeFixtureEnv(horizon=7), episode)

    class DriftingEnv(FakeFixtureEnv):
        def step(self, action):
            result = super().step(action)
            result[0]['joint_positions'][0] += .01
            return result

    with pytest.raises(ValueError, match='joint_positions'):
        replay_trajectory(DriftingEnv(), episode)


def test_real_topspin_episode_save_load_and_continuous_replay(tmp_path):
    pytest.importorskip('mujoco')
    from copuzzle.games.permutation.physics.env import PhysicalPermutationEnv
    from copuzzle.games.permutation.physics.scripted import FixtureSequencePolicy

    env = PhysicalPermutationEnv('topspin-8-4', max_episode_steps=500)
    try:
        start = np.arange(1, 9, dtype=np.int32)
        goal = np.roll(start, -1)
        trajectory = record_trajectory(
            env, FixtureSequencePolicy([0]), seed=3, options={'state': start, 'goal': goal},
            metadata={'purpose': 'single-operation-contact-replay-smoke'})
        assert trajectory.terminated[-1]
        assert trajectory.event_operations.tolist() == [0]
        assert np.any(trajectory.observations['board_valid'] == 0)
        assert np.any(trajectory.observations['mechanism_phase'] == 2)
        path = trajectory.save(tmp_path / 'physical-topspin.npz')
        loaded = load_physics_trajectory(path)
        report = replay_trajectory(env, loaded)
        assert report['max_absolute_observation_error'] < 1e-8
        assert PhysicsGoalSampler(loaded).sample(16)['goals'].shape == (16, 8)
    finally:
        env.close()
