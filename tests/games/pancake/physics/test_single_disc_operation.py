"""Physical flip-one evidence without adding an unsigned puzzle operation."""
from copy import deepcopy
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('mujoco')
from gymnasium.error import ResetNeeded
from copuzzle.games.permutation.rules import get_rule
from copuzzle.games.pancake.physics.env import PancakeManipulationEnv
from copuzzle.games.pancake.physics.scripted import PancakeFlipPolicy
from copuzzle.games.pancake.physics.data import (
    SCHEMA, OPERATION_SCHEMA, record_trajectory, load_trajectory, replay_trajectory,
)


@pytest.fixture(scope='module')
def operation_episode():
    env = PancakeManipulationEnv(task_mode='operation', operation_prefix=1)
    try:
        return record_trajectory(env, PancakeFlipPolicy(1), seed=3,
                                 options={'state': [1, 2, 3, 4]})
    finally:
        env.close()


def test_flip_one_does_not_expand_unsigned_rule_or_weaken_solved_reset():
    rule = get_rule('pancake-4')
    assert rule.n_actions == 3
    np.testing.assert_array_equal(rule.transition([1, 2, 3, 4], 0), [2, 1, 3, 4])
    env = PancakeManipulationEnv()
    try:
        obs, info = env.reset(options={'state': [1, 2, 3, 4], 'goal': [1, 2, 3, 4]})
        assert info['is_success']
        assert not any(name.startswith('operation_') for name in obs)
        with pytest.raises(ResetNeeded):
            env.step(np.zeros(7, dtype=np.float32))
    finally:
        env.close()


def test_operation_reset_can_equal_goal_but_noop_does_not_execute_flip():
    env = PancakeManipulationEnv(task_mode='operation', operation_prefix=1, max_episode_steps=8)
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        np.testing.assert_array_equal(obs['board'], obs['goal'])
        assert not info['is_success'] and not info['execution_success']
        for i in range(8):
            obs, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
            assert not terminated and reward == -1
            assert obs['operation_stage'][0] == 0
        assert truncated and not info['execution_success']
    finally:
        env.close()


@pytest.mark.parametrize('state', list(permutations(range(1, 5))))
def test_single_disc_grasp_flip_replace_for_all_24_initial_orders(state):
    env = PancakeManipulationEnv(task_mode='operation', operation_prefix=1)
    try:
        episode = record_trajectory(env, PancakeFlipPolicy(1), options={'state': list(state)})
        assert episode.schema_version == OPERATION_SCHEMA and episode.terminated[-1]
        obs, selected = episode.observations, state[0] - 1
        np.testing.assert_array_equal(obs['board'][-1], state)
        assert not episode.infos[-1]['permutation_changed']
        assert episode.infos[-1]['execution_success']
        assert obs['operation_stage'][0, 0] == 0 and obs['operation_stage'][-1, 0] == 3
        assert obs['operation_peak_lift'][-1, selected] > .12
        assert obs['operation_minimum_normal_dot'][-1, selected] < -.99
        assert np.max(np.delete(obs['operation_peak_lift'][-1], selected)) < .001
        assert np.max(np.delete(obs['operation_peak_displacement'][-1], selected)) < .003
        assert not np.any(obs['operation_violation'])
        assert env.model.neq == 0 and env.model.nmocap == 0
        assert not np.any(env.data.qfrc_applied) and not np.any(env.data.xfrc_applied)
    finally:
        env.close()


@pytest.mark.parametrize('prefix', [2, 3, 4])
def test_execution_mode_supports_other_prefix_counts(prefix):
    env = PancakeManipulationEnv(task_mode='operation', operation_prefix=prefix)
    try:
        episode = record_trajectory(env, PancakeFlipPolicy(prefix))
        assert episode.terminated[-1] and episode.infos[-1]['execution_success']
        assert episode.observations['operation_stage'][-1, 0] == 3
    finally:
        env.close()


def test_lift_without_inversion_remains_incomplete():
    class LiftOnly:
        def reset(self):
            self.policy, self.completed, self.phase = PancakeFlipPolicy(1), False, 'initial'

        def __call__(self, obs):
            action = self.policy(obs)
            self.phase = self.policy.phase
            if self.phase == 'flip':
                self.completed = True
            return action

    env = PancakeManipulationEnv(task_mode='operation', operation_prefix=1)
    try:
        episode = record_trajectory(env, LiftOnly())
        assert episode.collection_cutoff[-1] and not episode.terminated.any()
        assert episode.observations['operation_stage'][-1, 0] == 1
        assert episode.observations['operation_peak_lift'][-1, 0] > .12
    finally:
        env.close()


def test_lifting_extra_discs_is_a_diagnostic_violation():
    env = PancakeManipulationEnv(task_mode='operation', operation_prefix=1)
    try:
        episode = record_trajectory(env, PancakeFlipPolicy(2), max_steps=450)
        assert episode.observations['operation_violation'][-1, 0]
        assert episode.observations['operation_peak_lift'][-1, 1] > .10
        assert not episode.terminated.any()
    finally:
        env.close()


def test_execution_evidence_is_independent_of_collector_phase_labels(operation_episode):
    altered = deepcopy(operation_episode)
    altered.phases = ['untrusted collector text'] * altered.num_steps
    altered.validate()
    assert altered.terminated[-1]


@pytest.mark.parametrize('mutate', [
    lambda e: e.observations['operation_stage'].__setitem__((0, 0), 3),
    lambda e: e.observations['operation_selected'].__setitem__((0, 1), 1),
    lambda e: e.observations['operation_peak_lift'].__setitem__((10, 0), .5),
    lambda e: e.observations['operation_initial_positions'].__setitem__((20, 0, 0), .8),
    lambda e: e.observations['operation_minimum_normal_dot'].__setitem__((40, 0), -1.),
    lambda e: e.observations['operation_violation'].__setitem__((-1, 0), 1),
    lambda e: e.observations['tool_contacts'].fill(0),
    lambda e: e.observations['tool_contact_forces'].fill(0),
    lambda e: e.observations['token_orientations'].__setitem__(slice(None), e.observations['token_orientations'][0]),
    lambda e: e.infos[0].__setitem__('execution_success', True),
    lambda e: e.infos[-1].__setitem__('permutation_changed', True),
])
def test_reject_forged_operation_progress(operation_episode, mutate):
    altered = deepcopy(operation_episode)
    mutate(altered)
    with pytest.raises(ValueError):
        altered.validate()


def test_operation_save_load_replay_and_schema_separation(operation_episode, tmp_path):
    path = operation_episode.save(tmp_path / 'single-disc.npz')
    loaded = load_trajectory(path)
    assert loaded.schema_version == OPERATION_SCHEMA
    env = PancakeManipulationEnv(task_mode='operation', operation_prefix=1)
    try:
        for _ in range(2):
            assert replay_trajectory(env, loaded)['max_absolute_observation_error'] == 0
    finally:
        env.close()
    manifest_path = path.with_suffix('.json')
    manifest = json.loads(manifest_path.read_text())
    manifest['schema_version'] = SCHEMA
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='schema and task mode'):
        load_trajectory(path)


def test_operation_loader_remains_numpy_only(operation_episode, tmp_path):
    path = operation_episode.save(tmp_path / 'operation.npz')
    program = '''
import builtins,sys
original=builtins.__import__
def checked(name,*args,**kwargs):
    if name=='mujoco' or name.startswith('mujoco.'):
        raise AssertionError('operation loader imported physics')
    return original(name,*args,**kwargs)
builtins.__import__=checked
from copuzzle.games.pancake.physics.data import load_trajectory,OPERATION_SCHEMA
episode=load_trajectory(sys.argv[1]);assert episode.schema_version==OPERATION_SCHEMA
assert episode.infos[-1]['execution_success']
'''
    subprocess.run([sys.executable, '-c', program, str(path)], check=True, capture_output=True)


@pytest.mark.parametrize('name', ['1234-flip2', '1234-flip3', '1234-flip4', 'sequence-2-3-4'])
def test_archived_goal_episodes_still_replay_without_configuration_changes(name):
    path = Path(__file__).resolve().parents[4] / 'artifacts' / 'pancake-contact-v1' / 'episodes' / f'{name}.npz'
    if not path.is_file():
        pytest.skip('Archived local development artifact is not distributed with the package')
    episode = load_trajectory(path)
    env = PancakeManipulationEnv(max_episode_steps=episode.configuration['max_episode_steps'])
    try:
        assert episode.schema_version == SCHEMA
        assert replay_trajectory(env, episode)['max_absolute_observation_error'] == 0
    finally:
        env.close()


@pytest.mark.parametrize('kwargs', [
    {'task_mode': 'unknown'}, {'task_mode': 'operation'},
    {'task_mode': 'operation', 'operation_prefix': True},
    {'task_mode': 'operation', 'operation_prefix': 0},
    {'task_mode': 'operation', 'operation_prefix': 5}, {'operation_prefix': 1},
])
def test_invalid_operation_configuration(kwargs):
    with pytest.raises(ValueError):
        PancakeManipulationEnv(**kwargs)


def test_operation_rejects_unrelated_target_permutation():
    env = PancakeManipulationEnv(task_mode='operation', operation_prefix=1)
    try:
        with pytest.raises(ValueError, match='unsigned prefix reversal'):
            env.reset(options={'state': [1, 2, 3, 4], 'goal': [2, 1, 3, 4]})
    finally:
        env.close()
