"""Native contact grasping, physical release, and portable action replay."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from gymnasium.error import ResetNeeded
from copuzzle.games.pancake.physics.env import PancakeManipulationEnv
from copuzzle.games.pancake.physics.scripted import PancakeFlipPolicy
from copuzzle.games.pancake.physics.scripted import PancakeSequencePolicy
from copuzzle.games.pancake.physics.scene import STACK_XY, WITHDRAW_DISTANCE
from copuzzle.games.pancake.physics.data import record_trajectory, load_trajectory, replay_trajectory


def test_gravity_supported_real_disc_geometry_and_controller_observation():
    env = PancakeManipulationEnv()
    try:
        obs, info = env.reset(seed=4)
        assert env.observation_space.contains(obs)
        assert env.model.neq == 0 and env.model.nmocap == 0
        radii = [env.model.geom(f'pancake_disc_{i}').size[0] for i in range(4)]
        assert radii == sorted(set(radii))
        assert np.all(env.model.body_mass[env.scene.token_body_ids] > 0)
        assert np.all(env.model.body_gravcomp[env.scene.token_body_ids] == 0)
        original = obs['token_positions'].copy()
        for _ in range(80):
            obs, _, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
        assert obs['board_valid'][0]
        np.testing.assert_allclose(obs['token_positions'], original, atol=1e-5)
        np.testing.assert_array_equal(obs['arm_joint_targets'], env.controller._last_command)
        assert not np.any(env.data.qfrc_applied) and not np.any(env.data.xfrc_applied)
    finally:
        env.close()


@pytest.mark.parametrize('prefix', [2, 3, 4])
def test_contact_grasp_lifts_only_selected_prefix(prefix):
    env = PancakeManipulationEnv()
    try:
        obs, _ = env.reset(seed=0)
        initial = obs['token_positions'].copy()
        policy = PancakeFlipPolicy(prefix)
        for _ in range(650):
            action = policy(obs)
            if policy.phase == 'flip':
                break
            obs, _, _, _, _ = env.step(action)
        assert policy.phase == 'flip'
        assert np.all(obs['token_positions'][:prefix, 2] - initial[:prefix, 2] > .12)
        if prefix < 4:
            np.testing.assert_allclose(obs['token_positions'][prefix:], initial[prefix:], atol=4e-4)
        assert np.count_nonzero(obs['tool_contacts']) >= 2
        assert obs['tool_contact_forces'].sum() > 1
        assert not obs['board_valid'][0]
    finally:
        env.close()


@pytest.fixture(scope='module')
def completed_episode():
    env = PancakeManipulationEnv()
    try:
        return record_trajectory(env, PancakeFlipPolicy(2), seed=5,
                                 options={'state': [1, 2, 3, 4], 'goal': [2, 1, 3, 4]})
    finally:
        env.close()


@pytest.mark.parametrize('prefix', [2, 3, 4])
def test_full_real_flip_place_release_and_withdraw(prefix):
    env = PancakeManipulationEnv()
    try:
        goal = np.arange(1, 5, dtype=np.int32)
        goal[:prefix] = goal[:prefix][::-1]
        episode = record_trajectory(env, PancakeFlipPolicy(prefix), options={'goal': goal})
        assert episode.terminated[-1] and episode.num_steps < 1200
        obs = {k: v[-1] for k, v in episode.observations.items()}
        np.testing.assert_array_equal(obs['board'], goal)
        assert obs['board_valid'][0] and not np.any(obs['tool_contacts'])
        assert np.max(np.abs(obs['token_velocities'])) < .015
        assert np.linalg.norm(obs['end_effector_position'][:2] - STACK_XY) > WITHDRAW_DISTANCE
        assert {'insert', 'clamp', 'lift', 'flip', 'place', 'release', 'withdraw'} <= set(episode.phases)
        assert env.model.neq == 0
        with pytest.raises(ResetNeeded):
            env.step(np.zeros(7, dtype=np.float32))
    finally:
        env.close()


def test_same_environment_reset_replay_is_exact(completed_episode, tmp_path):
    path = completed_episode.save(tmp_path / 'contact.npz')
    loaded = load_trajectory(path)
    env = PancakeManipulationEnv()
    try:
        for _ in range(2):
            report = replay_trajectory(env, loaded)
            assert report['max_absolute_observation_error'] == 0
    finally:
        env.close()
    with pytest.raises(FileExistsError):
        completed_episode.save(path)
    path.write_bytes(path.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        load_trajectory(path)


def test_three_real_flips_chain_without_reset_and_replay_exactly():
    env = PancakeManipulationEnv(max_episode_steps=5000)
    try:
        episode = record_trajectory(env, PancakeSequencePolicy([2, 3, 4]),
                                    options={'goal': [4, 2, 1, 3]})
        assert episode.terminated[-1] and episode.num_steps < 4800
        np.testing.assert_array_equal(episode.observations['board'][-1], [4, 2, 1, 3])
        assert any('empty_lift' in phase for phase in episode.phases)
        assert any('empty_turn_down' in phase for phase in episode.phases)
        assert replay_trajectory(env, episode)['max_absolute_observation_error'] == 0
    finally:
        env.close()


def test_native_loader_does_not_import_mujoco(completed_episode, tmp_path):
    path = completed_episode.save(tmp_path / 'contact.npz')
    program = '''
import builtins,sys
original=builtins.__import__
def checked(name,*args,**kwargs):
    if name=='mujoco' or name.startswith('mujoco.'):
        raise AssertionError('Native data loader imported physics')
    return original(name,*args,**kwargs)
builtins.__import__=checked
from copuzzle.games.pancake.physics.data import load_trajectory
episode=load_trajectory(sys.argv[1]);assert episode.actions.shape[1]==7
assert 'mujoco' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', program, str(path)], check=True, capture_output=True)


@pytest.mark.parametrize('mutation', [
    lambda e: e.observations.pop('arm_joint_targets'),
    lambda e: e.observations['token_positions'].__setitem__((0, 0, 0), 99.),
    lambda e: e.observations['board'].__setitem__((0, 0), 2),
    lambda e: e.observations['jaw_target'].__setitem__((0, 0), 2.),
    lambda e: e.actions.__setitem__((0, 6), -1.),
    lambda e: e.rewards.__setitem__(0, 0.),
    lambda e: e.terminated.__setitem__(0, True),
    lambda e: e.metadata.__setitem__('configuration_sha256', 'a' * 64),
    lambda e: e.metadata.__setitem__('source_sha256', 'a' * 64),
])
def test_native_schema_rejects_corruption(completed_episode, mutation):
    candidate = deepcopy(completed_episode)
    mutation(candidate)
    with pytest.raises(ValueError):
        candidate.validate()


def test_partial_failed_controller_data_is_preserved():
    class FailedPolicy:
        def reset(self):
            self.calls = 0

        def __call__(self, obs):
            self.calls += 1
            if self.calls == 4:
                raise RuntimeError('deliberate test controller stop')
            return np.zeros(7, dtype=np.float32)

    env = PancakeManipulationEnv()
    try:
        result = record_trajectory(env, FailedPolicy(), on_policy_error='record')
        assert result.num_steps == 3 and result.collection_cutoff[-1]
        assert result.metadata['stop_reason'] == 'policy_error'
        assert 'deliberate' in result.metadata['policy_error']
    finally:
        env.close()


@pytest.mark.parametrize('action', [[0.] * 6, [0.] * 8, [0.] * 6 + [-.1], [np.nan] * 7,
                                    [2.] + [0.] * 6, [False] * 7])
def test_reject_invalid_actions(action):
    env = PancakeManipulationEnv()
    try:
        env.reset(seed=0)
        with pytest.raises(ValueError):
            env.step(action)
    finally:
        env.close()


def test_already_solved_reset_and_closed_environment():
    env = PancakeManipulationEnv()
    obs, info = env.reset(options={'state': [1, 2, 3, 4], 'goal': [1, 2, 3, 4]})
    assert info['is_success']
    with pytest.raises(ResetNeeded):
        env.step(np.zeros(7, dtype=np.float32))
    env.close()
    with pytest.raises(RuntimeError):
        env.reset()
    with pytest.raises(RuntimeError):
        env.step(np.zeros(7, dtype=np.float32))
