"""Continuous contact -> saved episode -> goal batch -> simulator replay."""
from __future__ import annotations

import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest

import copuzzle
from copuzzle.games.lights_out.presets import TASK_PRESETS
from copuzzle.games.lights_out.physics.scripted import ButtonSequencePolicy


def test_optional_imports_do_not_load_simulator():
    program = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('mujoco', 'dm_control', 'ogbench'):
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, Block())
import copuzzle
from copuzzle.games.lights_out.physics import PhysicsGoalSampler, load_physics_trajectory, ButtonSequencePolicy
env = copuzzle.make_env('grid-3x3')
env.reset(seed=0)
env.step(0)
assert 'mujoco' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("preset", TASK_PRESETS, ids=lambda preset: preset.task_id)
def test_scripted_episode_goal_sampling_and_replay(preset, tmp_path):
    pytest.importorskip("mujoco")
    from copuzzle.games.lights_out.physics import (
        PhysicsGoalSampler, load_physics_trajectory, record_episode, replay_trajectory,
    )
    env = copuzzle.make_physics_env(preset.task_id)
    try:
        state = np.zeros(env.rule.n_buttons, dtype=np.uint8)
        sequence = [0, env.rule.n_buttons - 1]
        goal = state.copy()
        for button in sequence:
            goal = env.rule.transition(goal, button)
        episode = record_episode(
            env, ButtonSequencePolicy(sequence, action_scale=env.action_scale), seed=9,
            options={"state": state, "goal": goal}, metadata={"controller": "scripted"})
        assert episode.terminated[-1]
        assert episode.event_buttons.tolist() == sequence
        for step, counts in enumerate(episode.event_counts):
            expected = env.rule.apply_event(episode.observations["board"][step], (counts % 2).astype(np.uint8))
            np.testing.assert_array_equal(expected, episode.observations["board"][step + 1])
        path = tmp_path / "episode.npz"
        episode.save(path)
        restored = load_physics_trajectory(path)
        report = replay_trajectory(env, restored)
        assert report["max_absolute_observation_error"] <= 1e-9
        assert report["press_events"] == len(sequence)
        batch = PhysicsGoalSampler(restored, seed=7).sample(32)
        assert np.all(np.any(batch["observations"]["board"] != batch["goals"], axis=1))
        assert batch["actions"].shape == (32, 3)
        np.testing.assert_array_equal(batch["terminated"], np.all(batch["next_observations"]["board"] == batch["goals"], axis=1))
        assert "joint_positions" in batch["observations"]
        with pytest.raises(ValueError, match="[Cc]onfiguration"):
            restored.metadata["configuration_sha256"] = "0" * 64
            replay_trajectory(env, restored)
    finally:
        env.close()


def test_physical_gym_id_has_separate_action_contract():
    pytest.importorskip("mujoco")
    env = gym.make("CoPuzzle/PhysicalGrid3x3-v0")
    try:
        observation, _ = env.reset(seed=0)
        assert env.action_space.shape == (3,)
        assert env.observation_space.contains(observation)
        env.step(np.zeros(3, dtype=np.float32))
    finally:
        env.close()


def test_scripted_executor_waits_for_observed_event():
    policy = ButtonSequencePolicy([0], phase_timeout=10)
    observation = dict(end_effector_position=np.array([0.5, 0., 0.15]),
                       button_rest_positions=np.array([[0.5, 0., 0.08]]),
                       previous_event_counts=np.array([0], dtype=np.int32),
                       button_armed=np.array([1], dtype=np.uint8))
    policy(observation)
    policy(observation)
    assert policy.phase == "press"
    assert policy(observation)[2] == -1
    assert not policy.confirmed_buttons
    observation["previous_event_counts"][0] = 1
    policy(observation)
    assert policy.confirmed_buttons == [0]
    assert policy.phase == "release"
