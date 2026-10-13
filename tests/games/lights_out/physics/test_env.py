"""Physical state, lifecycle and real-contact behavior of the UR5e environment."""
from copy import deepcopy
import hashlib
import json
import os

import numpy as np
import pytest
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env

pytest.importorskip('mujoco')

from copuzzle.games.lights_out.physics.env import PhysicalPuzzleEnv
from copuzzle.games.lights_out.rules import get_rule, grid_rule


@pytest.fixture(scope='module')
def env():
    environment = PhysicalPuzzleEnv(grid_rule(3, 3), max_episode_steps=500)
    yield environment
    environment.close()


def _unsolved_reset(env):
    start = np.zeros(env.rule.n_buttons, dtype=np.uint8)
    return env.reset(seed=7, options={'state': start, 'goal': env.rule.transition(start, env.rule.n_buttons - 1)})


def _drive(env, observation, target, steps):
    events = []
    for _ in range(steps):
        action = np.clip((target - observation['end_effector_position']) / env.action_scale, -1, 1).astype(np.float32)
        previous_board = observation['board'].copy()
        observation, _, terminated, truncated, info = env.step(action)
        counts = np.bincount([item['button'] for item in info['event_log']], minlength=env.rule.n_buttons)
        np.testing.assert_array_equal(counts, observation['previous_event_counts'])
        expected = env.rule.apply_event(previous_board, counts % 2)
        np.testing.assert_array_equal(observation['board'], expected)
        assert info['event_log'] == sorted(info['event_log'], key=lambda item: (item['substep'], item['button']))
        assert all(0 <= item['substep'] < env.substeps for item in info['event_log'])
        events.extend(info['event_log'])
        assert not terminated and not truncated
    return observation, events


def test_physical_gym_contract_and_state_fields(env):
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=9)
    assert env.observation_space.contains(observation)
    assert env.action_space.shape == (3,) and env.action_space.dtype == np.float32
    assert set(observation) == {
        'board', 'goal', 'grid_shape', 'grid_coordinates', 'joint_positions',
        'joint_velocities', 'end_effector_position', 'end_effector_orientation',
        'button_positions', 'button_rest_positions', 'button_displacements',
        'button_velocities', 'button_armed', 'previous_event_counts', 'steps_remaining',
    }
    assert set(info) == {'is_success', 'steps', 'elapsed_time', 'rule_id', 'event_log'}
    assert observation['previous_event_counts'].dtype == np.int32
    assert observation['steps_remaining'].tolist() == [env.max_episode_steps]
    assert observation['button_armed'].tolist() == [1] * 9
    assert info['event_log'] == [] and info['elapsed_time'] == 0.0
    np.testing.assert_allclose(np.linalg.norm(observation['end_effector_orientation']), 1)
    assert observation['end_effector_orientation'].shape == (4,)
    assert env._renderer is None and env.render() is None


def test_real_contact_hold_release_and_second_press(env):
    observation, _ = _unsolved_reset(env)
    start = observation['board'].copy()
    rest = observation['button_rest_positions'][0]
    high = rest + [0, 0, .10]
    down = rest - [0, 0, .010]
    observation, events = _drive(env, observation, high, 30)
    assert events == []  # A target above a button does not toggle it.
    np.testing.assert_array_equal(observation['board'], start)
    observation, events = _drive(env, observation, down, 40)
    assert len(events) == 1 and events[0]['button'] == 0
    assert env._contacts()[0]  # The live tool and button cap really touch.
    assert observation['button_displacements'][0] >= .008
    assert observation['button_armed'][0] == 0
    np.testing.assert_array_equal(observation['board'], env.rule.transition(start, 0))
    observation, held_events = _drive(env, observation, down, 25)
    assert held_events == []
    observation, released_events = _drive(env, observation, high, 35)
    assert released_events == [] and observation['button_displacements'][0] <= .003
    assert observation['button_armed'][0] == 1
    observation, second_events = _drive(env, observation, down, 40)
    assert len(second_events) == 1 and second_events[0]['button'] == 0
    np.testing.assert_array_equal(observation['board'], start)


def test_reset_clears_switch_memory_and_full_dynamics(env):
    original, info = env.reset(seed=31)
    commands = [np.array([.2, -.1, -.3], dtype=np.float32)] * 12
    first = [env.step(action) for action in commands]
    repeated, repeated_info = env.reset(seed=31)
    assert info == repeated_info
    for name in original:
        np.testing.assert_array_equal(original[name], repeated[name])
    second = [env.step(action) for action in commands]
    for left, right in zip(first, second):
        for name in left[0]:
            np.testing.assert_array_equal(left[0][name], right[0][name])
        assert left[1:] == right[1:]
    after, info = env.reset(seed=31)
    assert not after['previous_event_counts'].any()
    assert after['button_armed'].all()
    assert info['steps'] == 0 and info['event_log'] == [] and info['elapsed_time'] == 0


def test_idle_arm_stays_stable_without_false_events(env):
    observation, _ = _unsolved_reset(env)
    initial_tip = observation['end_effector_position'].copy()
    initial_board = observation['board'].copy()
    for _ in range(80):
        observation, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        assert info['event_log'] == []
        assert np.isfinite(observation['joint_positions']).all()
        assert np.isfinite(observation['joint_velocities']).all()
    np.testing.assert_allclose(observation['end_effector_position'], initial_tip, atol=1e-5)
    np.testing.assert_array_equal(observation['board'], initial_board)


def test_actions_reference_measured_pose_and_do_not_accumulate(env, monkeypatch):
    observation, _ = _unsolved_reset(env)
    targets = []
    solve = env._controller.solve
    def capture(data, target):
        targets.append(target.copy())
        return solve(data, target)
    monkeypatch.setattr(env._controller, 'solve', capture)
    action = np.array([.25, -.1, -.2], dtype=np.float32)
    for _ in range(4):
        measured = observation['end_effector_position'].copy()
        observation, *_ = env.step(action)
        np.testing.assert_allclose(targets[-1], measured + env.action_scale * action, atol=1e-10)


def test_returned_observations_and_configuration_are_independent(env):
    observation, _ = env.reset(seed=18)
    snapshot = {name: value.copy() for name, value in observation.items()}
    for value in observation.values():
        value.fill(0)
    unchanged = env._observation()
    for name, expected in snapshot.items():
        np.testing.assert_array_equal(unchanged[name], expected)
    configuration = env.configuration()
    canonical = json.dumps(configuration, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == env.configuration_sha256
    assert configuration['model_sha256'] == env.model_sha256
    assert configuration['goal_representation'] == 'binary_board_only'
    assert configuration['controller']['damping'] > 0
    configuration['scene'].clear()
    assert env.configuration()['scene']


def test_success_on_last_control_step_takes_precedence_and_requires_reset():
    rule = grid_rule(3, 3)
    with PhysicalPuzzleEnv(rule, max_episode_steps=200) as probe:
        start = np.zeros(9, dtype=np.uint8)
        ob, _ = probe.reset(seed=0, options={'state': start, 'goal': rule.transition(start, 0)})
        high = ob['button_rest_positions'][0] + [0, 0, .10]
        down = ob['button_rest_positions'][0] - [0, 0, .010]
        actions = []
        for step in range(150):
            target = high if step < 30 else down
            action = np.clip((target - ob['end_effector_position']) / probe.action_scale, -1, 1).astype(np.float32)
            actions.append(action)
            ob, reward, terminated, truncated, info = probe.step(action)
            if terminated:
                break
        assert terminated and reward == 0.0 and not truncated
    with PhysicalPuzzleEnv(rule, max_episode_steps=len(actions)) as final:
        final.reset(seed=0, options={'state': start, 'goal': rule.transition(start, 0)})
        for action in actions:
            ob, reward, terminated, truncated, info = final.step(action)
        assert terminated and not truncated and reward == 0.0 and info['is_success']
        assert ob['steps_remaining'].tolist() == [0]
        with pytest.raises(ResetNeeded):
            final.step(np.zeros(3))


def test_timeout_and_already_solved_reset():
    with PhysicalPuzzleEnv(grid_rule(2, 3), max_episode_steps=1) as environment:
        state = np.zeros(6, dtype=np.uint8)
        _, info = environment.reset(options={'state': state, 'goal': state})
        assert info['is_success']
        with pytest.raises(ResetNeeded):
            environment.step(np.zeros(3))
        environment.reset(seed=1)
        ob, reward, terminated, truncated, info = environment.step(np.zeros(3))
        assert reward == -1.0 and not terminated and truncated
        assert ob['steps_remaining'].tolist() == [0]
        with pytest.raises(ResetNeeded):
            environment.step(np.zeros(3))


def test_graph_structure_and_unreachable_goal():
    with PhysicalPuzzleEnv(get_rule('graph-cycle3')) as environment:
        observation, _ = environment.reset(seed=1)
        assert 'adjacency' in observation and 'grid_shape' not in observation
        assert environment.observation_space.contains(observation)
        with pytest.raises(ValueError, match='unreachable'):
            environment.reset(options={'state': [0, 0, 0], 'goal': [1, 0, 0]})


@pytest.mark.parametrize('action', [[0, 0], [np.nan, 0, 0], [np.inf, 0, 0],
                                    [1.01, 0, 0], [True, False, True], '0', None])
def test_invalid_action_does_not_advance_simulation(env, action):
    env.reset(seed=4)
    before = env.data.time
    with pytest.raises(ValueError):
        env.step(action)
    assert env.data.time == before


@pytest.mark.parametrize('kwargs', [{'physics_dt': 0}, {'control_dt': .003},
                                    {'action_scale': np.nan}, {'max_episode_steps': True},
                                    {'press_depth': .003, 'release_depth': .008},
                                    {'render_mode': 'human'}])
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        PhysicalPuzzleEnv(grid_rule(2, 3), **kwargs)


def test_rgb_render_is_lazy_and_does_not_advance_state():
    # Run in an explicitly selected headless backend; this test never changes it.
    if os.environ.get('MUJOCO_GL') not in {'egl', 'osmesa'}:
        pytest.skip('Set MUJOCO_GL=egl (or osmesa) before Python to test headless rendering')
    with PhysicalPuzzleEnv(grid_rule(2, 3), render_mode='rgb_array',
                           render_width=720, render_height=520) as environment:
        assert environment._renderer is None
        environment.reset(seed=2)
        before = deepcopy(environment._observation())
        when = environment.data.time
        frame = environment.render()
        assert frame.shape == (520, 720, 3) and frame.dtype == np.uint8
        assert frame.std() > 1
        after = environment._observation()
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        assert environment.data.time == when
        np.testing.assert_array_equal(frame, environment.render())
    environment.close()
