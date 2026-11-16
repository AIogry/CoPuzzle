"""Independent real-contact checks for the indexed UR5e puzzle fixtures.

These tests exercise dynamic token movement and ordered, measured completions.
They do not use rendering or set any live generalized coordinate during a step.
"""
import numpy as np
import pytest
from gymnasium.error import ResetNeeded

mujoco = pytest.importorskip('mujoco')

from copuzzle.games.permutation.physics.env import PhysicalPermutationEnv
from copuzzle.games.permutation.physics.scene import RACK_TRAVEL
from copuzzle.games.permutation.physics.scripted import FixtureSequencePolicy


@pytest.fixture(params=('topspin-8-4', 'pancake-4'))
def fixture_env(request):
    environment = PhysicalPermutationEnv(request.param, max_episode_steps=1600)
    yield environment
    environment.close()


def _arbitrary_state(env, seed=17):
    return np.random.default_rng(seed).permutation(np.arange(1, env.rule.n_tokens + 1, dtype=np.int32))


def _reset_off_path(env, state, operations):
    """Choose a reachable goal outside the scripted path, including its start."""
    states = {tuple(state)}
    expected = state.copy()
    for operation in operations:
        expected = env.rule.transition(expected, operation)
        states.add(tuple(expected))
    goal = next(state[relative - 1] for relative in env._oracle.reachable_states
                if tuple(state[relative - 1]) not in states)
    observation, _ = env.reset(seed=3, options={'state': state, 'goal': goal})
    return observation, expected


def _assert_event_consistency(env, before, observation, info):
    events = info['event_log']
    counts = np.bincount([event['operation'] for event in events], minlength=3)
    np.testing.assert_array_equal(observation['previous_event_counts'], counts)
    assert [event['substep'] for event in events] == sorted(event['substep'] for event in events)
    expected = before.copy()
    for event in events:
        assert 0 <= event['substep'] < env.substeps
        assert event['duration_seconds'] > 0
        expected = env.rule.transition(expected, event['operation'])
    np.testing.assert_array_equal(observation['board'], expected)
    assert env.observation_space.contains(observation)


@pytest.mark.parametrize('operation', (0, 1, 2))
def test_all_operations_need_contact_move_tokens_and_reach_measured_goal(fixture_env, operation):
    env = fixture_env
    state = _arbitrary_state(env, seed=17 + operation)
    goal = env.rule.transition(state, operation)
    observation, _ = env.reset(options={'state': state, 'goal': goal})
    initial_positions = observation['token_positions'].copy()
    policy = FixtureSequencePolicy([operation])
    ordered = []
    saw_contact = saw_invalid_intermediate = saw_moving_tokens = False
    for _ in range(450):
        before = observation['board'].copy()
        observation, reward, terminated, truncated, info = env.step(policy(observation))
        _assert_event_consistency(env, before, observation, info)
        ordered.extend(event['operation'] for event in info['event_log'])
        saw_contact |= operation in env._contact_racks()
        if observation['mechanism_phase'][0] == 1:
            saw_invalid_intermediate |= observation['board_valid'][0] == 0
            saw_moving_tokens |= np.linalg.norm(observation['token_positions'] - initial_positions) > .005
        if terminated or truncated:
            break
    assert saw_contact and saw_invalid_intermediate and saw_moving_tokens
    assert terminated and not truncated and reward == 0 and info['is_success']
    assert ordered == [operation]
    assert observation['mechanism_phase'].tolist() == [0]
    assert observation['board_valid'].tolist() == [1]
    np.testing.assert_array_equal(observation['board'], goal)
    # Completion must survive the latch's rebound, not merely enter phase zero.
    assert np.max(np.abs(observation['carrier_velocities'])) < .1
    assert np.max(np.abs(observation['rack_velocities'])) < .01
    assert np.max(np.abs(observation['token_velocities'])) < .01
    with pytest.raises(ResetNeeded):
        env.step(np.zeros(3, dtype=np.float32))


def test_ordered_sequences_and_inverse_operations(fixture_env):
    env = fixture_env
    operations = [0, 1, 2, 0, 2, 1] if env.rule.family == 'topspin' else [0, 0, 1, 2, 0]
    state = _arbitrary_state(env, seed=91)
    observation, expected = _reset_off_path(env, state, operations)
    policy = FixtureSequencePolicy(operations)
    ordered = []
    for _ in range(env.max_episode_steps - 1):
        before = observation['board'].copy()
        observation, _, terminated, truncated, info = env.step(policy(observation))
        _assert_event_consistency(env, before, observation, info)
        ordered.extend(event['operation'] for event in info['event_log'])
        assert not terminated and not truncated
        if policy.completed:
            break
    assert policy.completed
    assert ordered == operations and policy.confirmed == operations
    np.testing.assert_array_equal(observation['board'], expected)
    assert observation['board_valid'].tolist() == [1]


def test_no_python_generalized_state_edits_between_integrator_calls(fixture_env, monkeypatch):
    env = fixture_env
    state = _arbitrary_state(env)
    observation, _ = _reset_off_path(env, state, [2])
    expected_qpos, expected_qvel = env.data.qpos.copy(), env.data.qvel.copy()
    original_step = mujoco.mj_step
    calls = 0
    def audited_step(model, data, *args, **kwargs):
        nonlocal expected_qpos, expected_qvel, calls
        assert data is env.data
        np.testing.assert_array_equal(data.qpos, expected_qpos)
        np.testing.assert_array_equal(data.qvel, expected_qvel)
        original_step(model, data, *args, **kwargs)
        expected_qpos, expected_qvel = data.qpos.copy(), data.qvel.copy()
        calls += 1
    monkeypatch.setattr(mujoco, 'mj_step', audited_step)
    policy = FixtureSequencePolicy([2])
    for _ in range(450):
        observation, _, terminated, truncated, _ = env.step(policy(observation))
        assert not terminated and not truncated
        np.testing.assert_array_equal(env.data.qpos, expected_qpos)
        np.testing.assert_array_equal(env.data.qvel, expected_qvel)
        if policy.completed:
            break
    assert policy.completed and calls > 0


def test_zero_control_and_contact_disabled_attempts_emit_no_events(fixture_env):
    env = fixture_env
    state = _arbitrary_state(env)
    observation, _ = _reset_off_path(env, state, [0])
    for _ in range(30):
        observation, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        assert not terminated and not truncated
        assert not info['event_log'] and not observation['previous_event_counts'].any()
        np.testing.assert_array_equal(observation['board'], state)
    env.model.geom_contype[env.scene.tool_geom_ids] = 0
    env.model.geom_conaffinity[env.scene.tool_geom_ids] = 0
    policy = FixtureSequencePolicy([0])
    for _ in range(220):
        observation, _, terminated, truncated, info = env.step(policy(observation))
        assert not terminated and not truncated
        assert not info['event_log'] and not observation['previous_event_counts'].any()
        assert not env._contact_racks()
        assert observation['mechanism_phase'].tolist() == [0]
        np.testing.assert_array_equal(observation['board'], state)
    assert not policy.completed


def test_reset_restores_latches_limits_and_arbitrary_token_assignment(fixture_env):
    env = fixture_env
    state = _arbitrary_state(env)
    observation, _ = _reset_off_path(env, state, [2])
    policy = FixtureSequencePolicy([2])
    for _ in range(200):
        observation, _, _, _, _ = env.step(policy(observation))
        if observation['mechanism_phase'][0] == 1:
            break
    assert observation['mechanism_phase'][0] == 1
    assert env.data.eq_active[env.fixture['carry']].any()
    replacement = _arbitrary_state(env, seed=101)
    observation, info = env.reset(seed=8, options={'state': replacement, 'goal': env.rule.transition(replacement, 0)})
    assert env.observation_space.contains(observation)
    np.testing.assert_array_equal(observation['board'], replacement)
    assert observation['mechanism_phase'].tolist() == [0]
    assert observation['active_operation'].tolist() == [-1]
    assert observation['board_valid'].tolist() == [1]
    assert not info['event_log'] and not info['attempt_log'] and info['elapsed_time'] == 0
    assert env.data.eq_active[env.fixture['dock']].all()
    assert not env.data.eq_active[env.fixture['carry']].any()
    assert not env.data.eq_active[env.fixture['gear']].any()
    assert env.data.eq_active[env.fixture['carrier_lock']].all()
    assert not env.data.eq_active[env.fixture['rack_lock']].any()
    np.testing.assert_array_equal(env.model.jnt_range[env.scene.button_joint_ids, 1], np.full(3, RACK_TRAVEL))
    for slot, token in enumerate(replacement):
        np.testing.assert_allclose(observation['token_positions'][token - 1], env.fixture['slots'][slot], atol=1e-12)
    # Mutating returned arrays or copied metadata must never alter the mechanism.
    observation['board'][:] = 99
    observation['token_positions'][:] = 99
    configuration = env.configuration()
    configuration['rule_spec']['name'] = 'spoofed-intent'
    assert env.configuration()['rule_spec']['name'] == env.rule.name
    np.testing.assert_array_equal(env._observation()['board'], replacement)


def test_partial_stroke_release_returns_without_operation(fixture_env):
    env = fixture_env
    state = _arbitrary_state(env)
    observation, _ = _reset_off_path(env, state, [0])
    policy = FixtureSequencePolicy([0])
    for _ in range(200):
        observation, _, terminated, truncated, info = env.step(policy(observation))
        assert not terminated and not truncated
        assert not info['event_log']
        if observation['mechanism_phase'][0] == 1:
            break
    assert observation['mechanism_phase'][0] == 1
    attempts = []
    target = observation['rack_rest_positions'][0] + [0, 0, .12]
    for _ in range(300):
        action = np.clip((target - observation['end_effector_position']) / env.action_scale, -1, 1).astype(np.float32)
        observation, _, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated
        assert not info['event_log']
        attempts.extend(info['attempt_log'])
        if observation['mechanism_phase'][0] == 0:
            break
    assert observation['mechanism_phase'].tolist() == [0]
    assert attempts and attempts[-1]['result'] == 'aborted_and_returned'
    np.testing.assert_array_equal(observation['board'], state)
