"""Actual contact play, stopping evidence, and exact action replay."""
import numpy as np
import pytest

from copuzzle.manipulation.play.collector import collect_play, forward, replay_play
from copuzzle.manipulation.play.semantics import frame, safe_boundary


TASKS = ('grid-3x3', 'pancake-4', 'topspin-20-4')


class ZeroPolicy:
    completed = False

    def __init__(self, task_id):
        self.width = 7 if task_id == 'pancake-4' else 3

    def __call__(self, observation):
        result = np.zeros(self.width, dtype=np.float32)
        if self.width == 7:
            result[-1] = observation['jaw_target'][0]
        return result


@pytest.mark.parametrize('task_id,operations', [
    ('grid-3x3', [0, 1]), ('pancake-4', [2, 3]),
    ('topspin-20-4', ['reverse', 'shift_right']),
])
def test_two_real_operations_keep_one_physical_trajectory_and_replay(task_id, operations, tmp_path, monkeypatch):
    pytest.importorskip('mujoco')
    from copuzzle.manipulation.play.bridge import PlayBridge
    from copuzzle.manipulation.play.protocol import load_play_trace
    resets = []
    original_reset = PlayBridge.reset

    def counted_reset(self, **kwargs):
        resets.append(self)
        return original_reset(self, **kwargs)

    monkeypatch.setattr(PlayBridge, 'reset', counted_reset)
    trace = collect_play(task_id, seed=17, num_segments=2, operations=operations, max_episode_steps=8000)
    assert len(resets) == 1
    assert trace.metadata['stop_reason'] == 'requested_segments_complete'
    assert len(trace.segments) == 2
    assert all(segment['outcome'] == 'completed' for segment in trace.segments)
    assert all(segment['controller_completed'] and segment['switch_safe'] for segment in trace.segments)
    assert trace.collection_cutoff[-1] and not np.any(trace.timeouts)
    assert 'goal' not in trace.observations and 'success_streak' not in trace.observations
    count = len(trace.actions)
    assert trace.observations['board'].shape[0] == count + 1
    assert len(trace.infos) == count + 1
    assert np.array_equal(trace.observations['steps_remaining'][:, 0], 8000 - np.arange(count + 1))
    for index, segment in enumerate(trace.segments):
        start, end = segment['start_step'], segment['end_step']
        assert start < end
        assert np.all(trace.segment_ids[start:end] == index)
        assert np.all(trace.behavior_goals[start:end] == segment['goal'])
        assert np.array_equal(forward(task_id, trace.observations['board'][start], operations[index]), segment['goal'])
        assert np.array_equal(trace.observations['board'][end], segment['goal'])
        assert safe_boundary(task_id, frame(trace.observations, end), trace.configuration)
        assert trace.goal_reached[end - 1]
    boundary = trace.segments[0]['end_step']
    assert boundary == trace.segments[1]['start_step']
    assert not np.array_equal(trace.behavior_goals[boundary - 1], trace.behavior_goals[boundary])
    if task_id == 'grid-3x3':
        # Successful pressing must be followed by real release/re-arm actions.
        assert trace.segments[0]['goal_first_step'] + 1 < boundary
        assert np.all(trace.observations['button_armed'][boundary])
    path = trace.save(tmp_path / f'{task_id}.npz')
    loaded = load_play_trace(path)
    report = replay_play(loaded)
    assert len(resets) == 2  # Exactly one further reset, for independent replay.
    assert report['max_absolute_observation_error'] == 0
    assert report['actions_replayed'] and not report['policy_regenerated']


@pytest.mark.parametrize('task_id', TASKS)
def test_collection_cap_and_global_timeout_remain_distinct(task_id):
    pytest.importorskip('mujoco')
    factory = lambda task, operation: ZeroPolicy(task)
    cutoff = collect_play(task_id, num_segments=1, max_episode_steps=10, max_steps=2, policy_factory=factory)
    timeout = collect_play(task_id, num_segments=1, max_episode_steps=2, max_steps=10, policy_factory=factory)
    assert len(cutoff.actions) == len(timeout.actions) == 2
    assert cutoff.collection_cutoff.tolist() == [False, True]
    assert not np.any(cutoff.timeouts)
    assert timeout.timeouts.tolist() == [False, True]
    assert not np.any(timeout.collection_cutoff)
    assert cutoff.metadata['stop_reason'] == 'collection_cutoff'
    assert timeout.metadata['stop_reason'] == 'timeout'
    assert replay_play(cutoff)['max_absolute_observation_error'] == 0
    assert replay_play(timeout)['max_absolute_observation_error'] == 0


def test_controller_exception_keeps_only_executed_transitions():
    pytest.importorskip('mujoco')
    class BrokenPolicy(ZeroPolicy):
        def __init__(self):
            super().__init__('grid-3x3')
            self.calls = 0

        def __call__(self, observation):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError('deliberate controller abort')
            return super().__call__(observation)

    trace = collect_play('grid-3x3', num_segments=1, policy_factory=lambda task, op: BrokenPolicy())
    assert len(trace.actions) == 2
    assert trace.observations['board'].shape[0] == 3
    assert trace.metadata['stop_reason'] == 'controller_error'
    assert trace.metadata['error']['message'] == 'deliberate controller abort'
    assert trace.metadata['failed_action'] is None
    assert trace.segments[0]['outcome'] == 'controller_error'
    assert trace.collection_cutoff[-1] and not np.any(trace.goal_reached)
    assert replay_play(trace)['max_absolute_observation_error'] == 0


def test_final_action_is_executed_but_completed_is_not_goal_success():
    pytest.importorskip('mujoco')
    class EarlyComplete(ZeroPolicy):
        def __init__(self):
            super().__init__('grid-3x3')
            self.calls = 0

        def __call__(self, observation):
            self.calls += 1
            self.completed = self.calls == 2
            return super().__call__(observation)

    trace = collect_play('grid-3x3', num_segments=1, max_steps=5,
                         policy_factory=lambda task, op: EarlyComplete())
    assert len(trace.actions) == 2
    assert not np.any(trace.goal_reached)
    assert trace.metadata['stop_reason'] == 'mismatch'
    assert trace.segments[0]['controller_completed']
    assert trace.segments[0]['outcome'] == 'mismatch'


def test_first_action_controller_error_does_not_invent_a_trajectory():
    pytest.importorskip('mujoco')
    def factory(task, operation):
        def broken(observation):
            raise ValueError('nothing executed')
        return broken
    with pytest.raises(RuntimeError, match='without a complete transition'):
        collect_play('grid-3x3', policy_factory=factory)


def test_failed_native_action_has_no_fabricated_successor(monkeypatch):
    pytest.importorskip('mujoco')
    from copuzzle.manipulation.play.bridge import PlayBridge
    original = PlayBridge.step
    attempts = []

    def fail_third(self, action):
        attempts.append(action.copy())
        if len(attempts) == 3:
            raise RuntimeError('injected native failure')
        return original(self, action)

    monkeypatch.setattr(PlayBridge, 'step', fail_third)
    trace = collect_play('grid-3x3', num_segments=1,
                         policy_factory=lambda task, op: ZeroPolicy(task))
    assert len(attempts) == 3 and len(trace.actions) == 2
    assert len(trace.observations['board']) == 3 and len(trace.infos) == 3
    assert trace.metadata['stop_reason'] == 'native_error'
    assert trace.metadata['error']['stage'] == 'native_step'
    assert trace.metadata['failed_action'] == attempts[-1].tolist()
    assert trace.collection_cutoff[-1]
    monkeypatch.setattr(PlayBridge, 'step', original)
    assert replay_play(trace)['max_absolute_observation_error'] == 0


def test_next_factory_failure_retains_completed_segment_without_empty_segment():
    pytest.importorskip('mujoco')
    from copuzzle.games.lights_out.physics.scripted import ButtonSequencePolicy
    calls = []

    def factory(task, operation):
        calls.append(operation)
        if len(calls) == 2:
            raise RuntimeError('second controller unavailable')
        return ButtonSequencePolicy([operation])

    trace = collect_play('grid-3x3', num_segments=2, operations=[0, 1], policy_factory=factory)
    assert calls == [0, 1]
    assert len(trace.segments) == 1 and trace.segments[0]['outcome'] == 'completed'
    assert trace.segments[0]['end_step'] == len(trace.actions)
    assert trace.metadata['stop_reason'] == 'controller_error'
    assert trace.metadata['error']['stage'] == 'policy_factory'
    assert trace.collection_cutoff[-1]
    assert replay_play(trace)['max_absolute_observation_error'] == 0


def test_source_change_during_collection_is_a_provenance_error(monkeypatch):
    pytest.importorskip('mujoco')
    from copuzzle.manipulation.play import collector
    original = collector._sources
    snapshots = []

    def changing_sources():
        sources = original()
        snapshots.append(sources)
        if len(snapshots) > 1:
            name = next(iter(sources))
            sources[name] = '0' * 64 if sources[name] != '0' * 64 else '1' * 64
        return sources

    monkeypatch.setattr(collector, '_sources', changing_sources)
    with pytest.raises(RuntimeError, match='source files changed during collection'):
        collect_play('grid-3x3', num_segments=1, max_steps=2,
                     policy_factory=lambda task, operation: ZeroPolicy(task))
    assert len(snapshots) == 2


@pytest.mark.parametrize('kwargs', [
    {'num_segments': True}, {'max_episode_steps': 0}, {'max_steps': False},
    {'num_segments': 2, 'operations': [0]}, {'num_segments': 1, 'operations': [True]},
])
def test_invalid_collection_requests_fail_before_physics(kwargs):
    with pytest.raises((ValueError, TypeError)):
        collect_play('grid-3x3', **kwargs)


def test_pancake_identity_operation_is_not_play_goal_data():
    with pytest.raises(ValueError, match='Invalid requested'):
        collect_play('pancake-4', num_segments=1, operations=[1])
