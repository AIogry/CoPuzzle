"""Continuous local-operation play collection and action-only replay.

The reference factory is privileged: it receives one requested operation. This
is not the observation-only evaluation interface or a global-goal expert.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from .bridge import PlayBridge, TASK_IDS, _integer
from .semantics import public_observation, safe_boundary


OPERATIONS = {'grid-3x3': tuple(range(9)), 'pancake-4': (2, 3, 4),
              'topspin-20-4': ('reverse', 'shift_left', 'shift_right')}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _identity(value):
    value = value if inspect.isfunction(value) else type(value)
    return f'{value.__module__}.{value.__qualname__}'


def _sources():
    root = Path(__file__).parents[2]
    return {str(path.relative_to(root.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*.py'))}


def forward(task_id, state, operation):
    """One privileged local operation, never a solver or shortest-path label."""
    if task_id not in TASK_IDS:
        raise ValueError('Unsupported play task')
    if type(operation) is not (str if task_id == 'topspin-20-4' else int) or operation not in OPERATIONS[task_id]:
        raise ValueError('Operation is not in this task\'s play action set')
    result = np.array(state, copy=True)
    if task_id == 'grid-3x3':
        row, column = divmod(operation, 3)
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, column + dc
            if 0 <= r < 3 and 0 <= c < 3:
                result[3 * r + c] ^= 1
    elif task_id == 'pancake-4':
        result[:operation] = result[:operation][::-1]
    elif operation == 'reverse':
        result[:4] = result[:4][::-1]
    else:
        result = np.roll(result, -1 if operation == 'shift_left' else 1)
    return result


def _policy(task_id, operation, bridge, factory):
    if factory is not None:
        controller = factory(task_id, operation)
    elif task_id == 'grid-3x3':
        from copuzzle.games.lights_out.physics.scripted import ButtonSequencePolicy
        controller = ButtonSequencePolicy([operation], action_scale=bridge.env.action_scale)
    elif task_id == 'pancake-4':
        from copuzzle.games.pancake.physics.scripted import PancakeFlipPolicy
        controller = PancakeFlipPolicy(operation, translation_scale=bridge.env.translation_scale,
                                       rotation_scale=bridge.env.rotation_scale)
    else:
        from copuzzle.games.topspin.physics.scripted import SequencePolicy
        controller = SequencePolicy([operation], action_scale=bridge.env.action_scale)
    if not callable(controller):
        raise TypeError('policy_factory must return a callable controller')
    if callable(getattr(controller, 'reset', None)):
        controller.reset()
    return controller


def _segment(index, start, goal, operation):
    return {'id': index, 'start_step': start, 'end_step': start,
            'goal': goal.tolist(), 'requested_operation': operation,
            'goal_first_step': None, 'outcome': 'collection_cutoff',
            'switch_safe': False, 'controller_completed': False}


def collect_play(task_id, *, seed=0, num_segments=3, max_episode_steps=12000,
                 max_steps=None, operations=None, policy_factory=None):
    """Collect one physical trajectory with safe, local-operation boundaries.

The controller's final returned action is executed even when that call changes
its ``completed`` flag. Goal achievement alone never interrupts tool release.
Expected controller failures preserve every complete transition collected so
far; a failed native action has no fabricated successor and is diagnostic only.
"""
    if task_id not in TASK_IDS:
        raise ValueError('Unsupported play task')
    seed = _integer(seed, 'seed')
    num_segments = _integer(num_segments, 'num_segments', 1)
    horizon = _integer(max_episode_steps, 'max_episode_steps', 1)
    limit = horizon if max_steps is None else _integer(max_steps, 'max_steps', 1)
    if policy_factory is not None and not callable(policy_factory):
        raise TypeError('policy_factory must be callable')
    if operations is not None:
        if not isinstance(operations, (list, tuple)) or len(operations) != num_segments:
            raise ValueError('operations must contain exactly num_segments entries')
        for operation in operations:
            if type(operation) is not (str if task_id == 'topspin-20-4' else int) or operation not in OPERATIONS[task_id]:
                raise ValueError('Invalid requested play operation')
        operations = tuple(operations)
    rng = np.random.default_rng(seed)
    size = {'grid-3x3': 9, 'pancake-4': 4, 'topspin-20-4': 20}[task_id]
    initial = (rng.integers(0, 2, size=size, dtype=np.uint8) if task_id == 'grid-3x3'
               else rng.permutation(np.arange(1, size + 1, dtype=np.int32)))

    def choose(index):
        return operations[index] if operations is not None else OPERATIONS[task_id][int(rng.integers(len(OPERATIONS[task_id])))]

    operation = choose(0)
    goal = forward(task_id, initial, operation)
    sources = _sources()
    bridge = PlayBridge(task_id, horizon)
    config = bridge.configuration()
    metadata = {'seed': seed, 'initial_state': initial.tolist(), 'initial_goal': goal.tolist(),
                'global_horizon': horizon, 'collector_limit': limit, 'requested_segments': num_segments,
                'stop_reason': 'collection_cutoff', 'sources': sources,
                'source_sha256': hashlib.sha256(_canonical(sources)).hexdigest(),
                'configuration_sha256': hashlib.sha256(_canonical(config)).hexdigest(),
                'policy': _identity(policy_factory) if policy_factory is not None else 'native_single_operation_reference',
                'collection_semantics': 'continuous local-operation play; privileged reference executor, not a global-goal expert',
                'info_semantics': 'infos[i+1] is the diagnostic after action i under behavior_goals[i]; goal switches never rewrite infos at their boundary frame',
                'reset_count': 1, 'requested_operations': list(operations) if operations is not None else None,
                'rng_algorithm': 'numpy.PCG64',
                'operation_policy': 'explicit_local_operations' if operations is not None else 'uniform_local_operation',
                'error': None, 'failed_action': None, 'controller_identities': []}
    actions, rewards, reached_flags, timeout_flags, goals, ids, infos, segments = [], [], [], [], [], [], [], []
    histories = None
    current = _segment(0, 0, goal, operation)
    policy = None
    try:
        obs, info = bridge.reset(seed=seed, state=initial, goal=goal)
        histories = {key: [value] for key, value in public_observation(obs).items()}
        infos.append(deepcopy(info))
        while True:
            current['end_step'] = len(actions)
            current['switch_safe'] = bool(safe_boundary(task_id, obs, config))
            current['controller_completed'] = bool(policy is not None and getattr(policy, 'completed', False))
            if policy is not None:
                failure = getattr(policy, 'failure', None)
                finished = bool(getattr(policy, 'finished', False))
                if (failure and (finished or current['controller_completed'])) or (finished and not current['controller_completed']):
                    current['outcome'] = metadata['stop_reason'] = 'controller_failure'
                    metadata['error'] = {'stage': 'controller', 'type': 'ControllerFailure',
                                         'message': str(failure or 'Controller finished without completing its operation')}
                    break
                achieved = (current['goal_first_step'] is not None
                            and np.array_equal(obs['board'], goal))
                if current['controller_completed'] and achieved and current['switch_safe']:
                    current['outcome'] = 'completed'
                    segments.append(current)
                    if len(segments) == num_segments:
                        metadata['stop_reason'] = 'requested_segments_complete'
                        current = None
                        break
                    if len(actions) >= min(limit, horizon):
                        metadata['stop_reason'] = 'timeout' if len(actions) >= horizon else 'collection_cutoff'
                        current = None
                        break
                    operation = choose(len(segments))
                    goal = forward(task_id, obs['board'], operation)
                    current = _segment(len(segments), len(actions), goal, operation)
                    policy = None
                elif (current['controller_completed'] and current['switch_safe']
                      and not np.array_equal(obs['board'], goal)):
                    current['outcome'] = metadata['stop_reason'] = 'mismatch'
                    break
            if len(actions) >= horizon:
                current['outcome'] = metadata['stop_reason'] = 'timeout'
                break
            if len(actions) >= limit:
                current['outcome'] = metadata['stop_reason'] = 'collection_cutoff'
                break
            if policy is None:
                try:
                    policy = _policy(task_id, operation, bridge, policy_factory)
                except Exception as error:
                    current['outcome'] = metadata['stop_reason'] = 'controller_error'
                    metadata['error'] = {'stage': 'policy_factory', 'type': type(error).__name__, 'message': str(error)}
                    break
                metadata['controller_identities'].append(_identity(policy))
                if current['id']:
                    obs = bridge.set_goal(goal)
            try:
                action = np.asarray(policy(deepcopy(obs)), dtype=np.float32)
                space = bridge.env.action_space
                if (action.shape != space.shape or not np.isfinite(action).all()
                        or np.any(action < space.low) or np.any(action > space.high)):
                    raise ValueError('Controller returned an invalid continuous action')
            except Exception as error:
                current['outcome'] = metadata['stop_reason'] = 'controller_error'
                metadata['error'] = {'stage': 'controller', 'type': type(error).__name__, 'message': str(error)}
                break
            try:
                following, reward, reached, timeout, info = bridge.step(action)
            except Exception as error:
                current['outcome'] = 'collection_cutoff'
                metadata['stop_reason'] = 'native_error'
                metadata['error'] = {'stage': 'native_step', 'type': type(error).__name__, 'message': str(error)}
                metadata['failed_action'] = action.tolist()
                break
            physical = public_observation(following)
            if physical.keys() != histories.keys():
                raise ValueError('Physical observation layout changed during play')
            actions.append(action.copy()); rewards.append(reward)
            reached_flags.append(reached); timeout_flags.append(timeout)
            goals.append(goal.copy()); ids.append(current['id']); infos.append(deepcopy(info))
            for key, value in physical.items():
                histories[key].append(value)
            if reached and current['goal_first_step'] is None:
                current['goal_first_step'] = len(actions) - 1
            obs = following
            if timeout:
                current['end_step'] = len(actions)
                current['switch_safe'] = bool(safe_boundary(task_id, obs, config))
                current['controller_completed'] = bool(getattr(policy, 'completed', False))
                current['outcome'] = metadata['stop_reason'] = 'timeout'
                break
        if current is not None and current['end_step'] > current['start_step']:
            segments.append(current)
        if not actions:
            raise RuntimeError(f'Play stopped without a complete transition: {metadata["stop_reason"]}; {metadata["error"]}')
        cutoff = np.zeros(len(actions), dtype=bool)
        cutoff[-1] = not timeout_flags[-1]
        from .protocol import PlayTrace
        trace = PlayTrace(task_id=task_id, observations={key: np.stack(value) for key, value in histories.items()},
            actions=np.stack(actions), behavior_goals=np.stack(goals), segment_ids=np.asarray(ids, dtype=np.int32),
            native_rewards=np.asarray(rewards, dtype=np.float32), goal_reached=np.asarray(reached_flags, dtype=bool),
            timeouts=np.asarray(timeout_flags, dtype=bool), collection_cutoff=cutoff, infos=infos,
            segments=segments, configuration=config, metadata=metadata)
        trace.validate()
        if _sources() != sources:
            raise RuntimeError('Play source files changed during collection; refusing an ambiguous provenance artifact')
        return trace
    finally:
        bridge.close()


def _compare(expected, actual, *, atol, path):
    """Compare numeric snapshots and JSON diagnostics without regenerating policy."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f'Replay dictionary fields differ at {path}')
        return max((_compare(expected[key], actual[key], atol=atol, path=f'{path}.{key}')
                    for key in expected), default=0.)
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            raise ValueError(f'Replay sequence differs at {path}')
        return max((_compare(a, b, atol=atol, path=f'{path}[{i}]')
                    for i, (a, b) in enumerate(zip(expected, actual))), default=0.)
    if isinstance(expected, np.ndarray):
        actual = np.asarray(actual)
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise ValueError(f'Replay array contract differs at {path}')
        if expected.dtype.kind in 'bui':
            if not np.array_equal(expected, actual):
                raise ValueError(f'Replay integer observation differs at {path}')
            return 0.
        error = float(np.max(np.abs(expected - actual), initial=0))
        if not np.isfinite(actual).all() or error > atol:
            raise ValueError(f'Replay numerical observation differs at {path}: {error}')
        return error
    if isinstance(expected, (float, np.floating)):
        error = abs(float(expected) - float(actual))
        if not np.isfinite(error) or error > atol:
            raise ValueError(f'Replay diagnostic differs at {path}')
        return error
    if expected != actual:
        raise ValueError(f'Replay value differs at {path}: {expected!r} != {actual!r}')
    return 0.


def replay_play(trace, atol=1e-8):
    """Reset once and replay saved actions plus saved goal-boundary events."""
    if isinstance(atol, bool) or not np.isscalar(atol) or not np.isfinite(atol) or atol < 0:
        raise ValueError('atol must be finite and nonnegative')
    trace.validate()
    bridge = PlayBridge(trace.task_id, trace.metadata['global_horizon'])
    maximum = 0.
    boundaries = {segment['start_step']: segment for segment in trace.segments}
    try:
        if bridge.configuration() != trace.configuration:
            raise ValueError('Replay native configuration differs from the play trace')
        obs, info = bridge.reset(seed=trace.metadata['seed'], state=trace.metadata['initial_state'],
                                goal=trace.metadata['initial_goal'])
        maximum = max(maximum, _compare({key: value[0] for key, value in trace.observations.items()},
                                        public_observation(obs), atol=atol, path='observations[0]'))
        maximum = max(maximum, _compare(trace.infos[0], info, atol=atol, path='infos[0]'))
        for index, action in enumerate(trace.actions):
            if index in boundaries and index:
                obs = bridge.set_goal(boundaries[index]['goal'])
                maximum = max(maximum, _compare({key: value[index] for key, value in trace.observations.items()},
                    public_observation(obs), atol=atol, path=f'boundary[{index}]'))
            obs, reward, reached, timeout, info = bridge.step(action)
            maximum = max(maximum, _compare({key: value[index + 1] for key, value in trace.observations.items()},
                public_observation(obs), atol=atol, path=f'observations[{index + 1}]'))
            maximum = max(maximum, _compare(trace.infos[index + 1], info, atol=atol, path=f'infos[{index + 1}]'))
            if (reward != float(trace.native_rewards[index]) or reached != bool(trace.goal_reached[index])
                    or timeout != bool(trace.timeouts[index])):
                raise ValueError(f'Replay reward or ending flags differ at action {index}')
        return {'control_steps': len(trace.actions), 'segments': len(trace.segments), 'reset_count': 1,
                'max_absolute_observation_error': maximum, 'actions_replayed': True,
                'policy_regenerated': False}
    finally:
        bridge.close()
