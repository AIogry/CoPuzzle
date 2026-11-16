"""First-hit goal views of segmented play, without behavior-goal leakage.

Views start at a selected physical frame. They end at the first measured goal
success, or at their data/horizon boundary. An already satisfied start returns
no action supervision. Observed path length is never a shortest-distance label.
"""
from __future__ import annotations

import numpy as np
from .semantics import frame, goal_matches, physical_ready, settled_board_runs, validate_goal


def _index(value, name, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [0, {maximum}]')
    return value


def _features(trace, start, stop):
    fields = dict(trace.observations)
    if 'goal' in fields or 'success_streak' in fields:
        raise ValueError('Goal-dependent fields cannot enter play state features')
    if trace.task_id == 'topspin-20-4':
        fields['settled_board_steps'] = settled_board_runs(trace.task_id, fields, trace.configuration)[:, None]
    pieces, specification, offset = [], [], 0
    for name in sorted(fields):
        value = fields[name][start:stop + 1]
        width = int(np.prod(value.shape[1:]))
        with np.errstate(over='ignore', invalid='ignore'):
            flat = value.reshape(len(value), width).astype(np.float32)
        if not np.isfinite(flat).all():
            raise ValueError('Features must be representable as finite float32')
        pieces.append(flat)
        specification.append({'name': name, 'shape': list(value.shape[1:]),
                              'dtype': str(value.dtype), 'start': offset, 'stop': offset + width})
        offset += width
    return np.concatenate(pieces, axis=1), specification


def goal_episode(trace, start_index, goal, end_index=None, *, bootstrap_mode='finite_horizon'):
    """Recompute rewards/endings for one fixed goal from a physical play trace."""
    trace.validate()
    total = len(trace.actions)
    start = _index(start_index, 'start_index', total)
    end = total if end_index is None else _index(end_index, 'end_index', total)
    if end < start:
        raise ValueError('end_index must not precede start_index')
    if bootstrap_mode not in ('finite_horizon', 'terminated_only'):
        raise ValueError('Unsupported bootstrap mode')
    goal = validate_goal(trace.task_id, goal)
    reached = goal_matches(trace.task_id, trace.observations, goal, trace.configuration)
    initially_satisfied = bool(reached[start])
    confirmation_pending = bool(trace.task_id == 'topspin-20-4' and not initially_satisfied
        and np.array_equal(trace.observations['board'][start], goal)
        and physical_ready(trace.task_id, frame(trace.observations, start), trace.configuration))
    first = np.flatnonzero(reached[start + 1:end + 1])
    stop = start if initially_satisfied or confirmation_pending else start + 1 + int(first[0]) if len(first) else end
    count = stop - start
    terminated = np.zeros(count, dtype=bool)
    truncated = np.zeros(count, dtype=bool)
    cutoff = np.zeros(count, dtype=bool)
    rewards = np.full(count, -1., dtype=np.float32)
    if count:
        terminated[-1] = bool(reached[stop])
        truncated[-1] = bool(stop == total and trace.timeouts[-1] and not terminated[-1])
        cutoff[-1] = not (terminated[-1] or truncated[-1])
        if terminated[-1]:
            rewards[-1] = 0.
    states, feature_spec = _features(trace, start, stop)
    masks = ~(terminated | truncated) if bootstrap_mode == 'finite_horizon' else ~terminated
    return {'observations': states[:-1], 'next_observations': states[1:],
            'initial_observation': states[0].copy(), 'goals': np.tile(goal.astype(np.float32), (count, 1)),
            'actions': trace.actions[start:stop].copy(), 'rewards': rewards,
            'terminated': terminated, 'truncated': truncated, 'collection_cutoff': cutoff,
            'episode_end': terminated | truncated | cutoff, 'bootstrap_masks': masks.astype(np.float32),
            'initially_satisfied': initially_satisfied, 'source_start': start, 'source_end': stop,
            'initial_goal_confirmation_pending': confirmation_pending,
            'actor_supervision_valid': bool(count),
            'observed_control_steps': count, 'shortest_distance': None,
            'feature_spec': feature_spec, 'feature_profile': 'physical_play_goal_free_v1',
            'bootstrap_mode': bootstrap_mode,
            'source_segment_ids': trace.segment_ids[start:stop].copy(),
            'source_goal_reached': trace.goal_reached[start:stop].copy(),
            'source_timeouts': trace.timeouts[start:stop].copy(),
            'source_collection_cutoff': trace.collection_cutoff[start:stop].copy()}


def future_episode(trace, start_index, target_index, *, within_segment=True,
                   bootstrap_mode='finite_horizon'):
    """Use a physically achieved future board, truncating at its FIRST success.

    By default sampling remains in the behavior segment containing the start.
    Cross-segment paths require explicit opt-in; they are real paths, not expert
    efficiency labels. No view crosses the trace's single physical episode.
    """
    total = len(trace.actions)
    start = _index(start_index, 'start_index', total - 1)
    target = _index(target_index, 'target_index', total)
    if target <= start:
        raise ValueError('Future target must be later than the start')
    if type(within_segment) is not bool:
        raise ValueError('within_segment must be boolean')
    trace.validate()
    segment = trace.segments[int(trace.segment_ids[start])]
    if within_segment and target > segment['end_step']:
        raise ValueError('Future target crosses the behavior-goal segment')
    goal = trace.observations['board'][target]
    reached = goal_matches(trace.task_id, trace.observations, goal, trace.configuration)
    if not reached[target]:
        raise ValueError('Future frame is not a physically achieved goal')
    result = goal_episode(trace, start, goal, target, bootstrap_mode=bootstrap_mode)
    result.update(target_index=target, crosses_behavior_boundary=target > segment['end_step'],
                  sampling_semantics='real future goal, first physical hit, no optimality claim')
    return result
