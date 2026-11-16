"""Goal-independent physical evidence and explicit play goal semantics.

No simulator, solver, reference sequence or controller phase is used here.
The initial TopSpin settled-board counter is zero; each subsequent control
observation increments it only while the same board remains physically ready.
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache

TASK_IDS = ('grid-3x3', 'pancake-4', 'topspin-20-4')


@lru_cache(maxsize=16)
def _track_geometry(slot_values):
    slots = np.asarray(slot_values).reshape(20, 3)
    def center(points):
        a, b, c = points
        return np.linalg.solve(2 * np.stack((b - a, c - a)),
                               np.array([b @ b - a @ a, c @ c - a @ a]))
    right = center(slots[4:7, :2])
    left = center(slots[14:17, :2])
    return (right + left) / 2, np.linalg.norm(right - left) / 2, np.linalg.norm(slots[4, :2] - right), slots[0, 2]


def _topspin_board_matches(observation, configuration):
    center, half, radius, height = _track_geometry(tuple(np.asarray(configuration['scene']['slot_positions']).ravel()))
    p = observation['token_positions']
    xy = p[:, :2] - center
    distance = np.abs(np.hypot(np.maximum(np.abs(xy[:, 0]) - half, 0), xy[:, 1]) - radius)
    turn = observation['turntable_center']
    window = np.flatnonzero((np.abs(p[:, 0] - turn[0]) < .039) & (np.abs(p[:, 1] - turn[1]) < .002))
    if len(window) != 4 or distance.max() >= .0035 or np.max(np.abs(p[:, 2] - height)) >= .0015:
        return False
    first = window[np.argmin(p[window, 0])]
    order = np.argsort(-np.arctan2(xy[:, 1], xy[:, 0]))
    decoded = np.roll(order, -int(np.flatnonzero(order == first)[0])) + 1
    return np.array_equal(decoded, observation['board'])


def public_observation(observation):
    """Copy physical observations, excluding the native goal and its monitor."""
    return {name: np.array(value, copy=True) for name, value in observation.items()
            if name not in ('goal', 'success_streak')}


def validate_goal(task_id, value):
    if task_id not in TASK_IDS:
        raise ValueError('Unsupported play task')
    array = np.asarray(value)
    width = {'grid-3x3': 9, 'pancake-4': 4, 'topspin-20-4': 20}[task_id]
    if array.shape != (width,) or array.dtype.kind not in 'iu':
        raise ValueError('Goal must be an integer board of the declared size')
    if task_id == 'grid-3x3':
        if not np.all((array == 0) | (array == 1)):
            raise ValueError('Lights Out goal must be binary')
        return array.astype(np.uint8)
    if not np.array_equal(np.sort(array), np.arange(1, width + 1)):
        raise ValueError('Goal must be a permutation of the task tokens')
    return array.astype(np.int32)


def physical_ready(task_id, observation, configuration):
    """Whether the observed board can satisfy this family's physical goal."""
    o = observation
    if task_id == 'grid-3x3':
        return True  # Native Lights Out goal is lamp equality, not release.
    if task_id == 'topspin-20-4':
        return bool(o['board_valid'][0] and o['tool_clear'][0]
                    and o['end_effector_position'][2] > .15
                    and not np.any(o['tool_token_contact_mask'])
                    and not np.any(o['tool_token_contact_normal_force'] > 1e-6)
                    and np.max(np.abs(o['token_velocities'])) < .015
                    and abs(o['turntable_velocity'][0]) < .05
                    and min(abs(o['turntable_angle'][0]), abs(o['turntable_angle'][0] - np.pi)) < .005
                    and _topspin_board_matches(o, configuration))
    if task_id != 'pancake-4':
        raise ValueError('Unsupported play task')
    scene = configuration['scene']
    positions = o['token_positions']
    order = np.argsort(positions[:, 2])[::-1]
    heights = (scene['support_top_m'] + scene['symmetric_boss_height_m'] / 2
               + scene['stack_pitch_m'] * np.arange(3, -1, -1))
    xy = np.asarray(scene['stack_center_xy'])
    normal_z = 1 - 2 * np.sum(o['token_orientations'][:, 1:3] ** 2, axis=1)
    return bool(o['board_valid'][0] and np.array_equal(order + 1, o['board'])
                and np.max(np.linalg.norm(positions[:, :2] - xy, axis=1)) < .005
                and np.max(np.abs(positions[order, 2] - heights)) < .0025
                and np.min(np.abs(normal_z)) > .98
                and np.max(np.abs(o['token_velocities'])) < .015
                and not np.any(o['tool_contacts'])
                and not np.any(o['tool_contact_forces'] > 1e-6)
                and np.linalg.norm(o['end_effector_position'][:2] - xy) > scene['withdraw_distance_m'])


def safe_boundary(task_id, observation, configuration):
    """Physical gate for changing a behavior goal; distinct from success."""
    if task_id == 'grid-3x3':
        rest = observation['button_rest_positions']
        return bool(np.all(observation['button_armed'])
                    and np.max(np.abs(observation['button_displacements'])) < configuration['release_depth']
                    and np.max(np.abs(observation['button_velocities'])) < .01
                    and observation['end_effector_position'][2] >= np.max(rest[:, 2]) + .05)
    return physical_ready(task_id, observation, configuration)


def frame(observations, index):
    return {name: values[index] for name, values in observations.items()}


def settled_board_runs(task_id, observations, configuration):
    """Target-independent history; no reused native success_streak is allowed."""
    count = len(observations['board'])
    runs = np.zeros(count, dtype=np.int32)
    for index in range(1, count):
        current = frame(observations, index)
        if physical_ready(task_id, current, configuration):
            same = np.array_equal(observations['board'][index], observations['board'][index - 1])
            runs[index] = min(3, int(runs[index - 1]) + 1) if same else 1
    return runs


def goal_matches(task_id, observations, goals, configuration):
    """Static goal -> T+1 flags; per-action goals[T,N] -> post-action T flags.

    TopSpin uses consecutive stable observations of the SAME physical board,
    independent of the collector's current behavior goal. A goal reached before
    a sampled start is already satisfied, not a new positive actor trajectory.
    """
    boards = observations['board']
    goals = np.asarray(goals)
    if goals.ndim == 1:
        goal = validate_goal(task_id, goals)
        equal = np.all(boards == goal, axis=1)
        indices = np.arange(len(boards))
    elif goals.shape == (len(boards) - 1, boards.shape[1]):
        for goal in goals:
            validate_goal(task_id, goal)
        equal = np.all(boards[1:] == goals, axis=1)
        indices = np.arange(1, len(boards))
    else:
        raise ValueError('Goal history must be static or action-aligned')
    ready = np.array([physical_ready(task_id, frame(observations, int(i)), configuration)
                      for i in indices], dtype=bool)
    if task_id == 'topspin-20-4':
        ready &= settled_board_runs(task_id, observations, configuration)[indices] >= 3
    return equal & ready
