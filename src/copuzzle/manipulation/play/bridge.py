"""Explicit continuing lifecycle around unchanged native contact environments.

Only native goal/termination observer fields are changed. Robot and puzzle
physics, integration time, control targets, event state and the global control
budget continue across behavior-goal boundaries.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from copuzzle.manipulation.runner import make_environment
from .semantics import public_observation, safe_boundary


TASK_IDS = ('grid-3x3', 'pancake-4', 'topspin-20-4')


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}, not a boolean')
    return value


def _board(task_id, value, name):
    value = np.asarray(value)
    size = {'grid-3x3': 9, 'pancake-4': 4, 'topspin-20-4': 20}[task_id]
    if value.shape != (size,) or value.dtype.kind not in 'iu':
        raise ValueError(f'{name} must contain {size} integers')
    if task_id == 'grid-3x3':
        if not np.all((value == 0) | (value == 1)):
            raise ValueError(f'{name} must be a binary board')
        return value.astype(np.uint8, copy=True)
    if not np.array_equal(np.sort(value), np.arange(1, size + 1)):
        raise ValueError(f'{name} must be a permutation of 1..{size}')
    return value.astype(np.int32, copy=True)


class PlayBridge:
    """One reset followed by contact steps and explicit safe goal changes.

``native_goal_reached`` is a diagnostic achievement flag, not play termination.
The original native goal monitor is saturated at three observations for
TopSpin, because continuing after success is new behavior of this bridge.
"""

    def __init__(self, task_id, max_episode_steps):
        if not isinstance(task_id, str) or task_id not in TASK_IDS:
            raise ValueError(f'Unsupported play task: {task_id}')
        self.task_id = task_id
        self.max_episode_steps = _integer(max_episode_steps, 'max_episode_steps', 1)
        self.env = make_environment({'task_id': task_id, 'max_control_steps': max_episode_steps})
        self._started = self._closed = self._timeout = self._failed = False
        self._last_observation = None

    def configuration(self):
        """Return the unchanged native model/control configuration."""
        return deepcopy(self.env.configuration())

    def _check_open(self):
        if self._closed or self.env._closed:
            raise RuntimeError('Play bridge is closed')
        if self._failed:
            raise RuntimeError('Play bridge stopped after a native step failure')
        if self._timeout or (self._started and self.env._steps >= self.max_episode_steps):
            raise RuntimeError('Play global horizon has ended')

    def reset(self, *, seed, state, goal):
        self._check_open()
        if self._started:
            raise RuntimeError('A play trajectory allows exactly one physical reset')
        seed = _integer(seed, 'seed')
        state, goal = _board(self.task_id, state, 'state'), _board(self.task_id, goal, 'goal')
        if np.array_equal(state, goal):
            raise ValueError('Initial goal must differ from the reset board')
        observation, info = self.env.reset(seed=seed, options={'state': state, 'goal': goal})
        self._started = True
        self._last_observation = deepcopy(observation)
        return deepcopy(observation), deepcopy(info)

    def set_goal(self, goal):
        self._check_open()
        if not self._started:
            raise RuntimeError('Reset once before setting a play goal')
        goal = _board(self.task_id, goal, 'goal')
        observation = self.env._observation()
        if not safe_boundary(self.task_id, observation, self.configuration()):
            raise ValueError('A behavior goal can change only at a measured safe boundary')
        if np.array_equal(goal, observation['board']):
            raise ValueError('New behavior goal must differ from the measured current board')
        # This whitelist is intentional. Never touch MjData, robot controllers,
        # event re-arming, operation counters, or the global step count here.
        self.env._goal = goal.copy()
        if self.task_id == 'topspin-20-4':
            self.env._success_streak = 0
        self.env._needs_reset = False
        observation = self.env._observation()
        self._last_observation = deepcopy(observation)
        return deepcopy(observation)

    def step(self, action):
        self._check_open()
        if not self._started:
            raise RuntimeError('Reset once before stepping play')
        action = np.asarray(action)
        space = self.env.action_space
        if (action.shape != space.shape or action.dtype.kind not in 'iuf'
                or not np.isfinite(action).all()
                or np.any(action < space.low) or np.any(action > space.high)):
            raise ValueError('Action must be finite and inside the native continuous action space')
        # Native TopSpin increments a target-dependent counter and normally
        # stops at three. Saturation keeps its public range valid during dwell.
        if self.task_id == 'topspin-20-4' and self.env._success_streak >= 3:
            self.env._success_streak = 2
        try:
            observation, reward, reached, native_timeout, info = self.env.step(action.copy())
        except Exception:
            self._failed = True
            raise
        self._timeout = bool(native_timeout or self.env._steps >= self.max_episode_steps)
        # A successful last-budget action must end play too, even though native
        # episodic environments give success priority over their timeout flag.
        self.env._needs_reset = self._timeout
        self._last_observation = deepcopy(observation)
        return deepcopy(observation), float(reward), bool(reached), self._timeout, deepcopy(info)

    def close(self):
        if not self._closed:
            self.env.close()
            self._closed = True
