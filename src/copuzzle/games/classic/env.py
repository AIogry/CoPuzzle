"""Goal-conditioned Gymnasium environments with explicit legal-action masks."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import ResetNeeded
import numpy as np

from .rules import ClassicRule, _integer, get_rule


class ClassicEnv(gym.Env):
    """Discrete puzzle episode; consumers must sample from ``action_mask``.

    Invalid actions raise without advancing the episode. Goals are sampled
    by legal random walks; reset performs no exact graph search. A solved
    reset is terminal and requires another reset before stepping.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, rule: ClassicRule, max_episode_steps: int = 100, render_mode: str | None = None) -> None:
        super().__init__()
        if not isinstance(rule, ClassicRule):
            raise TypeError("rule must be a ClassicRule")
        max_episode_steps = _integer(max_episode_steps, name="max_episode_steps")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.rule = rule
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(rule.n_actions)
        high = rule.state_size - 1 if rule.family == "sliding" else 2
        self.observation_space = spaces.Dict({
            "board": spaces.Box(0, high, shape=(rule.state_size,), dtype=np.int32),
            "goal": spaces.Box(0, high, shape=(rule.state_size,), dtype=np.int32),
            "previous_event": spaces.Box(0, 1, shape=(rule.n_actions,), dtype=np.uint8),
            "action_mask": spaces.Box(0, 1, shape=(rule.n_actions,), dtype=np.uint8),
        })
        self._board: np.ndarray | None = None
        self._goal: np.ndarray | None = None
        self._previous_event = np.zeros(rule.n_actions, dtype=np.uint8)
        self._steps = 0
        self._needs_reset = True

    def _observation(self) -> dict[str, np.ndarray]:
        if self._board is None or self._goal is None:
            raise ResetNeeded("Call reset before requesting an observation")
        return {
            "board": self._board.copy(),
            "goal": self._goal.copy(),
            "previous_event": self._previous_event.copy(),
            "action_mask": self.rule.action_mask(self._board),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "is_success": bool(np.array_equal(self._board, self._goal)),
            "steps": self._steps,
            "rule_id": self.rule.rule_id,
        }

    def _walk(self, state: np.ndarray) -> np.ndarray:
        goal = state.copy()
        previous = None
        for _ in range(int(self.np_random.integers(1, 2 * self.rule.state_size + 1))):
            actions = np.flatnonzero(self.rule.action_mask(goal))
            if previous is not None and len(actions) > 1:
                actions = actions[actions != self.rule.inverse_action(previous)]
            action = int(self.np_random.choice(actions))
            goal = self.rule.transition(goal, action)
            previous = action
        if np.array_equal(goal, state):
            action = int(self.np_random.choice(np.flatnonzero(self.rule.action_mask(goal))))
            goal = self.rule.transition(goal, action)
        return goal

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if options is not None and not isinstance(options, dict):
            raise TypeError("options must be a dictionary or None")
        options = {} if options is None else options
        unknown = set(options) - {"state", "goal"}
        if unknown:
            raise ValueError(f"Unknown reset option(s): {sorted(unknown)}")
        state = self.rule.validate_state(options["state"]) if "state" in options else None
        goal = self.rule.validate_state(options["goal"], name="goal") if "goal" in options else None
        if state is None:
            state = self.rule.sample_state(self.np_random) if goal is None else self._walk(goal)
        if goal is None:
            goal = self._walk(state)
        if not self.rule.is_reachable(state, goal):
            raise ValueError("goal is unreachable from state under this rule")
        self._board = state.copy()
        self._goal = goal.copy()
        self._previous_event = np.zeros(self.rule.n_actions, dtype=np.uint8)
        self._steps = 0
        self._needs_reset = bool(np.array_equal(state, goal))
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise ResetNeeded("Call reset before stepping or after an episode ends")
        successor = self.rule.transition(self._board, action)
        self._board = successor
        self._previous_event = self.rule.event(action)
        self._steps += 1
        success = bool(np.array_equal(self._board, self._goal))
        terminated = success
        truncated = self._steps >= self.max_episode_steps and not success
        self._needs_reset = terminated or truncated
        return self._observation(), 0.0 if success else -1.0, terminated, truncated, self._info()

    def render(self) -> str | None:
        if self.render_mode is None:
            return None
        if self._board is None or self._goal is None:
            raise ResetNeeded("Call reset before rendering")
        board = " ".join(map(str, self._board.tolist()))
        goal = " ".join(map(str, self._goal.tolist()))
        return f"{self.rule.name} (step {self._steps}/{self.max_episode_steps})\nboard: {board}\ngoal:  {goal}"


def make_env(task_id: str, **kwargs: Any) -> ClassicEnv:
    return ClassicEnv(get_rule(task_id), **kwargs)
