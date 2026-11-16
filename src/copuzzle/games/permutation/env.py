"""Gymnasium lifecycle for goal-conditioned Pancake and atomic TopSpin."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import ResetNeeded
import numpy as np

from .oracle import ExactOracle
from .rules import PermutationRule, _integer, get_rule


class PermutationEnv(gym.Env):
    """Operate a fixed permutation rule without exposing its operation maps.

    Every action is legal. Observations contain int32 board/goal arrays and
    uint8 previous_event/action_mask arrays, whose widths are action count.
    Random goals are sampled uniformly from reachable nontrivial states.
    An explicitly solved reset finishes the episode and requires a reset.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        rule: PermutationRule,
        max_episode_steps: int = 30,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(rule, PermutationRule):
            raise TypeError("rule must be a PermutationRule")
        max_episode_steps = _integer(max_episode_steps, name="max_episode_steps")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.rule = rule
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(rule.n_actions)
        self.observation_space = spaces.Dict({
            "board": spaces.Box(1, rule.n_tokens, shape=(rule.n_tokens,), dtype=np.int32),
            "goal": spaces.Box(1, rule.n_tokens, shape=(rule.n_tokens,), dtype=np.int32),
            "previous_event": spaces.Box(0, 1, shape=(rule.n_actions,), dtype=np.uint8),
            "action_mask": spaces.Box(0, 1, shape=(rule.n_actions,), dtype=np.uint8),
        })
        self._oracle = ExactOracle(rule)
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
            "action_mask": np.ones(self.rule.n_actions, dtype=np.uint8),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "is_success": bool(np.array_equal(self._board, self._goal)),
            "steps": self._steps,
            "rule_id": self.rule.rule_id,
        }

    def _sample_goal(self, state: np.ndarray) -> np.ndarray:
        relative = self._oracle.reachable_states[int(self.np_random.integers(1, len(self._oracle.reachable_states)))]
        return state[relative - 1]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if options is not None and not isinstance(options, dict):
            raise TypeError("options must be a dictionary or None")
        options = {} if options is None else options
        unknown = set(options) - {"state", "goal"}
        if unknown:
            raise ValueError(f"Unknown reset option(s): {sorted(unknown)}")
        if "state" in options:
            state = self.rule.validate_state(options["state"])
        else:
            state = self.np_random.permutation(np.arange(1, self.rule.n_tokens + 1, dtype=np.int32))
        if "goal" in options:
            goal = self.rule.validate_state(options["goal"], name="goal")
            if self._oracle.distance(state, goal) is None:
                raise ValueError("goal is unreachable from state under this rule")
        else:
            goal = self._sample_goal(state)
        self._board = state.copy()
        self._goal = goal.copy()
        self._previous_event = np.zeros(self.rule.n_actions, dtype=np.uint8)
        self._steps = 0
        self._needs_reset = bool(np.array_equal(state, goal))
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise ResetNeeded("Call reset before stepping or after an episode ends")
        action = self.rule._action(action)
        self._board = self.rule.transition(self._board, action)
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


def make_env(task_id: str, **kwargs: Any) -> PermutationEnv:
    """Create a supported permutation game by its canonical task identifier."""
    return PermutationEnv(get_rule(task_id), **kwargs)
