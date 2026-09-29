"""Gymnasium environment for a fixed Lights Out rule.

Actions are intended single-button presses. Each ideal operation succeeds, so
the returned previous_event is the one-hot result of that action. It is zero
at reset and never reveals the action that the policy has yet to choose.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import ResetNeeded

from .oracle import is_reachable
from .rules import PuzzleRule


def _binary_space(shape: int | tuple[int, ...]) -> spaces.MultiBinary:
    """Keep binary observations and the declared space in uint8."""
    space = spaces.MultiBinary(shape)
    space.dtype = np.dtype(np.uint8)
    return space


class PuzzleEnv(gym.Env):
    """Operate one fixed rule with a reachable, goal-conditioned board.

    ``reset(options={"state": ..., "goal": ...})`` accepts explicit binary
    vectors. Otherwise the initial state is random and a nontrivial reachable
    goal is sampled. Explicit equal state/goal pairs are allowed, but already
    finish the episode; inspect ``info["is_success"]`` and reset again.

    The default observation hides graph structure. ``reveal_graph=True`` adds
    adjacency to the observation, but no oracle solution or distance is ever
    included in observations or info.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        rule: PuzzleRule,
        max_episode_steps: int = 30,
        reveal_graph: bool = False,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if (
            isinstance(max_episode_steps, (bool, np.bool_))
            or not isinstance(max_episode_steps, (int, np.integer))
            or max_episode_steps <= 0
        ):
            raise ValueError("max_episode_steps must be a positive integer")
        if not isinstance(reveal_graph, (bool, np.bool_)):
            raise TypeError("reveal_graph must be a boolean")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")

        self.rule = rule
        self.max_episode_steps = int(max_episode_steps)
        self.reveal_graph = bool(reveal_graph)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(rule.n_buttons)
        observation_fields = {
            name: _binary_space(rule.n_buttons)
            for name in ("board", "goal", "previous_event")
        }
        if self.reveal_graph:
            observation_fields["adjacency"] = _binary_space(
                (rule.n_buttons, rule.n_buttons)
            )
        self.observation_space = spaces.Dict(observation_fields)

        self._board: np.ndarray | None = None
        self._goal: np.ndarray | None = None
        self._previous_event = np.zeros(rule.n_buttons, dtype=np.uint8)
        self._steps = 0
        self._needs_reset = True

    def _observation(self) -> dict[str, np.ndarray]:
        if self._board is None or self._goal is None:
            raise ResetNeeded("Call reset before requesting an observation")
        observation = {
            "board": self._board.copy(),
            "goal": self._goal.copy(),
            "previous_event": self._previous_event.copy(),
        }
        if self.reveal_graph:
            observation["adjacency"] = np.array(
                self.rule.adjacency, dtype=np.uint8, copy=True
            )
        return observation

    def _info(self) -> dict[str, Any]:
        return {
            "is_success": bool(np.array_equal(self._board, self._goal)),
            "steps": self._steps,
            "rule_id": self.rule.rule_id,
        }

    def _sample_goal(self, state: np.ndarray) -> np.ndarray:
        matrix = self.rule.toggle_matrix
        if not np.any(matrix):
            raise ValueError("This rule has no reachable goal distinct from state")
        while True:
            parity = self.np_random.integers(
                0, 2, size=self.rule.n_buttons, dtype=np.uint8
            )
            effect = np.asarray((matrix @ parity) % 2, dtype=np.uint8)
            if np.any(effect):
                return np.bitwise_xor(state, effect)

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
            state = self.rule.validate_state(options["state"], name="state")
        else:
            state = self.np_random.integers(
                0, 2, size=self.rule.n_buttons, dtype=np.uint8
            )
        if "goal" in options:
            goal = self.rule.validate_state(options["goal"], name="goal")
            if not is_reachable(self.rule, state, goal):
                raise ValueError("goal is unreachable from state under this rule")
        else:
            goal = self._sample_goal(state)

        self._board = state.copy()
        self._goal = goal.copy()
        self._previous_event = np.zeros(self.rule.n_buttons, dtype=np.uint8)
        self._steps = 0
        self._needs_reset = bool(np.array_equal(self._board, self._goal))
        return self._observation(), self._info()

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise ResetNeeded("Call reset before stepping or after an episode ends")
        # Discrete accepts integer NumPy scalars and scalar integer arrays;
        # reject booleans explicitly because Python bool is an int subclass.
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {self.rule.n_buttons})")
        action = int(action)

        self._board = self.rule.transition(self._board, action)
        self._previous_event = np.zeros(self.rule.n_buttons, dtype=np.uint8)
        self._previous_event[action] = 1
        self._steps += 1
        success = bool(np.array_equal(self._board, self._goal))
        terminated = success
        truncated = self._steps >= self.max_episode_steps and not success
        self._needs_reset = terminated or truncated
        reward = 0.0 if success else -1.0
        return self._observation(), reward, terminated, truncated, self._info()

    def render(self) -> str | None:
        if self.render_mode is None:
            return None
        if self._board is None or self._goal is None:
            raise ResetNeeded("Call reset before rendering")
        board = " ".join(map(str, self._board.tolist()))
        goal = " ".join(map(str, self._goal.tolist()))
        return (
            f"{self.rule.name} (step {self._steps}/{self.max_episode_steps})\n"
            f"board: {board}\ngoal:  {goal}"
        )
