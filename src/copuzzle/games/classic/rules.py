"""Bounded Sliding Puzzle and Tower of Hanoi with state-dependent legality.

Sliding actions move the blank up, down, left, or right. Hanoi states assign
each disk to a peg, smallest disk first; a peg's stack order is implicit.
Invalid moves raise rather than silently becoming self-loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import operator
import re
from typing import Iterator

import numpy as np


RULE_SCHEMA = "copuzzle.classic-rule.v1"
HANOI_MOVES = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _immutable(array: np.ndarray) -> np.ndarray:
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class ClassicRule:
    """Immutable rule and portable checked semantic specification."""

    family: str
    state_size: int
    rows: int | None = None
    cols: int | None = None
    name: str = field(init=False)
    n_actions: int = field(init=False)
    rule_id: str = field(init=False)

    def __post_init__(self) -> None:
        size = _integer(self.state_size, name="state_size")
        object.__setattr__(self, "state_size", size)
        if self.family == "sliding":
            rows = _integer(self.rows, name="rows")
            cols = _integer(self.cols, name="cols")
            if (rows, cols) not in ((2, 3), (3, 3)):
                raise ValueError("supported sliding shapes are 2x3 and 3x3")
            if size != rows * cols:
                raise ValueError("state_size must equal rows * cols")
            object.__setattr__(self, "rows", rows)
            object.__setattr__(self, "cols", cols)
            name, n_actions = f"sliding-{rows}x{cols}", 4
        elif self.family == "hanoi":
            if self.rows is not None or self.cols is not None:
                raise ValueError("Hanoi has no rows or cols")
            if not 2 <= size <= 6:
                raise ValueError("Hanoi supports 2..6 disks")
            name, n_actions = f"hanoi-{size}", 6
        else:
            raise ValueError("family must be 'sliding' or 'hanoi'")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "n_actions", n_actions)
        payload = json.dumps(self._spec_fields(), sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "rule_id", "classic-v1-" + hashlib.sha256(payload.encode("utf-8")).hexdigest())

    def _spec_fields(self) -> dict[str, object]:
        return {
            "schema": RULE_SCHEMA,
            "name": self.name,
            "family": self.family,
            "state_size": self.state_size,
            "rows": self.rows,
            "cols": self.cols,
            "n_actions": self.n_actions,
            "state_encoding": "tiles-0-to-N-1-blank-0-int32" if self.family == "sliding" else "peg-0-to-2-smallest-disk-first-int32",
            "action_encoding": "blank-up-down-left-right" if self.family == "sliding" else "peg-01-02-10-12-20-21",
            "invalid_action": "raise",
            "goal_equivalence": "exact-state",
        }

    def spec(self) -> dict[str, object]:
        return {**self._spec_fields(), "rule_id": self.rule_id}

    @property
    def canonical_goal(self) -> np.ndarray:
        """Independent goal array; arbitrary reachable goals are also valid."""
        if self.family == "hanoi":
            return np.full(self.state_size, 2, dtype=np.int32)
        return np.asarray([*range(1, self.state_size), 0], dtype=np.int32)

    def validate_state(self, value: object, name: str = "state") -> np.ndarray:
        try:
            array = np.asarray(value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{name} must be an integer state") from error
        if array.shape != (self.state_size,):
            raise ValueError(f"{name} must have shape ({self.state_size},), got {array.shape}")
        if array.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integers")
        if isinstance(value, (list, tuple)) and any(isinstance(item, (bool, np.bool_)) for item in value):
            raise ValueError(f"{name} must contain integers, not booleans")
        if self.family == "sliding":
            if not np.array_equal(np.sort(array), np.arange(self.state_size)):
                raise ValueError(f"{name} must contain every tile 0..{self.state_size - 1} exactly once")
        elif np.any(array > 2) or np.any(array < 0):
            raise ValueError(f"{name} must assign each disk to peg 0, 1, or 2")
        return array.astype(np.int32, copy=True)

    def _action(self, action: object) -> int:
        action = _integer(action, name="action")
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions})")
        return action

    def _successors(self, state: tuple[int, ...]) -> Iterator[tuple[int, tuple[int, ...]]]:
        """Fast internal graph edges; callers must supply a validated state."""
        if self.family == "sliding":
            blank = state.index(0)
            row, col = divmod(blank, self.cols)
            for action, target in enumerate((blank - self.cols, blank + self.cols, blank - 1, blank + 1)):
                if (action == 0 and row == 0) or (action == 1 and row == self.rows - 1):
                    continue
                if (action == 2 and col == 0) or (action == 3 and col == self.cols - 1):
                    continue
                successor = list(state)
                successor[blank], successor[target] = successor[target], successor[blank]
                yield action, tuple(successor)
        else:
            top = [self.state_size] * 3
            for disk, peg in enumerate(state):
                if top[peg] == self.state_size:
                    top[peg] = disk
            for action, (source, target) in enumerate(HANOI_MOVES):
                if top[source] < top[target]:
                    successor = list(state)
                    successor[top[source]] = target
                    yield action, tuple(successor)

    def action_mask(self, state: object) -> np.ndarray:
        board = tuple(self.validate_state(state).tolist())
        mask = np.zeros(self.n_actions, dtype=np.uint8)
        for action, _ in self._successors(board):
            mask[action] = 1
        return mask

    def transition(self, state: object, action: int) -> np.ndarray:
        board = tuple(self.validate_state(state).tolist())
        action = self._action(action)
        for candidate, successor in self._successors(board):
            if candidate == action:
                return np.asarray(successor, dtype=np.int32)
        raise ValueError(f"action {action} is illegal in the current state")

    def event(self, action: int) -> np.ndarray:
        action = self._action(action)
        event = np.zeros(self.n_actions, dtype=np.uint8)
        event[action] = 1
        return event

    def inverse_action(self, action: int) -> int:
        action = self._action(action)
        return ((1, 0, 3, 2) if self.family == "sliding" else (2, 4, 0, 5, 1, 3))[action]

    def is_reachable(self, state: object, goal: object) -> bool:
        """Check reachability without running an exact search.

        Sliding parity excludes the blank. For even-width boards the blank
        row from the bottom also contributes; current bounded shapes are odd.
        All valid Hanoi assignments are reachable under the stack convention.
        """
        state = self.validate_state(state)
        goal = self.validate_state(goal, name="goal")
        if self.family == "hanoi":
            return True

        def signature(board: np.ndarray) -> int:
            tiles = board[board != 0].tolist()
            parity = sum(tiles[i] > tiles[j] for i in range(len(tiles)) for j in range(i + 1, len(tiles)))
            if self.cols % 2 == 0:
                parity += self.rows - int(np.flatnonzero(board == 0)[0]) // self.cols
            return parity % 2

        return signature(state) == signature(goal)

    def sample_state(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a valid state; sliding samples both parity components."""
        if self.family == "hanoi":
            return rng.integers(0, 3, size=self.state_size, dtype=np.int32)
        return rng.permutation(np.arange(self.state_size, dtype=np.int32))


def get_rule(task_id: str) -> ClassicRule:
    if not isinstance(task_id, str):
        raise ValueError("task_id must be a string")
    match = re.fullmatch(r"sliding-([1-9][0-9]*)x([1-9][0-9]*)", task_id)
    if match:
        rows, cols = map(int, match.groups())
        return ClassicRule("sliding", rows * cols, rows, cols)
    match = re.fullmatch(r"hanoi-([1-9][0-9]*)", task_id)
    if match:
        return ClassicRule("hanoi", int(match.group(1)))
    raise ValueError(f"unknown task_id {task_id!r}; expected sliding-2x3, sliding-3x3 or hanoi-N")


def rule_from_spec(spec: object) -> ClassicRule:
    """Validate every field and identity without transitions or graph search."""
    if not isinstance(spec, dict):
        raise ValueError("rule spec must be a dictionary")
    try:
        rule = ClassicRule(spec["family"], spec["state_size"], spec["rows"], spec["cols"])
        supplied = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid classic rule spec") from error
    expected = json.dumps(rule.spec(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    if supplied != expected:
        raise ValueError("classic rule spec or rule_id does not match the canonical rule")
    return rule


def validate_rule_spec(spec: object) -> dict[str, object]:
    return rule_from_spec(spec).spec()
