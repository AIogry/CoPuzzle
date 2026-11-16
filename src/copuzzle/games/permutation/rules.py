"""Versioned position-permutation rules for Pancake and atomic TopSpin.

The bounded first version accepts 2..8 distinct integer tokens numbered 1..N.
TopSpin uses a fixed position marker: rotations are operations, not equivalent
goals. Every operation costs one step. These rules expose no XOR effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import operator
import re

import numpy as np


MIN_TOKENS = 2
MAX_TOKENS = 8
RULE_SCHEMA = "copuzzle.permutation-rule.v1"


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _immutable(array: np.ndarray) -> np.ndarray:
    """Use a bytes owner so callers cannot re-enable writing on cached arrays."""
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class PermutationRule:
    """One immutable rule with a canonical, portable semantic identity."""

    family: str
    n_tokens: int
    window: int | None = None
    name: str = field(init=False)
    n_actions: int = field(init=False)
    rule_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.family not in ("pancake", "topspin"):
            raise ValueError("family must be 'pancake' or 'topspin'")
        n_tokens = _integer(self.n_tokens, name="n_tokens")
        if not MIN_TOKENS <= n_tokens <= MAX_TOKENS:
            raise ValueError(f"n_tokens must be in [{MIN_TOKENS}, {MAX_TOKENS}]")
        object.__setattr__(self, "n_tokens", n_tokens)
        if self.family == "pancake":
            if self.window is not None:
                raise ValueError("Pancake has no fixed window")
            name = f"pancake-{n_tokens}"
            n_actions = n_tokens - 1
        else:
            window = _integer(self.window, name="window")
            if not 2 <= window <= n_tokens:
                raise ValueError("window must be in [2, n_tokens]")
            object.__setattr__(self, "window", window)
            name = f"topspin-{n_tokens}-{window}"
            n_actions = 3
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "n_actions", n_actions)
        payload = json.dumps(self._spec_fields(), sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "rule_id", "perm-v1-" + hashlib.sha256(payload.encode("utf-8")).hexdigest())

    def _spec_fields(self) -> dict[str, object]:
        return {
            "schema": RULE_SCHEMA,
            "name": self.name,
            "family": self.family,
            "n_tokens": self.n_tokens,
            "n_actions": self.n_actions,
            "window": self.window,
            "action_encoding": "prefix-flip-a+2" if self.family == "pancake" else "left-right-prefix-flip",
            "goal_equivalence": "fixed-position",
            "state_encoding": "tokens-1-to-N-int32",
        }

    def spec(self) -> dict[str, object]:
        """Return an independent JSON-compatible rule specification."""
        return {**self._spec_fields(), "rule_id": self.rule_id}

    def validate_state(self, value: object, name: str = "state") -> np.ndarray:
        """Validate integer permutation values before converting to int32."""
        try:
            array = np.asarray(value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{name} must be an integer permutation") from error
        if array.shape != (self.n_tokens,):
            raise ValueError(f"{name} must have shape ({self.n_tokens},), got {array.shape}")
        if array.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integer tokens")
        if isinstance(value, (list, tuple)) and any(isinstance(item, (bool, np.bool_)) for item in value):
            raise ValueError(f"{name} must contain integer tokens, not booleans")
        if not np.array_equal(np.sort(array), np.arange(1, self.n_tokens + 1)):
            raise ValueError(f"{name} must contain each token 1..{self.n_tokens} exactly once")
        return array.astype(np.int32, copy=True)

    def _action(self, action: object) -> int:
        action = _integer(action, name="action")
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions})")
        return action

    def transition(self, state: object, action: int) -> np.ndarray:
        """Apply an atomic operation, returning an independent int32 state."""
        board = self.validate_state(state)
        action = self._action(action)
        if self.family == "pancake":
            length = action + 2
            board[:length] = board[:length][::-1]
        elif action == 0:
            board = np.roll(board, -1)
        elif action == 1:
            board = np.roll(board, 1)
        else:
            board[:self.window] = board[:self.window][::-1]
        return board

    def event(self, action: int) -> np.ndarray:
        """Actual operation event: one-hot over actions, not token positions."""
        action = self._action(action)
        event = np.zeros(self.n_actions, dtype=np.uint8)
        event[action] = 1
        return event


def get_rule(task_id: str) -> PermutationRule:
    """Resolve canonical pancake-N or topspin-N-K task identifiers."""
    if not isinstance(task_id, str):
        raise ValueError("task_id must be a string")
    match = re.fullmatch(r"pancake-([1-9][0-9]*)", task_id)
    if match:
        return PermutationRule("pancake", int(match.group(1)))
    match = re.fullmatch(r"topspin-([1-9][0-9]*)-([1-9][0-9]*)", task_id)
    if match:
        return PermutationRule("topspin", int(match.group(1)), int(match.group(2)))
    raise ValueError(f"unknown task_id {task_id!r}; expected pancake-N or topspin-N-K")


def rule_from_spec(spec: object) -> PermutationRule:
    """Reconstruct a rule only when all fields and its checked identity match.

    This performs static metadata validation only, without a transition or
    search. Extra fields and type changes are rejected rather than ignored.
    """
    if not isinstance(spec, dict):
        raise ValueError("rule spec must be a dictionary")
    try:
        rule = PermutationRule(spec["family"], spec["n_tokens"], spec["window"])
        supplied = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid permutation rule spec") from error
    expected = json.dumps(rule.spec(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    if supplied != expected:
        raise ValueError("permutation rule spec or rule_id does not match the canonical rule")
    return rule


def validate_rule_spec(spec: object) -> dict[str, object]:
    """Validate portable metadata without executing transitions or a solver."""
    return rule_from_spec(spec).spec()
