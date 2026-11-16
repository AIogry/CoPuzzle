"""Exact bounded graph searches for puzzles with state-dependent legality.

Sliding normalizes only nonzero tile labels. The blank is never relabeled,
so different goal blank locations use different searches. A two-entry cache
bounds large 3x3 searches; Hanoi searches are rooted at the actual goal.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .rules import ClassicRule, _immutable, _integer


MAX_EXACT_STATES = 181_440


@dataclass(frozen=True)
class _Search:
    states: np.ndarray
    indices: Mapping[tuple[int, ...], int]
    parents: np.ndarray
    toward_goal: np.ndarray
    distances: np.ndarray


@lru_cache(maxsize=2)
def _search(rule: ClassicRule, goal: tuple[int, ...], max_states: int) -> _Search:
    states = [goal]
    indices = {goal: 0}
    parents = [-1]
    toward_goal = [-1]
    distances = [0]
    inverse = tuple(rule.inverse_action(action) for action in range(rule.n_actions))
    head = 0
    while head < len(states):
        state = states[head]
        for action, successor in rule._successors(state):
            if successor in indices:
                continue
            if len(states) >= max_states:
                raise ValueError(f"exact BFS exceeds max_states={max_states}; no partial distances are returned")
            indices[successor] = len(states)
            states.append(successor)
            parents.append(head)
            toward_goal.append(inverse[action])
            distances.append(distances[head] + 1)
        head += 1
    return _Search(
        states=_immutable(np.asarray(states, dtype=np.int32)),
        indices=MappingProxyType(indices),
        parents=_immutable(np.asarray(parents, dtype=np.int32)),
        toward_goal=_immutable(np.asarray(toward_goal, dtype=np.int32)),
        distances=_immutable(np.asarray(distances, dtype=np.int32)),
    )


class ExactOracle:
    """Shortest paths with a finite search budget and immutable reference arrays.

    ``reachable_states`` and ``distances`` refer to ``rule.canonical_goal``.
    Their maximum is the reference goal's eccentricity, not an assumption
    that arbitrary goals have the same distance distribution.
    """

    def __init__(self, rule: ClassicRule, *, max_states: int = MAX_EXACT_STATES) -> None:
        if not isinstance(rule, ClassicRule):
            raise TypeError("rule must be a ClassicRule")
        max_states = _integer(max_states, name="max_states")
        if not 1 <= max_states <= MAX_EXACT_STATES:
            raise ValueError(f"max_states must be in [1, {MAX_EXACT_STATES}]")
        self.rule = rule
        self.max_states = max_states
        self._reference = _search(rule, tuple(rule.canonical_goal.tolist()), max_states)

    @property
    def reachable_states(self) -> np.ndarray:
        return self._reference.states

    @property
    def distances(self) -> np.ndarray:
        return self._reference.distances

    @property
    def max_distance(self) -> int:
        return int(self.distances.max())

    def _goal_search(self, goal: np.ndarray) -> tuple[_Search, np.ndarray | None]:
        if self.rule.family == "hanoi":
            return _search(self.rule, tuple(goal.tolist()), self.max_states), None
        # The canonical nonzero labels depend on the goal's blank position.
        normalized_goal = np.zeros(self.rule.state_size, dtype=np.int32)
        normalized_goal[goal != 0] = np.arange(1, self.rule.state_size, dtype=np.int32)
        encode = np.empty(self.rule.state_size, dtype=np.int32)
        encode[goal] = normalized_goal
        return _search(self.rule, tuple(normalized_goal.tolist()), self.max_states), encode

    def _locate(self, state: object, goal: object) -> tuple[_Search | None, int | None]:
        state = self.rule.validate_state(state)
        goal = self.rule.validate_state(goal, name="goal")
        if not self.rule.is_reachable(state, goal):
            return None, None
        search, encode = self._goal_search(goal)
        normalized = state if encode is None else encode[state]
        return search, search.indices.get(tuple(normalized.tolist()))

    def distance(self, state: object, goal: object) -> int | None:
        search, index = self._locate(state, goal)
        return None if index is None else int(search.distances[index])

    def solve(self, state: object, goal: object) -> list[int] | None:
        search, index = self._locate(state, goal)
        if index is None:
            return None
        result = []
        while index:
            result.append(int(search.toward_goal[index]))
            index = int(search.parents[index])
        return result

    def states_at_distance(self, goal: object, depth: int) -> np.ndarray:
        """Independent rows exactly ``depth`` legal moves from an arbitrary goal."""
        goal = self.rule.validate_state(goal, name="goal")
        depth = _integer(depth, name="depth")
        if depth < 0:
            raise ValueError("depth must be nonnegative")
        search, encode = self._goal_search(goal)
        selected = search.states[search.distances == depth]
        if encode is None:
            return selected.copy()
        decode = np.empty_like(encode)
        decode[encode] = np.arange(self.rule.state_size, dtype=np.int32)
        return decode[selected]
