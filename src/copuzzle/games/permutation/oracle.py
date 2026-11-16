"""Bounded exact BFS diagnostics for permutation games.

The search explores only the identity component. Relabeling tokens relative
to the source reduces any requested pair to this search, including pairs in
other TopSpin components. Unreachable pairs return None, never a scramble
length or a distance upper bound. These are privileged diagnostics, not a
learned baseline or part of environment observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .rules import PermutationRule, _immutable, _integer


MAX_EXACT_STATES = 40_320


@dataclass(frozen=True)
class _Search:
    states: np.ndarray
    indices: Mapping[tuple[int, ...], int]
    parents: np.ndarray
    actions: np.ndarray
    distances: np.ndarray


@lru_cache(maxsize=8)
def _search(rule: PermutationRule, max_states: int) -> _Search:
    identity = tuple(range(1, rule.n_tokens + 1))
    # Build generator position maps once. The BFS itself uses Python tuples,
    # avoiding repeated ndarray validation and allocation at every edge.
    maps = tuple(tuple((rule.transition(identity, action) - 1).tolist()) for action in range(rule.n_actions))
    states = [identity]
    indices = {identity: 0}
    parents = [-1]
    actions = [-1]
    distances = [0]
    head = 0
    while head < len(states):
        state = states[head]
        for action, position_map in enumerate(maps):
            successor = tuple(state[position] for position in position_map)
            if successor in indices:
                continue
            if len(states) >= max_states:
                raise ValueError(f"exact BFS exceeds max_states={max_states}; no partial distances are returned")
            indices[successor] = len(states)
            states.append(successor)
            parents.append(head)
            actions.append(action)
            distances.append(distances[head] + 1)
        head += 1
    return _Search(
        states=_immutable(np.asarray(states, dtype=np.int32)),
        indices=MappingProxyType(indices),
        parents=_immutable(np.asarray(parents, dtype=np.int32)),
        actions=_immutable(np.asarray(actions, dtype=np.int32)),
        distances=_immutable(np.asarray(distances, dtype=np.int32)),
    )


class ExactOracle:
    """Cached exact distances and shortest action sequences for a fixed rule.

    ``reachable_states`` contains relative states reached from identity in
    deterministic BFS order. ``distances`` is aligned with those rows. Both
    arrays are immutable. The first row is identity and has distance zero.
    """

    def __init__(self, rule: PermutationRule, *, max_states: int = MAX_EXACT_STATES) -> None:
        if not isinstance(rule, PermutationRule):
            raise TypeError("rule must be a PermutationRule")
        max_states = _integer(max_states, name="max_states")
        if not 1 <= max_states <= MAX_EXACT_STATES:
            raise ValueError(f"max_states must be in [1, {MAX_EXACT_STATES}]")
        self.rule = rule
        self.max_states = max_states
        self._search = _search(rule, max_states)

    @property
    def reachable_states(self) -> np.ndarray:
        return self._search.states

    @property
    def distances(self) -> np.ndarray:
        return self._search.distances

    @property
    def diameter(self) -> int:
        """Exact diameter of any component, using token-label symmetry."""
        return int(self._search.distances.max())

    def _index(self, state: object, goal: object) -> int | None:
        state = self.rule.validate_state(state)
        goal = self.rule.validate_state(goal, name="goal")
        positions = np.empty(self.rule.n_tokens + 1, dtype=np.int32)
        positions[state] = np.arange(1, self.rule.n_tokens + 1, dtype=np.int32)
        relative_goal = tuple(positions[goal].tolist())
        return self._search.indices.get(relative_goal)

    def distance(self, state: object, goal: object) -> int | None:
        """Return the exact atomic-operation distance, or None if unreachable."""
        index = self._index(state, goal)
        return None if index is None else int(self._search.distances[index])

    def solve(self, state: object, goal: object) -> list[int] | None:
        """Return a shortest ordered action list, [] if solved, else None."""
        index = self._index(state, goal)
        if index is None:
            return None
        actions = []
        while index:
            actions.append(int(self._search.actions[index]))
            index = int(self._search.parents[index])
        actions.reverse()
        return actions
