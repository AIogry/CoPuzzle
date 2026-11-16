"""Evaluator-only finite graphs and deterministic leakage-aware partitions."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from types import MappingProxyType

import numpy as np

from .spec import PROTOCOL, REGIMES, canonical, instance, integer


def immutable(array):
    array = np.asarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class Graph:
    task: str
    states: np.ndarray
    successors: np.ndarray  # Native action columns; -1 means illegal.
    indices: object

    @property
    def n_actions(self):
        return self.successors.shape[1]

    def locate(self, state) -> int:
        array = np.asarray(state)
        if array.shape != self.states.shape[1:] or array.dtype.kind not in "iu":
            raise ValueError("state must be an integer board of the declared width")
        try:
            return self.indices[tuple(array.tolist())]
        except KeyError as error:
            raise ValueError("state is outside the benchmark's canonical component") from error


@lru_cache(maxsize=6)
def graph(task: str) -> Graph:
    """Enumerate the canonical component without altering any native game API."""
    from copuzzle import get_rule
    spec = instance(task)
    rule = get_rule(task)
    if spec.family == "lights_out":
        root = (0,) * rule.n_buttons
        effects = rule.toggle_matrix.T.tolist()

        def moves(state):
            return ((a, tuple(x ^ y for x, y in zip(state, effect))) for a, effect in enumerate(effects))

        n_actions = rule.n_buttons
    elif spec.family in ("pancake", "topspin"):
        root = tuple(range(1, rule.n_tokens + 1))
        maps = [(rule.transition(root, a) - 1).tolist() for a in range(rule.n_actions)]

        def moves(state):
            return ((a, tuple(state[i] for i in positions)) for a, positions in enumerate(maps))

        n_actions = rule.n_actions
    else:
        root = tuple(rule.canonical_goal.tolist())
        moves = rule._successors
        n_actions = rule.n_actions
    states, indices, rows = [root], {root: 0}, []
    for state in states:
        row = [-1] * n_actions
        for action, successor in moves(state):
            if successor not in indices:
                if len(states) >= spec.component_states:
                    raise ValueError("reachable component exceeds the protocol's exact state count")
                indices[successor] = len(states)
                states.append(successor)
            row[action] = indices[successor]
        rows.append(row)
    if len(states) != spec.component_states:
        raise ValueError("reachable component size differs from the protocol")
    return Graph(task, immutable(np.asarray(states, dtype=np.int32)),
                 immutable(np.asarray(rows, dtype=np.int32)), MappingProxyType(indices))


def distances(g: Graph, goal: int, allowed: np.ndarray | None = None) -> np.ndarray:
    """BFS on an undirected graph, optionally restricted to symmetric edges."""
    result = np.full(len(g.states), -1, dtype=np.int32)
    result[goal] = 0
    queue = deque([goal])
    while queue:
        source = queue.popleft()
        for action, successor in enumerate(g.successors[source]):
            if successor < 0 or result[successor] >= 0 or (allowed is not None and not allowed[source, action]):
                continue
            result[successor] = result[source] + 1
            queue.append(int(successor))
    return result


@lru_cache(maxsize=32)
def exact_distances(task: str, goal: int) -> np.ndarray:
    return immutable(distances(graph(task), goal))


@lru_cache(maxsize=6)
def checked_diameter(task: str) -> int:
    g, spec = graph(task), instance(task)
    if spec.family in ("lights_out", "pancake", "topspin"):
        # XOR translation or token relabeling acts transitively on this component.
        diameter = int(exact_distances(task, 0).max())
    else:
        # Arbitrary Sliding blank locations / Hanoi configurations are not transitive.
        diameter = max(int(distances(g, i).max()) for i in range(len(g.states)))
    if diameter != spec.diameter:
        raise ValueError(f"diameter mismatch: {task}: expected {spec.diameter}, measured {diameter}")
    return diameter


def _bucket(seed: int, task: str, key) -> int:
    value = int.from_bytes(hashlib.sha256(canonical([PROTOCOL, seed, task, key])).digest()[:8], "big") % 100
    return 0 if value < 70 else 1 if value < 85 else 2


@dataclass(frozen=True)
class Partition:
    graph: Graph
    regime: str
    seed: int
    action_map: np.ndarray  # Opaque ID -> native ID. Evaluator-only.
    state_split: np.ndarray
    owners: np.ndarray  # Native columns; -1 marks a discarded cross-split transition.

    def queries(self, split: int) -> np.ndarray:
        # Return opaque actions in the learner-facing tables.
        return np.argwhere(self.owners[:, self.action_map] == split).astype(np.int32)

    def successor(self, source: int, action: int) -> tuple[int, bool]:
        target = int(self.graph.successors[source, self.action_map[action]])
        return (source, False) if target < 0 else (target, True)


@lru_cache(maxsize=12)
def partition(task: str, regime: str, seed: int) -> Partition:
    seed = integer(seed, "seed")
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}")
    g = graph(task)
    action_map = sorted(range(g.n_actions), key=lambda a: hashlib.sha256(canonical([PROTOCOL, seed, task, "action", a])).digest())
    labels = [_bucket(seed, task, ["state", s.tolist()]) for s in g.states]
    owners = np.full_like(g.successors, -1, dtype=np.int8)
    for source, row in enumerate(g.successors):
        for action, target in enumerate(row):
            if regime == "state_holdout":
                if target < 0 or labels[source] == labels[target]:
                    owners[source, action] = labels[source]
            elif target < 0:
                owners[source, action] = _bucket(seed, task, ["illegal", g.states[source].tolist(), action])
            else:
                # Group both directions AND every action reaching the same neighbor.
                edge = sorted((g.states[source].tolist(), g.states[target].tolist()))
                owners[source, action] = _bucket(seed, task, ["edge", *edge])
    return Partition(g, regime, seed, immutable(np.asarray(action_map, dtype=np.int32)),
                     immutable(np.asarray(labels, dtype=np.int8)), immutable(owners))


def training_components(p: Partition) -> list[int]:
    """Sizes of empirical legal-transition components, including isolated sources."""
    allowed = p.owners == 0
    seen = set()
    sizes = []
    sources = np.flatnonzero(np.any(allowed, axis=1))
    for root in sources:
        if int(root) in seen:
            continue
        queue = [int(root)]
        seen.add(int(root))
        for source in queue:
            for action, target in enumerate(p.graph.successors[source]):
                if target >= 0 and allowed[source, action] and int(target) not in seen:
                    queue.append(int(target))
                    seen.add(int(target))
        sizes.append(len(queue))
    return sorted(sizes, reverse=True)
