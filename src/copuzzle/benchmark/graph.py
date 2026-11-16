"""Evaluator-only graph distances and independently versioned split identities."""
from __future__ import annotations

from functools import lru_cache
import hashlib

import numpy as np

from copuzzle.research.graph import Graph, Partition, graph, immutable
from copuzzle.research.spec import REGIMES, canonical, instance, integer


PARTITION_RECIPE = "copuzzle.partition.v1"


def distances(g: Graph, goal: int, allowed=None, max_depth=None):
    """Exact vectorized BFS; -1 is unreachable or beyond an explicit depth bound."""
    goal = integer(goal, "goal index")
    if goal >= len(g.states):
        raise ValueError("goal index is outside the component")
    if max_depth is not None:
        max_depth = integer(max_depth, "max_depth")
    if allowed is not None and (np.asarray(allowed).dtype.kind != "b" or np.shape(allowed) != g.successors.shape):
        raise ValueError("allowed edges must be a boolean matrix aligned with successors")
    result = np.full(len(g.states), -1, dtype=np.int32)
    result[goal] = 0
    frontier = np.asarray([goal], dtype=np.int32)
    depth = 0
    while len(frontier) and (max_depth is None or depth < max_depth):
        edges = g.successors[frontier]
        candidates = edges.ravel() if allowed is None else edges[allowed[frontier]]
        candidates = np.unique(candidates[candidates >= 0])
        frontier = candidates[result[candidates] < 0]
        depth += 1
        result[frontier] = depth
    return result


@lru_cache(maxsize=6)
def _reference(task):
    g, spec = graph(task), instance(task)
    root_distance = distances(g, 0)
    if spec.family in ("lights_out", "pancake", "topspin"):
        base = 2 if spec.family == "lights_out" else g.states.shape[1] + 1
        powers = np.asarray([base**i for i in range(g.states.shape[1])], dtype=np.int64)
        codes = g.states @ powers
        table = np.full(base**g.states.shape[1], -1, dtype=np.int8)
        table[codes] = root_distance
        return immutable(powers), immutable(codes), immutable(table)
    # These graphs are small and do not share the transitive state symmetry.
    return immutable(np.stack([distances(g, i) for i in range(len(g.states))]))


@lru_cache(maxsize=64)
def _full_distances(task: str, goal: int):
    g, spec = graph(task), instance(task)
    reference = _reference(task)
    if spec.family == "lights_out":
        _, codes, table = reference
        result = table[codes ^ codes[goal]]
    elif spec.family in ("pancake", "topspin"):
        powers, _, table = reference
        inverse = np.zeros(g.states.shape[1] + 1, dtype=np.int32)
        inverse[g.states[goal]] = np.arange(1, g.states.shape[1] + 1)
        result = table[inverse[g.states] @ powers]
    else:
        result = reference[goal]
    if np.any(result < 0):
        raise ValueError("distance lookup left the declared canonical component")
    return immutable(np.asarray(result, dtype=np.int16))


def full_distances(task: str, goal: int):
    goal = integer(goal, "goal index")
    if goal >= len(graph(task).states):
        raise ValueError("goal index is outside the component")
    return _full_distances(task, goal)


def _bucket(seed, task, key):
    number = int.from_bytes(hashlib.sha256(canonical([PARTITION_RECIPE, seed, task, key])).digest()[:8], "big") % 100
    return 0 if number < 70 else 1 if number < 85 else 2


@lru_cache(maxsize=12)
def _partition(task, regime, seed):
    g = graph(task)
    actions = sorted(range(g.n_actions), key=lambda a: hashlib.sha256(canonical([PARTITION_RECIPE, seed, task, "action", a])).digest())
    labels = np.asarray([_bucket(seed, task, ["state", state.tolist()]) for state in g.states], dtype=np.int8)
    owners = np.full_like(g.successors, -1, dtype=np.int8)
    for source, row in enumerate(g.successors):
        for action, successor in enumerate(row):
            if regime == "state_holdout":
                if successor < 0 or labels[source] == labels[successor]:
                    owners[source, action] = labels[source]
            elif successor < 0:
                owners[source, action] = _bucket(seed, task, ["illegal", g.states[source].tolist(), action])
            else:
                edge = sorted((g.states[source].tolist(), g.states[successor].tolist()))
                owners[source, action] = _bucket(seed, task, ["edge", *edge])
    return Partition(g, regime, seed, immutable(np.asarray(actions, dtype=np.int32)),
                     immutable(labels), immutable(owners))


def partition(task, regime, seed):
    seed = integer(seed, "seed")
    if regime not in REGIMES:
        raise ValueError("unknown partition condition")
    return _partition(task, regime, seed)
