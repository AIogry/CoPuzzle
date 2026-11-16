"""Evaluator-side observed-graph context for the actual selected training view."""
from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

import numpy as np


def observed_graph_context(states, successors, legal, starts, goals):
    """Use recorded legal edges only; never consult a simulator or true graph.

    CoPuzzle's selected legal edges are inverse-closed. Verify this property
    before using connected components for reachability; use directed BFS if it
    does not hold. A weak component alone is not proof of a directed path.
    """
    states, successors, legal, starts, goals = map(np.asarray, (states, successors, legal, starts, goals))
    if (states.ndim != 2 or successors.shape != states.shape or legal.shape != (len(states),)
            or starts.ndim != 2 or goals.shape != starts.shape or starts.shape[1] != states.shape[1]
            or any(a.dtype.kind not in "iu" for a in (states, successors, starts, goals))
            or np.any((legal != 0) & (legal != 1))):
        raise ValueError("invalid observed-graph arrays")
    identifiers = {}
    def vertex(state):
        key = tuple(state.tolist())
        if key not in identifiers:
            identifiers[key] = len(identifiers)
        return identifiers[key]
    sources, destinations = [vertex(s) for s in states], [vertex(s) for s in successors]
    edges = {(s, t) for s, t, ok in zip(sources, destinations, legal) if ok}
    symmetric = all((t, s) in edges for s, t in edges)
    parent = list(range(len(identifiers)))
    size = [1]*len(parent)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    adjacency = {i: set() for i in range(len(parent))}
    for s, t in edges:
        adjacency[s].add(t)
        a, b = find(s), find(t)
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
    components = Counter(find(i) for i in range(len(parent)))
    reached = {}
    flags, observed_starts, observed_goals = [], [], []
    for source, target in zip(starts, goals):
        s, t = identifiers.get(tuple(source.tolist())), identifiers.get(tuple(target.tolist()))
        observed_starts.append(s is not None)
        observed_goals.append(t is not None)
        if s is None or t is None:
            flags.append(False)
        elif symmetric:
            flags.append(find(s) == find(t))
        else:
            if s not in reached:
                seen, queue = {s}, deque([s])
                while queue:
                    for next_state in adjacency[queue.popleft()]:
                        if next_state not in seen:
                            seen.add(next_state)
                            queue.append(next_state)
                reached[s] = seen
            flags.append(t in reached[s])
    return {"selected_queries": len(states), "observed_states": len(identifiers),
            "observed_legal_directed_edges": len(edges), "legal_edges_inverse_closed": symmetric,
            "weak_components": len(components), "largest_weak_component": max(components.values(), default=0),
            "case_start_observed": observed_starts, "case_goal_observed": observed_goals,
            "case_observed_path_exists": flags, "cases_with_observed_paths": sum(flags)}


def goal_support(learner, *, split, coverage, coverage_seed):
    from ..data import load_training
    from ..protocols import public_api
    learner = Path(learner)
    training = load_training(learner, coverage=coverage, coverage_seed=coverage_seed)
    queries = public_api(learner).load_probe(learner, "G", split)
    result = observed_graph_context(training.vocabulary.decode(training.states), training.vocabulary.decode(training.next_states),
                                    training.legal, queries["board"], queries["goal"])
    return {"role": "observed training-view context, not a claim that a reference was fitted or that an unobserved path is impossible",
            "training_view_sha256": training.digest, "coverage": coverage, "coverage_seed": coverage_seed,
            "source": "selected training records and public G inputs only",
            "stratum_score_semantics": "case-pooled diagnostic within each support stratum; not the primary equal-tier G score",
            "frozen_G_reachability_flag": "original evaluation reports use the full retained corpus; this field uses the actual selected view",
            **result}


def goal_support_scores(context, cases):
    flags = context["case_observed_path_exists"]
    if cases is not None and (len(cases) != len(flags) or [c["case_index"] for c in cases] != list(range(len(flags)))):
        raise ValueError("goal support and evaluation cases are unaligned")
    result = {}
    for name, flag in (("observed_path", True), ("no_observed_path", False)):
        selected = [i for i, value in enumerate(flags) if value == flag]
        successes = sum(bool(cases[i]["success"]) for i in selected) if cases is not None else 0
        result[name] = {"cases": len(selected), "success_rate": successes/len(selected) if selected else None}
    return result
