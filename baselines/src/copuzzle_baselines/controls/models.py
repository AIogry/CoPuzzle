"""Fit controls using retained training records, never querying puzzle mechanics."""
from __future__ import annotations

import numpy as np

from ..data import load_training


class Lookup:
    def __init__(self, data):
        self.rows = {(tuple(s.tolist()), int(a)): (t.copy(), bool(legal))
                     for s, a, t, legal in zip(data.vocabulary.decode(data.states), data.actions,
                                               data.vocabulary.decode(data.next_states), data.legal)}

    def predict(self, states, actions):
        values = [self.rows.get((tuple(s.tolist()), int(a)), (s, False)) for s, a in zip(states, actions)]
        return np.asarray([t for t, _ in values], np.int32), np.asarray([flag for _, flag in values], np.uint8)


class FittedOperators:
    """Explicit prior: globally applicable constant XOR or position operators."""
    def __init__(self, data, kind):
        self.kind = kind
        self.operators = {}
        states, successors = data.vocabulary.decode(data.states), data.vocabulary.decode(data.next_states)
        if not np.all(data.legal):
            raise ValueError("the globally-applicable operator prior does not fit rejected training attempts")
        for action in range(data.n_actions):
            rows = np.flatnonzero(data.actions == action)
            if not len(rows):
                continue
            if kind == "fitted_xor":
                if np.any((states[rows] != 0) & (states[rows] != 1)) or np.any((successors[rows] != 0) & (successors[rows] != 1)):
                    raise ValueError("XOR prior requires binary observations")
                effects = states[rows] ^ successors[rows]
                if not np.all(effects == effects[0]):
                    raise ValueError("observations contradict a constant XOR operator")
                operator = effects[0].copy()
            elif kind == "fitted_position":
                first = int(rows[0])
                if len(np.unique(states[first])) != data.width:
                    raise ValueError("position-map fitting requires distinct token labels")
                indices = {int(value): i for i, value in enumerate(states[first])}
                try:
                    operator = np.asarray([indices[int(value)] for value in successors[first]], np.int32)
                except KeyError as error:
                    raise ValueError("observed output contains a new token") from error
                if len(np.unique(operator)) != data.width or not np.array_equal(states[rows][:, operator], successors[rows]):
                    raise ValueError("observations contradict a constant permutation operator")
            else:
                raise ValueError("unknown structural prior")
            operator.flags.writeable = False
            self.operators[action] = operator

    def predict(self, states, actions):
        states = np.asarray(states)
        result, legal = states.copy(), np.zeros(len(states), np.uint8)
        for i, action in enumerate(actions):
            operator = self.operators.get(int(action))
            if operator is not None:
                result[i] = states[i] ^ operator if self.kind == "fitted_xor" else states[i, operator]
                legal[i] = 1
        return result, legal


def fit_control(learner, kind, *, coverage=1.0, coverage_seed=0):
    data = load_training(learner, coverage=coverage, coverage_seed=coverage_seed)
    if kind == "lookup":
        model = Lookup(data)
    elif kind in ("fitted_xor", "fitted_position"):
        task = data.identity["task"]
        if kind == "fitted_xor" and not task.startswith("grid-"):
            raise ValueError("XOR structural control is declared only for the Lights Out instances")
        if kind == "fitted_position" and not task.startswith(("pancake-", "topspin-")):
            raise ValueError("position structural control is declared only for Pancake/TopSpin")
        model = FittedOperators(data, kind)
    else:
        raise ValueError("unknown data-only control")
    return model, data
