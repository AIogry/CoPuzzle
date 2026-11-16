"""Offline position-map lookup: a fixed-instance diagnostic, not an RL agent.

Each recorded state contains unique tokens, so a transition identifies the
position permutation for its operation. Composition applies those learned maps
in order; XOR/parity would be incorrect for these noncommuting games.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def dataset_identity(dataset):
    """Bind metadata and actual array contents, independently of NPZ compression."""
    digest = hashlib.sha256(json.dumps(dataset.manifest, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode())
    for name in ("states", "actions", "events"):
        array = getattr(dataset, name)
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes(order="C"))
    return {"content_sha256": digest.hexdigest(), "manifest": dict(dataset.manifest)}


class PositionMapLookup:
    def __init__(self, maps, counts, rule_id):
        self.maps = np.array(maps, dtype=np.int64, copy=True)
        self.counts = np.array(counts, dtype=np.int64, copy=True)
        self.rule_id = rule_id

    @classmethod
    def fit(cls, dataset):
        dataset.validate()
        maps = np.full((dataset.n_actions, dataset.n_tokens), -1, dtype=np.int64)
        counts = np.zeros(dataset.n_actions, dtype=np.int64)
        for before, after, action in zip(
            dataset.states[:, :-1].reshape(-1, dataset.n_tokens),
            dataset.states[:, 1:].reshape(-1, dataset.n_tokens),
            dataset.actions.reshape(-1),
        ):
            inferred = np.argsort(before)[after - 1]
            if counts[action] and not np.array_equal(maps[action], inferred):
                raise ValueError("Recorded transitions disagree on an operation position map")
            maps[action] = inferred
            counts[action] += 1
        return cls(maps, counts, dataset.manifest["rule_id"])

    def predict(self, board, action):
        if (isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer))
                or not 0 <= action < len(self.maps)):
            raise ValueError("action must be a legal integer operation ID")
        if not self.counts[action]:
            raise ValueError("Operation is absent from the fitted offline data")
        board = np.asarray(board)
        if (board.shape != (self.maps.shape[1],) or board.dtype.kind not in "iu"
                or not np.array_equal(np.sort(board), np.arange(1, len(board) + 1))):
            raise ValueError("board must be a token permutation")
        return board[self.maps[action]].copy()


def evaluate_lookup(train, test, *, lengths=(2, 4, 8), num_examples=128, seed=0):
    """Score recorded single steps and windows, retaining sampled provenance."""
    train.validate()
    test.validate()
    if train.manifest["rule_id"] != test.manifest["rule_id"]:
        raise ValueError("Lookup evaluation requires the same rule in train and test")
    if type(num_examples) is not int or num_examples < 1:
        raise ValueError("num_examples must be a positive integer")
    lengths = tuple(lengths)
    if not lengths or any(type(length) is not int or not 1 <= length <= test.horizon for length in lengths):
        raise ValueError("Each sequence length must fit within the test horizon")
    lookup = PositionMapLookup.fit(train)
    if np.any(lookup.counts == 0):
        raise ValueError("Training data must cover every operation")
    rng = np.random.default_rng(seed)
    predicted = np.stack([
        lookup.predict(state, action)
        for state, action in zip(test.states[:, :-1].reshape(-1, test.n_tokens), test.actions.flat)
    ])
    truth = test.states[:, 1:].reshape(-1, test.n_tokens)
    sequences = {}
    for length in lengths:
        episodes = rng.integers(test.num_episodes, size=num_examples)
        starts = rng.integers(test.horizon - length + 1, size=num_examples)
        exact = np.empty((num_examples, length), dtype=bool)
        for row, (episode, start) in enumerate(zip(episodes, starts)):
            state = test.states[episode, start].copy()
            for offset in range(length):
                state = lookup.predict(state, test.actions[episode, start + offset])
                exact[row, offset] = np.array_equal(state, test.states[episode, start + offset + 1])
        sequences[str(length)] = {
            "endpoint_exact_accuracy": float(exact[:, -1].mean()),
            "all_steps_exact_accuracy": float(exact.all(axis=1).mean()),
            "per_step_exact_accuracy": exact.mean(axis=0).tolist(),
            "episode_indices": episodes.tolist(), "start_indices": starts.tolist(),
        }
    return {
        "schema_version": "copuzzle.permutation_lookup.v1",
        "rule_id": lookup.rule_id, "is_rl": False, "privileged_rule_access": False,
        "training_data": dataset_identity(train), "validation_data": dataset_identity(test),
        "method": "offline_position_map_lookup", "seed": seed,
        "operation_counts": lookup.counts.tolist(),
        "single_step": {"exact_accuracy": float(np.all(predicted == truth, axis=1).mean()),
                        "token_accuracy": float(np.mean(predicted == truth))},
        "sequences": sequences,
        "scope": "Recorded validation windows; no held-out composition or learned-transfer claim",
    }


def dataset_coverage(dataset):
    """Descriptive observed coverage; never assumes all permutations reachable."""
    states = dataset.states.reshape(-1, dataset.n_tokens)
    state_actions = np.concatenate((dataset.states[:, :-1].reshape(-1, dataset.n_tokens),
                                    dataset.actions.reshape(-1, 1)), axis=1)
    return {
        "episodes": dataset.num_episodes, "horizon": dataset.horizon,
        "transitions": int(dataset.actions.size),
        "unique_states": len(np.unique(states, axis=0)),
        "unique_state_actions": len(np.unique(state_actions, axis=0)),
        "operation_counts": np.bincount(dataset.actions.flat, minlength=dataset.n_actions).tolist(),
    }
