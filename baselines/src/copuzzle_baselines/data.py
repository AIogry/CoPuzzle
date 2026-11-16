"""Training-only adaptation of one-step experience, without rule/mask queries."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import Config, canonical, fingerprint, integer, number


def immutable(value):
    a = np.asarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class Vocabulary:
    symbols: np.ndarray
    width: int

    @property
    def size(self):
        return len(self.symbols)

    def encode(self, boards):
        a = np.asarray(boards)
        if a.ndim != 2 or a.shape[1] != self.width or a.dtype.kind not in "iu":
            raise ValueError("boards must be a batched integer matrix with the fitted width")
        ids = np.searchsorted(self.symbols, a)
        known = (ids < self.size) & (self.symbols[np.minimum(ids, self.size - 1)] == a)
        # Unknown symbols receive a fixed input-only category; never fit on queries.
        return np.where(known, ids, self.size).astype(np.int32)

    def decode(self, codes):
        a = np.asarray(codes)
        if a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a >= self.size):
            raise ValueError("predictions must use fitted output symbols")
        return self.symbols[a].copy()


@dataclass(frozen=True)
class TrainingSet:
    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    legal: np.ndarray
    n_actions: int
    vocabulary: Vocabulary
    goal_pool: np.ndarray
    source_goal_ids: np.ndarray
    progress_rows: np.ndarray
    selected_rows: np.ndarray
    identity: dict

    @property
    def width(self):
        return self.states.shape[1]

    @property
    def digest(self):
        return fingerprint(self.identity)


def _group_hash(state, successor, action, legal, seed):
    group = ["edge", *sorted((state.tolist(), successor.tolist()))] if legal else ["rejected", state.tolist(), int(action)]
    return int.from_bytes(hashlib.sha256(canonical(["copuzzle.coverage.v1", seed, group])).digest()[:8], "big")


def load_training(learner_directory, *, coverage=1.0, coverage_seed=0):
    """Read only metadata.json and train.npz, even when probes exist nearby."""
    from copuzzle.research.data import load_training as load_core, metadata

    path = Path(learner_directory)
    coverage = number(coverage, "coverage", 0, 1, open_low=True)
    coverage_seed = integer(coverage_seed, "coverage_seed")
    meta = metadata(path)
    arrays = load_core(path)
    observations = arrays["observations"]
    bounds = np.iinfo(np.int32)
    if np.any(observations < bounds.min) or np.any(observations > bounds.max):
        raise ValueError("board values do not fit the lossless int32 encoding")
    states, successors = observations[:, 0].astype(np.int32), observations[:, 1].astype(np.int32)
    actions, legal = arrays["actions"][:, 0].astype(np.int32), arrays["legal"][:, 0].astype(np.uint8)
    keys = np.concatenate((states, actions[:, None]), axis=1)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("each state-action query must occur exactly once")
    threshold = int(coverage * (1 << 64))
    selected = np.asarray([i for i, (s, t, a, flag) in enumerate(zip(states, successors, actions, legal))
                           if coverage == 1 or _group_hash(s, t, a, flag, coverage_seed) < threshold], dtype=np.int32)
    if not len(selected):
        raise ValueError("coverage selection is empty; no automatic backfilling")
    states, successors, actions, legal = states[selected], successors[selected], actions[selected], legal[selected]
    symbols = np.unique(np.concatenate((states.ravel(), successors.ravel())))
    vocabulary = Vocabulary(immutable(symbols), states.shape[1])
    encoded, following = vocabulary.encode(states), vocabulary.encode(successors)
    pool = np.unique(np.concatenate((encoded, following), axis=0), axis=0)
    if len(pool) < 2:
        raise ValueError("goal relabeling requires at least two observed training states")
    lookup = {tuple(s.tolist()): i for i, s in enumerate(pool)}
    source_ids = np.asarray([lookup[tuple(s.tolist())] for s in encoded], dtype=np.int32)
    progress = np.flatnonzero(np.any(encoded != following, axis=1)).astype(np.int32)
    identity = {
        "schema": "copuzzle.baseline_training.v1", "protocol": meta["protocol"],
        "task": meta["task"], "regime": meta["regime"], "n_actions": meta["n_actions"],
        "state_width": states.shape[1], "symbols": symbols.tolist(),
        "train_sha256": hashlib.sha256((path / "train.npz").read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256((path / "metadata.json").read_bytes()).hexdigest(),
        "coverage": coverage, "coverage_seed": coverage_seed,
        "coverage_unit": "observed unordered legal edge or rejected state-action query",
        "selected_rows_sha256": hashlib.sha256(selected.astype("<i4").tobytes()).hexdigest(),
        "available_rows": len(keys), "selected_rows": len(selected),
        "progress_rows": len(progress), "rejected_rows": int(np.sum(legal == 0)),
        "observed_goal_states": len(pool), "goal_pool": "unique states in selected training inputs and successors",
        "legality_supervision": "selected attempted action only; other actions unknown",
        "actions_observed": np.bincount(actions, minlength=meta["n_actions"]).tolist(),
    }
    return TrainingSet(*(immutable(x) for x in (encoded, following, actions, legal)),
                       meta["n_actions"], vocabulary, immutable(pool), immutable(source_ids),
                       immutable(progress), immutable(selected), identity)


class Sampler:
    """Goal equality labels, collection-cutoff bootstrap and replayable RNG state."""
    def __init__(self, data: TrainingSet, config: Config, seed: int):
        if data.identity["coverage"] != config.coverage or data.identity["coverage_seed"] != config.coverage_seed:
            raise ValueError("sampler configuration differs from the selected training corpus")
        if config.method == "gcbc" and not len(data.progress_rows):
            raise ValueError("GCBC has no nontrivial observed successor goals")
        self.data, self.config = data, config
        self.rng = np.random.default_rng(integer(seed, "sampler seed"))
        self.counts = {key: 0 for key in ("batches", "rows", "rejected_attempts", "actor_future_choices",
                                         "value_future_choices", "value_goal_successes")}

    def _goals(self, rows, probability):
        d = self.data
        # Uniform distinct observed goals, with no rejection loop or solved starts.
        ids = self.rng.integers(len(d.goal_pool) - 1, size=len(rows))
        ids += ids >= d.source_goal_ids[rows]
        goals = d.goal_pool[ids].copy()
        eligible = np.any(d.states[rows] != d.next_states[rows], axis=1)
        future = (self.rng.random(len(rows)) < probability) & eligible
        goals[future] = d.next_states[rows[future]]
        return goals, future

    def sample(self, size=None):
        d, c = self.data, self.config
        size = c.batch_size if size is None else integer(size, "batch size", 1)
        if c.method == "gcbc":
            rows = self.rng.choice(d.progress_rows, size=size)
            self.counts["batches"] += 1
            self.counts["rows"] += size
            self.counts["actor_future_choices"] += size
            return {"states": d.states[rows], "actions": d.actions[rows], "actor_goals": d.next_states[rows]}
        rows = self.rng.integers(len(d.states), size=size)
        self.counts["batches"] += 1
        self.counts["rows"] += size
        self.counts["rejected_attempts"] += int(np.sum(d.legal[rows] == 0))
        base = {"states": d.states[rows], "actions": d.actions[rows], "next_states": d.next_states[rows]}
        if c.method == "dynamics":
            return {**base, "legal": d.legal[rows].astype(np.float32)}
        value_goals, value_future = self._goals(rows, c.value_future_probability)
        actor_goals, actor_future = self._goals(rows, c.actor_future_probability)
        success = np.all(d.next_states[rows] == value_goals, axis=1)
        self.counts["value_future_choices"] += int(value_future.sum())
        self.counts["actor_future_choices"] += int(actor_future.sum())
        self.counts["value_goal_successes"] += int(success.sum())
        return {**base, "value_goals": value_goals, "actor_goals": actor_goals,
                "rewards": np.where(success, 0, -1).astype(np.float32),
                "bootstrap": (~success).astype(np.float32)}

    def state_dict(self):
        return {"schema": "copuzzle.baseline_sampler.v1", "data_sha256": self.data.digest,
                "config": self.config.to_dict(), "rng": deepcopy(self.rng.bit_generator.state), "counts": dict(self.counts)}

    def restore(self, state):
        if set(state) != {"schema", "data_sha256", "config", "rng", "counts"} or state["schema"] != "copuzzle.baseline_sampler.v1":
            raise ValueError("invalid sampler checkpoint")
        if state["data_sha256"] != self.data.digest or canonical(state["config"]) != canonical(self.config.to_dict()):
            raise ValueError("sampler checkpoint has different data or sampling semantics")
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(state["rng"])
        if set(state["counts"]) != set(self.counts):
            raise ValueError("sampler counter fields differ")
        counts = {key: integer(value, key) for key, value in state["counts"].items()}
        self.rng = rng
        self.counts = counts
