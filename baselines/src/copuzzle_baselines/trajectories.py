"""Development training on checked multi-step symbolic experience.

This adapter retains native episode boundaries and action IDs. It is deliberately
separate from the frozen single-query benchmark and accepts no evaluation labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .config import integer, number
from .data import Sampler, TrainingSet, Vocabulary, immutable


@dataclass(frozen=True)
class TrajectoryTrainingSet(TrainingSet):
    trajectory_states: np.ndarray

    @property
    def horizon(self):
        return self.trajectory_states.shape[1] - 1


def load_trajectories(path, *, family, coverage=1.0, coverage_seed=0):
    """Adapt one native train archive, without exposing masks or rule metadata.

    Repeated transitions remain repeated experience. Partial edge coverage would
    destroy observed paths, so this interface rejects it rather than inventing
    a new trajectory subsampling protocol. Native readers verify file checksums
    and static schema; they are never given a rule for transition validation.
    """
    coverage = number(coverage, "coverage", 0, 1, open_low=True)
    coverage_seed = integer(coverage_seed, "coverage_seed")
    if coverage != 1:
        raise ValueError("multi-step trajectory training requires full coverage")
    if family == "lights_out":
        from copuzzle.games.lights_out.data import load_dataset
    elif family == "permutation":
        from copuzzle.games.permutation.data import load_dataset
    elif family == "classic":
        from copuzzle.games.classic.data import load_dataset
    else:
        raise ValueError("trajectory family must be lights_out, permutation or classic")
    path = Path(path).expanduser().absolute()
    if path.suffix != ".npz":
        raise ValueError("trajectory archive must have the .npz suffix")
    # Bind the exact files which the checked reader consumes, detecting concurrent
    # replacement instead of associating loaded arrays with later file contents.
    paths = (path, path.with_suffix(".json"))
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    native = load_dataset(path)
    if hashes != [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]:
        raise ValueError("trajectory archive changed while loading")
    if native.manifest["split"] != "train":
        raise ValueError("trajectory training requires a source with split='train'")
    observations = native.states
    bounds = np.iinfo(np.int32)
    if np.any(observations < bounds.min) or np.any(observations > bounds.max):
        raise ValueError("board values do not fit the lossless int32 encoding")
    symbols = np.unique(observations).astype(np.int32)
    width = observations.shape[-1]
    vocabulary = Vocabulary(immutable(symbols), width)
    encoded = vocabulary.encode(observations.reshape(-1, width)).reshape(observations.shape)
    states = encoded[:, :-1].reshape(-1, width)
    successors = encoded[:, 1:].reshape(-1, width)
    actions = native.actions.reshape(-1).astype(np.int32)
    n_actions = native.events.shape[-1]
    # Native symbolic archives contain executed legal operations only. Classic's
    # full masks are checked by its reader but are not copied into learner data.
    legal = np.ones(len(actions), dtype=np.uint8)
    pool = np.unique(encoded.reshape(-1, width), axis=0)
    if len(pool) < 2:
        raise ValueError("goal relabeling requires at least two observed training states")
    lookup = {tuple(s.tolist()): i for i, s in enumerate(pool)}
    source_ids = np.asarray([lookup[tuple(s.tolist())] for s in states], dtype=np.int32)
    # A row has a distinct future iff some subsequent adjacent pair changes.
    # This O(E L) eligibility cache avoids materializing O(E L^2) future pairs.
    changes = np.any(encoded[:, :-1] != encoded[:, 1:], axis=-1)
    eligible = np.logical_or.accumulate(changes[:, ::-1], axis=1)[:, ::-1]
    progress = np.flatnonzero(eligible.reshape(-1)).astype(np.int64)
    selected = np.arange(len(actions), dtype=np.int64)
    identity = {
        "schema": "copuzzle.trajectory_baseline_training.v1",
        "protocol": "copuzzle.symbolic-trajectories.dev1", "purpose": "development",
        "task": native.manifest["task_name"], "regime": "native_trajectory_split",
        "family": family, "source_schema": native.manifest["schema_version"],
        "n_actions": n_actions, "state_width": width, "symbols": symbols.tolist(),
        "train_sha256": hashes[0], "metadata_sha256": hashes[1],
        "coverage": coverage, "coverage_seed": coverage_seed,
        "coverage_unit": "all recorded transitions; no subsampling",
        "available_rows": len(actions), "selected_rows": len(actions),
        "selected_rows_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
        "episodes": native.num_episodes, "horizon": native.horizon,
        "progress_rows": len(progress), "progress_definition": "has a strictly later distinct board in the same episode",
        "rejected_rows": 0, "observed_goal_states": len(pool),
        "goal_pool": "unique states observed in training episodes including final frames",
        "future_goals": "uniform later distinct-state indices in the same episode; repeated indices retain multiplicity",
        "legality_supervision": "recorded executed actions only; no unattempted-action masks",
        "action_encoding": "native archive IDs; not frozen benchmark opaque IDs",
        "actions_observed": np.bincount(actions, minlength=n_actions).tolist(),
    }
    return TrajectoryTrainingSet(
        *(immutable(x) for x in (states, successors, actions, legal)),
        n_actions, vocabulary, immutable(pool), immutable(source_ids),
        immutable(progress), immutable(selected), identity, immutable(encoded))


class TrajectorySampler(Sampler):
    """Uniform observed future indices, with checkpointable independent mixtures.

    Future candidates are scanned only for sampled rows, using O(L * width) temporary
    memory per row rather than caching all O(L^2) pairs. Goal rewards always use
    successor equality, including when the chosen future index is a later repeat.
    """

    def __init__(self, data: TrajectoryTrainingSet, config, seed):
        if not isinstance(data, TrajectoryTrainingSet):
            raise TypeError("TrajectorySampler requires trajectory training data")
        super().__init__(data, config, seed)

    def _goals(self, rows, probability):
        d = self.data
        ids = self.rng.integers(len(d.goal_pool) - 1, size=len(rows))
        ids += ids >= d.source_goal_ids[rows]
        goals = d.goal_pool[ids].copy()
        future = self.rng.random(len(rows)) < probability
        for position in np.flatnonzero(future):
            episode, step = divmod(int(rows[position]), d.horizon)
            tail = d.trajectory_states[episode, step + 1:]
            candidates = np.flatnonzero(np.any(tail != d.states[rows[position]], axis=1))
            if len(candidates):
                goals[position] = tail[candidates[self.rng.integers(len(candidates))]]
            else:
                future[position] = False
        return goals, future

    def sample(self, size=None):
        if self.config.method != "gcbc":
            return super().sample(size)
        size = self.config.batch_size if size is None else integer(size, "batch size", 1)
        rows = self.rng.choice(self.data.progress_rows, size=size)
        goals, future = self._goals(rows, 1.0)
        self.counts["batches"] += 1
        self.counts["rows"] += size
        self.counts["actor_future_choices"] += int(future.sum())
        return {"states": self.data.states[rows], "actions": self.data.actions[rows], "actor_goals": goals}
