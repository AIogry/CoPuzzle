"""Learner-only loading: no graph construction, oracle, or evaluator file reads."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .spec import PROTOCOL, REGIMES, instance, integer


def metadata(learner_directory) -> dict:
    value = json.loads((Path(learner_directory) / "metadata.json").read_text())
    if value.get("protocol") != PROTOCOL or value.get("regime") not in REGIMES:
        raise ValueError("unsupported learner metadata")
    spec = instance(value["task"])
    if integer(value["control_budget"], "control_budget", 1) != spec.control_budget:
        raise ValueError("changed control budget")
    integer(value["state_width"], "state_width", 1)
    integer(value["n_actions"], "n_actions", 1)
    return value


def _arrays(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(keys) or len(archive.files) != len(keys):
            raise ValueError("unexpected learner fields")
        result = {key: archive[key] for key in keys}
    if any(a.dtype.kind not in "iu" for a in result.values()):
        raise ValueError("learner arrays must have integer dtype")
    return result


def load_training(learner_directory) -> dict[str, np.ndarray]:
    meta = metadata(learner_directory)
    result = _arrays(Path(learner_directory) / "train.npz", ("observations", "actions", "legal"))
    observations, actions, legal = (result[k] for k in ("observations", "actions", "legal"))
    n = len(actions)
    if not n or observations.shape != (n, 2, meta["state_width"]) or actions.shape != (n, 1) or legal.shape != (n, 1):
        raise ValueError("training data must contain one-step episodes with two observations")
    if np.any(actions < 0) or np.any(actions >= meta["n_actions"]) or np.any(legal > 1) or np.any(legal < 0):
        raise ValueError("action or legality label out of bounds")
    invalid = legal[:, 0] == 0
    if not np.array_equal(observations[invalid, 0], observations[invalid, 1]):
        raise ValueError("rejected attempts must leave the board unchanged")
    return result


def load_probe(learner_directory, probe: str, split: str = "dev") -> dict[str, np.ndarray]:
    meta = metadata(learner_directory)
    if probe not in meta["probes"] or split not in ("dev", "test"):
        raise ValueError("unsupported probe/split for this regime")
    keys = {"R": ("board", "action"), "C": ("board", "actions", "lengths"), "G": ("board", "goal")}[probe]
    result = _arrays(Path(learner_directory) / f"{probe}_{split}.npz", keys)
    n = len(result["board"])
    if not n or result["board"].shape != (n, meta["state_width"]):
        raise ValueError("invalid probe board shape")
    if probe == "R":
        a = result["action"]
        if a.shape != (n,) or np.any(a < 0) or np.any(a >= meta["n_actions"]):
            raise ValueError("invalid rule actions")
    elif probe == "G":
        if result["goal"].shape != result["board"].shape:
            raise ValueError("invalid goal shape")
    else:
        actions, lengths = result["actions"], result["lengths"]
        if actions.ndim != 2 or len(actions) != n or lengths.shape != (n,):
            raise ValueError("invalid composition shape")
        if np.any(lengths < 1) or np.any(lengths > actions.shape[1]):
            raise ValueError("invalid composition lengths")
        active = np.arange(actions.shape[1])[None, :] < lengths[:, None]
        if np.any(actions[~active] != -1) or np.any(actions[active] < 0) or np.any(actions[active] >= meta["n_actions"]):
            raise ValueError("invalid actions or composition padding")
    return result
