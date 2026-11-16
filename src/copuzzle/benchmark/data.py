"""Public learner arrays only; importing this module does not build true graphs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from copuzzle.research.data import _arrays
from . import PROTOCOL
from .spec import REGIMES, instance, integer


def metadata(directory):
    value = json.loads((Path(directory) / "metadata.json").read_text())
    if value.get("protocol") != PROTOCOL or value.get("regime") not in REGIMES:
        raise ValueError("unsupported benchmark metadata")
    if value.get("purpose") not in ("development", "benchmark"):
        raise ValueError("unsupported artifact purpose")
    spec = instance(value["task"])
    if integer(value["control_budget"], "control_budget", 1) != spec.control_budget:
        raise ValueError("changed control budget")
    integer(value["state_width"], "state_width", 1)
    integer(value["n_actions"], "n_actions", 1)
    return value


def load_training(directory):
    meta = metadata(directory)
    result = _arrays(Path(directory) / "train.npz", ("observations", "actions", "legal"))
    obs, actions, legal = (result[k] for k in ("observations", "actions", "legal"))
    n = len(actions)
    if not n or obs.shape != (n, 2, meta["state_width"]) or actions.shape != (n, 1) or legal.shape != (n, 1):
        raise ValueError("invalid one-step training shapes")
    if np.any(actions < 0) or np.any(actions >= meta["n_actions"]) or np.any((legal != 0) & (legal != 1)):
        raise ValueError("invalid action or legality label")
    rejected = legal[:, 0] == 0
    if not np.array_equal(obs[rejected, 0], obs[rejected, 1]):
        raise ValueError("rejected attempts must preserve the board")
    return result


def load_probe(directory, probe, split="dev"):
    meta = metadata(directory)
    if probe not in ("R", "C", "G") or split not in ("dev", "test"):
        raise ValueError("unknown probe or split")
    fields = {"R": ("board", "action"), "C": ("board", "actions", "lengths"), "G": ("board", "goal")}[probe]
    result = _arrays(Path(directory) / f"{probe}_{split}.npz", fields)
    boards = result["board"]
    if boards.ndim != 2 or not len(boards) or boards.shape[1] != meta["state_width"]:
        raise ValueError("invalid query board shape")
    if probe == "R":
        actions = result["action"]
        if actions.shape != (len(boards),) or np.any(actions < 0) or np.any(actions >= meta["n_actions"]):
            raise ValueError("invalid R actions")
    elif probe == "G":
        if result["goal"].shape != boards.shape:
            raise ValueError("invalid G goals")
    else:
        words, lengths = result["actions"], result["lengths"]
        if words.ndim != 2 or len(words) != len(boards) or lengths.shape != (len(boards),):
            raise ValueError("invalid C shapes")
        if np.any(lengths < 1) or np.any(lengths > words.shape[1]):
            raise ValueError("invalid C lengths")
        active = np.arange(words.shape[1])[None, :] < lengths[:, None]
        if np.any(words[~active] != -1) or np.any(words[active] < 0) or np.any(words[active] >= meta["n_actions"]):
            raise ValueError("invalid C action/padding")
    return result
