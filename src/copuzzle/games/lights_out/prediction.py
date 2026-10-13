"""Offline prediction evaluation with explicit public-structure inputs.

The evaluator gives a predictor only a board, the public structure fields and
one candidate button index. It never passes a goal, recorded successor, effect,
rule object, or solver answer. Python callables are not sandboxed; callers must
still declare any analytic rules, solvers or external environment queries.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from .data import TrajectoryDataset
from .diagnostics import _binary_array, _integer, prediction_metrics
from .evaluation import _method_declarations
from .structure import PublicStructure


PREDICTION_SCHEMA = "copuzzle.prediction.v1"


def _dataset_hash(dataset):
    """Identity of the manifest and typed arrays, independent of NPZ encoding."""
    digest = hashlib.sha256()
    digest.update(json.dumps(dataset.manifest, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode())
    for name in ("states", "actions", "events"):
        array = getattr(dataset, name)
        digest.update(json.dumps([name, array.dtype.name, list(array.shape)]).encode())
        digest.update(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order="C"))
    return digest.hexdigest()


def evaluate_predictor(
    predictor, dataset: TrajectoryDataset, *, structure: PublicStructure,
    sequence_length=1, num_examples=128, seed=0,
    predictor_label=None, method_metadata=None,
):
    """Measure next-board prediction and autoregressive sequence error.

    ``predictor(observation, action)`` returns a binary vector of length N.
    observation contains ``board`` and ``structure.observation_fields()`` only.
    Later calls receive the previous prediction, not a recorded intermediate
    state. Optional predictor.reset() runs before each sampled window.

    Windows are sampled uniformly with replacement within complete recorded
    episodes. No unseen-state or unseen-combination claim follows from this
    sampling; callers must establish train/evaluation separation separately.
    Per-step effect metrics describe the cumulative effect from the initial
    board, and have the same exact accuracy as the corresponding next board.
    This does not evaluate operation selection or goal-solving ability.
    """
    if not isinstance(dataset, TrajectoryDataset):
        raise TypeError("dataset must be a TrajectoryDataset")
    dataset.validate()
    if not isinstance(structure, PublicStructure):
        raise TypeError("structure must be a PublicStructure")
    if (dataset.n_buttons != structure.n_buttons
            or dataset.manifest["rule_id"] != structure.rule_id):
        raise ValueError("Dataset and public structure must identify the same ordered rule")
    length = _integer(sequence_length, "sequence_length", 1)
    count = _integer(num_examples, "num_examples", 1)
    seed = _integer(seed, "seed", 0)
    if length > dataset.horizon:
        raise ValueError("sequence_length cannot exceed the dataset trajectory horizon")
    if method_metadata is not None and isinstance(method_metadata, dict):
        reserved = {
            "predictor", "dataset_sha256", "dataset_hash_semantics", "sequence_length",
            "num_examples", "seed", "sampling", "target_semantics", "by_step", "endpoint",
            "examples", "predictor_calls", "inference_seconds", "candidate_operation",
        }
        if reserved & method_metadata.keys():
            raise ValueError("method_metadata cannot overwrite core result fields")
    declarations = _method_declarations(predictor, method_metadata)

    windows_per_episode = dataset.horizon - length + 1
    window_ids = np.random.default_rng(seed).integers(
        dataset.num_episodes * windows_per_episode, size=count)
    episodes = window_ids // windows_per_episode
    starts = window_ids % windows_per_episode
    offsets = starts[:, None] + np.arange(length + 1)[None, :]
    targets = dataset.states[episodes[:, None], offsets].copy()
    actions = dataset.actions[episodes[:, None], offsets[:, :-1]].copy()
    predictions = np.empty((count, length, dataset.n_buttons), dtype=np.uint8)
    public_fields = set(structure.observation_fields())
    inference_seconds = 0.0
    # Record identity before invoking an arbitrary user callable.
    dataset_sha256 = _dataset_hash(dataset)
    for example in range(count):
        reset = getattr(predictor, "reset", None)
        if callable(reset):
            reset()
        board = targets[example, 0].copy()
        for step in range(length):
            observation = {"board": board.copy(), **structure.observation_fields()}
            began = time.perf_counter()
            output = predictor(observation, int(actions[example, step]))
            inference_seconds += time.perf_counter() - began
            output = _binary_array(output, "predicted next board", 1)
            if output.shape != (dataset.n_buttons,):
                raise ValueError("Predicted next board has the wrong button count")
            board = output.copy()
            predictions[example, step] = board

    by_step = {
        str(step + 1): prediction_metrics(predictions[:, step], targets[:, 0], targets[:, step + 1])
        for step in range(length)
    }
    bit_errors = np.count_nonzero(predictions != targets[:, 1:], axis=-1)
    return {
        "schema_version": PREDICTION_SCHEMA,
        "predictor": predictor_label or getattr(predictor, "label", type(predictor).__name__),
        "rule_id": structure.rule_id,
        "observation_profile": "structure_v1",
        "actual_observation_fields": sorted({"board"} | public_fields),
        "candidate_operation": "integer button index supplied as the second argument",
        "structure_sha256": structure.structure_sha256,
        "dataset_sha256": dataset_sha256,
        "dataset_hash_semantics": "canonical manifest plus named typed array contents; not an artifact file hash",
        "sequence_length": length, "num_examples": count, "seed": seed,
        "sampling": "uniform recorded windows with replacement within episodes; novelty is not guaranteed",
        "target_semantics": "autoregressive endpoints; cumulative effects measured from the initial board",
        "method_metadata": declarations,
        "method_information_source": "caller declarations or predictor attributes; not independently verified",
        "parameter_freezing_enforced": False,
        "by_step": by_step, "endpoint": dict(by_step[str(length)]),
        "examples": [
            {"episode": int(episodes[i]), "start_step": int(starts[i]),
             "bit_errors_by_step": bit_errors[i].tolist(),
             "exact_by_step": (bit_errors[i] == 0).tolist(),
             "endpoint_bit_errors": int(bit_errors[i, -1])}
            for i in range(count)
        ],
        "predictor_calls": count * length,
        "inference_seconds": inference_seconds,
    }
