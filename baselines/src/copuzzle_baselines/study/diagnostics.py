"""Evaluator-only composition context/support diagnostics; never fitting inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data import load_training
from ..protocols import public_api


def composition_diagnostics(bundle, predictor, *, split, coverage, coverage_seed, batch_size=512):
    from copuzzle.research.bundle import _read_npz
    learner = Path(bundle) / "learner"
    query = public_api(learner).load_probe(learner, "C", split)
    truth = _read_npz(Path(bundle) / "evaluator/answers.npz")[f"C_{split}_prefix_boards"]
    prior = np.concatenate((query["board"][:, None, :], truth[:, :-1]), axis=1)
    active = np.arange(query["actions"].shape[1])[None, :] < query["lengths"][:, None]
    states, actions, targets = prior[active], query["actions"][active], truth[active]
    exact, legal = [], []
    for start in range(0, len(states), batch_size):
        end = start + batch_size
        predicted, flags = predictor(states[start:end], actions[start:end])
        exact.extend(np.all(predicted == targets[start:end], axis=1).tolist())
        legal.extend(np.asarray(flags, bool).tolist())
    exact, legal = np.asarray(exact, bool), np.asarray(legal, bool)
    data = load_training(learner, coverage=coverage, coverage_seed=coverage_seed)
    observed = {(tuple(state.tolist()), int(action)) for state, action in zip(data.vocabulary.decode(data.states), data.actions)}
    supported = np.asarray([(tuple(state.tolist()), int(action)) in observed for state, action in zip(states, actions)], bool)
    missing = np.zeros(active.shape, bool)
    missing[active] = ~supported
    return {
        "role": "teacher-forced context diagnostic; separate from the open-loop C score",
        "steps": len(states), "teacher_forced_state_exact": float(exact.mean()),
        "teacher_forced_legality_accuracy": float(legal.mean()), "teacher_forced_joint_exact": float((exact & legal).mean()),
        "case_unseen_transition_counts": missing.sum(axis=1).astype(int).tolist(),
        "cases_with_all_primitive_queries_observed": int(np.sum(~missing.any(axis=1))),
        "by_training_support": {name: {"steps": int(np.sum(supported == flag)),
                                        "state_exact": float(exact[supported == flag].mean()) if np.any(supported == flag) else None,
                                        "joint_exact": float((exact & legal)[supported == flag].mean()) if np.any(supported == flag) else None}
                                for name, flag in (("observed", True), ("unobserved", False))},
    }
