"""Separated evaluator: exact prediction scores and failure-inclusive control."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bundle import _read_npz
from .data import load_probe, load_training, metadata
from .graph import exact_distances, partition
from .spec import integer


def _prediction(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "iub":
        raise ValueError(f"{name} must be an integer array of shape {shape}")
    return value


def score_predictions(bundle, probe: str, predictions: dict, *, split: str = "dev") -> dict:
    """Score an already frozen prediction file; never supply truth to a model."""
    bundle = Path(bundle)
    inputs = load_probe(bundle / "learner", probe, split)
    truth = _read_npz(bundle / "evaluator" / "answers.npz")
    if probe == "R":
        if set(predictions) != {"next_board", "legal"}:
            raise ValueError("R predictions require next_board and legal only")
        target, legal = (truth[f"R_{split}_{key}"] for key in ("next_board", "legal"))
        board = _prediction(predictions["next_board"], target.shape, "next_board")
        flag = _prediction(predictions["legal"], legal.shape, "legal")
        if np.any((flag != 0) & (flag != 1)):
            raise ValueError("predicted legality must be binary")
        correct = np.all(board == target, axis=1)
        joint = correct & (flag == legal)
        groups = {label: {"cases": int(np.sum(legal == value)),
                          "joint_exact": float(joint[legal == value].mean()) if np.any(legal == value) else None}
                  for label, value in (("legal", 1), ("illegal", 0))}
        return {"cases": len(legal), "next_board_exact": float(correct.mean()),
                "legality_accuracy": float((flag == legal).mean()), "joint_exact": float(joint.mean()),
                "balanced_joint_exact": float(np.mean([v["joint_exact"] for v in groups.values() if v["cases"]])),
                "by_legality": groups}
    if probe != "C" or set(predictions) != {"prefix_boards"}:
        raise ValueError("C predictions require prefix_boards only")
    target = truth[f"C_{split}_prefix_boards"]
    pred = _prediction(predictions["prefix_boards"], target.shape, "prefix_boards")
    lengths = inputs["lengths"]
    active = np.arange(target.shape[1])[None, :] < lengths[:, None]
    exact = np.all(pred == target, axis=2)
    full = np.all(exact | ~active, axis=1)
    final = exact[np.arange(len(lengths)), lengths - 1]
    return {"cases": len(lengths), "whole_prefix_exact": float(full.mean()), "final_board_exact": float(final.mean()),
            "by_length": {str(int(k)): {"cases": int(np.sum(lengths == k)),
                                       "whole_prefix_exact": float(full[lengths == k].mean()),
                                       "final_board_exact": float(final[lengths == k].mean())}
                          for k in sorted(set(lengths.tolist()))}}


class QueryLimitExceeded(RuntimeError):
    """A policy may catch this and select its best action so far."""


class ModelBudget:
    """Count single-state/action model queries, including lookups, per decision.

    Route every planning transition through predict. This is an accounting API,
    not isolation against a malicious policy; neural inference and fit cost must
    also be disclosed. One call must compute exactly one candidate transition.
    """
    def __init__(self, limit):
        self.limit = integer(limit, "model_query_limit")
        self.used = 0

    def predict(self, predictor, board, action):
        if self.used >= self.limit:
            raise QueryLimitExceeded("per-decision model transition budget exhausted")
        self.used += 1
        return predictor(np.asarray(board).copy(), action)


def evaluate_policy(bundle, policy_factory, *, split="dev", model_query_limit=0) -> dict:
    """Reset episode memory per case; weights must be frozen by the caller.

    factory() -> policy(observation, ModelBudget) -> opaque action ID.
    No evaluator object, oracle labels, tier, or true legal mask reaches policy.
    """
    bundle = Path(bundle)
    inputs = load_probe(bundle / "learner", "G", split)
    meta = metadata(bundle / "learner")
    manifest = json.loads((bundle / "manifest.json").read_text())
    p = partition(meta["task"], meta["regime"], manifest["seed"])
    truth = _read_npz(bundle / "evaluator" / "answers.npz")
    limit = integer(model_query_limit, "model_query_limit")
    rows = []
    for i, (start, goal) in enumerate(zip(inputs["board"], inputs["goal"])):
        source, target = p.graph.locate(start), p.graph.locate(goal)
        policy = policy_factory()
        illegal, queries, used, last_legal = 0, 0, 0, None
        while source != target and used < meta["control_budget"]:
            budget = ModelBudget(limit)
            obs = {"board": p.graph.states[source].copy(), "goal": goal.copy(),
                   "steps_remaining": meta["control_budget"] - used, "previous_legal": last_legal}
            action = integer(policy(obs, budget), "policy action")
            if action >= p.graph.n_actions:
                raise ValueError("policy returned an out-of-range action; this is not an in-range illegal attempt")
            source, last_legal = p.successor(source, action)
            used += 1
            illegal += not last_legal
            queries += budget.used
        solved = source == target
        distance = int(truth[f"G_{split}_distance"][i])
        rows.append({"success": solved, "attempts": used, "illegal_attempts": illegal,
                     "penalized_cost": used if solved else meta["control_budget"],
                     "success_excess": used - distance if solved else None,
                     "distance": distance, "tier": int(truth[f"G_{split}_tier"][i]), "model_queries": queries,
                     "empirically_reachable": bool(truth[f"G_{split}_empirical_distance"][i] >= 0)})

    def summarize(selected):
        if not selected:
            return {"cases": 0}
        successful = [row for row in selected if row["success"]]
        attempts = sum(row["attempts"] for row in selected)
        return {"cases": len(selected), "success_rate": sum(row["success"] for row in selected) / len(selected),
                "mean_attempts_all": float(np.mean([row["attempts"] for row in selected])),
                "mean_penalized_cost_all": float(np.mean([row["penalized_cost"] for row in selected])),
                "illegal_attempt_rate": sum(row["illegal_attempts"] for row in selected) / max(1, attempts),
                "mean_success_excess": float(np.mean([row["success_excess"] for row in successful])) if successful else None,
                "model_queries": sum(row["model_queries"] for row in selected)}

    return {"control_budget": meta["control_budget"], "model_query_limit_per_decision": limit,
            "aggregate": summarize(rows),
            "by_tier": {name: summarize([row for row in rows if row["tier"] == i])
                        for i, name in enumerate(("easy", "medium", "hard"))},
            "by_empirical_reachability": {name: summarize([row for row in rows if row["empirically_reachable"] == flag])
                                          for name, flag in (("reachable", True), ("unreachable", False))},
            "cases": rows}


def empirical_predictions(learner_directory, probe, split="dev"):
    """Training-only lookup control; demonstrates the limited claim of supported C."""
    train = load_training(learner_directory)
    model = {(tuple(states[0].tolist()), int(actions[0])): (states[1].copy(), int(legal[0]))
             for states, actions, legal in zip(train["observations"], train["actions"], train["legal"])}
    inputs = load_probe(learner_directory, probe, split)
    if probe == "R":
        results = [model.get((tuple(board.tolist()), int(action)), (board, 0))
                   for board, action in zip(inputs["board"], inputs["action"])]
        return {"next_board": np.asarray([r[0] for r in results]), "legal": np.asarray([r[1] for r in results])}
    if probe != "C":
        raise ValueError("empirical_predictions supports R or C")
    prefixes = np.full((*inputs["actions"].shape, inputs["board"].shape[1]), -1, dtype=np.int32)
    for i, (start, word, length) in enumerate(zip(inputs["board"], inputs["actions"], inputs["lengths"])):
        board = start.copy()
        for j, action in enumerate(word[:length]):
            board, _ = model.get((tuple(board.tolist()), int(action)), (board, 0))
            prefixes[i, j] = board
    return {"prefix_boards": prefixes}


def reference_report(bundle, *, split="dev", seed=0):
    """Calibration only: data lookup prediction, random control, privileged oracle."""
    bundle = Path(bundle)
    meta = metadata(bundle / "learner")
    manifest = json.loads((bundle / "manifest.json").read_text())
    p = partition(meta["task"], meta["regime"], manifest["seed"])
    rng = np.random.default_rng(integer(seed, "seed"))

    def random_factory():
        case_rng = np.random.default_rng(int(rng.integers(2**63)))
        return lambda observation, budget: int(case_rng.integers(meta["n_actions"]))

    def oracle_factory():
        def policy(observation, budget):
            # Deliberately privileged calibration; never presented as learned.
            source = p.graph.locate(observation["board"])
            target = p.graph.locate(observation["goal"])
            d = exact_distances(p.graph.task, target)
            return next(a for a in range(p.graph.n_actions)
                        if p.successor(source, a)[1] and d[p.successor(source, a)[0]] == d[source] - 1)
        return policy

    return {"purpose": "qualification controls; not learned benchmark results", "split": split,
            "training_lookup": {probe: score_predictions(bundle, probe, empirical_predictions(bundle / "learner", probe, split), split=split)
                                for probe in meta["probes"] if probe != "G"},
            "random_G": evaluate_policy(bundle, random_factory, split=split),
            "privileged_oracle_G": evaluate_policy(bundle, oracle_factory, split=split)}
