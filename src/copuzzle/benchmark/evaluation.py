"""V1 exact prediction metrics, batch-aware planning budgets and retained failures."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import time

import numpy as np

from copuzzle.research.bundle import _read_npz
from copuzzle.research.evaluation import QueryLimitExceeded
from .data import load_probe, metadata
from . import PROTOCOL
from .graph import partition
from .spec import digest, instance, integer, specification


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("benchmark/evaluation.py", "benchmark/data.py", "benchmark/spec.py", "benchmark/graph.py",
             "research/data.py", "research/bundle.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _finish(report, source):
    if source_hashes() != source:
        raise RuntimeError("metric source changed during evaluation")
    return {"protocol": PROTOCOL, "specification_sha256": digest(specification()), "metric_sources": source, **report}


class ModelBudget:
    """Charge requested state-action candidates, not Python calls or batch count."""
    def __init__(self, limit):
        self.limit = integer(limit, "model query limit")
        self.used = 0
        self.calls = 0

    def _charge(self, count):
        count = integer(count, "query count", 1)
        if self.used + count > self.limit:
            raise QueryLimitExceeded("model transition budget exhausted")
        self.used += count
        self.calls += 1

    def predict(self, predictor, board, action):
        self._charge(1)
        return predictor(np.asarray(board).copy(), action)

    def predict_batch(self, predictor, boards, actions):
        boards, actions = np.asarray(boards), np.asarray(actions)
        if boards.ndim != 2 or actions.shape != (len(boards),):
            raise ValueError("batched model requests must align")
        self._charge(len(boards))
        return predictor(boards.copy(), actions.copy())


def _prediction(value, expected, name):
    value = np.asarray(value)
    if value.shape != expected.shape or value.dtype.kind not in "iub":
        raise ValueError(f"{name} has invalid shape or dtype")
    return value


def score_predictions(bundle, probe, predictions, *, split="dev"):
    source = source_hashes()
    bundle = Path(bundle)
    inputs = load_probe(bundle / "learner", probe, split)
    truth = _read_npz(bundle / "evaluator/answers.npz")
    spec = instance(metadata(bundle / "learner")["task"])
    if probe == "R":
        if set(predictions) != {"next_board", "legal"}:
            raise ValueError("R requires next_board and legal")
        actual, legal = truth[f"R_{split}_next_board"], truth[f"R_{split}_legal"]
        predicted = _prediction(predictions["next_board"], actual, "next_board")
        flags = _prediction(predictions["legal"], legal, "legal")
        if np.any((flags != 0) & (flags != 1)):
            raise ValueError("predicted legality must be binary")
        exact = np.all(predicted == actual, axis=1)
        joint = exact & (flags == legal)
        groups = {label: {"cases": int(np.sum(legal == flag)),
                          "next_board_exact": float(exact[legal == flag].mean()) if np.any(legal == flag) else None,
                          "legality_accuracy": float((flags[legal == flag] == flag).mean()) if np.any(legal == flag) else None,
                          "joint_exact": float(joint[legal == flag].mean()) if np.any(legal == flag) else None}
                  for label, flag in (("legal", 1), ("illegal", 0))}
        return _finish({"cases": len(actual), "next_board_exact": float(exact.mean()), "legality_accuracy": float((flags == legal).mean()),
                "joint_exact": float(joint.mean()), "balanced_joint_exact": float(np.mean([v["joint_exact"] for v in groups.values() if v["cases"]])),
                "by_legality": groups}, source)
    if probe != "C" or set(predictions) != {"prefix_boards"}:
        raise ValueError("C requires prefix_boards")
    actual = truth[f"C_{split}_prefix_boards"]
    predicted = _prediction(predictions["prefix_boards"], actual, "prefix_boards")
    lengths = inputs["lengths"]
    active = np.arange(actual.shape[1])[None, :] < lengths[:, None]
    exact = np.all(predicted == actual, axis=-1)
    whole = np.all(exact | ~active, axis=1)
    final = exact[np.arange(len(lengths)), lengths - 1]
    groups = {}
    for length in spec.composition_lengths:
        rows = lengths == length
        if not np.any(rows):
            raise ValueError("required composition length is missing")
        groups[str(length)] = {"cases": int(rows.sum()), "whole_prefix_exact": float(whole[rows].mean()),
                               "final_board_exact": float(final[rows].mean())}
    return _finish({"cases": len(lengths), "whole_prefix_exact": float(whole.mean()), "final_board_exact": float(final.mean()),
            "length_macro_whole_prefix_exact": float(np.mean([row["whole_prefix_exact"] for row in groups.values()])),
            "length_macro_final_board_exact": float(np.mean([row["final_board_exact"] for row in groups.values()])), "by_length": groups}, source)


def _summary(rows):
    if not rows:
        return {"cases": 0}
    total = sum(row["attempts"] for row in rows)
    successes = [row for row in rows if row["success"]]
    return {"cases": len(rows), "success_rate": sum(row["success"] for row in rows) / len(rows),
            "mean_attempts_all": float(np.mean([row["attempts"] for row in rows])),
            "mean_penalized_cost_all": float(np.mean([row["penalized_cost"] for row in rows])),
            "illegal_attempt_rate": sum(row["illegal_attempts"] for row in rows) / max(1, total),
            "policy_error_rate": sum(row["outcome"] == "policy_error" for row in rows) / len(rows),
            "mean_success_excess": float(np.mean([row["success_excess"] for row in successes])) if successes else None,
            "model_queries": sum(row["model_queries"] for row in rows), "model_calls": sum(row["model_calls"] for row in rows),
            "policy_seconds": sum(row["policy_seconds"] for row in rows)}


def evaluate_policy(bundle, policy_factory, *, split="dev", model_query_limit=0):
    """Record policy failures per case, preserving every expected denominator.

    Data/protocol errors remain fatal. KeyboardInterrupt/SystemExit propagate so
    cancellation cannot masquerade as a completed evaluation. Policy exceptions
    and out-of-range actions are retained as failures with full budget cost.
    """
    metric_source = source_hashes()
    bundle = Path(bundle)
    inputs = load_probe(bundle / "learner", "G", split)
    meta = metadata(bundle / "learner")
    manifest = json.loads((bundle / "manifest.json").read_text())
    p = partition(meta["task"], meta["regime"], manifest["seed"])
    truth = _read_npz(bundle / "evaluator/answers.npz")
    limit = integer(model_query_limit, "model_query_limit")
    horizon = meta["control_budget"]
    rows = []
    for index, (board, goal) in enumerate(zip(inputs["board"], inputs["goal"])):
        source, target = p.graph.locate(board), p.graph.locate(goal)
        used, illegal, queries, calls, seconds = 0, 0, 0, 0, 0.0
        last_legal, error, policy = None, None, None
        started = time.perf_counter()
        try:
            policy = policy_factory()
        except Exception as failure:
            error = {"type": type(failure).__name__, "message": str(failure)}
        finally:
            seconds += time.perf_counter() - started
        while source != target and used < horizon and error is None:
            budget = ModelBudget(limit)
            observation = {"board": p.graph.states[source].copy(), "goal": goal.copy(),
                           "steps_remaining": horizon - used, "previous_legal": last_legal}
            started = time.perf_counter()
            try:
                action = integer(policy(observation, budget), "policy action")
                if action >= p.graph.n_actions:
                    raise ValueError("out-of-range action; not an in-range illegal attempt")
            except Exception as failure:
                error = {"type": type(failure).__name__, "message": str(failure)}
            finally:
                seconds += time.perf_counter() - started
                queries += budget.used
                calls += budget.calls
            if error is None:
                source, last_legal = p.successor(source, action)
                used += 1
                illegal += not last_legal
        solved = source == target and error is None
        distance = int(truth[f"G_{split}_distance"][index])
        rows.append({"case_index": index, "success": solved,
                     "outcome": "policy_error" if error is not None else "success" if solved else "budget_exhausted",
                     "error": error, "attempts": used, "illegal_attempts": illegal,
                     "penalized_cost": used if solved else horizon, "success_excess": used - distance if solved else None,
                     "distance": distance, "tier": int(truth[f"G_{split}_tier"][index]),
                     "empirically_reachable": bool(truth[f"G_{split}_empirical_distance"][index] >= 0),
                     "model_queries": queries, "model_calls": calls, "policy_seconds": seconds})
    tiers = {name: _summary([row for row in rows if row["tier"] == i]) for i, name in enumerate(("easy", "medium", "hard"))}
    if any(row["cases"] == 0 for row in tiers.values()):
        raise ValueError("required goal tier is missing")
    return _finish({"control_budget": horizon, "model_query_limit_per_decision": limit, "aggregate": _summary(rows),
            "tier_macro_success_rate": float(np.mean([row["success_rate"] for row in tiers.values()])),
            "tier_macro_penalized_cost": float(np.mean([row["mean_penalized_cost_all"] for row in tiers.values()])),
            "by_tier": tiers,
            "by_empirical_reachability": {name: _summary([row for row in rows if row["empirically_reachable"] == flag])
                                          for name, flag in (("reachable", True), ("unreachable", False))}, "cases": rows}, metric_source)
