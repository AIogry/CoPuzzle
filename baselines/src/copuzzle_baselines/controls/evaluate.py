"""Matched-planner controls for separating model errors from search limitations."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import time

import numpy as np

from ..config import fingerprint, integer
from ..planning import ModelPlanner
from .models import fit_control


KINDS = ("random", "lookup", "fitted_xor", "fitted_position", "oracle_dynamics", "optimal_oracle")


def source_sha256():
    root = Path(__file__).parents[1]
    names = ("controls/evaluate.py", "controls/models.py", "planning.py", "data.py", "protocols.py", "config.py")
    return fingerprint({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names})


def _predictions(model, inputs, probe):
    if probe == "R":
        boards, legal = model.predict(inputs["board"], inputs["action"])
        return {"next_board": boards, "legal": legal}
    words, lengths = inputs["actions"], inputs["lengths"]
    current = inputs["board"].copy()
    result = np.full((*words.shape, current.shape[1]), -1, np.int32)
    for step in range(words.shape[1]):
        rows = np.flatnonzero(lengths > step)
        current[rows], _ = model.predict(current[rows], words[rows, step])
        result[rows, step] = current[rows]
    return {"prefix_boards": result}


def evaluate_control(bundle, kind, *, split="dev", model_query_limit=32, coverage=1.0, coverage_seed=0, seed=0, test_lock=None):
    """Development/selected dev references. Final-test orchestration is separate."""
    from copuzzle.benchmark.data import load_probe, metadata
    from copuzzle.benchmark.evaluation import evaluate_policy, score_predictions
    from copuzzle.benchmark.generation import validate_bundle
    from copuzzle.benchmark.graph import full_distances, partition

    if kind not in KINDS:
        raise ValueError("unknown reference control")
    bundle = Path(bundle)
    meta = metadata(bundle / "learner")
    seed = integer(seed, "rollout seed")
    limit = integer(model_query_limit, "model_query_limit")
    settings = {"kind": kind, "model_query_limit": None if kind == "optimal_oracle" else 0 if kind == "random" else limit,
                "coverage": coverage, "coverage_seed": coverage_seed, "seed": seed}
    authorization = None
    if meta["purpose"] == "benchmark" and split == "test":
        from ..study.access import authorize_control_test
        authorization = authorize_control_test(test_lock, bundle, settings, source_sha256())
    validate_bundle(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    model, data, p = None, None, None
    started = time.perf_counter()
    if kind in ("lookup", "fitted_xor", "fitted_position"):
        model, data = fit_control(bundle / "learner", kind, coverage=coverage, coverage_seed=coverage_seed)
    elif kind in ("oracle_dynamics", "optimal_oracle"):
        p = partition(meta["task"], meta["regime"], manifest["seed"])
        class ExactModel:
            def predict(self, boards, actions):
                values = [p.successor(p.graph.locate(s), int(a)) for s, a in zip(boards, actions)]
                return p.graph.states[[i for i, _ in values]], np.asarray([legal for _, legal in values], np.uint8)
        model = ExactModel()
    fit_seconds = time.perf_counter() - started
    report = {"kind": kind, "purpose": "diagnostic reference", "split": split, "settings": settings,
              "sources_sha256": source_sha256(), "test_authorization": authorization,
              "privileged": kind in ("oracle_dynamics", "optimal_oracle"),
              "structural_prior": {"fitted_xor": "constant globally applicable XOR updates", "fitted_position": "constant globally applicable position permutations"}.get(kind),
              "fit_seconds": fit_seconds, "data": data.identity if data else None,
              "R": {"supported": False}, "C": {"supported": False}}
    if model is not None:
        for probe in ("R", "C"):
            predictions = _predictions(model, load_probe(bundle / "learner", probe, split), probe)
            report[probe] = {"supported": True, **score_predictions(bundle, probe, predictions, split=split)}
    rng = np.random.default_rng(seed)

    def forward(board, action):
        following, legal = model.predict(np.asarray(board)[None], np.asarray([action], np.int32))
        return following[0], bool(legal[0])

    def factory():
        if kind == "random":
            local = np.random.default_rng(int(rng.integers(2**63)))
            return lambda observation, budget: int(local.integers(meta["n_actions"]))
        if kind == "optimal_oracle":
            def oracle(observation, budget):
                source, target = p.graph.locate(observation["board"]), p.graph.locate(observation["goal"])
                d = full_distances(meta["task"], target)
                return next(a for a in range(p.graph.n_actions) if p.successor(source, a)[1] and d[p.successor(source, a)[0]] == d[source] - 1)
            return oracle
        return ModelPlanner(forward, meta["n_actions"], batch_predictor=model.predict)

    if kind not in ("random", "optimal_oracle") and limit == 0:
        raise ValueError("model planners require a positive query budget")
    report["G"] = evaluate_policy(bundle, factory, split=split, model_query_limit=0 if kind in ("random", "optimal_oracle") else limit)
    if kind == "optimal_oracle":
        report["G"]["budget_exempt_privileged_solver"] = True
        report["G"]["model_query_limit_per_decision"] = None
        for row in [report["G"]["aggregate"], *report["G"]["by_tier"].values(), *report["G"]["by_empirical_reachability"].values(), *report["G"]["cases"]]:
            if row.get("cases", 1):
                row["model_queries"] = row["model_calls"] = None
    return report
