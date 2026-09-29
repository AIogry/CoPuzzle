"""Versioned goal tasks and model-agnostic evaluation.

Exact distances and the reference solver are evaluator-only information. Policies
receive the environment observation, never the task record or its answer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .env import PuzzleEnv
from .oracle import rank, solve
from .rules import PuzzleRule, get_rule

TASK_SCHEMA = "copuzzle.goal_tasks.v0"
RESULT_SCHEMA = "copuzzle.evaluation.v0"


def write_json(path, value):
    """Write a new artifact, refusing accidental replacement."""
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        out.write(text)
    return path


def generate_task_suite(
    rule: PuzzleRule, *, per_depth=16, depths=(1, 2, 3, 4), seed=100,
    max_episode_steps=30, max_attempts_per_depth=10000,
):
    """Sample reachable goals in exact minimum-press strata.

    These are independently sampled pairs, not a claim of unseen states or
    unseen pairs relative to training data. Audit that separately.
    """
    for name, value in (("per_depth", per_depth), ("max_episode_steps", max_episode_steps),
                        ("max_attempts_per_depth", max_attempts_per_depth)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    depths = tuple(depths)
    max_rank = rank(rule.toggle_matrix)
    if not depths or len(set(depths)) != len(depths):
        raise ValueError("depths must be nonempty and distinct")
    for depth in depths:
        if isinstance(depth, (bool, np.bool_)) or not isinstance(depth, (int, np.integer)):
            raise ValueError("depths must contain integers")
        if not 1 <= depth <= min(max_rank, max_episode_steps):
            raise ValueError("depth must be positive and no greater than rank or episode budget")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 71329]))
    cases, seen = [], set()
    for depth in depths:
        accepted = 0
        for _ in range(max_attempts_per_depth):
            start = rng.integers(0, 2, rule.n_buttons, dtype=np.uint8)
            event = np.zeros(rule.n_buttons, dtype=np.uint8)
            event[rng.choice(rule.n_buttons, int(depth), replace=False)] = 1
            goal = rule.apply_event(start, event)
            solution = solve(rule, start, goal)
            if solution is None or len(solution) != depth:
                continue
            key = start.tobytes() + goal.tobytes()
            if key in seen:
                continue
            seen.add(key)
            case_id = hashlib.sha256(rule.rule_id.encode() + key).hexdigest()[:20]
            cases.append({"case_id": case_id, "state": start.tolist(), "goal": goal.tolist(),
                          "optimal_steps": int(depth)})
            accepted += 1
            if accepted == per_depth:
                break
        if accepted != per_depth:
            raise ValueError(
                f"Could generate only {accepted}/{per_depth} distinct tasks at depth {depth}; "
                "this distance may be unattainable or the attempt budget is insufficient"
            )
    return {
        "schema_version": TASK_SCHEMA, "generator_version": "0.1.0",
        "task_name": rule.name, "rule_id": rule.rule_id, "seed": int(seed),
        "max_episode_steps": int(max_episode_steps), "reveal_graph": False,
        "difficulty": "exact_minimum_number_of_presses",
        "sampling": "distinct random starts plus random distinct-button subsets, accepted by exact distance",
        "split_semantics": "independent_pairs; state and training-pair novelty require an overlap audit",
        "cases": cases,
    }


def validate_task_suite(suite, rule=None):
    if not isinstance(suite, dict) or suite.get("schema_version") != TASK_SCHEMA:
        raise ValueError("Unsupported goal-task schema")
    if not isinstance(suite.get("task_name"), str):
        raise ValueError("Task suite must name a rule instance")
    rule = get_rule(suite["task_name"]) if rule is None else rule
    if suite.get("rule_id") != rule.rule_id:
        raise ValueError("Task suite and rule identity differ")
    horizon = suite.get("max_episode_steps")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Invalid episode budget")
    if type(suite.get("reveal_graph")) is not bool:
        raise ValueError("reveal_graph must be explicit")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Task suite must contain cases")
    ids, pairs = set(), set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ValueError("Invalid case record")
        if "state" not in case or "goal" not in case:
            raise ValueError("Case must contain state and goal")
        start = rule.validate_state(case["state"])
        goal = rule.validate_state(case["goal"])
        key = start.tobytes() + goal.tobytes()
        if case["case_id"] in ids or key in pairs:
            raise ValueError("Duplicate evaluation case")
        ids.add(case["case_id"])
        pairs.add(key)
        distance = case.get("optimal_steps")
        if type(distance) is not int or not 1 <= distance <= horizon:
            raise ValueError("Invalid optimal_steps")
        solution = solve(rule, start, goal)
        if solution is None or len(solution) != distance:
            raise ValueError("Unreachable case or incorrect shortest distance")
    return rule


def load_task_suite(path):
    with Path(path).open(encoding="utf-8") as inp:
        suite = json.load(inp)
    validate_task_suite(suite)
    return suite


class RandomPolicy:
    """An untrained baseline with its own reproducible RNG."""
    label = "random"
    privileged_rule_access = False

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, observation):
        return int(self.rng.integers(len(observation["board"])))


class OraclePolicy:
    """Privileged exact-rule diagnostic, never a learned baseline."""
    label = "exact_rule_oracle"
    privileged_rule_access = True

    def __init__(self, rule):
        self.rule = rule

    def __call__(self, observation):
        solution = solve(self.rule, observation["board"], observation["goal"])
        if not solution:
            raise ValueError("Oracle called on an already solved or unreachable task")
        return solution[0]


def evaluate_policy(policy: Callable, suite, *, rule=None, policy_label=None):
    """Run a frozen policy on the fixed cases. No training data are added.

    A callable sees only observation dictionaries. A stateful callable may
    expose reset(), which is invoked before each case without task answers.
    """
    rule = validate_task_suite(suite, rule)
    env = PuzzleEnv(rule, max_episode_steps=suite["max_episode_steps"],
                    reveal_graph=suite["reveal_graph"])
    results = []
    try:
        for case in suite["cases"]:
            reset = getattr(policy, "reset", None)
            if callable(reset):
                reset()
            obs, _ = env.reset(seed=0, options={"state": case["state"], "goal": case["goal"]})
            total_reward, inference_seconds = 0.0, 0.0
            for step in range(1, suite["max_episode_steps"] + 1):
                began = time.perf_counter()
                action = policy(obs)
                inference_seconds += time.perf_counter() - began
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            results.append({
                "case_id": case["case_id"], "optimal_steps": case["optimal_steps"],
                "success": bool(info["is_success"]), "presses": step,
                "timeout": bool(truncated), "return": float(total_reward),
                "policy_calls": step, "inference_seconds": inference_seconds,
                "excess_presses_on_success": step - case["optimal_steps"] if terminated else None,
            })
    finally:
        env.close()
    def summarize(rows):
        successful = [r for r in rows if r["success"]]
        return {
            "cases": len(rows), "success_rate": sum(r["success"] for r in rows) / len(rows),
            "timeout_rate": sum(r["timeout"] for r in rows) / len(rows),
            "mean_presses": sum(r["presses"] for r in rows) / len(rows),
            "mean_successful_presses": (sum(r["presses"] for r in successful) / len(successful)
                                        if successful else None),
            "mean_excess_presses_on_success": (sum(r["excess_presses_on_success"] for r in successful)
                                              / len(successful) if successful else None),
            "policy_calls": sum(r["policy_calls"] for r in rows),
            "inference_seconds": sum(r["inference_seconds"] for r in rows),
        }
    canonical = json.dumps(suite, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": RESULT_SCHEMA, "rule_id": rule.rule_id,
        "task_suite_sha256": hashlib.sha256(canonical).hexdigest(),
        "policy": policy_label or getattr(policy, "label", type(policy).__name__),
        "privileged_rule_access": bool(getattr(policy, "privileged_rule_access", False)),
        "reveal_graph": suite["reveal_graph"], "max_episode_steps": suite["max_episode_steps"],
        "summary": summarize(results),
        "by_optimal_steps": {str(d): summarize([r for r in results if r["optimal_steps"] == d])
                             for d in sorted({r["optimal_steps"] for r in results})},
        "cases": results,
    }


def audit_overlap(dataset, suite, *, rule=None):
    """Measure novelty; independent seeds alone do not establish a held-out state."""
    dataset.validate()
    rule = validate_task_suite(suite, rule)
    if dataset.n_buttons != rule.n_buttons:
        raise ValueError("Dataset and task suite have different board dimensions")
    if dataset.manifest.get("rule_id") != suite.get("rule_id"):
        raise ValueError("Overlap audit requires the same rule instance")
    starts = [rule.validate_state(case["state"]).tobytes() for case in suite["cases"]]
    goals = [rule.validate_state(case["goal"]).tobytes() for case in suite["cases"]]
    all_states = set()
    forward_pairs = np.zeros(len(starts), dtype=bool)
    for episode in dataset.states:
        first, last = {}, {}
        for i, state in enumerate(episode):
            key = state.tobytes()
            all_states.add(key)
            first.setdefault(key, i)
            last[key] = i
        for c, (start, goal) in enumerate(zip(starts, goals)):
            if start in first and goal in last and first[start] < last[goal]:
                forward_pairs[c] = True
    rows = [{"case_id": case["case_id"], "start_seen": start in all_states,
             "goal_seen": goal in all_states, "forward_pair_seen_in_one_trajectory": bool(pair)}
            for case, start, goal, pair in zip(suite["cases"], starts, goals, forward_pairs)]
    return {
        "definition": "A forward pair occurs if start appears strictly before goal in one training trajectory",
        "cases": rows, "training_unique_states": len(all_states),
        "start_seen_fraction": sum(r["start_seen"] for r in rows) / len(rows),
        "goal_seen_fraction": sum(r["goal_seen"] for r in rows) / len(rows),
        "forward_pair_seen_fraction": float(forward_pairs.mean()),
    }
