"""Exact, versioned evaluation for the bounded Sliding Puzzle and Hanoi families.

Tasks contain evaluator-only distances. Policies receive copied environment
observations, never task records, rule objects, or shortest solutions. Uniform
sampling is over directed pairs at each exact distance; independent seeds do
not establish novelty relative to recorded training trajectories.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .env import ClassicEnv
from .oracle import ExactOracle
from .rules import rule_from_spec


TASK_SCHEMA = "copuzzle.classic_goal_tasks.v1"
RESULT_SCHEMA = "copuzzle.classic_evaluation.v1"
GENERATOR_VERSION = "copuzzle.classic_exact_pairs.v1"
OBSERVATION_PROFILE = "classic_state_v1"
OBSERVATION_FIELDS = ("action_mask", "board", "goal", "previous_event")


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _suite_hash(suite):
    return _hash({key: value for key, value in suite.items() if key != "task_suite_sha256"})


def _pair(rule, state, goal):
    start = tuple(int(x) for x in rule.validate_state(state, name="state"))
    end = tuple(int(x) for x in rule.validate_state(goal, name="goal"))
    return start, end


def _case_id(rule, pair):
    return _hash({"rule_id": rule.rule_id, "state": pair[0], "goal": pair[1]})[:24]


def _goal_pool(rule, oracle):
    """Small, fixed target population; all pairs remain exactly stratified.

    Sliding goals fix the blank at its canonical cell, which permits reuse of
    one exact BFS table even for 3x3. Token order varies. This deliberately
    restricted development distribution is recorded in every suite.
    """
    states = oracle.reachable_states
    if rule.family == "sliding":
        blank = int(np.flatnonzero(rule.canonical_goal == 0)[0])
        states = states[states[:, blank] == 0]
    rng = np.random.default_rng(73191)
    picks = rng.choice(len(states), size=min(16, len(states)), replace=False)
    return [tuple(int(x) for x in states[index]) for index in picks]


SAMPLING = "uniform_directed_pairs_without_replacement_per_exact_distance_in_fixed_goal_pool"
GOAL_DISTRIBUTION = "16_seeded_goals_or_all_if_fewer_in_canonical_component; sliding_blank_at_canonical_cell"


def generate_task_suite(
    rule, *, depths=(1, 2, 3, 4), tasks_per_depth=16,
    max_episode_steps=30, seed=100, exclude_pairs=(),
):
    """Sample exact directed pairs from a declared, bounded goal population.

    Distances and exclusions are evaluator metadata, never policy inputs.
    Sampling uses complete finite candidate lists, so exhausted strata raise
    before retry loops or partial results. Training novelty needs an audit.
    """
    count = _integer(tasks_per_depth, "tasks_per_depth", 1)
    budget = _integer(max_episode_steps, "max_episode_steps", 1)
    seed = _integer(seed, "seed")
    depths = tuple(_integer(value, "depth", 1) for value in depths)
    if not depths or len(set(depths)) != len(depths):
        raise ValueError("depths must be nonempty and distinct")
    if max(depths) > budget:
        raise ValueError("Requested depth exceeds the episode budget")
    oracle = ExactOracle(rule)
    goals = _goal_pool(rule, oracle)
    excluded_pairs = set()
    for item in exclude_pairs:
        if isinstance(item, dict):
            if "state" not in item or "goal" not in item:
                raise ValueError("Excluded case must contain state and goal")
            state, goal = item["state"], item["goal"]
        else:
            try:
                state, goal = item
            except (TypeError, ValueError) as error:
                raise ValueError("Excluded pair must contain state and goal") from error
        excluded_pairs.add(_pair(rule, state, goal))
    buckets = {depth: [] for depth in depths}
    for goal in goals:
        for depth in depths:
            for state in oracle.states_at_distance(goal, depth):
                pair = tuple(int(x) for x in state), goal
                if pair not in excluded_pairs:
                    buckets[depth].append(pair)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 91873]))
    cases = []
    for depth in depths:
        candidates = buckets[depth]
        if count > len(candidates):
            raise ValueError(f"Only {len(candidates)} nonexcluded pairs at distance {depth} in the fixed goal pool; requested {count}")
        for index in rng.choice(len(candidates), size=count, replace=False):
            state, goal = candidates[int(index)]
            cases.append({"case_id": _case_id(rule, (state, goal)),
                          "state": list(state), "goal": list(goal), "optimal_steps": depth})
    suite = {
        "schema_version": TASK_SCHEMA, "generator_version": GENERATOR_VERSION,
        "task_name": rule.name, "family": rule.family,
        "state_size": rule.state_size, "n_actions": rule.n_actions,
        "rule_id": rule.rule_id, "rule_spec": rule.spec(),
        "observation_profile": OBSERVATION_PROFILE,
        "max_episode_steps": budget, "seed": seed,
        "difficulty": "exact_minimum_atomic_operations", "sampling": SAMPLING,
        "goal_distribution": GOAL_DISTRIBUTION, "goal_pool": [list(goal) for goal in goals],
        "split_semantics": "excluded_directed_pairs_only; training/state novelty requires a separate audit",
        "depths": list(depths), "tasks_per_depth": count,
        "excluded_pairs_count": len(excluded_pairs), "excluded_pairs_sha256": _hash(sorted(excluded_pairs)),
        "cases": cases,
    }
    suite["task_suite_sha256"] = _suite_hash(suite)
    validate_task_suite(suite, rule=rule)
    return suite


def validate_task_suite(suite, rule=None):
    """Check identity, integrity, exact distances, uniqueness and budgets."""
    if not isinstance(suite, dict) or suite.get("schema_version") != TASK_SCHEMA:
        raise ValueError("Unsupported classic goal-task schema")
    required = {
        "schema_version", "generator_version", "task_name", "family", "state_size", "n_actions",
        "rule_id", "rule_spec", "observation_profile", "max_episode_steps", "seed", "difficulty",
        "sampling", "split_semantics", "depths", "tasks_per_depth", "excluded_pairs_count",
        "excluded_pairs_sha256", "cases", "task_suite_sha256", "goal_distribution", "goal_pool",
    }
    if set(suite) != required:
        raise ValueError("Task suite has missing or unexpected fields")
    try:
        expected_hash = _suite_hash(suite)
    except (TypeError, ValueError) as error:
        raise ValueError("Task suite must contain finite JSON values") from error
    if suite["task_suite_sha256"] != expected_hash:
        raise ValueError("Task-suite checksum mismatch")
    resolved = rule_from_spec(suite["rule_spec"])
    if rule is not None and (rule.spec() != resolved.spec() or rule.rule_id != resolved.rule_id):
        raise ValueError("Supplied rule differs from the task suite")
    rule = resolved if rule is None else rule
    for field, expected in (("task_name", rule.name), ("family", rule.family),
                            ("state_size", rule.state_size), ("n_actions", rule.n_actions),
                            ("rule_id", rule.rule_id)):
        if suite[field] != expected or type(suite[field]) is not type(expected):
            raise ValueError(f"Task-suite {field} differs from rule identity")
    expected_metadata = {
        "generator_version": GENERATOR_VERSION,
        "observation_profile": OBSERVATION_PROFILE,
        "difficulty": "exact_minimum_atomic_operations",
        "sampling": SAMPLING,
        "goal_distribution": GOAL_DISTRIBUTION,
        "split_semantics": "excluded_directed_pairs_only; training/state novelty requires a separate audit",
    }
    if any(suite[key] != value for key, value in expected_metadata.items()):
        raise ValueError("Unsupported task generation or information contract")
    budget = _integer(suite["max_episode_steps"], "max_episode_steps", 1)
    count = _integer(suite["tasks_per_depth"], "tasks_per_depth", 1)
    _integer(suite["seed"], "seed")
    _integer(suite["excluded_pairs_count"], "excluded_pairs_count")
    digest = suite["excluded_pairs_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Invalid excluded-pair checksum")
    if not isinstance(suite["depths"], list):
        raise ValueError("depths must be a list")
    depths = [_integer(value, "depth", 1) for value in suite["depths"]]
    if not depths or len(set(depths)) != len(depths) or max(depths) > budget:
        raise ValueError("Invalid distance strata or episode budget")
    if not isinstance(suite["cases"], list) or len(suite["cases"]) != count * len(depths):
        raise ValueError("Task suite case count differs from strata")

    oracle = ExactOracle(rule)
    expected_goals = _goal_pool(rule, oracle)
    if suite["goal_pool"] != [list(goal) for goal in expected_goals]:
        raise ValueError("Task suite goal population differs from declared distribution")
    seen = set()
    depth_counts = defaultdict(int)
    for case in suite["cases"]:
        if not isinstance(case, dict) or set(case) != {"case_id", "state", "goal", "optimal_steps"}:
            raise ValueError("Invalid evaluation case fields")
        pair = _pair(rule, case["state"], case["goal"])
        if pair[1] not in expected_goals:
            raise ValueError("Goal is outside the declared target population")
        if pair in seen:
            raise ValueError("Duplicate directed evaluation pair")
        seen.add(pair)
        if case["case_id"] != _case_id(rule, pair):
            raise ValueError("Case identity does not match its state and goal")
        distance = _integer(case["optimal_steps"], "optimal_steps", 1)
        if distance not in depths or distance > budget or oracle.distance(*pair) != distance:
            raise ValueError("Unreachable case, incorrect shortest distance, or invalid episode budget")
        depth_counts[distance] += 1
    if any(depth_counts[depth] != count for depth in depths):
        raise ValueError("Task suite does not populate the declared distance strata")
    return rule


def write_json(path, value):
    """Write JSON without replacing any existing artifact."""
    path = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)
    return path


def save_task_suite(suite, path):
    validate_task_suite(suite)
    return write_json(path, suite)


def load_task_suite(path):
    def unique_fields(pairs):
        fields = {}
        for key, value in pairs:
            if key in fields:
                raise ValueError(f"Duplicate task JSON field: {key}")
            fields[key] = value
        return fields

    suite = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_fields)
    validate_task_suite(suite)
    return suite


class RandomPolicy:
    """Uniform untrained control with an independent random stream."""

    label = "random"
    privileged_rule_access = False
    uses_analytic_rules = False
    uses_solver = False
    uses_environment_queries = False

    def __init__(self, n_actions, seed=0):
        self.n_actions = _integer(n_actions, "n_actions", 1)
        self.rng = np.random.default_rng(_integer(seed, "seed"))

    def __call__(self, observation):
        if np.asarray(observation["action_mask"]).shape != (self.n_actions,):
            raise ValueError("Policy and observation action dimensions differ")
        legal = np.flatnonzero(observation["action_mask"])
        if not len(legal):
            raise ValueError("No legal action available")
        return int(self.rng.choice(legal))


class OraclePolicy:
    """Privileged exact-rule reference; this is not a learned policy."""

    label = "exact_rule_oracle"
    privileged_rule_access = True
    uses_analytic_rules = True
    uses_solver = True
    uses_environment_queries = False

    def __init__(self, rule):
        self.rule = rule
        self.oracle = ExactOracle(rule)

    def __call__(self, observation):
        solution = self.oracle.solve(observation["board"], observation["goal"])
        if not solution:
            raise ValueError("Oracle called on a solved or unreachable case")
        return solution[0]


def _summarize(rows):
    successful = [row for row in rows if row["success"]]
    return {
        "cases": len(rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "timeout_rate": sum(row["timeout"] for row in rows) / len(rows),
        "mean_steps": sum(row["steps"] for row in rows) / len(rows),
        "mean_successful_steps": (sum(row["steps"] for row in successful) / len(successful) if successful else None),
        "mean_excess_steps_on_success": (sum(row["excess_steps_on_success"] for row in successful) / len(successful) if successful else None),
        "policy_calls": sum(row["policy_calls"] for row in rows),
        "inference_seconds": sum(row["inference_seconds"] for row in rows),
    }


def evaluate_policy(policy, suite, *, policy_label=None):
    """Evaluate a callable on copied observations, with reset before each case.

    Arbitrary Python policies are not sandboxed and parameter freezing is the
    caller's responsibility. Unknown method declarations remain null instead
    of being reported as evidence that no privileged information was used.
    """
    if not callable(policy):
        raise TypeError("policy must be callable")
    if policy_label is not None and (not isinstance(policy_label, str) or not policy_label.strip()):
        raise ValueError("policy_label must be a nonempty string or None")
    # Snapshot protects the task/report identity from accidental caller mutation.
    suite = deepcopy(suite)
    rule = validate_task_suite(suite)
    declarations = {}
    for name in ("privileged_rule_access", "uses_analytic_rules", "uses_solver", "uses_environment_queries"):
        value = getattr(policy, name, None)
        if value is not None and type(value) is not bool:
            raise ValueError(f"{name} must be a boolean or None")
        declarations[name] = value
    if isinstance(policy, OraclePolicy) and policy.rule.rule_id != rule.rule_id:
        raise ValueError("Oracle policy and evaluation suite have different rules")
    results = []
    env = ClassicEnv(rule, max_episode_steps=suite["max_episode_steps"])
    try:
        for case in suite["cases"]:
            reset = getattr(policy, "reset", None)
            if callable(reset):
                reset()
            observation, _ = env.reset(seed=0, options={"state": case["state"], "goal": case["goal"]})
            elapsed, total_reward = 0.0, 0.0
            for step in range(1, suite["max_episode_steps"] + 1):
                if set(observation) != set(OBSERVATION_FIELDS):
                    raise RuntimeError("Observation fields differ from the classic information contract")
                inputs = {key: np.array(value, copy=True) for key, value in observation.items()}
                began = time.perf_counter()
                action = policy(inputs)
                elapsed += time.perf_counter() - began
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            results.append({
                "case_id": case["case_id"], "optimal_steps": case["optimal_steps"],
                "success": bool(info["is_success"]), "timeout": bool(truncated),
                "steps": step, "policy_calls": step, "return": float(total_reward),
                "inference_seconds": elapsed,
                "excess_steps_on_success": step - case["optimal_steps"] if terminated else None,
            })
    finally:
        env.close()
    return {
        "schema_version": RESULT_SCHEMA,
        "task_name": rule.name, "family": rule.family,
        "rule_id": rule.rule_id, "task_suite_sha256": suite["task_suite_sha256"],
        "policy": policy_label or getattr(policy, "label", type(policy).__name__),
        "observation_profile": OBSERVATION_PROFILE,
        "actual_observation_fields": list(OBSERVATION_FIELDS),
        "max_episode_steps": suite["max_episode_steps"],
        "method_declarations": declarations,
        "privileged_rule_access": declarations["privileged_rule_access"],
        "method_information_source": "policy attributes; not independently verified",
        "parameter_freezing_enforced": False,
        "summary": _summarize(results),
        "by_distance": {str(depth): _summarize([row for row in results if row["optimal_steps"] == depth])
                        for depth in sorted(suite["depths"])},
        "cases": results,
    }


def audit_overlap(suite, dataset):
    """Audit starts, goals and strictly ordered within-trajectory pairs."""
    rule = validate_task_suite(suite)
    dataset.validate()
    if (dataset.state_size != rule.state_size or dataset.n_actions != rule.n_actions
            or dataset.manifest.get("rule_id") != rule.rule_id):
        raise ValueError("Overlap audit requires matching dataset and task rule identities")
    pairs = [_pair(rule, case["state"], case["goal"]) for case in suite["cases"]]
    all_states, observed = set(), set()
    for episode in dataset.states:
        first, last = {}, {}
        for index, state in enumerate(episode):
            key = tuple(int(x) for x in state)
            all_states.add(key)
            first.setdefault(key, index)
            last[key] = index
        for start, goal in pairs:
            if start in first and goal in last and first[start] < last[goal]:
                observed.add((start, goal))
    rows = [
        {"case_id": case["case_id"], "start_seen": start in all_states,
         "goal_seen": goal in all_states, "forward_pair_seen_in_one_trajectory": (start, goal) in observed}
        for case, (start, goal) in zip(suite["cases"], pairs)
    ]
    return {
        "definition": "A forward pair occurs if start appears strictly before goal in one training trajectory",
        "task_suite_sha256": suite["task_suite_sha256"], "rule_id": rule.rule_id,
        "training_unique_states": len(all_states),
        "start_seen_fraction": sum(row["start_seen"] for row in rows) / len(rows),
        "goal_seen_fraction": sum(row["goal_seen"] for row in rows) / len(rows),
        "forward_pair_seen_fraction": len(observed) / len(rows),
        "cases": rows,
    }
