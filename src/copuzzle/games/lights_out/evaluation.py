"""Versioned goal tasks and model-agnostic evaluation.

Exact distances and the reference solver are evaluator-only information. Policies
receive the environment observation, never the task record or its answer.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .env import PuzzleEnv
from .oracle import rank, solve
from .rules import PuzzleRule, get_rule

TASK_SCHEMA = "copuzzle.goal_tasks.v0"
RESULT_SCHEMA = "copuzzle.evaluation.v0"

STRUCTURE_TASK_SCHEMA = "copuzzle.goal_tasks.v1"
STRUCTURE_RESULT_SCHEMA = "copuzzle.evaluation.v1"


class TaskSuite(dict):
    """JSON mapping with a private, validated public-structure snapshot.

    The snapshot is outside the mapping so canonical hashing preserves the
    task-file payload. Bind v1 mappings with load_task_suite or with_structure.
    """

    def __init__(self, value, *, structure):
        super().__init__(value)
        self._structure = structure


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_structure_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Structure path must be a relative .npz path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".npz":
        raise ValueError("Structure path must stay within the task-suite directory")
    return path


def _structure_snapshot(suite):
    from .structure import PublicStructure

    structure = getattr(suite, "_structure", None)
    if not isinstance(structure, PublicStructure):
        raise ValueError("Structure-v1 suites must be bound with load_task_suite or with_structure")
    reference = suite.get("structure")
    if not isinstance(reference, dict) or set(reference) != {
        "path", "structure_sha256", "file_sha256", "manifest_sha256"
    }:
        raise ValueError("Invalid task-suite structure reference")
    if reference["structure_sha256"] != structure.structure_sha256:
        raise ValueError("Task-suite structure semantic hash mismatch")
    for name in ("task_name", "family", "n_buttons", "rule_id"):
        if suite.get(name) != getattr(structure, name):
            raise ValueError(f"Task-suite {name} differs from its public structure")
    location = reference["path"]
    hashes = (reference["file_sha256"], reference["manifest_sha256"])
    if location is None:
        if any(value is not None for value in hashes):
            raise ValueError("An unbound structure reference cannot have file hashes")
    else:
        _relative_structure_path(location)
        for value in hashes:
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)):
                raise ValueError("Invalid structure file hash")
    return structure


def with_structure(suite, *, rule):
    """Change only information protocol/artifact identity, preserving cases.

    The returned in-memory v1 suite can be evaluated immediately. Saving binds
    file references in a serialized copy; load that copy when a report must
    identify the exact persisted artifacts.
    """
    from .structure import export_structure

    validate_task_suite(suite, rule)
    structure = export_structure(rule)
    upgraded = deepcopy(dict(suite))
    upgraded.pop("reveal_graph", None)
    upgraded.update({
        "schema_version": STRUCTURE_TASK_SCHEMA,
        "observation_profile": "structure_v1",
        "task_name": structure.task_name,
        "family": structure.family,
        "n_buttons": structure.n_buttons,
        "structure": {
            "path": None, "structure_sha256": structure.structure_sha256,
            "file_sha256": None, "manifest_sha256": None,
        },
    })
    return TaskSuite(upgraded, structure=structure)


def save_task_suite(suite, path, *, structure=None, structure_path=None):
    """Save tasks and publish or verify their public-structure sidecar.

    A PublicStructure upgrades a v0 suite. Existing sidecars are reused only
    after content validation and a matching semantic hash. structure_path must
    lie within the output directory. Inputs and existing tasks are preserved.
    """
    from .structure import PublicStructure, load_structure, rule_from_structure, save_structure

    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Task suite already exists: {path}")
    if structure is not None and not isinstance(structure, PublicStructure):
        raise TypeError("structure must be a PublicStructure")
    if suite.get("schema_version") == TASK_SCHEMA and structure is not None:
        suite = with_structure(suite, rule=rule_from_structure(structure))
    if suite.get("schema_version") == TASK_SCHEMA:
        if structure_path is not None:
            raise ValueError("structure_path requires a structure-v1 suite")
        validate_task_suite(suite)
        return write_json(path, suite)
    if suite.get("schema_version") != STRUCTURE_TASK_SCHEMA:
        raise ValueError("Unsupported goal-task schema")
    if path.suffix != ".json":
        raise ValueError("Structure-v1 task files must have a .json suffix")
    if structure is not None:
        suite = TaskSuite(deepcopy(dict(suite)), structure=structure)
    validate_task_suite(suite)
    snapshot = _structure_snapshot(suite)
    base = path.parent.resolve()
    artifact = Path("structure.npz") if structure_path is None else Path(structure_path)
    if not artifact.is_absolute():
        artifact = base / artifact
    artifact = artifact.resolve()
    try:
        relative = artifact.relative_to(base)
    except ValueError as error:
        raise ValueError("structure_path must lie within the task-suite directory") from error
    _relative_structure_path(relative.as_posix())
    if path.resolve() == artifact.with_suffix(".json"):
        raise ValueError("Task-suite path conflicts with the structure manifest")
    if (artifact.exists() or artifact.is_symlink()
            or artifact.with_suffix(".json").exists() or artifact.with_suffix(".json").is_symlink()):
        if not artifact.with_suffix(".json").resolve().is_relative_to(base):
            raise ValueError("Structure sidecar escapes the task-suite directory")
        existing = load_structure(artifact)
        if existing.structure_sha256 != snapshot.structure_sha256:
            raise ValueError("Existing structure artifact has a different semantic hash")
    else:
        save_structure(snapshot, artifact)
    payload = deepcopy(dict(suite))
    payload["structure"] = {
        "path": relative.as_posix(), "structure_sha256": snapshot.structure_sha256,
        "file_sha256": _file_hash(artifact),
        "manifest_sha256": _file_hash(artifact.with_suffix(".json")),
    }
    return write_json(path, payload)


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
    max_episode_steps=30, max_attempts_per_depth=10000, observation_profile=None,
):
    """Sample reachable goals in exact minimum-press strata.

    These are independently sampled pairs, not a claim of unseen states or
    unseen pairs relative to training data. Audit that separately. Omitting
    observation_profile preserves the legacy v0 format.
    """
    if observation_profile not in {None, "legacy_hidden", "legacy_adjacency", "structure_v1"}:
        raise ValueError("Unsupported observation_profile")
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
    suite = {
        "schema_version": TASK_SCHEMA, "generator_version": "0.1.0",
        "task_name": rule.name, "rule_id": rule.rule_id, "seed": int(seed),
        "max_episode_steps": int(max_episode_steps), "reveal_graph": False,
        "difficulty": "exact_minimum_number_of_presses",
        "sampling": "distinct random starts plus random distinct-button subsets, accepted by exact distance",
        "split_semantics": "independent_pairs; state and training-pair novelty require an overlap audit",
        "cases": cases,
    }
    if observation_profile == "structure_v1":
        return with_structure(suite, rule=rule)
    suite["reveal_graph"] = observation_profile == "legacy_adjacency"
    return suite


def validate_task_suite(suite, rule=None):
    if not isinstance(suite, dict) or suite.get("schema_version") not in {TASK_SCHEMA, STRUCTURE_TASK_SCHEMA}:
        raise ValueError("Unsupported goal-task schema")
    if not isinstance(suite.get("task_name"), str):
        raise ValueError("Task suite must name a rule instance")
    if suite["schema_version"] == STRUCTURE_TASK_SCHEMA:
        from .structure import export_structure, rule_from_structure
        json.dumps(suite, allow_nan=False)
        if suite.get("observation_profile") != "structure_v1" or "reveal_graph" in suite:
            raise ValueError("Structure-v1 tasks require the structure_v1 observation profile")
        if type(suite.get("n_buttons")) is not int:
            raise ValueError("Task-suite n_buttons must be an integer")
        structure = _structure_snapshot(suite)
        if rule is None:
            rule = rule_from_structure(structure)
        elif export_structure(rule).structure_sha256 != structure.structure_sha256:
            raise ValueError("Supplied rule differs from the task-suite public structure")
    else:
        rule = get_rule(suite["task_name"]) if rule is None else rule
        if type(suite.get("reveal_graph")) is not bool:
            raise ValueError("reveal_graph must be explicit")
    if suite.get("rule_id") != rule.rule_id:
        raise ValueError("Task suite and rule identity differ")
    horizon = suite.get("max_episode_steps")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Invalid episode budget")
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
    """Load v0 tasks or bind v1 tasks to validated portable structure files."""
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    suite = json.loads(content)
    if isinstance(suite, dict) and suite.get("schema_version") == STRUCTURE_TASK_SCHEMA:
        from .structure import load_structure

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate task-suite JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError(f"Nonfinite task-suite JSON constant: {value}")

        suite = json.loads(content, object_pairs_hook=unique_object, parse_constant=reject_constant)
        reference = suite.get("structure")
        if not isinstance(reference, dict) or set(reference) != {
            "path", "structure_sha256", "file_sha256", "manifest_sha256"
        }:
            raise ValueError("Invalid task-suite structure reference")
        relative = _relative_structure_path(reference["path"])
        base = path.parent.resolve()
        artifact = (base / relative).resolve()
        if not artifact.is_relative_to(base):
            raise ValueError("Structure path escapes the task-suite directory")
        for artifact_path, key in ((artifact, "file_sha256"),
                                   (artifact.with_suffix(".json"), "manifest_sha256")):
            if not artifact_path.resolve().is_relative_to(base):
                raise ValueError("Structure sidecar escapes the task-suite directory")
            if _file_hash(artifact_path) != reference[key]:
                raise ValueError(f"Task-suite structure {key} mismatch")
        suite = TaskSuite(suite, structure=load_structure(artifact))
    validate_task_suite(suite)
    return suite


class RandomPolicy:
    """An untrained baseline with its own reproducible RNG."""
    label = "random"
    privileged_rule_access = False
    uses_analytic_rules = False
    uses_solver = False
    uses_environment_queries = False

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, observation):
        return int(self.rng.integers(len(observation["board"])))


class OraclePolicy:
    """Privileged exact-rule diagnostic, never a learned baseline."""
    label = "exact_rule_oracle"
    privileged_rule_access = True
    uses_analytic_rules = True
    uses_solver = True
    uses_environment_queries = False

    def __init__(self, rule):
        self.rule = rule

    def __call__(self, observation):
        solution = solve(self.rule, observation["board"], observation["goal"])
        if not solution:
            raise ValueError("Oracle called on an already solved or unreachable task")
        return solution[0]


def _method_declarations(policy, metadata):
    reserved = {
        "schema_version", "rule_id", "task_suite_sha256", "policy", "privileged_rule_access",
        "reveal_graph", "max_episode_steps", "summary", "by_optimal_steps", "cases",
        "observation_profile", "actual_observation_fields", "structure_sha256",
        "structure_file_sha256", "structure_manifest_sha256", "evaluation_mode",
        "training_rule_ids", "method_metadata", "parameter_freezing_enforced",
        "evaluation_mode_semantics", "method_information_source",
    }
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("method_metadata must be a dictionary or None")
    declared = {} if metadata is None else deepcopy(metadata)
    if any(not isinstance(key, str) for key in declared):
        raise ValueError("method_metadata keys must be strings")
    if reserved & declared.keys():
        raise ValueError("method_metadata cannot overwrite core result fields")
    for key in ("uses_analytic_rules", "uses_solver", "uses_environment_queries"):
        known = getattr(policy, key, None)
        value = declared.setdefault(key, known)
        if value is not None and type(value) is not bool:
            raise ValueError(f"{key} must be a boolean or None (unknown)")
        if known is True and value is not True:
            raise ValueError(f"Cannot deny the policy's declared {key}")
    # Check serializability before executing any policy, including nonfinite values.
    json.dumps(declared, allow_nan=False)
    return declared


def evaluate_policy(
    policy: Callable, suite, *, rule=None, policy_label=None,
    evaluation_mode="fixed_instance", training_rule_ids=None, method_metadata=None,
):
    """Evaluate declared fixed-instance or frozen-transfer behavior.

    Only profile-defined observations reach the callable. Optional reset() is
    invoked before each case without task answers. Freezing parameters, model
    inference mode, policy RNG, and honest method declarations remain the
    caller's responsibility: arbitrary Python callables are not sandboxed.
    The evaluator does not append evaluation transitions to training data.

    Default calls on v0 suites preserve the v0 result payload. Structure suites
    or explicit new declarations produce a v1 report with protocol provenance.
    """
    if evaluation_mode not in {"fixed_instance", "frozen_transfer"}:
        raise ValueError("evaluation_mode must be fixed_instance or frozen_transfer")
    if training_rule_ids is not None:
        if (not isinstance(training_rule_ids, (list, tuple))
                or any(not isinstance(value, str) or not value.strip() for value in training_rule_ids)):
            raise ValueError("training_rule_ids must be a list of nonempty rule IDs or None")
        training_rule_ids = list(training_rule_ids)
    declarations = _method_declarations(policy, method_metadata)
    rule = validate_task_suite(suite, rule)
    structure = _structure_snapshot(suite) if suite["schema_version"] == STRUCTURE_TASK_SCHEMA else None
    if structure is not None:
        profile = "structure_v1"
        env = PuzzleEnv(rule, max_episode_steps=suite["max_episode_steps"], observation_profile=profile)
        public_fields = set(structure.observation_fields())
    else:
        profile = "legacy_adjacency" if suite["reveal_graph"] else "legacy_hidden"
        env = PuzzleEnv(rule, max_episode_steps=suite["max_episode_steps"],
                        reveal_graph=suite["reveal_graph"])
        public_fields = {"adjacency"} if suite["reveal_graph"] else set()
    allowed_fields = {"board", "goal", "previous_event"} | public_fields
    results = []
    try:
        for case in suite["cases"]:
            reset = getattr(policy, "reset", None)
            if callable(reset):
                reset()
            obs, _ = env.reset(seed=0, options={"state": case["state"], "goal": case["goal"]})
            total_reward, inference_seconds = 0.0, 0.0
            for step in range(1, suite["max_episode_steps"] + 1):
                if set(obs) != allowed_fields:
                    raise RuntimeError("Environment observation does not match the declared information profile")
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
    report = {
        "schema_version": RESULT_SCHEMA, "rule_id": rule.rule_id,
        "task_suite_sha256": hashlib.sha256(canonical).hexdigest(),
        "policy": policy_label or getattr(policy, "label", type(policy).__name__),
        "privileged_rule_access": bool(getattr(policy, "privileged_rule_access", False)),
        "reveal_graph": "adjacency" in allowed_fields, "max_episode_steps": suite["max_episode_steps"],
        "summary": summarize(results),
        "by_optimal_steps": {str(d): summarize([r for r in results if r["optimal_steps"] == d])
                             for d in sorted({r["optimal_steps"] for r in results})},
        "cases": results,
    }
    if (structure is not None or evaluation_mode != "fixed_instance"
            or training_rule_ids is not None or method_metadata is not None):
        reference = suite.get("structure", {}) if structure is not None else {}
        report.update({
            "schema_version": STRUCTURE_RESULT_SCHEMA,
            "observation_profile": profile,
            "actual_observation_fields": sorted(allowed_fields),
            "structure_sha256": None if structure is None else structure.structure_sha256,
            "structure_file_sha256": reference.get("file_sha256"),
            "structure_manifest_sha256": reference.get("manifest_sha256"),
            "evaluation_mode": evaluation_mode,
            "training_rule_ids": training_rule_ids,
            "method_metadata": declarations,
            "method_information_source": "caller declarations or policy attributes; not independently verified",
            "parameter_freezing_enforced": False,
            "evaluation_mode_semantics": "caller-declared; no inspection of parameter updates",
        })
    return report


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
