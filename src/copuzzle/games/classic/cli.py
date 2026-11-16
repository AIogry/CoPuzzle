"""Prepare and audit bounded Sliding Puzzle/Hanoi development artifacts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from .data import generate_dataset, load_dataset, dataset_coverage, dataset_identity
from .evaluation import (OraclePolicy, RandomPolicy, audit_overlap, evaluate_policy,
                         generate_task_suite, load_task_suite, save_task_suite, write_json)
from .oracle import ExactOracle
from .rules import get_rule


PRESETS = {
    "sliding-2x3": {"gym_id": "CoPuzzle/Sliding2x3-v0", "depths": (1, 3, 6, 10)},
    "sliding-3x3": {"gym_id": "CoPuzzle/Sliding3x3-v0", "depths": (1, 4, 8, 12, 16)},
    "hanoi-2": {"gym_id": "CoPuzzle/Hanoi2-v0", "depths": (1, 2, 3)},
    "hanoi-3": {"gym_id": "CoPuzzle/Hanoi3-v0", "depths": (1, 3, 5, 7)},
    "hanoi-4": {"gym_id": "CoPuzzle/Hanoi4-v0", "depths": (1, 4, 8, 12, 15)},
    "hanoi-5": {"gym_id": "CoPuzzle/Hanoi5-v0", "depths": (1, 5, 10, 15, 23)},
    "hanoi-6": {"gym_id": "CoPuzzle/Hanoi6-v0", "depths": (1, 5, 10, 20, 30)},
}
DEFAULT_TASKS = ("sliding-2x3", "hanoi-3", "hanoi-4", "hanoi-5")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _versions():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ("copuzzle", "numpy", "gymnasium")
    }}


def _depths(value):
    if value == "preset":
        return value
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected positive comma-separated distances") from error
    if not result or min(result) < 1 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Distances must be positive and distinct")
    return result


def _selected_depths(task, value):
    if value != "preset":
        return value
    if task not in PRESETS:
        raise ValueError("Custom task sizes require explicit --depths")
    return PRESETS[task]["depths"]


def _generate(rule, output, episodes, val_episodes, horizon, seed):
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    train = generate_dataset(rule, episodes, horizon, seed, "train")
    val = generate_dataset(rule, val_episodes, horizon, seed, "val")
    output.mkdir(parents=True)
    train.save(output / "train.npz")
    val.save(output / "val.npz")
    # Audit the saved representation, not just the in-memory collector.
    train, val = load_dataset(output / "train.npz"), load_dataset(output / "val.npz")
    train.validate(rule)
    val.validate(rule)
    record = {"task": rule.name, "rule_id": rule.rule_id, "seed": seed,
              "versions": _versions(), "train": dataset_coverage(train), "val": dataset_coverage(val),
              "split_semantics": "independent trajectories, not state-disjoint",
              "files": {p.name: _hash(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "generation.json", record)
    return record


def _run_evaluation(suite, policy_name, seed, train=None):
    rule = get_rule(suite["task_name"])
    policy = OraclePolicy(rule) if policy_name == "oracle" else RandomPolicy(rule.n_actions, seed=seed)
    report = evaluate_policy(policy, suite, policy_label=policy_name)
    report["versions"] = _versions()
    report["policy_seed"] = seed if policy_name == "random" else None
    if train is not None:
        report["training_data"] = dataset_identity(train)
        report["training_overlap"] = audit_overlap(suite, train)
    return report


def build_collection(output, *, tasks=DEFAULT_TASKS, episodes=1000, val_episodes=100,
                     horizon=30, per_depth=16, seed=0):
    """Write a complete development collection; final manifest signals success.

    Data are independent trajectories, while dev/test goal pairs are disjoint.
    Neither property implies unseen training states or pairs. Each split's
    overlap with training is therefore reported separately.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    tasks = tuple(tasks)
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in PRESETS for task in tasks):
        raise ValueError("Collection tasks must be distinct classic presets")
    for value, name, lower in ((episodes, "episodes", 1), (val_episodes, "val_episodes", 1),
                               (horizon, "horizon", 1), (per_depth, "per_depth", 1), (seed, "seed", 0)):
        if type(value) is not int or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if horizon < max(max(PRESETS[task]["depths"]) for task in tasks):
        raise ValueError("horizon must cover every selected exact task distance")
    # Generate both suites before creating output so infeasible pair populations
    # fail cleanly. Small goals reuse exact BFS tables on the larger 3x3 task.
    suites = {}
    for task in tasks:
        rule = get_rule(task)
        dev = generate_task_suite(rule, depths=PRESETS[task]["depths"], tasks_per_depth=per_depth,
                                  max_episode_steps=horizon, seed=seed + 100)
        test = generate_task_suite(rule, depths=PRESETS[task]["depths"], tasks_per_depth=per_depth,
                                   max_episode_steps=horizon, seed=seed + 200, exclude_pairs=dev["cases"])
        suites[task] = dev, test
    # Read source identity before collection and confirm it did not change.
    source_root = Path(__file__).resolve().parent
    source_files = {str(p.relative_to(source_root)): _hash(p) for p in sorted(source_root.rglob("*.py"))}
    output.mkdir(parents=True)
    records = {}
    for task in tasks:
        rule, folder = get_rule(task), output / task
        record = _generate(rule, folder, episodes, val_episodes, horizon, seed)
        train, val = load_dataset(folder / "train.npz"), load_dataset(folder / "val.npz")
        dev, test = suites[task]
        save_task_suite(dev, folder / "dev_tasks.json")
        save_task_suite(test, folder / "test_tasks.json")
        for split in ("dev", "test"):
            suite = load_task_suite(folder / f"{split}_tasks.json")
            for policy_name in ("random", "oracle"):
                report = _run_evaluation(suite, policy_name, seed, train)
                write_json(folder / f"{split}_{policy_name}.json", report)
                if policy_name == "oracle" and report["summary"]["success_rate"] != 1.0:
                    raise RuntimeError("Exact oracle failed a generated task")
        oracle = ExactOracle(rule)
        record.update(exact_identity_component_size=len(oracle.reachable_states),
                      canonical_goal_max_distance=oracle.max_distance,
                      dev_cases=len(dev["cases"]), test_cases=len(test["cases"]),
                      dev_test_directed_pair_overlap=0)
        records[task] = record
    if source_files != {str(p.relative_to(source_root)): _hash(p) for p in sorted(source_root.rglob("*.py"))}:
        raise RuntimeError("Package source changed during generation; collection is incomplete")
    manifest = {
        "schema_version": "copuzzle.classic_collection.v1", "status": "complete",
        "purpose": "development; not a frozen public benchmark", "versions": _versions(),
        "configuration": {"tasks": list(tasks), "episodes": episodes, "val_episodes": val_episodes,
                          "horizon": horizon, "per_depth": per_depth, "seed": seed,
                          "dev_seed": seed + 100, "test_seed": seed + 200},
        "tasks": records, "source_sha256": source_files,
        "files": {str(p.relative_to(output)): _hash(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "evaluation_policy": "Train on train.npz; tune on dev_tasks.json; freeze before test_tasks.json. Baselines are development checks.",
        "split_semantics": "Disjoint dev/test directed pairs; training overlap measured, not excluded.",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="copuzzle classic", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the development presets")
    generate = commands.add_parser("generate", help="Generate independent train/validation trajectories")
    generate.add_argument("--task", default="sliding-2x3")
    collection = commands.add_parser("collection", help="Generate data, dev/test tasks, baselines and checksums")
    collection.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=tuple(PRESETS))
    collection.add_argument("--per-depth", type=int, default=16)
    for command in (generate, collection):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--episodes", type=int, default=1000)
        command.add_argument("--val-episodes", type=int, default=100)
        command.add_argument("--horizon", type=int, default=30)
        command.add_argument("--seed", type=int, default=0)
    tasks = commands.add_parser("tasks", help="Generate evaluator-only tasks with exact distances")
    tasks.add_argument("--task", default="sliding-2x3")
    tasks.add_argument("--output", type=Path, required=True)
    tasks.add_argument("--depths", type=_depths, default="preset")
    tasks.add_argument("--per-depth", type=int, default=16)
    tasks.add_argument("--horizon", type=int, default=30)
    tasks.add_argument("--seed", type=int, default=100)
    tasks.add_argument("--exclude-suite", type=Path, help="Exclude every directed pair in a prior suite")
    validate = commands.add_parser("validate", help="Validate a classic trajectory archive")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--check-rule", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Run random or privileged exact-oracle evaluation")
    evaluate.add_argument("--tasks", type=Path, required=True)
    evaluate.add_argument("--policy", choices=("random", "oracle"), default="random")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--training-data", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            result = [{"task_id": name, "gym_id": spec["gym_id"], "state_size": get_rule(name).state_size,
                       "n_actions": get_rule(name).n_actions, "evaluation_depths": list(spec["depths"])}
                      for name, spec in PRESETS.items()]
        elif args.command == "generate":
            result = _generate(get_rule(args.task), args.output, args.episodes, args.val_episodes,
                               args.horizon, args.seed)
        elif args.command == "collection":
            result = build_collection(args.output, tasks=args.tasks, episodes=args.episodes,
                                      val_episodes=args.val_episodes, horizon=args.horizon,
                                      per_depth=args.per_depth, seed=args.seed)
            result = {"output": str(args.output), "status": result["status"],
                      "tasks": list(result["tasks"]), "manifest": str(args.output / "manifest.json")}
        elif args.command == "tasks":
            rule = get_rule(args.task)
            excluded = load_task_suite(args.exclude_suite) if args.exclude_suite else None
            if excluded is not None and excluded["rule_id"] != rule.rule_id:
                raise ValueError("Excluded suite uses another rule")
            result = generate_task_suite(rule, depths=_selected_depths(args.task, args.depths),
                                         tasks_per_depth=args.per_depth, max_episode_steps=args.horizon,
                                         seed=args.seed, exclude_pairs=excluded["cases"] if excluded else ())
            save_task_suite(result, args.output)
        elif args.command == "validate":
            dataset = load_dataset(args.dataset)
            dataset.validate(get_rule(dataset.manifest["task_name"]) if args.check_rule else None)
            result = {"valid": True, "rule_checked": args.check_rule, **dataset_coverage(dataset)}
        elif args.command == "evaluate":
            result = _run_evaluation(load_task_suite(args.tasks), args.policy, args.seed,
                                     load_dataset(args.training_data) if args.training_data else None)
            write_json(args.output, result)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
