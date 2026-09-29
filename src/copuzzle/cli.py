"""Generate and evaluate CoPuzzle artifacts for the implemented Lights Out family."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

from copuzzle.games.lights_out.data import generate_dataset, load_dataset
from copuzzle.games.lights_out.evaluation import (
    OraclePolicy, RandomPolicy, audit_overlap, evaluate_policy, generate_task_suite,
    load_task_suite, write_json,
)
from copuzzle.games.lights_out.rules import get_rule


def _integers(text):
    try:
        values = tuple(int(item) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("Expected positive integers")
    return values


def _versions():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ("copuzzle", "numpy", "gymnasium")
    }}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="copuzzle", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Create fixed offline train/validation trajectories")
    generate.add_argument("--task", default="grid-3x3")
    generate.add_argument("--output", type=Path, required=True, help="New output directory")
    generate.add_argument("--episodes", type=int, default=1000)
    generate.add_argument("--val-episodes", type=int, default=100)
    generate.add_argument("--horizon", type=int, default=30)
    generate.add_argument("--seed", type=int, default=0)

    tasks = commands.add_parser("tasks", help="Generate evaluator-only goals with exact difficulty")
    tasks.add_argument("--task", default="grid-3x3")
    tasks.add_argument("--output", type=Path, required=True)
    tasks.add_argument("--per-depth", type=int, default=16)
    tasks.add_argument("--depths", type=_integers, default=(1, 2, 3, 4))
    tasks.add_argument("--horizon", type=int, default=30)
    tasks.add_argument("--seed", type=int, default=100)
    tasks.add_argument("--reveal-graph", action="store_true", help="Explicit alternative information condition")

    validate = commands.add_parser("validate", help="Check an archive and optional true-rule consistency")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--check-rule", action="store_true", help="Privileged data-integrity audit")

    evaluate = commands.add_parser("evaluate", help="Run an untrained baseline or privileged oracle")
    evaluate.add_argument("--tasks", type=Path, required=True)
    evaluate.add_argument("--policy", choices=("random", "oracle"), default="random")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--training-data", type=Path, help="Audit overlaps with this training archive")

    diagnose = commands.add_parser("diagnose", help="Fit an offline effect lookup and test predictions")
    diagnose.add_argument("--train", type=Path, required=True)
    diagnose.add_argument("--test", type=Path, required=True)
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--lengths", type=_integers, default=(2, 4, 8))
    diagnose.add_argument("--samples", type=int, default=128)
    diagnose.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            rule = get_rule(args.task)
            if args.output.exists():
                raise FileExistsError(f"Output directory already exists: {args.output}")
            train = generate_dataset(rule, args.episodes, args.horizon, args.seed, "train")
            val = generate_dataset(rule, args.val_episodes, args.horizon, args.seed, "val")
            args.output.mkdir(parents=True)
            train.save(args.output / "train.npz")
            val.save(args.output / "val.npz")
            summary = {"task": args.task, "rule_id": rule.rule_id, "horizon": args.horizon,
                       "train_episodes": args.episodes, "val_episodes": args.val_episodes,
                       "seed": args.seed, "versions": _versions()}
            write_json(args.output / "generation.json", summary)
        elif args.command == "tasks":
            suite = generate_task_suite(get_rule(args.task), per_depth=args.per_depth,
                                        depths=args.depths, seed=args.seed,
                                        max_episode_steps=args.horizon)
            suite["reveal_graph"] = args.reveal_graph
            write_json(args.output, suite)
            summary = {"task": args.task, "cases": len(suite["cases"]),
                       "depths": list(args.depths), "reveal_graph": args.reveal_graph}
        elif args.command == "validate":
            dataset = load_dataset(args.dataset)
            if args.check_rule:
                dataset.validate(get_rule(dataset.manifest["task_name"]))
            summary = {"valid": True, "rule_checked": args.check_rule,
                       "episodes": dataset.num_episodes, "horizon": dataset.horizon,
                       "buttons": dataset.n_buttons, "sha256": _hash(args.dataset)}
        elif args.command == "evaluate":
            suite = load_task_suite(args.tasks)
            rule = get_rule(suite["task_name"])
            policy = OraclePolicy(rule) if args.policy == "oracle" else RandomPolicy(args.seed)
            report = evaluate_policy(policy, suite, rule=rule)
            report.update({"versions": _versions(), "policy_seed": args.seed,
                           "task_file_sha256": _hash(args.tasks)})
            if args.training_data:
                report["training_overlap"] = audit_overlap(load_dataset(args.training_data), suite)
                report["training_archive_sha256"] = _hash(args.training_data)
            write_json(args.output, report)
            summary = {"policy": report["policy"], **report["summary"]}
        else:
            from copuzzle.games.lights_out.diagnostics import EffectLookup, composition_examples, prediction_metrics
            train, test = load_dataset(args.train), load_dataset(args.test)
            if train.manifest["rule_id"] != test.manifest["rule_id"]:
                raise ValueError("The lookup diagnostic requires the same fixed rule in train and test")
            model = EffectLookup().fit(train)
            states = test.states[:, :-1].reshape(-1, test.n_buttons)
            next_states = test.states[:, 1:].reshape(-1, test.n_buttons)
            events = test.events.reshape(-1, test.n_buttons)
            single = prediction_metrics(model.predict(states, events), states, next_states)
            composed = {}
            for length in args.lengths:
                examples = composition_examples(test, length=length, num_examples=args.samples, seed=args.seed)
                predicted = model.predict_sequence(examples["states"], examples["events"])
                composed[str(length)] = prediction_metrics(predicted, examples["states"], examples["next_states"])
            report = {"schema_version": "copuzzle.lookup_diagnostic.v0", "versions": _versions(),
                      "baseline": "effect_lookup_from_fixed_offline_data", "is_rl": False,
                      "privileged_rule_access": False, "rule_id": train.manifest["rule_id"],
                      "train_sha256": _hash(args.train), "test_sha256": _hash(args.test),
                      "single_step": single, "composition": composed,
                      "scope": "same fixed rule; independent trajectories; no claim of unseen states or compositions"}
            write_json(args.output, report)
            summary = report
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
