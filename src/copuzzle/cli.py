"""Generate and evaluate CoPuzzle artifacts for the Lights Out family."""
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
    load_task_suite, validate_task_suite, write_json,
)
from copuzzle.games.lights_out.rules import get_rule, graph_rule
from copuzzle.games.lights_out.presets import TASK_PRESETS, get_task_preset


def _integers(text):
    try:
        values = tuple(int(item) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("Expected positive integers")
    return values


def _depths(text):
    return "preset" if text == "preset" else _integers(text)


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


def _rule_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task", help="Named instance (default: grid-3x3)")
    group.add_argument("--structure", type=Path, help="Public structure NPZ, including a custom graph")


def _resolve_rule(args, fallback="grid-3x3"):
    if args.structure is not None:
        from copuzzle.games.lights_out.structure import load_structure, rule_from_structure
        return rule_from_structure(load_structure(args.structure))
    return get_rule(args.task or fallback)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="copuzzle", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List standard task sizes and suggested evaluation depths")
    generate = commands.add_parser("generate", help="Create fixed offline train/validation trajectories")
    _rule_arguments(generate)
    generate.add_argument("--output", type=Path, required=True, help="New output directory")
    generate.add_argument("--episodes", type=int, default=1000)
    generate.add_argument("--val-episodes", type=int, default=100)
    generate.add_argument("--horizon", type=int, default=30)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--observation-profile", choices=("legacy_hidden", "structure_v1"),
                          default="legacy_hidden", help="structure_v1 also writes a public offline bundle")

    structure = commands.add_parser("structure", help="Export Grid geometry or a Graph adjacency artifact")
    source = structure.add_mutually_exclusive_group(required=True)
    source.add_argument("--task")
    source.add_argument("--adjacency", type=Path, help="JSON array defining a custom undirected graph")
    structure.add_argument("--name", default="custom", help="Name for a custom graph")
    structure.add_argument("--output", type=Path, required=True, help="New .npz path; also writes .json")

    bundle = commands.add_parser("bundle", help="Copy existing trajectories into a checked structure bundle")
    _rule_arguments(bundle)
    bundle.add_argument("--train", type=Path, required=True)
    bundle.add_argument("--val", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True, help="New output directory")

    tasks = commands.add_parser("tasks", help="Generate evaluator-only goals with exact difficulty")
    _rule_arguments(tasks)
    tasks.add_argument("--output", type=Path, required=True)
    tasks.add_argument("--per-depth", type=int, default=16)
    tasks.add_argument("--depths", type=_depths, default=(1, 2, 3, 4),
                       help="Comma-separated exact distances, or preset (default: 1,2,3,4)")
    tasks.add_argument("--horizon", type=int, default=30)
    tasks.add_argument("--seed", type=int, default=100)
    tasks.add_argument("--reveal-graph", action="store_true", default=None,
                       help="Legacy adjacency observation; use separately from a profile")
    tasks.add_argument("--observation-profile", choices=("legacy_hidden", "legacy_adjacency", "structure_v1"))

    migrate = commands.add_parser("migrate-tasks", help="Attach structure to existing goals without resampling")
    migrate.add_argument("--tasks", type=Path, required=True)
    migrate.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate", help="Check a trajectory archive or offline bundle")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--check-rule", action="store_true", help="Privileged transition-integrity audit")

    evaluate = commands.add_parser("evaluate", help="Run an untrained baseline or privileged oracle")
    evaluate.add_argument("--tasks", type=Path, required=True)
    evaluate.add_argument("--policy", choices=("random", "oracle"), default="random")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--training-data", type=Path, help="Audit overlaps with this same-rule training archive")

    diagnose = commands.add_parser("diagnose", help="Fit an offline effect lookup and test predictions")
    diagnose.add_argument("--train", type=Path, required=True)
    diagnose.add_argument("--test", type=Path, required=True)
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--lengths", type=_integers, default=(2, 4, 8))
    diagnose.add_argument("--samples", type=int, default=128)
    diagnose.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            summary = [{"task_id": preset.task_id, "gym_id": preset.gym_id,
                        "buttons": get_rule(preset.task_id).n_buttons,
                        "evaluation_depths": list(preset.evaluation_depths)}
                       for preset in TASK_PRESETS]
        elif args.command == "generate":
            rule = _resolve_rule(args)
            if args.output.exists():
                raise FileExistsError(f"Output directory already exists: {args.output}")
            train = generate_dataset(rule, args.episodes, args.horizon, args.seed, "train")
            val = generate_dataset(rule, args.val_episodes, args.horizon, args.seed, "val")
            args.output.mkdir(parents=True)
            train.save(args.output / "train.npz")
            val.save(args.output / "val.npz")
            summary = {"task": rule.name, "rule_id": rule.rule_id, "horizon": args.horizon,
                       "train_episodes": args.episodes, "val_episodes": args.val_episodes,
                       "seed": args.seed, "versions": _versions(),
                       "observation_profile": args.observation_profile}
            if args.observation_profile == "structure_v1":
                from copuzzle.games.lights_out.bundle import write_offline_bundle
                bundle_path = write_offline_bundle(args.output, train_path=args.output / "train.npz",
                                                   val_path=args.output / "val.npz", rule=rule)
                summary["bundle"] = bundle_path.name
            write_json(args.output / "generation.json", summary)
        elif args.command == "structure":
            from copuzzle.games.lights_out.structure import export_structure, save_structure
            if args.adjacency is not None:
                with args.adjacency.open(encoding="utf-8") as stream:
                    rule = graph_rule(json.load(stream), name=args.name)
            else:
                rule = get_rule(args.task)
            public = export_structure(rule)
            save_structure(public, args.output)
            summary = {"task": rule.name, "family": public.family,
                       "structure_sha256": public.structure_sha256, "output": str(args.output)}
        elif args.command == "bundle":
            from copuzzle.games.lights_out.bundle import create_offline_bundle, load_offline_bundle
            fallback = load_dataset(args.train).manifest["task_name"]
            rule = _resolve_rule(args, fallback=fallback)
            path = create_offline_bundle(args.output, train_path=args.train, val_path=args.val, rule=rule)
            public = load_offline_bundle(path).structure
            summary = {"bundle": str(path), "structure_sha256": public.structure_sha256}
        elif args.command == "tasks":
            from copuzzle.games.lights_out.evaluation import save_task_suite, with_structure
            if args.observation_profile is not None and args.reveal_graph is not None:
                raise ValueError("Use --observation-profile separately from --reveal-graph")
            rule = _resolve_rule(args)
            if args.depths == "preset":
                args.depths = get_task_preset(rule.name).evaluation_depths
            profile = args.observation_profile or ("legacy_adjacency" if args.reveal_graph else "legacy_hidden")
            suite = generate_task_suite(rule, per_depth=args.per_depth, depths=args.depths,
                                        seed=args.seed, max_episode_steps=args.horizon)
            if profile == "structure_v1":
                suite = with_structure(suite, rule=rule)
                save_task_suite(suite, args.output)
            else:
                suite["reveal_graph"] = profile == "legacy_adjacency"
                # The legacy loader resolves task names; custom instances need v1 artifacts.
                try:
                    named_rule = get_rule(rule.name)
                except ValueError as error:
                    raise ValueError("Custom graphs require --observation-profile structure_v1") from error
                if named_rule.rule_id != rule.rule_id:
                    raise ValueError("This custom rule requires --observation-profile structure_v1")
                write_json(args.output, suite)
            summary = {"task": rule.name, "cases": len(suite["cases"]),
                       "depths": list(args.depths), "observation_profile": profile}
        elif args.command == "migrate-tasks":
            from copuzzle.games.lights_out.evaluation import save_task_suite, with_structure
            suite = load_task_suite(args.tasks)
            suite = with_structure(suite, rule=validate_task_suite(suite))
            save_task_suite(suite, args.output)
            summary = {"cases": len(suite["cases"]), "observation_profile": "structure_v1",
                       "output": str(args.output)}
        elif args.command == "validate":
            is_bundle = args.dataset.is_dir() or args.dataset.suffix == ".json"
            if is_bundle:
                from copuzzle.games.lights_out.bundle import load_offline_bundle
                from copuzzle.games.lights_out.structure import rule_from_structure
                loaded = load_offline_bundle(args.dataset)
                dataset = loaded.dataset
                if args.check_rule:
                    rule = rule_from_structure(loaded.structure)
                    dataset.validate(rule)
                    load_offline_bundle(args.dataset, split="val").dataset.validate(rule)
                summary = {"valid": True, "rule_checked": args.check_rule,
                           "rule_checked_splits": ["train", "val"] if args.check_rule else [],
                           "episodes": dataset.num_episodes, "horizon": dataset.horizon,
                           "buttons": dataset.n_buttons,
                           "structure_sha256": loaded.structure.structure_sha256,
                           "observation_profile": "structure_v1"}
            else:
                dataset = load_dataset(args.dataset)
                if args.check_rule:
                    dataset.validate(get_rule(dataset.manifest["task_name"]))
                summary = {"valid": True, "rule_checked": args.check_rule,
                           "episodes": dataset.num_episodes, "horizon": dataset.horizon,
                           "buttons": dataset.n_buttons, "sha256": _hash(args.dataset)}
        elif args.command == "evaluate":
            suite = load_task_suite(args.tasks)
            rule = validate_task_suite(suite)
            policy = OraclePolicy(rule) if args.policy == "oracle" else RandomPolicy(args.seed)
            report = evaluate_policy(policy, suite, rule=rule)
            report.update({"versions": _versions(), "policy_seed": args.seed,
                           "task_file_sha256": _hash(args.tasks)})
            if args.training_data:
                report["training_overlap"] = audit_overlap(load_dataset(args.training_data), suite, rule=rule)
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
