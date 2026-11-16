"""Benchmark v1 specification, exact populations and immutable data inventories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .spec import INSTANCES, REGIMES, specification


def main(argv=None):
    parser = argparse.ArgumentParser(prog="copuzzle benchmark", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("spec")
    build = commands.add_parser("build", help="Build a development bundle with explicit population parameters")
    build.add_argument("--task", required=True, choices=[s.task for s in INSTANCES])
    build.add_argument("--regime", required=True, choices=REGIMES)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--max-per-cell", type=int, default=32)
    build.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("population")
    audit.add_argument("--task", required=True, choices=[s.task for s in INSTANCES])
    audit.add_argument("--regime", required=True, choices=REGIMES)
    audit.add_argument("--seed", type=int, required=True)
    check = commands.add_parser("validate")
    check.add_argument("bundle", type=Path)
    score = commands.add_parser("score", help="Score frozen R/C prediction arrays")
    score.add_argument("bundle", type=Path)
    score.add_argument("--probe", choices=("R", "C"), required=True)
    score.add_argument("--split", choices=("dev", "test"), default="dev")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("freeze-data", help="Freeze all 36 prespecified task/condition/seed datasets")
    freeze.add_argument("--output", type=Path, required=True)
    inventory = commands.add_parser("validate-inventory")
    inventory.add_argument("directory", type=Path)
    inventory.add_argument("--semantic", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "spec":
            result = specification()
        elif args.command == "build":
            from .generation import write_bundle
            result = {"bundle": str(write_bundle(args.output, args.task, args.regime, args.seed, max_per_cell=args.max_per_cell))}
        elif args.command == "population":
            from .population import population
            result = population(args.task, args.regime, args.seed)
        elif args.command == "validate":
            from .generation import validate_bundle
            result = validate_bundle(args.bundle)
        elif args.command == "score":
            from copuzzle.research.bundle import _read_npz
            from .evaluation import score_predictions
            from .generation import file_hash, validate_bundle
            if args.output.exists():
                raise FileExistsError(args.output)
            validate_bundle(args.bundle)
            result = score_predictions(args.bundle, args.probe, _read_npz(args.predictions), split=args.split)
            result.update(prediction_sha256=file_hash(args.predictions), bundle_manifest_sha256=file_hash(args.bundle / "manifest.json"), split=args.split)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
        elif args.command == "freeze-data":
            from .collection import freeze_data, public_descriptor
            freeze_data(args.output, progress=lambda row: print(json.dumps(row), flush=True))
            result = public_descriptor(args.output)
        else:
            from .collection import validate_inventory
            result = validate_inventory(args.directory, semantic=args.semantic)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))
    return 0
