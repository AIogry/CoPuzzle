"""Versioned research scope, held-out data and exact-difficulty qualification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .spec import INSTANCES, REGIMES, specification


def main(argv=None):
    parser = argparse.ArgumentParser(prog="copuzzle research", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("spec", help="Print the machine-readable release-candidate protocol")
    build = commands.add_parser("build", help="Generate one independent task/regime bundle")
    build.add_argument("--task", choices=[i.task for i in INSTANCES], required=True)
    build.add_argument("--regime", choices=REGIMES, required=True)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--per-cell", type=int, default=8, help="C cases per length and G cases per tier; R is exhaustive")
    build.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("validate", help="Recompute splits, transitions, distances and labels")
    check.add_argument("bundle", type=Path)
    refs = commands.add_parser("references", help="Run data-only lookup, random and privileged oracle controls")
    refs.add_argument("bundle", type=Path)
    refs.add_argument("--split", choices=("dev", "test"), default="dev")
    refs.add_argument("--seed", type=int, default=0)
    refs.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score", help="Score a frozen R/C prediction NPZ")
    score.add_argument("bundle", type=Path)
    score.add_argument("--probe", choices=("R", "C"), required=True)
    score.add_argument("--split", choices=("dev", "test"), default="dev")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "spec":
            result = specification()
        elif args.command == "build":
            from .bundle import write_bundle
            result = {"bundle": str(write_bundle(args.output, args.task, args.regime, seed=args.seed, per_cell=args.per_cell))}
        elif args.command == "validate":
            from .bundle import validate_bundle
            result = validate_bundle(args.bundle)
        else:
            if args.output.exists():
                raise FileExistsError(f"output already exists: {args.output}")
            from .bundle import validate_bundle
            validate_bundle(args.bundle)
            if args.command == "references":
                from .evaluation import reference_report
                result = reference_report(args.bundle, split=args.split, seed=args.seed)
            else:
                from .bundle import _read_npz
                from .evaluation import score_predictions
                result = score_predictions(args.bundle, args.probe, _read_npz(args.predictions), split=args.split)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    return 0
