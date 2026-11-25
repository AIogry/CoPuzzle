"""Standalone experience-only training and frozen evaluation commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import Config, METHODS


def main(argv=None):
    parser = argparse.ArgumentParser(prog="copuzzle-baselines", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    source = fit.add_mutually_exclusive_group(required=True)
    source.add_argument("--learner", type=Path)
    source.add_argument("--trajectories", type=Path, help="checked native train.npz (development only)")
    fit.add_argument("--trajectory-family", choices=("lights_out", "permutation", "classic"))
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--config", type=Path)
    fit.add_argument("--method", choices=METHODS)
    fit.add_argument("--seed", type=int)
    fit.add_argument("--steps", type=int, default=1000)
    fit.add_argument("--checkpoint-every", type=int, default=1000)
    fit.add_argument("--log-every", type=int, default=100)
    fit.add_argument("--resume", type=Path)
    pred = commands.add_parser("predict")
    pred.add_argument("--checkpoint", type=Path, required=True)
    pred.add_argument("--learner", type=Path, required=True)
    pred.add_argument("--probe", choices=("R", "C"), required=True)
    pred.add_argument("--split", choices=("dev", "test"), default="dev")
    pred.add_argument("--output", type=Path, required=True)
    pred.add_argument("--test-lock", type=Path)
    test = commands.add_parser("evaluate")
    test.add_argument("--checkpoint", type=Path, required=True)
    test.add_argument("--bundle", type=Path, required=True)
    test.add_argument("--split", choices=("dev", "test"), default="dev")
    test.add_argument("--model-query-limit", type=int)
    test.add_argument("--output", type=Path, required=True)
    test.add_argument("--test-lock", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        if args.command == "train":
            if (args.trajectories is not None) != (args.trajectory_family is not None):
                raise ValueError("--trajectories and --trajectory-family must be supplied together")
            from .runtime import load_checkpoint, train
            settings = json.loads(args.config.read_text()) if args.config else {}
            seed = args.seed
            if args.resume:
                restored, _, _ = load_checkpoint(args.resume)
                if args.config is None:
                    settings = restored.config.to_dict()
                if seed is None:
                    seed = restored.seed
            if args.method is not None:
                settings["method"] = args.method
            result = train(args.trajectories if args.trajectories is not None else args.learner,
                           args.output, config=Config.from_dict(settings), seed=0 if seed is None else seed,
                           steps=args.steps, checkpoint_every=args.checkpoint_every, log_every=args.log_every,
                           resume=args.resume, trajectory_family=args.trajectory_family)
            print(json.dumps({key: result[key] for key in ("status", "completed_step", "train_seconds", "checkpoints")}, indent=2))
        elif args.command == "predict":
            from .runtime import predict
            result = predict(args.checkpoint, args.learner, args.probe, split=args.split, test_lock=args.test_lock)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as stream:
                np.savez_compressed(stream, **result)
            print(json.dumps({"output": str(args.output), "probe": args.probe, "split": args.split}))
        else:
            from .runtime import evaluate
            result = evaluate(args.checkpoint, args.bundle, split=args.split, model_query_limit=args.model_query_limit, test_lock=args.test_lock)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
            print(json.dumps({"output": str(args.output), "method": result["method"], "G": result["G"]["aggregate"]}, indent=2))
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
