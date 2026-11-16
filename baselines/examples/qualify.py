"""Exercise all core instances/methods; outputs are engineering qualification only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from copuzzle.research import INSTANCES, REGIMES
from copuzzle.research.bundle import write_bundle
from copuzzle_baselines.config import Config, METHODS
from copuzzle_baselines.runtime import evaluate, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    rows = []
    for task in INSTANCES:
        for regime in REGIMES:
            name = f"{task.task}-{regime}"
            bundle = write_bundle(args.output / "data" / name, task.task, regime, seed=0, per_cell=1)
            for method in METHODS:
                run_path = args.output / "runs" / name / method
                config = Config(method=method, hidden_dims=(16,), batch_size=32)
                run = train(bundle / "learner", run_path, config=config, seed=0, steps=args.steps,
                            checkpoint_every=args.steps, log_every=args.steps)
                checkpoint = run_path / run["checkpoints"][-1]["file"]
                report = evaluate(checkpoint, bundle, split="dev", model_query_limit=32 if method == "dynamics" else 0)
                report_path = run_path / "evaluation-dev.json"
                report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                row = {"task": task.task, "regime": regime, "method": method,
                       "training_episodes": run["data"]["selected_rows"], "updates": run["completed_step"],
                       "train_seconds": run["train_seconds"], "eval_seconds": report["evaluation_seconds"],
                       "frozen": report["learner_state_unchanged"], "G_cases": report["G"]["aggregate"]["cases"],
                       "G_success": report["G"]["aggregate"]["success_rate"],
                       "R_supported": report["R"]["supported"], "C_supported": report["C"]["supported"]}
                rows.append(row)
                print(json.dumps(row), flush=True)
    summary = {"purpose": "engineering qualification; short development fits, not formal baseline results",
               "fit_seed": 0, "data_seed": 0, "per_cell": 1, "fits": len(rows), "formal_fits": 0,
               "all_frozen": all(row["frozen"] for row in rows), "results": rows}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    files = sorted(path for path in args.output.rglob("*") if path.is_file())
    (args.output / "inventory.json").write_text(json.dumps({path.relative_to(args.output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    main()
