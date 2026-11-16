"""Development-only reference audit; never evaluates a learned final-test model."""
import argparse
import hashlib
import json
from pathlib import Path

from copuzzle.benchmark.generation import write_bundle
from copuzzle.benchmark.spec import INSTANCES, REGIMES
from copuzzle_baselines.controls.evaluate import evaluate_control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "reports").mkdir()
    rows = []
    for item in INSTANCES:
        for regime in REGIMES:
            name = f"{item.task}-{regime}"
            bundle = write_bundle(args.output / "data" / name, item.task, regime, 0, max_per_cell=8)
            specs = [("random", 0), ("optimal_oracle", 0)]
            specs += [(kind, budget) for kind in ("lookup", "oracle_dynamics") for budget in (32, 128)]
            prior = "fitted_xor" if item.family == "lights_out" else "fitted_position" if item.family in ("pancake", "topspin") else None
            if prior:
                specs += [(prior, budget) for budget in (32, 128)]
            for kind, budget in specs:
                report = evaluate_control(bundle, kind, split="dev", model_query_limit=budget)
                destination = args.output / "reports" / f"{name}-{kind}-b{budget}.json"
                destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                row = {"task": item.task, "regime": regime, "kind": kind, "budget": report["settings"]["model_query_limit"],
                       "G_tier_macro": report["G"]["tier_macro_success_rate"],
                       "R_balanced_joint": report["R"].get("balanced_joint_exact"),
                       "C_length_macro": report["C"].get("length_macro_whole_prefix_exact")}
                rows.append(row)
                print(json.dumps(row), flush=True)
    summary = {"purpose": "development reference audit", "data_seed": 0, "split": "dev", "formal_fits": 0, "results": rows}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    files = sorted(path for path in args.output.rglob("*") if path.is_file())
    (args.output / "inventory.json").write_text(json.dumps({path.relative_to(args.output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}, indent=2) + "\n")


if __name__ == "__main__":
    main()
