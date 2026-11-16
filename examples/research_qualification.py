"""Generate and qualify all research candidates; never a learned-method benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from copuzzle.research import INSTANCES, PROTOCOL, REGIMES, load_probe, specification
from copuzzle.research.bundle import validate_bundle, write_bundle
from copuzzle.research.evaluation import reference_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-cell", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "reports").mkdir()
    rows = []
    for item in INSTANCES:
        for regime in REGIMES:
            name = f"{item.task}-{regime}"
            path = write_bundle(args.output / "bundles" / name, item.task, regime,
                                seed=args.seed, per_cell=args.per_cell)
            audit = validate_bundle(path)
            row = {"task": item.task, "regime": regime, "training_episodes": audit["training_episodes"],
                   "query_counts": audit["audit"]["query_counts"],
                   "discarded_cross_state_queries": audit["audit"]["discarded_cross_state_queries"],
                   "training_components": len(audit["audit"]["training_legal_components"]),
                   "largest_training_component": max(audit["audit"]["training_legal_components"]),
                   "composition_effect_overlap": audit["audit"]["composition_dev_test_effect_overlap"],
                   "scores": {}, "evaluation_cases": {}}
            for split in ("dev", "test"):
                report = reference_report(path, split=split, seed=args.seed)
                (args.output / "reports" / f"{name}-{split}.json").write_text(json.dumps(report, indent=2) + "\n")
                oracle = report["privileged_oracle_G"]["aggregate"]
                assert oracle["success_rate"] == 1 and oracle["mean_success_excess"] == 0
                if regime == "transition_holdout":
                    assert report["training_lookup"]["C"]["whole_prefix_exact"] == 1
                row["scores"][split] = {"lookup_R": report["training_lookup"]["R"]["balanced_joint_exact"],
                                         "lookup_C": report["training_lookup"]["C"]["whole_prefix_exact"],
                                         "random_G": report["random_G"]["aggregate"]["success_rate"],
                                         "oracle_G": oracle["success_rate"]}
                row["evaluation_cases"][split] = {probe: len(load_probe(path / "learner", probe, split)["board"])
                                                   for probe in ("R", "C", "G")}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("task", "regime", "training_episodes", "scores")}), flush=True)
    summary = {"protocol": PROTOCOL, "purpose": "development qualification; no trained models",
               "seed": args.seed, "per_cell": args.per_cell, "instances": rows,
               "total_training_episodes": sum(row["training_episodes"] for row in rows),
               "total_evaluation_cases": sum(sum(counts.values()) for row in rows for counts in row["evaluation_cases"].values())}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "specification.json").write_text(json.dumps(specification(), indent=2) + "\n")
    files = sorted(p for p in args.output.rglob("*") if p.is_file())
    manifest = {p.relative_to(args.output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output / "inventory.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "instances"}), flush=True)


if __name__ == "__main__":
    main()
