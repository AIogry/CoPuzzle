"""Qualify report rendering with explicitly synthetic data, never experiment outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

from copuzzle_baselines.study.report import LEARNED, REGIMES, TASK_NAMES, build_report
from copuzzle_baselines.study.statistics import summarize


def synthetic_analysis(root):
    """Rendering fixture only: none of these values are experimental outcomes."""
    root.mkdir()
    families = {task: "lights_out" if task.startswith("grid-") else task.split("-")[0] for task in TASK_NAMES}
    rows = []
    for coverage in (.25, .5, 1.0):
        for regime in REGIMES:
            for t, task in enumerate(TASK_NAMES):
                for d, data_seed in enumerate((101, 202, 303)):
                    for fit_seed in (0, 1, 2):
                        for i, (method, budget, _) in enumerate(LEARNED):
                            success = .08+.07*i+.03*t+.08*coverage+.05*d+.025*fit_seed
                            rows.append({"task": task, "regime": regime, "method": method, "budget": budget, "coverage": coverage,
                                         "data_seed": data_seed, "fit_seed": fit_seed, "status": "completed",
                                         "metrics": {"R": .8 if method == "dynamics" else None, "C": .4 if method == "dynamics" else None,
                                                     "G": success, "G_normalized_cost": 1-success/2}})
    statistics = summarize(rows, data_seeds=[101, 202, 303], fit_seeds=[0, 1, 2], families=families, repetitions=20)
    costs = {"known_process_work_seconds": 0, "attempts_with_unknown_process_duration": 0, "unlinked_attempt_directories": []}
    for name, value in (("rows.json", rows), ("statistics.json", statistics), ("costs.json", costs)):
        (root / name).write_text(json.dumps(value))
    manifest = {"schema": "copuzzle.study_analysis.v1", "synthetic": True, "study_sha256": "synthetic-fixture",
                "role": "qualification", "split": "dev", "evaluation_jobs": len(rows), "outcomes": {"completed": len(rows)},
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = synthetic_analysis(args.output / "analysis")
    result = build_report(source, args.output / "report", figures=True)
    assert result["synthetic"] and result["role"] == "qualification"
    assert len(list((args.output / "report/figures").glob("*.png"))) == 3
    print(json.dumps({"qualified": True, "synthetic": True, "formal_results": False,
                      "render_versions": result["render_versions"]}))


if __name__ == "__main__":
    main()
