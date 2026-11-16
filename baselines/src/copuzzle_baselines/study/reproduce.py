"""Independent installed-package replay of a fixed, performance-blind subset."""
from __future__ import annotations

import argparse
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time


def selected_jobs(study):
    """Data seed 101, full coverage, learned fit seed 0, every task/condition.

    Reproduce both learned model budgets. Reference replay uses seed 0 and
    32-query planning, plus random and budget-exempt optimal controls.
    Selection never reads performance values.
    """
    data = {d["id"]: d for d in study["datasets"]}
    fits = {f["id"]: f for f in study["fit_jobs"]}
    result = []
    for job in study["evaluation_jobs"]:
        fit = fits[job["fit_id"]]
        if data[job["dataset_id"]]["seed"] == 101 and fit["fit_seed"] == 0 and fit["coverage"] == 1:
            result.append(("test", job))
    for job in study["control_jobs"]:
        settings = job["settings"]
        if (data[job["dataset_id"]]["seed"] == 101 and settings["coverage"] == 1 and settings["seed"] == 0
                and settings["model_query_limit"] in (None, 0, 32)):
            result.append(("control-test", job))
    if len(result) != 104 or sum(phase == "test" for phase, _ in result) != 48:
        raise ValueError("independent replay subset differs from its declared 48 learned / 56 reference evaluations")
    return result


def comparable(value):
    """Retain scores, every case, budgets and provenance; exclude measured time."""
    if isinstance(value, dict):
        return {k: comparable(v) for k, v in value.items() if k not in {"policy_seconds", "evaluation_seconds", "fit_seconds"}}
    if isinstance(value, list):
        return [comparable(v) for v in value]
    return value


def reproduce(root, output, *, forbid_source):
    # Set before importing the core, evaluator or learner. This process must
    # never depend on a sibling research checkout or optional physics engine.
    class BlockUnrelated(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"copuzzle_lab", "mujoco", "dm_control"}:
                raise AssertionError(f"unexpected reproduction dependency: {fullname}")
    finder = BlockUnrelated()
    sys.meta_path.insert(0, finder)
    try:
        return _reproduce(root, output, forbid_source=forbid_source)
    finally:
        sys.meta_path.remove(finder)


def _reproduce(root, output, *, forbid_source):
    import copuzzle
    import copuzzle_baselines
    from ..runtime import _json, evaluate, sources, versions
    from ..controls.evaluate import evaluate_control, source_sha256
    from .analysis import collect_rows
    from .evidence import verify_fit
    from .registry import data_directory, dataset, job_for, load_study
    from .storage import active, file_hash, inside, write_new
    root, output, forbidden = Path(root).resolve(), Path(output).resolve(), Path(forbid_source).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(data_directory(root)) or output.is_relative_to(root / "source"):
        raise ValueError("reproduction output must stay outside frozen data/source directories")
    for module in (copuzzle, copuzzle_baselines):
        if Path(module.__file__).resolve().is_relative_to(forbidden):
            raise ValueError("independent replay must use installed packages outside the checkout")
    study, checksum = load_study(root)
    if study["role"] != "formal" or sources() != study["learner_sources"] or versions() != study["dependency_versions"] or source_sha256() != study["control_source_sha256"]:
        raise ValueError("installed numerical implementation/runtime differs from the formal study")
    if os.environ.get("JAX_PLATFORMS") != study["backend"]:
        raise ValueError("set JAX_PLATFORMS to the registered backend before replay; automatic device selection is not used")
    import jax
    if jax.default_backend() != study["backend"]:
        raise ValueError("reproduction backend differs from the registered study")
    if active(root / ".controller.lock") or any(active(p) for p in (root / "jobs").glob("*/*/.lease")):
        raise ValueError("independent replay requires an idle, complete study")
    jobs = selected_jobs(study)
    _, originals = collect_rows(root, "test")
    output.mkdir(parents=True)
    write_new(output / "selection.json", {"study_sha256": checksum, "data_seed": 101, "coverage": 1.0, "fit_seed": 0,
                                         "reference_seed": 0, "reference_planning_budget": 32,
                                         "jobs": [{"phase": phase, "id": j["id"]} for phase, j in jobs]})
    records, started = [], time.perf_counter()
    for phase, job in jobs:
        record = {"job_id": job["id"], "phase": phase}
        try:
            original = originals.get(job["id"])
            if original is None:
                raise ValueError("selected original evaluation was unavailable; cannot claim an independent score reproduction")
            bundle = data_directory(root) / dataset(study, job["dataset_id"])["path"]
            lock = root / "selection/test-lock.json"
            if job["kind"] == "learned":
                fit = job_for(study, "fit", job["fit_id"])
                fitted = verify_fit(root, study, checksum, fit)
                report = evaluate(inside(root, fitted["checkpoint"]), bundle, split="test", model_query_limit=job["budget"], test_lock=lock)
            else:
                settings = job["settings"]
                report = evaluate_control(bundle, settings["kind"], split="test", model_query_limit=settings["model_query_limit"] or 0,
                    coverage=settings["coverage"], coverage_seed=settings["coverage_seed"], seed=settings["seed"], test_lock=lock)
            path = output / f"{job['id']}.json"
            write_new(path, report)
            record.update(report=path.name, report_sha256=file_hash(path))
            if comparable(report) != comparable(original):
                raise ValueError("replayed scores/cases/provenance differ; inspect the retained fresh report")
            record["status"] = "verified"
        except Exception as error:
            record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        records.append(record)
        _json(output / "progress.json", {"expected": len(jobs), "finished": len(records), "failed": sum(r["status"] == "failed" for r in records)})
        print(json.dumps({"job": job["id"], "status": record["status"], "finished": len(records), "expected": len(jobs)}), flush=True)
    result = {"schema": "copuzzle.independent_reproduction.v1", "study_sha256": checksum,
              "valid": all(r["status"] == "verified" for r in records), "expected_evaluations": len(jobs), "records": records,
              "elapsed_seconds": time.perf_counter()-started, "versions": versions(), "learner_sources": sources(),
              "execution": {"backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()]},
              "core_path": copuzzle.__file__, "baseline_path": copuzzle_baselines.__file__,
              "sibling_and_physics_imports_blocked": True, "excluded_comparison_fields": ["policy_seconds", "evaluation_seconds", "fit_seconds"],
              "scope": "installed-package replay of fixed checkpoints/cases on this runtime; no refitting or cross-platform equivalence claim"}
    (output / "reproduce_source.py").write_bytes(Path(__file__).read_bytes())
    result["files"] = {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_new(output / "manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forbid-source", type=Path, required=True)
    args = parser.parse_args(argv)
    result = reproduce(args.study, args.output, forbid_source=args.forbid_source)
    print(json.dumps({"valid": result["valid"], "expected_evaluations": result["expected_evaluations"]}))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
