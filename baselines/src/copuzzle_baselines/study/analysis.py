"""Validate complete study outcomes, retain failures, and produce paired summaries."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path
import shutil

import numpy as np

from ..config import fingerprint
from .evidence import verify_fit, verify_seal
from .registry import data_directory, dataset, fit_context, job_for, load_study, validate_study
from .statistics import summarize
from .storage import active, file_hash, inside, read_json, verified_record, write_new


def _close(actual, expected, name):
    if expected is None:
        valid = actual is None
    elif isinstance(expected, (bool, str, dict, list)):
        valid = actual == expected
    else:
        valid = isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12)
    if not valid:
        raise ValueError(f"report differs from recomputed {name}")


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"invalid {name}")


def _goal_summary(rows, *, exempt=False):
    if not rows:
        return {"cases": 0}
    n, attempts = len(rows), sum(r["attempts"] for r in rows)
    successes = [r for r in rows if r["success"]]
    return {"cases": n, "success_rate": len(successes)/n,
            "mean_attempts_all": sum(r["attempts"] for r in rows)/n,
            "mean_penalized_cost_all": sum(r["penalized_cost"] for r in rows)/n,
            "illegal_attempt_rate": sum(r["illegal_attempts"] for r in rows)/max(1, attempts),
            "policy_error_rate": sum(r["outcome"] == "policy_error" for r in rows)/n,
            "mean_success_excess": sum(r["success_excess"] for r in successes)/len(successes) if successes else None,
            "model_queries": None if exempt else sum(r["model_queries"] for r in rows),
            "model_calls": None if exempt else sum(r["model_calls"] for r in rows),
            "policy_seconds": sum(r["policy_seconds"] for r in rows)}


def audit_goals(goal, *, expected_cases, horizon, budget, labels=None):
    """Recompute all G aggregates from retained cases, including failed policies."""
    rows = goal["cases"]
    if len(rows) != expected_cases or [r["case_index"] for r in rows] != list(range(expected_cases)):
        raise ValueError("goal cases omitted, reordered or duplicated")
    _close(goal["control_budget"], horizon, "control budget")
    _close(goal["model_query_limit_per_decision"], budget, "model budget")
    exempt = budget is None
    for i, row in enumerate(rows):
        if type(row["success"]) is not bool or type(row["empirically_reachable"]) is not bool:
            raise ValueError("goal boolean fields are malformed")
        _integer(row["tier"], 0, 2, "goal tier")
        _integer(row["distance"], 1, horizon, "exact distance")
        _integer(row["attempts"], 0, horizon, "attempt count")
        _integer(row["illegal_attempts"], 0, row["attempts"], "illegal attempts")
        if row["outcome"] not in ("success", "budget_exhausted", "policy_error"):
            raise ValueError("unknown goal outcome")
        if row["success"] != (row["outcome"] == "success") or (row["outcome"] == "policy_error") != bool(row["error"]):
            raise ValueError("goal failure labels disagree")
        if row["outcome"] == "budget_exhausted" and row["attempts"] != horizon:
            raise ValueError("budget exhaustion has an incomplete attempt count")
        if row["success"] and row["attempts"] < row["distance"]:
            raise ValueError("goal solved faster than the exact distance")
        _close(row["penalized_cost"], row["attempts"] if row["success"] else horizon, "failure-inclusive cost")
        _close(row["success_excess"], row["attempts"]-row["distance"] if row["success"] else None, "success excess")
        if exempt:
            if row["model_queries"] is not None or row["model_calls"] is not None:
                raise ValueError("privileged optimal solver must not claim zero query cost")
        else:
            decisions = row["attempts"] + int(row["outcome"] == "policy_error")
            _integer(row["model_queries"], 0, decisions*budget, "model queries")
            _integer(row["model_calls"], 0, row["model_queries"], "model calls")
        if not math.isfinite(row["policy_seconds"]) or row["policy_seconds"] < 0:
            raise ValueError("invalid policy time")
        if labels is not None:
            for name in ("distance", "tier", "empirically_reachable"):
                _close(row[name], labels[name][i].item(), f"frozen {name}")
    groups = {"aggregate": _goal_summary(rows, exempt=exempt)}
    tiers = {name: _goal_summary([r for r in rows if r["tier"] == i], exempt=exempt)
             for i, name in enumerate(("easy", "medium", "hard"))}
    if any(not r["cases"] for r in tiers.values()):
        raise ValueError("missing goal difficulty tier")
    for name, summary in [("aggregate", groups["aggregate"]), *[(f"tier/{k}", v) for k, v in tiers.items()],
                          *[(f"reach/{name}", _goal_summary([r for r in rows if r["empirically_reachable"] == flag], exempt=exempt))
                            for name, flag in (("reachable", True), ("unreachable", False))]]:
        actual = goal["aggregate"] if name == "aggregate" else goal["by_tier"][name[5:]] if name.startswith("tier/") else goal["by_empirical_reachability"][name[6:]]
        for key, value in summary.items():
            _close(actual.get(key), value, f"G {name}/{key}")
    _close(goal["tier_macro_success_rate"], np.mean([r["success_rate"] for r in tiers.values()]), "equal-tier success")
    _close(goal["tier_macro_penalized_cost"], np.mean([r["mean_penalized_cost_all"] for r in tiers.values()]), "equal-tier cost")
    return groups["aggregate"]


def _rate(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("prediction score must be a finite rate")
    return value


def audit_prediction_scores(report, probe, expected_cases):
    row = report[probe]
    _close(row["cases"], expected_cases, f"{probe} cases")
    groups = row["by_legality"] if probe == "R" else row["by_length"]
    if sum(g["cases"] for g in groups.values()) != expected_cases:
        raise ValueError("prediction groups omit cases")
    metrics = ("next_board_exact", "legality_accuracy", "joint_exact") if probe == "R" else ("whole_prefix_exact", "final_board_exact")
    for metric in metrics:
        for group in groups.values():
            _integer(group["cases"], 0, expected_cases, "prediction group cases")
            if group["cases"]:
                _rate(group[metric])
            elif group[metric] is not None:
                raise ValueError("empty prediction groups must use null")
        _close(_rate(row[metric]), sum(g[metric]*g["cases"] for g in groups.values() if g["cases"])/expected_cases, f"{probe} pooled {metric}")
    if probe == "R":
        _close(_rate(row["balanced_joint_exact"]), np.mean([g["joint_exact"] for g in groups.values() if g["cases"]]), "R legality-balanced mean")
    else:
        for metric in metrics:
            _close(_rate(row[f"length_macro_{metric}"]), np.mean([g[metric] for g in groups.values()]), f"C equal-length {metric}")


def _check_metric_source(probe, study):
    from copuzzle.benchmark.evaluation import source_hashes
    if probe.get("protocol") != "copuzzle.rule-composition.v1":
        raise ValueError("wrong metric protocol")
    sources = probe["metric_sources"]
    if set(sources) != set(source_hashes()) or any(study["source_snapshot"].get(f"copuzzle/{name}") != digest for name, digest in sources.items()):
        raise ValueError("metric source differs from registered snapshot")


def validate_report(report, record, study, checksum, job, data, *, split, fit_record=None, lock_sha=None, labels=None):
    from copuzzle.benchmark.spec import instance
    learned = job["kind"] == "learned"
    fit = job_for(study, "fit", job["fit_id"]) if learned else None
    method = fit["method"] if learned else job["settings"]["kind"]
    supported = method not in ("gcbc", "gciql", "random")
    budget = job["budget"] if learned else job["settings"]["model_query_limit"]
    if report["split"] != split:
        raise ValueError("evaluation split differs")
    if learned:
        expected = {"method": method, "fit_seed": fit["fit_seed"], "step": fit["steps"], "config": fit["config"],
                    "bundle_manifest_sha256": data["manifest_sha256"], "versions": study["dependency_versions"],
                    "checkpoint_sha256": fit_record["checkpoint_evidence"]["checkpoint_sha256"],
                    "learner_state_unchanged": True,
                    "fit_context": fit_context(study, checksum, fit) if study["role"] == "formal" else None}
        if any(report.get(k) != v for k, v in expected.items()) or fingerprint(report["data"]) != fit["data_sha256"]:
            raise ValueError("evaluation/checkpoint registration binding differs")
    elif (report["kind"] != method or report["settings"] != job["settings"] or report["sources_sha256"] != study["control_source_sha256"]):
        raise ValueError("reference control differs from registration")
    if split == "test":
        auth = {"selection_lock_sha256": lock_sha, "job_id": fit["id"] if learned else job["id"]}
        if report["test_authorization"] != auth:
            raise ValueError("report lacks exact final-test authorization")
    for probe in ("R", "C"):
        if report[probe].get("supported") is not supported:
            raise ValueError("probe support does not match method capabilities")
        if supported:
            _check_metric_source(report[probe], study)
            audit_prediction_scores(report, probe, data["evaluation_cases"][f"{probe}_{split}"])
    _check_metric_source(report["G"], study)
    spec = instance(data["task"])
    audit_goals(report["G"], expected_cases=data["evaluation_cases"][f"G_{split}"], horizon=spec.control_budget, budget=budget, labels=labels)
    return report


def collect_rows(root, split, *, include_controls=True):
    """Require all registered jobs terminal before emitting a scientific table."""
    from copuzzle.benchmark.spec import instance
    root = Path(root)
    validate_study(root, verify_data=True, verify_snapshot=True)
    study, checksum = load_study(root)
    lock_sha = None
    if split == "test":
        verify_seal(root / "selection/test-lock.json", verify_receipts=True)
        lock_sha = file_hash(root / "selection/test-lock.json")
    elif split != "dev":
        raise ValueError("unknown analysis split")
    if study["role"] == "formal" and not include_controls:
        raise ValueError("formal analysis requires the complete reference cohort")
    @lru_cache(maxsize=None)
    def labels(dataset_id):
        from copuzzle.research.bundle import _read_npz
        bundle = data_directory(root) / dataset(study, dataset_id)["path"]
        truth = _read_npz(bundle / "evaluator/answers.npz")
        return {"distance": truth[f"G_{split}_distance"], "tier": truth[f"G_{split}_tier"],
                "empirically_reachable": truth[f"G_{split}_empirical_distance"] >= 0}
    rows, reports = [], {}
    phases = [(split, study["evaluation_jobs"])]
    if include_controls:
        phases.append((f"control-{split}", study["control_jobs"]))
    for phase, jobs in phases:
        for job in jobs:
            if active(root / f"jobs/{phase}/{job['id']}/.lease"):
                raise ValueError("analysis cannot include active evaluations")
            record = verified_record(root, phase, job["id"])
            if record is None or record["status"] not in ("completed", "failed"):
                raise ValueError(f"nonterminal evaluation: {job['id']}")
            if record["study_sha256"] != checksum:
                raise ValueError("evaluation record belongs to another study")
            data = dataset(study, job["dataset_id"])
            fit = job_for(study, "fit", job["fit_id"]) if job["kind"] == "learned" else None
            fit_record = verify_fit(root, study, checksum, fit) if fit else None
            settings = fit["config"] if fit else job["settings"]
            method = fit["method"] if fit else settings["kind"]
            supported = method not in ("gcbc", "gciql", "random")
            spec = instance(data["task"])
            row = {"job_id": job["id"], "phase": phase, "kind": job["kind"], "method": method,
                   "task": data["task"], "family": spec.family, "regime": data["regime"], "data_seed": data["seed"],
                   "fit_seed": fit["fit_seed"] if fit else settings["seed"] if method == "random" else None,
                   "coverage": settings["coverage"], "budget": job["budget"] if fit else settings["model_query_limit"],
                   "status": record["status"], "fit_status": fit_record["status"] if fit_record else None,
                   "expected_goal_cases": data["evaluation_cases"][f"G_{split}"],
                   "report": record.get("report"), "report_sha256": record.get("report_sha256"),
                   "metrics": {"R": 0.0 if supported else None, "C": 0.0 if supported else None,
                               "G": 0.0, "G_normalized_cost": 1.0}, "error": record.get("error")}
            if record["status"] == "completed":
                if file_hash(inside(root, record["report"])) != record["report_sha256"]:
                    raise ValueError("evaluation report changed")
                report = validate_report(read_json(inside(root, record["report"])), record, study, checksum, job, data,
                                         split=split, fit_record=fit_record, lock_sha=lock_sha, labels=labels(data["id"]))
                reports[job["id"]] = report
                row["metrics"] = {"R": report["R"]["balanced_joint_exact"] if supported else None,
                                  "C": report["C"]["length_macro_whole_prefix_exact"] if supported else None,
                                  "G": report["G"]["tier_macro_success_rate"],
                                  "G_normalized_cost": report["G"]["tier_macro_penalized_cost"]/spec.control_budget}
                row["probe_strata"] = {"R": report["R"].get("by_legality"), "C": report["C"].get("by_length"),
                                       "G": report["G"]["by_tier"]}
                row["goal_diagnostics"] = report["G"]["aggregate"]
                row["composition_context"] = report.get("C_context_diagnostic")
            elif not record.get("error"):
                raise ValueError("failed evaluation has no error record")
            rows.append(row)
    return rows, reports


def attempt_costs(root):
    """Count every attempt once. Interrupted unknown durations are not zero work."""
    root = Path(root).resolve()
    result, tracked = [], set()
    for status in sorted((root / "jobs").glob("*/*/status.json")):
        receipt, prior_attempt = read_json(status), math.inf
        phase, job_id = status.parent.parent.name, status.parent.name
        while receipt is not None:
            path = inside(root, receipt["record"])
            if file_hash(path) != receipt["record_sha256"]:
                raise ValueError("historical cost record changed")
            record = read_json(path)
            if (record["phase"] != phase or record["job_id"] != job_id or record["attempt"] >= prior_attempt
                    or any(record[k] != receipt[k] for k in ("phase", "job_id", "attempt", "status"))):
                raise ValueError("invalid attempt history chain")
            prior_attempt = record["attempt"]
            tracked.add(path.parent.resolve())
            training = path.parent / "training/run.json"
            run = read_json(training) if training.exists() else {}
            if record.get("training_manifest") is not None and (
                    inside(root, record["training_manifest"]) != training.resolve()
                    or file_hash(training) != record["training_manifest_sha256"]):
                raise ValueError("recorded training cost manifest changed")
            for duration in (record.get("process_work_seconds"), run.get("train_seconds"), run.get("setup_seconds")):
                if duration is not None and (not math.isfinite(duration) or duration < 0):
                    raise ValueError("invalid recorded work duration")
            result.append({"phase": phase, "job_id": job_id, "attempt": record["attempt"],
                           "status": record["status"], "record": path.relative_to(root).as_posix(),
                           "record_sha256": file_hash(path), "training_manifest_sha256": file_hash(training) if training.exists() else None,
                           "process_work_seconds": record.get("process_work_seconds"),
                           "train_seconds": run.get("train_seconds"), "setup_seconds": run.get("setup_seconds"),
                           "completed_step": run.get("completed_step"), "start_step": run.get("start_step"),
                           "duration_is_partial": record["status"] == "running"})
            receipt = record.get("previous_receipt")
    unlinked = [p.relative_to(root).as_posix() for p in sorted((root / "jobs").glob("*/*/attempts/*"))
                if p.is_dir() and p.resolve() not in tracked]
    return {"attempts": result,
            "known_process_work_seconds": sum(r["process_work_seconds"] for r in result if r["process_work_seconds"] is not None),
            "attempts_with_unknown_process_duration": sum(r["process_work_seconds"] is None for r in result),
            "unlinked_attempt_directories": unlinked,
            "note": "sum of worker elapsed work, not wall time/CPU time; process launch and unrecorded crash work excluded; shared fits counted once"}


def analyze(root, output, *, split="test", repetitions=10000, seed=0, predictions=None, plan=None):
    from copuzzle.benchmark.spec import INSTANCES
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, reports = collect_rows(root, split)
    study, checksum = load_study(root)
    if study["role"] == "formal" and split == "test" and (repetitions != 10000 or seed != 0 or predictions is None or plan is None):
        raise ValueError("formal test analysis requires the registered plan, 10000 draws, seed 0 and per-case prediction evidence")
    if plan is not None:
        contract = read_json(plan)
        uncertainty = contract.get("uncertainty", {})
        if (contract.get("schema") != "copuzzle.analysis_plan.v1" or contract.get("study_sha256") != checksum
                or uncertainty.get("repetitions") != repetitions or uncertainty.get("seed") != seed
                or uncertainty.get("confidence") != .95):
            raise ValueError("analysis plan differs from the study/resampling settings")
    prediction_evidence = None
    if predictions is not None:
        from .prediction_archive import verify_export
        prediction_evidence = verify_export(root, predictions, split=split, rows=rows, reports=reports)
    data_seeds = sorted({d["seed"] for d in study["datasets"]})
    fit_seeds = sorted({j["fit_seed"] for j in study["fit_jobs"]})
    families = {s.task: s.family for s in INSTANCES}
    statistics = summarize(rows, data_seeds=data_seeds, fit_seeds=fit_seeds, families=families, repetitions=repetitions, seed=seed)
    output.mkdir(parents=True)
    if plan is not None:
        shutil.copyfile(plan, output / "analysis_plan.json")
    write_new(output / "rows.json", rows)
    write_new(output / "statistics.json", statistics)
    write_new(output / "costs.json", attempt_costs(root))
    with (output / "goal_cases.jsonl").open("x") as stream:
        for row in rows:
            report = reports.get(row["job_id"])
            if report:
                for case in report["G"]["cases"]:
                    stream.write(json.dumps({"job_id": row["job_id"], **case}, allow_nan=False)+"\n")
            else:
                # Preserve every missing deployment outcome without inventing an
                # attempted action, exception timing, or achieved trajectory.
                for i in range(row["expected_goal_cases"]):
                    stream.write(json.dumps({"job_id": row["job_id"], "case_index": i, "success": False,
                                             "outcome": "evaluation_unavailable", "normalized_penalized_cost": 1.0,
                                             "error": row["error"]}, allow_nan=False)+"\n")
    snapshot = output / "source"
    snapshot.mkdir()
    for name in ("analysis.py", "statistics.py", "prediction_archive.py"):
        shutil.copyfile(Path(__file__).with_name(name), snapshot / name)
    result = {"schema": "copuzzle.study_analysis.v1", "study_sha256": checksum, "role": study["role"], "split": split,
              "outcomes": dict(Counter(r["status"] for r in rows)), "evaluation_jobs": len(rows),
              "failure_policy": "unavailable supported scores=0 and normalized goal cost=1; unsupported probes=null",
              "prediction_evidence": prediction_evidence,
              "analysis_plan_sha256": file_hash(plan) if plan is not None else None,
              "files": {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()},
              "per_case_evidence_complete": prediction_evidence is not None}
    write_new(output / "manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.study, args.output, split=args.split, repetitions=args.repetitions, seed=args.seed,
                             predictions=args.predictions, plan=args.plan), indent=2))


if __name__ == "__main__":
    main()
