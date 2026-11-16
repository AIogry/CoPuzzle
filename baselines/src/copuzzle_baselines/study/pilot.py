"""Execute the registered development-only selection plan in frozen subprocesses."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from ..config import Config, fingerprint
from ..runtime import _json, evaluate, sources, train, versions


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(index_path, job_id):
    index_path = Path(index_path)
    index = json.loads(index_path.read_text())
    root = index_path.parent
    row = next(item for item in index["jobs"] if item["id"] == job_id)
    destination = root / "jobs" / job_id
    result = {"id": job_id, "candidate": row["candidate"], "task": row["task"], "regime": row["regime"],
              "status": "running", "formal_fits_contributed": 0, "pid": os.getpid()}
    _json(destination / "result.json", result)
    started = time.perf_counter()
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if file_hash(root / "plan.json") != index["plan_sha256"]:
            raise ValueError("registered pilot plan changed")
        if sources() != index["sources"] or versions() != index["versions"]:
            raise ValueError("worker source/dependency identity differs")
        config_path = root / row["config_path"]
        if file_hash(config_path) != row["config_sha256"]:
            raise ValueError("registered candidate changed")
        bundle = Path(row["bundle"])
        if file_hash(bundle / "manifest.json") != row["bundle_sha256"]:
            raise ValueError("development data changed")
        metadata = json.loads((bundle / "learner/metadata.json").read_text())
        if metadata.get("purpose") != "development":
            raise ValueError("pilot refuses benchmark-purpose data")
        config = Config.from_dict(json.loads(config_path.read_text()))
        run = train(bundle / "learner", destination / "training", config=config,
                    seed=index["fit_seed"], steps=index["steps"], checkpoint_every=1000, log_every=100)
        checkpoint = destination / "training" / run["checkpoints"][-1]["file"]
        scores, evaluations = {}, {}
        for budget in ([0] if config.method != "dynamics" else index["model_query_budgets"]):
            report = evaluate(checkpoint, bundle, split="dev", model_query_limit=budget)
            _json(destination / f"evaluation-dev-b{budget}.json", report)
            evaluations[str(budget)] = {"file": str((destination / f"evaluation-dev-b{budget}.json").relative_to(root)),
                                       "sha256": file_hash(destination / f"evaluation-dev-b{budget}.json")}
            scores[str(budget)] = {"G_tier_macro": report["G"]["tier_macro_success_rate"],
                                   "R_balanced_joint": report["R"].get("balanced_joint_exact"),
                                   "C_length_macro": report["C"].get("length_macro_whole_prefix_exact")}
        result.update(status="completed", step=run["completed_step"], checkpoint=str(checkpoint.relative_to(root)),
                      checkpoint_sha256=file_hash(checkpoint), scores=scores, evaluations=evaluations,
                      training_manifest_sha256=file_hash(destination / "training/run.json"),
                      source_sha256=fingerprint(sources()), plan_sha256=index["plan_sha256"])
    except BaseException as error:
        result.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error={"type": type(error).__name__, "message": str(error)})
    finally:
        result["process_work_seconds"] = time.perf_counter() - started
        _json(destination / "result.json", result)
    return 0 if result["status"] == "completed" else 1


def aggregate_selection(index, results):
    if len(results) != len(index["jobs"]) or any(row["status"] != "completed" for row in results):
        raise ValueError("selection requires all registered development fits; failures need recorded diagnosis")
    expected = {row["id"]: row for row in index["jobs"]}
    if len({row["id"] for row in results}) != len(expected) or {row["id"] for row in results} != set(expected):
        raise ValueError("duplicate or missing pilot job")
    for row in results:
        if any(row[key] != expected[row["id"]][key] for key in ("candidate", "task", "regime")):
            raise ValueError("pilot job identity differs")
    from copuzzle.research.spec import instance
    scores = {}
    for candidate in index["candidates"]:
        selected = [row for row in results if row["candidate"] == candidate]
        if len(selected) != 12:
            raise ValueError("missing candidate task/condition cells")
        by_condition = {}
        for regime in index["conditions"]:
            family = {}
            for row in selected:
                if row["regime"] == regime:
                    value = row["scores"].get("0", row["scores"].get("128"))["G_tier_macro"]
                    family.setdefault(instance(row["task"]).family, []).append(value)
            by_condition[regime] = sum(sum(values) / len(values) for values in family.values()) / len(family)
        scores[candidate] = {"by_condition": by_condition, "G_macro": sum(by_condition.values()) / len(by_condition)}
    a, b = scores["gciql_future"]["G_macro"], scores["gciql_mixed"]["G_macro"]
    choice = "gciql_future" if a > b + 1e-12 else "gciql_mixed"
    return {"schema": "copuzzle.dev_selection.v1", "purpose": "development selection; not final-test authorization",
            "data_inventory_sha256": index["data_inventory_sha256"], "plan_sha256": index["plan_sha256"],
            "selected_candidates": {"gcbc": "gcbc", "gciql": choice, "dynamics": "dynamics"},
            "scores": scores, "completed_pilot_fits": len(results), "formal_fits_contributed": 0}


def run(plan_path, data_root, frozen_data_root, output, *, workers=2):
    from copuzzle.benchmark.collection import validate_inventory
    from ..runtime import sources
    plan_path, data_root, frozen_data_root, output = map(Path, (plan_path, data_root, frozen_data_root, output))
    if output.exists():
        raise FileExistsError(output)
    if workers not in (1, 2):
        raise ValueError("pilot concurrency is limited to one or two CPU workers")
    plan = json.loads(plan_path.read_text())
    from copuzzle.benchmark.spec import INSTANCES, REGIMES
    inventory = validate_inventory(frozen_data_root)
    if plan["schema"] != "copuzzle.method_selection_plan.v1" or plan["split"] != "dev" or plan["data_inventory_sha256"] != inventory["inventory_sha256"]:
        raise ValueError("pilot plan is not bound to the frozen inventory")
    if plan["instances"] != [item.task for item in INSTANCES] or plan["conditions"] != list(REGIMES) or set(plan["candidate_configs"]) != {"gcbc", "gciql_future", "gciql_mixed", "dynamics"}:
        raise ValueError("registered pilot scope differs")
    jobs = []
    for task in plan["instances"]:
        for regime in plan["conditions"]:
            bundle = (data_root / f"{task}-{regime}").resolve()
            meta = json.loads((bundle / "learner/metadata.json").read_text())
            manifest = json.loads((bundle / "manifest.json").read_text())
            if meta["purpose"] != "development" or manifest["seed"] != plan["development_data_seed"]:
                raise ValueError("pilot data identity differs")
            for candidate, description in plan["candidate_configs"].items():
                path = Path(description["path"])
                if file_hash(path) != description["sha256"]:
                    raise ValueError("candidate configuration changed")
                cfg = Config.from_dict(json.loads(path.read_text()))
                if cfg.method != description["method"] or cfg.coverage != 1:
                    raise ValueError("candidate method/coverage differs")
                jobs.append({"id": f"{task}-{regime}-{candidate}", "task": task, "regime": regime,
                             "candidate": candidate, "bundle": str(bundle), "bundle_sha256": file_hash(bundle / "manifest.json"),
                             "config_path": f"configs/{candidate}.json", "config_sha256": description["sha256"]})
    if len(jobs) != plan["expected_pilot_fits"]:
        raise ValueError("registered pilot matrix differs")
    output.mkdir(parents=True)
    (output / "configs").mkdir()
    shutil.copyfile(plan_path, output / "plan.json")
    for candidate, row in plan["candidate_configs"].items():
        shutil.copyfile(row["path"], output / "configs" / f"{candidate}.json")
    shutil.copytree(frozen_data_root / "source/copuzzle", output / "source/copuzzle")
    source = Path(__file__).resolve().parents[1]
    for path in source.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        target = output / "source/copuzzle_baselines" / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    index = {"schema": "copuzzle.pilot_index.v1", "plan_sha256": file_hash(output / "plan.json"),
             "data_inventory_sha256": inventory["inventory_sha256"], "candidates": list(plan["candidate_configs"]),
             "conditions": plan["conditions"], "fit_seed": plan["development_fit_seed"], "steps": plan["training_steps"],
             "model_query_budgets": plan["model_query_budgets"], "sources": sources(), "versions": versions(), "jobs": jobs}
    _json(output / "index.json", index)
    state = {"status": "running", "controller_pid": os.getpid(), "expected": len(jobs), "completed": 0, "failed": 0, "formal_fits": 0}
    _json(output / "status.json", state)
    env = os.environ.copy()
    env.update(PYTHONPATH=str((output / "source").resolve()), JAX_PLATFORMS="cpu", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    queue, live, results = list(jobs), {}, []
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        while queue or live:
            while queue and len(live) < workers and not state["failed"]:
                job = queue.pop(0)
                destination = output / "jobs" / job["id"]
                destination.mkdir(parents=True)
                log = (destination / "process.log").open("xb")
                try:
                    process = subprocess.Popen([sys.executable, "-m", "copuzzle_baselines.study.pilot", "--worker",
                                                "--index", str((output / "index.json").resolve()), "--job", job["id"]],
                                               env=env, stdout=log, stderr=subprocess.STDOUT)
                except Exception as error:
                    log.close()
                    result = {"id": job["id"], "status": "failed", "error": {"type": type(error).__name__, "message": str(error)}}
                    _json(destination / "result.json", result)
                    results.append(result)
                    state["failed"] += 1
                    break
                live[job["id"]] = (process, log)
            for job_id, (process, log) in list(live.items()):
                status = process.poll()
                if status is None:
                    continue
                log.close()
                result_path = output / "jobs" / job_id / "result.json"
                result = json.loads(result_path.read_text()) if result_path.exists() else {"id": job_id, "status": "failed", "error": {"type": "WorkerStartupError", "message": f"worker exited {status} without a result"}}
                if status != 0 or result.get("status") != "completed":
                    result["status"] = "failed"
                    state["failed"] += 1
                else:
                    checkpoint = output / result["checkpoint"]
                    if result.get("step") != index["steps"] or not checkpoint.exists() or file_hash(checkpoint) != result["checkpoint_sha256"]:
                        result.update(status="failed", error={"type": "CompletionMismatch", "message": "worker did not supply the registered final checkpoint"})
                        state["failed"] += 1
                    else:
                        state["completed"] += 1
                results.append(result)
                _json(result_path, result)
                del live[job_id]
                print(json.dumps({"job": job_id, "status": result["status"], "completed": state["completed"], "failed": state["failed"]}), flush=True)
                _json(output / "status.json", state)
            if state["failed"] and not live:
                break
            time.sleep(0.2)
        if state["failed"]:
            state["status"] = "failed"
            state["not_started"] = [row["id"] for row in queue]
        else:
            selection = aggregate_selection(index, results)
            _json(output / "selection.json", selection)
            state["status"] = "completed"
    except BaseException:
        state["status"] = "interrupted"
        for process, log in live.values():
            process.terminate()
        for process, log in live.values():
            process.wait()
            log.close()
        raise
    finally:
        _json(output / "status.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--job")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--frozen-data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.index, args.job)
    if any(value is None for value in (args.plan, args.data, args.frozen_data, args.output)):
        parser.error("controller requires --plan, --data, --frozen-data and --output")
    result = run(args.plan, args.data, args.frozen_data, args.output, workers=args.workers)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
