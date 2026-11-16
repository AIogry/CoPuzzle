"""Bounded local controllers, resumable jobs and complete-cohort test sealing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..runtime import _json
from .evidence import seal, verify_fit, verify_seal
from .jobs import record_startup_failure, run_job
from .registry import job_for, load_study, register, validate_study
from .storage import BusyStudy, active, file_hash, inside, lease, verified_record, write_new


def phase_jobs(study, phase):
    return study["fit_jobs"] if phase == "fit" else study["control_jobs"] if phase.startswith("control") else study["evaluation_jobs"]


def snapshot(root, phase):
    root = Path(root)
    study, checksum = load_study(root)
    counts = {key: 0 for key in ("pending", "active", "completed", "failed", "interrupted", "abandoned")}
    primary = 0
    for job in phase_jobs(study, phase):
        job_active = active(root / f"jobs/{phase}/{job['id']}/.lease")
        record = verified_record(root, phase, job["id"])
        if job_active:
            counts["active"] += 1
        elif record is None:
            counts["pending"] += 1
        elif record["status"] == "running":
            counts["abandoned"] += 1
        else:
            counts[record["status"]] += 1
            if study["role"] == "formal" and phase == "fit" and job["primary"] and record["status"] == "completed":
                primary += 1
    return {"phase": phase, "study_sha256": checksum, "expected": len(phase_jobs(study, phase)),
            "counts": counts, "primary_fits_completed": primary,
            "controller_active": active(root / ".controller.lock"), "stop_requested": (root / "STOP").exists()}


def run(root, phase, *, workers=2, retry_failed=False, clear_stop=False):
    root = Path(root).resolve()
    if phase not in ("fit", "dev", "test", "control-dev", "control-test"):
        raise ValueError("unknown phase")
    checked = validate_study(root, verify_data=True, verify_snapshot=True)
    study, checksum = load_study(root)
    if type(workers) is not int or not 1 <= workers <= study["max_workers"]:
        raise ValueError("worker count exceeds the registered resource limit")
    if phase.endswith("test") or phase == "test":
        verify_seal(root / "selection/test-lock.json", verify_receipts=True)
    with lease(root / ".controller.lock", {"phase": phase}):
        if clear_stop:
            (root / "STOP").unlink(missing_ok=True)
        if (root / "STOP").exists():
            raise ValueError("stop request remains; clear it explicitly when resuming")
        if phase == "fit" and (root / "selection/test-lock.json").exists():
            raise ValueError("fit cohort is sealed")
        env = os.environ.copy()
        env.update(PYTHONPATH=str(root / "source"), JAX_PLATFORMS=study["backend"], OMP_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
        live, scheduled = {}, set()
        invocation = root / "controllers" / str(time.time_ns())
        invocation.mkdir(parents=True)
        controller_source = Path(__file__).read_bytes()
        (invocation / "runner.py").write_bytes(controller_source)
        state = {"phase": phase, "status": "running", "workers": workers, "started_unix": time.time(),
                 "pid_in_current_namespace": os.getpid(), "study_sha256": checksum,
                 "invocation": invocation.relative_to(root).as_posix(), "controller_source_sha256": file_hash(invocation / "runner.py")}
        write_new(invocation / "start.json", state)
        _json(root / "controller.json", state)
        try:
            while True:
                active_elsewhere = False
                reaped = False
                for job in phase_jobs(study, phase):
                    if len(live) >= workers or (root / "STOP").exists():
                        break
                    job_id = job["id"]
                    if job_id in live or job_id in scheduled:
                        continue
                    if active(root / f"jobs/{phase}/{job_id}/.lease"):
                        active_elsewhere = True
                        continue
                    record = verified_record(root, phase, job_id)
                    if record and (record["status"] == "completed" or (record["status"] == "failed" and not retry_failed)):
                        continue
                    logs = root / "logs" / phase
                    logs.mkdir(parents=True, exist_ok=True)
                    suffix = time.time_ns()
                    log = (logs / f"{job_id}-{suffix}.log").open("xb")
                    command = [sys.executable, "-m", "copuzzle_baselines.study.runner", "worker", "--study", str(root), "--phase", phase, "--job", job_id]
                    if retry_failed:
                        command.append("--retry-failed")
                    try:
                        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    except Exception as error:
                        log.close()
                        record_startup_failure(root, phase, job_id, error)
                        scheduled.add(job_id)
                        continue
                    live[job_id] = (process, log)
                    scheduled.add(job_id)
                for job_id, (process, log) in list(live.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    del live[job_id]
                    reaped = True
                    record = verified_record(root, phase, job_id)
                    if record is None or record["status"] == "running":
                        if code == 75 and active(root / f"jobs/{phase}/{job_id}/.lease"):
                            scheduled.discard(job_id)
                            continue
                        record = record_startup_failure(root, phase, job_id, RuntimeError(f"worker exit {code} without a terminal receipt"))
                    print(json.dumps({"phase": phase, "job": job_id, "status": record["status"]}), flush=True)
                if (root / "STOP").exists():
                    state["status"] = "interrupted"
                    for process, _ in live.values():
                        process.terminate()
                    for process, log in live.values():
                        process.wait()
                        log.close()
                    live.clear()
                    break
                if not live:
                    # Probe every job lock before concluding that no work remains.
                    active_elsewhere = any(active(root / f"jobs/{phase}/{j['id']}/.lease") for j in phase_jobs(study, phase))
                    if not active_elsewhere and not reaped:
                        break
                state["last_progress"] = snapshot(root, phase)
                _json(root / "controller.json", state)
                time.sleep(0.5)
            if state["status"] == "running":
                remaining = snapshot(root, phase)["counts"]
                state["status"] = "completed" if remaining["completed"] == len(phase_jobs(study, phase)) else "attention_required"
        except BaseException as error:
            state.update(status="interrupted", error=f"{type(error).__name__}: {error}")
            for process, _ in live.values():
                process.terminate()
            for process, log in live.values():
                process.wait()
                log.close()
            raise
        finally:
            state["finished_unix"] = time.time()
            state["last_progress"] = snapshot(root, phase)
            _json(root / "controller.json", state)
            write_new(invocation / "finish.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("register")
    create.add_argument("--data", type=Path, required=True)
    create.add_argument("--methods", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    for name in ("run", "worker", "status", "stop", "seal", "validate"):
        p = commands.add_parser(name)
        p.add_argument("--study", type=Path, required=True)
        if name in ("run", "worker", "status"):
            p.add_argument("--phase", choices=("fit", "dev", "test", "control-dev", "control-test"), default="fit")
        if name in ("run", "worker"):
            p.add_argument("--retry-failed", action="store_true")
        if name == "worker":
            p.add_argument("--job", required=True)
        if name == "run":
            p.add_argument("--workers", type=int, default=2)
            p.add_argument("--clear-stop", action="store_true")
        if name == "seal":
            p.add_argument("--include-failed", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            result = register(args.data, args.methods, args.output, progress=lambda row: print(json.dumps(row), flush=True))
        elif args.command == "worker":
            result = run_job(args.study, args.phase, args.job, retry_failed=args.retry_failed)
            print(json.dumps({"job": args.job, "status": result["status"]}), flush=True)
            return 0 if result["status"] == "completed" else 1
        elif args.command == "run":
            result = run(args.study, args.phase, workers=args.workers, retry_failed=args.retry_failed, clear_stop=args.clear_stop)
        elif args.command == "status":
            result = snapshot(args.study, args.phase)
        elif args.command == "validate":
            result = validate_study(args.study, verify_data=True, verify_snapshot=True)
        elif args.command == "seal":
            lock = seal(args.study, allow_failed=args.include_failed)
            result = {"sealed": True, "fits": lock["expected_fit_count"], "failed": len(lock["failed_fit_ids"])}
        else:
            (args.study / "STOP").write_text("Stop requested; preserve checkpoints and attempt history.\n")
            result = {"stop_requested": True}
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result.get("status", "completed") == "completed" else 1
    except BusyStudy as error:
        print(str(error), file=sys.stderr)
        return 75
    except (ValueError, OSError, KeyError, StopIteration) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
