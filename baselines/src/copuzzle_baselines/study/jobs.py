"""Resumable registered jobs with immutable attempt history and checked receipts."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import threading
import time

from ..config import Config, fingerprint
from ..runtime import _json, evaluate, sources, train, versions
from ..controls.evaluate import evaluate_control, source_sha256 as control_source_sha256
from .evidence import checkpoint_evidence, verify_fit, verify_seal
from .registry import data_directory, dataset, fit_context, job_for, load_study, validate_study
from .storage import (BusyStudy, file_hash, inside, lease, publish_receipt,
                      read_receipt, verified_record, write_new)


def _previous_checkpoint(root, job, study, study_hash):
    candidates, rejected = [], []
    folder = inside(root, f"jobs/fit/{job['id']}/attempts")
    for path in sorted(folder.glob("*/training/checkpoint-*.zip")):
        try:
            evidence = checkpoint_evidence(path, study, study_hash, job)
            candidates.append((evidence["step"], path.as_posix(), path, evidence))
        except Exception as error:
            rejected.append({"file": path.relative_to(root).as_posix(), "error": f"{type(error).__name__}: {error}"})
    if not candidates:
        return None, None, rejected
    _, _, path, evidence = max(candidates)
    return path, evidence, rejected


def _fit(root, study, checksum, job, attempt, record):
    bundle = data_directory(root) / dataset(study, job["dataset_id"])["path"]
    resume, evidence, rejected = _previous_checkpoint(root, job, study, checksum)
    record["rejected_resume_candidates"] = rejected
    record["resume_checkpoint"] = resume.relative_to(root).as_posix() if resume else None
    if evidence and evidence["step"] == job["steps"]:
        checkpoint = resume
        record["execution_mode"] = "recover_registered_final_checkpoint"
    else:
        context = fit_context(study, checksum, job) if study["role"] == "formal" else None
        run = train(bundle / "learner", attempt / "training", config=Config.from_dict(job["config"]),
                    seed=job["fit_seed"], steps=job["steps"], checkpoint_every=job["checkpoint_every"],
                    log_every=job["log_every"], resume=resume, fit_context=context)
        checkpoint = attempt / "training" / run["checkpoints"][-1]["file"]
        record["execution_mode"] = "resume" if resume else "fresh"
        record["training_manifest"] = (attempt / "training/run.json").relative_to(root).as_posix()
        record["training_manifest_sha256"] = file_hash(attempt / "training/run.json")
    record["checkpoint"] = checkpoint.relative_to(root).as_posix()
    record["checkpoint_evidence"] = checkpoint_evidence(checkpoint, study, checksum, job, final=True, restore=True)


def _evaluation(root, study, checksum, phase, job, attempt, record):
    split = "test" if phase.endswith("test") or phase == "test" else "dev"
    lock = root / "selection/test-lock.json" if split == "test" else None
    if split == "test":
        verify_seal(lock)
    bundle = data_directory(root) / dataset(study, job["dataset_id"])["path"]
    if job["kind"] == "control":
        settings = job["settings"]
        report = evaluate_control(bundle, settings["kind"], split=split,
            model_query_limit=settings["model_query_limit"] or 0, coverage=settings["coverage"],
            coverage_seed=settings["coverage_seed"], seed=settings["seed"], test_lock=lock)
    else:
        fit_job = job_for(study, "fit", job["fit_id"])
        fitted = verify_fit(root, study, checksum, fit_job)
        record["fit_record"] = read_receipt(root, "fit", fit_job["id"])
        if fitted["status"] == "failed":
            raise RuntimeError("registered fit unavailable; evaluation failure must remain in the cohort")
        report = evaluate(inside(root, fitted["checkpoint"]), bundle, split=split,
                          model_query_limit=job["budget"], test_lock=lock)
    if report.get("learner_state_unchanged", True) is not True:
        raise ValueError("evaluation modified learner state")
    expected_cases = dataset(study, job["dataset_id"])["evaluation_cases"][f"G_{split}"]
    if report["G"]["aggregate"]["cases"] != expected_cases:
        raise ValueError("evaluation omitted registered goal cases")
    destination = attempt / "evaluation.json"
    write_new(destination, report)
    record.update(report=destination.relative_to(root).as_posix(), report_sha256=file_hash(destination),
                  split=split, expected_goal_cases=expected_cases)


def run_job(root, phase, job_id, *, retry_failed=False):
    root = Path(root).resolve()
    if phase not in ("fit", "dev", "test", "control-dev", "control-test"):
        raise ValueError("unknown job phase")
    validate_study(root, verify_snapshot=True)
    study, checksum = load_study(root)
    if study["role"] == "formal" and not Path(__file__).resolve().is_relative_to(root / "source"):
        raise ValueError("formal workers must execute the registered source snapshot")
    job = job_for(study, phase, job_id)
    base = inside(root, f"jobs/{phase}/{job_id}")
    with lease(base / ".lease", {"job_id": job_id, "phase": phase}):
        existing = verified_record(root, phase, job_id)
        if existing and existing["status"] == "completed":
            if phase == "fit":
                verify_fit(root, study, checksum, job)
            elif file_hash(inside(root, existing["report"])) != existing["report_sha256"]:
                raise ValueError("completed evaluation changed")
            return existing
        if phase == "fit" and ((root / "selection/test-lock.json").exists() or (root / ".sealing").exists()):
            raise ValueError("fitting/retries are forbidden after final-test selection is sealed")
        if existing and existing["status"] == "failed" and not retry_failed:
            raise ValueError("failed jobs require an explicit diagnosed retry")
        attempts = base / "attempts"
        attempts.mkdir(exist_ok=True)
        number = max([int(p.name) for p in attempts.iterdir() if p.is_dir() and p.name.isdigit()], default=0) + 1
        attempt = attempts / f"{number:04d}"
        attempt.mkdir()
        record = {"schema": "copuzzle.job_attempt.v1", "study_sha256": checksum, "job_id": job_id, "phase": phase,
                  "attempt": number, "status": "running", "started_unix": time.time(),
                  "pid_in_current_namespace": os.getpid(), "previous_receipt": read_receipt(root, phase, job_id),
                  "executor_sources_sha256": fingerprint(study["source_snapshot"]), "checkpoint_selection": study["checkpoint_selection"]}
        package = Path(__file__).resolve().parents[1]
        executing = {p.relative_to(package).as_posix(): file_hash(p) for p in sorted(package.rglob("*.py")) if "__pycache__" not in p.parts}
        record["executing_baseline_sources"] = executing
        record["executing_baseline_sources_sha256"] = fingerprint(executing)
        path = attempt / "record.json"
        # Receipts never point to a file that will later be rewritten.
        publish_receipt(root, phase, job_id, attempt / "started.json", record)
        started = time.perf_counter()
        watcher_stop = threading.Event()
        old_handler = None
        can_signal = threading.current_thread() is threading.main_thread()
        if can_signal:
            def stop_handler(signum, frame):
                raise KeyboardInterrupt("study stop requested")
            old_handler = signal.signal(signal.SIGTERM, stop_handler)
            def watch_stop():
                while not watcher_stop.wait(0.25):
                    if (root / "STOP").exists():
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
            threading.Thread(target=watch_stop, daemon=True).start()
        try:
            if sources() != study["learner_sources"] or versions() != study["dependency_versions"]:
                raise ValueError("worker learner/dependency identity differs")
            if control_source_sha256() != study["control_source_sha256"]:
                raise ValueError("worker reference-control source differs")
            if (root / "STOP").exists():
                raise KeyboardInterrupt("study stopped before job startup")
            if phase == "fit":
                _fit(root, study, checksum, job, attempt, record)
            else:
                _evaluation(root, study, checksum, phase, job, attempt, record)
            record["status"] = "completed"
        except BaseException as error:
            record.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                          error={"type": type(error).__name__, "message": str(error)})
        finally:
            watcher_stop.set()
            if can_signal:
                signal.signal(signal.SIGTERM, old_handler)
            record["finished_unix"] = time.time()
            record["process_work_seconds"] = time.perf_counter() - started
            publish_receipt(root, phase, job_id, path, record)
        return record


def record_startup_failure(root, phase, job_id, error):
    """Record a worker that failed before importing/entering run_job."""
    root = Path(root).resolve()
    study, checksum = load_study(root)
    job_for(study, phase, job_id)
    base = inside(root, f"jobs/{phase}/{job_id}")
    with lease(base / ".lease", "recording failed startup"):
        previous = verified_record(root, phase, job_id)
        if previous and previous["status"] in ("completed", "failed", "interrupted"):
            return previous
        folder = base / "attempts"
        folder.mkdir(exist_ok=True)
        number = max([int(p.name) for p in folder.iterdir() if p.is_dir() and p.name.isdigit()], default=0) + 1
        path = folder / f"{number:04d}" / "record.json"
        path.parent.mkdir()
        record = {"schema": "copuzzle.job_attempt.v1", "study_sha256": checksum, "job_id": job_id,
                  "phase": phase, "attempt": number, "status": "failed", "failure_stage": "worker_startup",
                  "error": {"type": type(error).__name__, "message": str(error)},
                  "previous_receipt": read_receipt(root, phase, job_id), "process_work_seconds": 0.0}
        publish_receipt(root, phase, job_id, path, record)
        return record
