"""Independent binding checks for checkpoints, completed jobs and test seals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from ..config import canonical, fingerprint
from ..runtime import load_checkpoint
from .registry import dataset, fit_context, load_study, validate_study
from .storage import file_hash, inside, read_json, verified_record, write_new


def checkpoint_evidence(path, study, study_sha256, job, *, final=False, restore=False):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != ["header.json", "state.msgpack"]:
            raise ValueError("checkpoint members differ")
        header = json.loads(archive.read("header.json"))
        payload = archive.read("state.msgpack")
    if header.get("schema") != "copuzzle.baseline_checkpoint.v2" or hashlib.sha256(payload).hexdigest() != header["agent_state_sha256"]:
        raise ValueError("checkpoint schema/state checksum differs")
    expected_context = fit_context(study, study_sha256, job) if study["role"] == "formal" else None
    if (canonical(header["config"]) != canonical(job["config"]) or header["fit_seed"] != job["fit_seed"]
            or fingerprint(header["data"]) != job["data_sha256"] or header["sources"] != study["learner_sources"]
            or header["versions"] != study["dependency_versions"] or canonical(header["fit_context"]) != canonical(expected_context)):
        raise ValueError("checkpoint is not the registered data/config/source/seed/runtime")
    if type(header["step"]) is not int or not 0 < header["step"] <= job["steps"] or (final and header["step"] != job["steps"]):
        raise ValueError("checkpoint update count differs from the registered rule")
    if restore:
        agent, _, restored = load_checkpoint(path)
        if agent.step != header["step"] or restored["agent_state_sha256"] != header["agent_state_sha256"]:
            raise ValueError("checkpoint restoration differs")
    return {"checkpoint_sha256": file_hash(path), "agent_state_sha256": header["agent_state_sha256"],
            "config_sha256": fingerprint(header["config"]), "sources_sha256": fingerprint(header["sources"]),
            "data_sha256": fingerprint(header["data"]), "fit_seed": header["fit_seed"], "step": header["step"]}


def verify_fit(root, study, study_sha256, job, *, restore=False):
    record = verified_record(root, "fit", job["id"])
    if record is None or record["status"] not in ("completed", "failed"):
        raise ValueError(f"fit is not terminal: {job['id']}")
    if record["study_sha256"] != study_sha256 or record["job_id"] != job["id"]:
        raise ValueError("fit receipt belongs to another study")
    if record["status"] == "failed":
        if not record.get("error"):
            raise ValueError("failed fit has no error evidence")
        return record
    info = checkpoint_evidence(inside(root, record["checkpoint"]), study, study_sha256, job, final=True, restore=restore)
    if record["checkpoint_evidence"] != info:
        raise ValueError("completed fit checkpoint changed")
    return record


def seal(root, *, allow_failed=False):
    root = Path(root)
    validate_study(root, verify_data=True, verify_snapshot=True)
    study, checksum = load_study(root)
    if study["role"] != "formal":
        raise ValueError("qualification studies cannot authorize benchmark test")
    if (root / "selection/test-lock.json").exists():
        raise FileExistsError("test selection is already sealed")
    from .storage import active, lease, receipt_path
    with lease(root / ".controller.lock", "sealing final test selection"):
        barrier = root / ".sealing"
        if barrier.exists():
            # Kernel lease ownership proves the earlier sealer is no longer live.
            barrier.unlink()
        barrier.write_text("No fitting/retry may begin while the cohort is being sealed.\n")
        try:
            _build_seal(root, study, checksum, allow_failed)
        finally:
            barrier.unlink(missing_ok=True)
    return verify_seal(root / "selection/test-lock.json", verify_receipts=True)


def _build_seal(root, study, checksum, allow_failed):
    from .storage import active, receipt_path
    temporary = Path(tempfile.mkdtemp(prefix="selection-pending-", dir=root))
    try:
        rows, entries, failures = [], [], []
        for job in study["fit_jobs"]:
            if active(root / f"jobs/fit/{job['id']}/.lease"):
                raise ValueError("a fit is still active")
            record = verify_fit(root, study, checksum, job, restore=True)
            receipt = read_json(receipt_path(root, "fit", job["id"]))
            row = {"job_id": job["id"], "status": record["status"], "record": receipt["record"],
                   "record_sha256": receipt["record_sha256"], "receipt_sha256": file_hash(receipt_path(root, "fit", job["id"]))}
            if record["status"] == "failed":
                failures.append(job["id"])
                row["error"] = record["error"]
            else:
                entry = {"job_id": job["id"], "checkpoint": record["checkpoint"], **record["checkpoint_evidence"],
                         "bundle_manifest_sha256": dataset(study, job["dataset_id"])["manifest_sha256"]}
                entries.append(entry)
                row["checkpoint"] = entry
            rows.append(row)
        if failures and not allow_failed:
            raise ValueError("failed fits require diagnosis/retry or explicit failure-inclusive sealing")
        cohort = {"schema": "copuzzle.fit_cohort.v1", "study_sha256": checksum,
                  "expected_fit_count": len(study["fit_jobs"]), "fits": rows}
        controls = [{"job_id": row["id"], "bundle_manifest_sha256": dataset(study, row["dataset_id"])["manifest_sha256"],
                     "settings": row["settings"], "sources_sha256": study["control_source_sha256"]} for row in study["control_jobs"]]
        write_new(temporary / "cohort.json", cohort)
        lock = {"schema": "copuzzle.test_selection.v2", "selection_complete": True,
                "study_sha256": checksum, "data_inventory_sha256": study["data_inventory_sha256"],
                "selected_methods_sha256": study["selected_methods_sha256"], "cohort_sha256": file_hash(temporary / "cohort.json"),
                "expected_fit_count": len(study["fit_jobs"]), "failed_fit_ids": failures,
                "failure_policy": "registered failed fits remain unavailable and receive zero primary performance, never dropped",
                "checkpoints": entries, "controls": controls}
        write_new(temporary / "test-lock.json", lock)
        if (root / "selection").exists():
            raise FileExistsError("selection directory already exists")
        temporary.rename(root / "selection")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_seal(lock_path, *, verify_receipts=False):
    lock_path = Path(lock_path)
    root = lock_path.parent.parent
    if lock_path.name != "test-lock.json" or lock_path.parent.name != "selection":
        raise ValueError("test lock must belong to a registered study selection directory")
    validate_study(root)
    study, checksum = load_study(root)
    if study["role"] != "formal":
        raise ValueError("qualification studies do not authorize final test")
    lock, cohort = read_json(lock_path), read_json(lock_path.parent / "cohort.json")
    if (lock.get("schema") != "copuzzle.test_selection.v2" or lock.get("selection_complete") is not True
            or lock.get("study_sha256") != checksum or cohort.get("study_sha256") != checksum
            or file_hash(lock_path.parent / "cohort.json") != lock.get("cohort_sha256")
            or lock.get("data_inventory_sha256") != study["data_inventory_sha256"]
            or lock.get("selected_methods_sha256") != study["selected_methods_sha256"]):
        raise ValueError("test seal has inconsistent provenance")
    expected = {row["id"]: row for row in study["fit_jobs"]}
    rows = cohort.get("fits", [])
    if (len(rows) != len(expected) or {row["job_id"] for row in rows} != set(expected)
            or cohort.get("expected_fit_count") != len(expected) or lock.get("expected_fit_count") != len(expected)):
        raise ValueError("test seal omits or duplicates registered fits")
    checkpoints, failed = [], []
    for row in rows:
        job = expected[row["job_id"]]
        if row["status"] == "failed":
            if not row.get("error"):
                raise ValueError("missing failure evidence")
            failed.append(job["id"])
        elif row["status"] == "completed":
            entry = row["checkpoint"]
            required = {"job_id": job["id"], "config_sha256": fingerprint(job["config"]),
                        "data_sha256": job["data_sha256"], "sources_sha256": fingerprint(study["learner_sources"]),
                        "step": job["steps"], "fit_seed": job["fit_seed"],
                        "bundle_manifest_sha256": dataset(study, job["dataset_id"])["manifest_sha256"]}
            if any(entry.get(key) != value for key, value in required.items()):
                raise ValueError("test seal checkpoint bindings differ from registration")
            checkpoints.append(entry)
        else:
            raise ValueError("test seal includes a nonterminal fit")
        if verify_receipts:
            from .storage import receipt_path
            if file_hash(inside(root, row["record"])) != row["record_sha256"] or file_hash(receipt_path(root, "fit", job["id"])) != row["receipt_sha256"]:
                raise ValueError("fit history changed after test sealing")
            verify_fit(root, study, checksum, job)
    expected_controls = [{"job_id": row["id"], "bundle_manifest_sha256": dataset(study, row["dataset_id"])["manifest_sha256"],
                          "settings": row["settings"], "sources_sha256": study["control_source_sha256"]} for row in study["control_jobs"]]
    if lock["checkpoints"] != checkpoints or lock["failed_fit_ids"] != failed or lock["controls"] != expected_controls:
        raise ValueError("test authorization does not match the complete cohort")
    return lock
