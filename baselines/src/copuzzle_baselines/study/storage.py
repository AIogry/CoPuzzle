"""Atomic receipts and kernel-backed leases; a status file is not a liveness test."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

from ..runtime import _json


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def inside(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path must stay inside its study")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("artifact path escapes its study")
    return path


class BusyStudy(RuntimeError):
    pass


@contextmanager
def lease(path, owner=None):
    """Linux/POSIX advisory lock released by the kernel on process exit.

    PID metadata is diagnostic only, including under PID namespaces. Never kill
    a process or restart work merely because a JSON status/lease file exists.
    """
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BusyStudy(f"active lease: {path}") from error
        stream.seek(0)
        stream.truncate()
        json.dump({"pid_in_current_namespace": os.getpid(), "acquired_unix": time.time(), "owner": owner}, stream)
        stream.flush()
        yield
    finally:
        stream.close()


def active(path):
    """A read-only-in-effect kernel lock probe, not a stale metadata guess."""
    path = Path(path)
    if not path.exists():
        return False
    import fcntl
    with path.open("r+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return False


def receipt_path(root, phase, job_id):
    return inside(root, f"jobs/{phase}/{job_id}/status.json")


def read_receipt(root, phase, job_id):
    path = receipt_path(root, phase, job_id)
    return read_json(path) if path.exists() else None


def publish_receipt(root, phase, job_id, record_path, record):
    record_path = Path(record_path)
    _json(record_path, record)
    receipt = {"schema": "copuzzle.job_receipt.v1", "job_id": job_id, "phase": phase,
               "status": record["status"], "attempt": record["attempt"],
               "record": record_path.resolve().relative_to(Path(root).resolve()).as_posix(),
               "record_sha256": file_hash(record_path)}
    _json(receipt_path(root, phase, job_id), receipt)
    return receipt


def verified_record(root, phase, job_id):
    receipt = read_receipt(root, phase, job_id)
    if receipt is None:
        return None
    if receipt.get("schema") != "copuzzle.job_receipt.v1" or receipt.get("job_id") != job_id or receipt.get("phase") != phase:
        raise ValueError("invalid job receipt identity")
    path = inside(root, receipt["record"])
    if file_hash(path) != receipt["record_sha256"]:
        raise ValueError("job record changed after receipt")
    record = read_json(path)
    if any(record.get(key) != receipt[key] for key in ("job_id", "phase", "status", "attempt")):
        raise ValueError("job record/receipt disagree")
    return record
