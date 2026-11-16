"""Final test access requires a checkpoint registered in a sealed selection file."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import fingerprint


def validate_fit_context(context, config, data, sources, versions, seed, steps, backend):
    keys = {"schema", "study_sha256", "job_id", "config_sha256", "data_sha256", "sources_sha256", "versions_sha256", "fit_seed", "steps", "backend"}
    if not isinstance(context, dict) or set(context) != keys or context["schema"] != "copuzzle.formal_fit_context.v1":
        raise ValueError("invalid formal fit context")
    if data.get("purpose") != "benchmark":
        raise ValueError("formal fits require frozen benchmark training data")
    expected = {"config_sha256": fingerprint(config), "data_sha256": fingerprint(data), "sources_sha256": fingerprint(sources),
                "versions_sha256": fingerprint(versions), "fit_seed": seed, "steps": steps, "backend": backend}
    if any(context[key] != value for key, value in expected.items()):
        raise ValueError("formal fit configuration/data/source/runtime binding differs")
    if not isinstance(context["study_sha256"], str) or len(context["study_sha256"]) != 64 or not isinstance(context["job_id"], str) or not context["job_id"]:
        raise ValueError("formal fit has no registered study/job identity")


def authorize_test(lock_path, checkpoint, bundle, header):
    if lock_path is None:
        raise ValueError("benchmark test evaluation requires a sealed study selection lock")
    lock = json.loads(Path(lock_path).read_text())
    if lock.get("schema") != "copuzzle.test_selection.v1" or lock.get("selection_complete") is not True:
        raise ValueError("test selection is not sealed")
    checksum = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    matches = [row for row in lock.get("checkpoints", []) if row.get("checkpoint_sha256") == checksum]
    if len(matches) != 1:
        raise ValueError("checkpoint is not uniquely registered for final test")
    selected = matches[0]
    expected = {
        "bundle_manifest_sha256": hashlib.sha256((Path(bundle) / "manifest.json").read_bytes()).hexdigest(),
        "config_sha256": fingerprint(header["config"]), "sources_sha256": fingerprint(header["sources"]),
        "data_sha256": fingerprint(header["data"]), "fit_seed": header["fit_seed"], "step": header["step"],
    }
    if any(selected.get(key) != value for key, value in expected.items()):
        raise ValueError("selected checkpoint/data/configuration binding differs")
    return {"selection_lock_sha256": hashlib.sha256(Path(lock_path).read_bytes()).hexdigest(), "job_id": selected["job_id"]}


def authorize_control_test(lock_path, bundle, settings, source_sha256):
    if lock_path is None:
        raise ValueError("benchmark test controls require a sealed study selection lock")
    lock = json.loads(Path(lock_path).read_text())
    if lock.get("schema") != "copuzzle.test_selection.v1" or lock.get("selection_complete") is not True:
        raise ValueError("test selection is not sealed")
    expected = {"bundle_manifest_sha256": hashlib.sha256((Path(bundle) / "manifest.json").read_bytes()).hexdigest(),
                "settings": settings, "sources_sha256": source_sha256}
    matches = [row for row in lock.get("controls", []) if all(row.get(key) == value for key, value in expected.items())]
    if len(matches) != 1:
        raise ValueError("reference control is not uniquely registered for final test")
    return {"selection_lock_sha256": hashlib.sha256(Path(lock_path).read_bytes()).hexdigest(), "job_id": matches[0]["job_id"]}
