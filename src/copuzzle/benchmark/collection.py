"""Frozen benchmark data inventories; configuration/method selection is separate."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
import shutil

from . import PROTOCOL
from .generation import file_hash, source_hashes, validate_bundle, write_bundle
from .spec import BANK_LIMIT, EXHAUSTIVE_LIMIT, FINAL_DATA_SEEDS, INSTANCES, MAX_PER_CELL, REGIMES, digest, specification


SCHEMA = "copuzzle.benchmark_inventory.v1"


def _files(directory):
    return sorted(p for p in Path(directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")


def freeze_data(output, progress=None):
    from .evaluation import source_hashes as metric_hashes
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source = source_hashes()
    metrics = metric_hashes()
    output.mkdir(parents=True)
    core = Path(__file__).resolve().parents[1]
    snapshot = output / "source" / "copuzzle"
    for path in (p for p in _files(core) if p.suffix == ".py"):
        if path.is_symlink():
            raise ValueError("source snapshots must not follow symlinks")
        target = snapshot / path.relative_to(core)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    bundles = []
    for seed in FINAL_DATA_SEEDS:
        for spec in INSTANCES:
            for regime in REGIMES:
                name = f"{spec.task}-{regime}-s{seed}"
                relative = f"data/{name}"
                path = write_bundle(output / relative, spec.task, regime, seed, purpose="benchmark")
                checked = validate_bundle(path)
                row = {"id": name, "task": spec.task, "regime": regime, "seed": seed, "path": relative,
                       "manifest_sha256": file_hash(path / "manifest.json"),
                       "train_sha256": file_hash(path / "learner/train.npz"),
                       "metadata_sha256": file_hash(path / "learner/metadata.json"),
                       "training_episodes": checked["training_episodes"], "evaluation_cases": checked["evaluation_cases"],
                       "cell_populations": checked["cell_populations"]}
                bundles.append(row)
                if progress is not None:
                    progress({key: row[key] for key in ("id", "training_episodes", "evaluation_cases")})
    (output / "specification.json").write_text(json.dumps(specification(), indent=2) + "\n", encoding="utf-8")
    if source_hashes() != source or metric_hashes() != metrics:
        raise RuntimeError("generator source changed while freezing the inventory")
    inventory = {"schema": SCHEMA, "protocol": PROTOCOL, "purpose": "benchmark data frozen before method selection",
                 "test_policy": "method/configuration/checkpoint selection must be locked before learned test evaluation",
                 "data_seeds": list(FINAL_DATA_SEEDS), "specification_sha256": digest(specification()),
                 "generator_sources": source, "metric_sources": metrics, "bundles": bundles,
                 "generation_versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("numpy", "gymnasium")}},
                 "files": {path.relative_to(output).as_posix(): file_hash(path) for path in _files(output)}}
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return inventory


def validate_inventory(directory, *, semantic=False):
    from .evaluation import source_hashes as metric_hashes
    directory = Path(directory)
    inventory = json.loads((directory / "inventory.json").read_text())
    if set(inventory) != {"schema", "protocol", "purpose", "test_policy", "data_seeds", "specification_sha256", "generator_sources", "metric_sources", "bundles", "generation_versions", "files"}:
        raise ValueError("invalid inventory fields")
    if inventory["schema"] != SCHEMA or inventory["protocol"] != PROTOCOL or inventory["specification_sha256"] != digest(specification()):
        raise ValueError("unsupported inventory protocol")
    if inventory["purpose"] != "benchmark data frozen before method selection" or inventory["test_policy"] != "method/configuration/checkpoint selection must be locked before learned test evaluation":
        raise ValueError("inventory purpose or test policy differs")
    if inventory["data_seeds"] != list(FINAL_DATA_SEEDS) or inventory["generator_sources"] != source_hashes() or inventory["metric_sources"] != metric_hashes():
        raise ValueError("inventory seed/source identity differs")
    expected = {(item.task, regime, seed) for item in INSTANCES for regime in REGIMES for seed in FINAL_DATA_SEEDS}
    rows = inventory["bundles"]
    observed = [(row["task"], row["regime"], row["seed"]) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("missing or duplicated task/condition/seed combination")
    actual_files = {path.relative_to(directory).as_posix() for path in _files(directory)} - {"inventory.json"}
    if set(inventory["files"]) != actual_files:
        raise ValueError("inventory file set differs")
    for name, checksum in inventory["files"].items():
        if file_hash(directory / name) != checksum:
            raise ValueError(f"inventory checksum mismatch: {name}")
    for row in rows:
        expected_path = f"data/{row['task']}-{row['regime']}-s{row['seed']}"
        if row["path"] != expected_path or row["id"] != expected_path.removeprefix("data/"):
            raise ValueError("unexpected dataset path/identity")
        path = directory / row["path"]
        if file_hash(path / "manifest.json") != row["manifest_sha256"] or file_hash(path / "learner/train.npz") != row["train_sha256"] or file_hash(path / "learner/metadata.json") != row["metadata_sha256"]:
            raise ValueError("dataset identity differs from frozen inventory")
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["purpose"] != "benchmark" or any(manifest[key] != row[key] for key in ("task", "regime", "seed")):
            raise ValueError("dataset purpose/scope differs")
        if manifest["protocol"] != PROTOCOL or manifest["specification_sha256"] != inventory["specification_sha256"] or manifest["generator_sources"] != inventory["generator_sources"]:
            raise ValueError("dataset protocol/source differs")
        if manifest["sampling"] != {"max_per_cell": MAX_PER_CELL, "bank_limit": BANK_LIMIT, "exhaustive_limit": EXHAUSTIVE_LIMIT}:
            raise ValueError("dataset does not use the frozen population settings")
        if semantic:
            checked = validate_bundle(path)
            if any(row[key] != checked[key] for key in ("training_episodes", "evaluation_cases", "cell_populations")):
                raise ValueError("inventory summary disagrees with data semantics")
    return {"valid": True, "protocol": PROTOCOL, "bundles": len(rows), "inventory_sha256": file_hash(directory / "inventory.json"),
            "training_episodes": sum(row["training_episodes"] for row in rows),
            "evaluation_cases": sum(sum(row["evaluation_cases"].values()) for row in rows), "semantic": semantic}


def public_descriptor(directory):
    """A tracked, label-free reference to the immutable local/distributed inventory."""
    directory = Path(directory)
    checked = validate_inventory(directory)
    inventory = json.loads((directory / "inventory.json").read_text())
    return {"schema": "copuzzle.public_data_freeze.v1", **checked,
            "purpose": inventory["purpose"], "test_policy": inventory["test_policy"],
            "data_seeds": inventory["data_seeds"], "generation_versions": inventory["generation_versions"],
            "specification_sha256": inventory["specification_sha256"], "generator_sources": inventory["generator_sources"], "metric_sources": inventory["metric_sources"],
            "datasets": inventory["bundles"]}
