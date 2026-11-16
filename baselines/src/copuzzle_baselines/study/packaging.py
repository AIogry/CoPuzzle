"""Deterministic, checksum-verified local distribution staging; never publishing."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from .analysis import attempt_costs, collect_rows
from .evidence import verify_seal
from .registry import load_study
from .statistics import summarize
from .storage import active, file_hash, inside, read_json, write_new


def _safe_name(name):
    path = PurePosixPath(name)
    if not name or not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in path.parts[0] or path.as_posix() != name:
        raise ValueError("archive members must have normalized relative POSIX paths")
    return name


def verify_archives(directory):
    """Inspect regular members without extracting or following links."""
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema") != "copuzzle.distribution_stage.v1":
        raise ValueError("unsupported distribution stage")
    if file_hash(directory / "builder.py") != manifest["builder_sha256"]:
        raise ValueError("distribution builder snapshot differs")
    expected = manifest["members"]
    for name in expected:
        _safe_name(name)
    seen, archive_names = set(), set()
    for part in manifest["archives"]:
        if part["file"] in archive_names:
            raise ValueError("duplicated distribution archive")
        archive_names.add(part["file"])
        archive = inside(directory, part["file"])
        if file_hash(archive) != part["sha256"] or archive.stat().st_size != part["bytes"]:
            raise ValueError("distribution archive checksum/size differs")
        actual_part = []
        with tarfile.open(archive, "r:gz") as stream:
            for entry in stream:
                name = _safe_name(entry.name)
                if not entry.isfile() or name in seen or name not in expected:
                    raise ValueError("unexpected/duplicate/nonregular archive member")
                if entry.size != expected[name]["bytes"]:
                    raise ValueError("archive member size differs")
                digest = hashlib.sha256()
                file = stream.extractfile(entry)
                size = 0
                for block in iter(lambda: file.read(1024*1024), b""):
                    size += len(block)
                    digest.update(block)
                if size != entry.size or digest.hexdigest() != expected[name]["sha256"]:
                    raise ValueError("archive member contents differ")
                seen.add(name)
                actual_part.append(name)
        if actual_part != part["members"]:
            raise ValueError("archive part inventory differs")
    if seen != set(expected):
        raise ValueError("distribution omits expected members")
    return {"valid": True, "archives": len(archive_names), "members": len(seen),
            "manifest_sha256": file_hash(directory / "manifest.json")}


def build_archives(files, output, *, identity, part_bytes=1024**3):
    """Stable path ordering, zero metadata timestamps, bounded uncompressed parts."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if type(part_bytes) is not int or part_bytes < 1 or not files:
        raise ValueError("nonempty files and a positive part size are required")
    members, groups, current, size = {}, [], [], 0
    for name, value in sorted(files.items()):
        _safe_name(name)
        path = Path(value)
        if path.is_symlink() or not path.is_file():
            raise ValueError("stage only regular files, not symlinks")
        count = path.stat().st_size
        if count > part_bytes:
            raise ValueError("one artifact exceeds the selected archive part size")
        if current and size+count > part_bytes:
            groups.append(current)
            current, size = [], 0
        current.append((name, path))
        size += count
        members[name] = {"bytes": count, "sha256": file_hash(path)}
    groups.append(current)
    output.mkdir(parents=True)
    builder_source = Path(__file__).read_bytes()
    (output / "builder.py").write_bytes(builder_source)
    archives = []
    for number, group in enumerate(groups, 1):
        destination = output / f"part-{number:03d}.tar.gz"
        with destination.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    for name, path in group:
                        info = tarfile.TarInfo(name)
                        info.size, info.mode, info.mtime = members[name]["bytes"], 0o644, 0
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
        archives.append({"file": destination.name, "sha256": file_hash(destination),
                         "bytes": destination.stat().st_size, "members": [name for name, _ in group]})
    manifest = {"schema": "copuzzle.distribution_stage.v1", "identity": identity,
                "publication_status": "local draft; publication/license/acceptance gates are separate",
                "members": members, "archives": archives, "part_uncompressed_byte_limit": part_bytes,
                "builder_sha256": hashlib.sha256(builder_source).hexdigest()}
    write_new(output / "manifest.json", manifest)
    verified = verify_archives(output)
    write_new(output / "verification.json", verified)
    return verified


def stage_data(data_root, output, *, part_bytes=1024**3):
    from copuzzle.benchmark.collection import validate_inventory
    data_root = Path(data_root).resolve()
    verified = validate_inventory(data_root)
    inventory = read_json(data_root / "inventory.json")
    files = {f"data/{name}": inside(data_root, name) for name in inventory["files"]}
    files["data/inventory.json"] = data_root / "inventory.json"
    return build_archives(files, output, identity={"kind": "frozen benchmark data", "protocol": inventory["protocol"],
        "inventory_sha256": verified["inventory_sha256"]}, part_bytes=part_bytes)


def stage_physical(manifest, inputs, qualification, script, output, *, part_bytes=1024**3):
    manifest, inputs, qualification, script = Path(manifest), Path(inputs), Path(qualification), Path(script)
    inventory, report = read_json(manifest), read_json(qualification)
    if (inventory.get("schema") != "copuzzle.physical_history_inputs.v1"
            or report.get("schema") != "copuzzle.physical_history_qualification.v1"
            or report.get("valid") is not True or report["manifest_sha256"] != file_hash(manifest)
            or report["script_sha256"] != file_hash(script)):
        raise ValueError("physical input qualification binding differs")
    cases = inventory["cases"]
    if len(cases) != 3 or {r["task"] for r in cases} != {"grid-3x3", "pancake-4", "topspin-20-4"} or len(report["cases"]) != 3:
        raise ValueError("physical qualification omits a required family")
    files = {"physical-history/inputs.json": manifest, "physical-history/qualification.json": qualification,
             "physical-history/verify_physical_history.py": script}
    for case, row in zip(cases, report["cases"]):
        if row["status"] != "passed" or any(row[k] != v for k, v in case.items()):
            raise ValueError("physical replay evidence differs from the input inventory")
        for path_key, hash_key in (("path", "sha256"), ("metadata_path", "metadata_sha256")):
            path = inside(inputs, case[path_key])
            if file_hash(path) != case[hash_key]:
                raise ValueError("physical input or sidecar changed")
            files[f"physical-history/{case[path_key]}"] = path
    return build_archives(files, output, identity={"kind": "historical native physical action archives",
        "manifest_sha256": file_hash(manifest), "qualification_sha256": file_hash(qualification)}, part_bytes=part_bytes)


def stage_study(root, predictions, analysis, output, *, report=None, part_bytes=1024**3):
    """Stage selected checkpoints, all attempt evidence and reproducible case outputs."""
    from .prediction_archive import verify_export
    root, predictions, analysis = Path(root).resolve(), Path(predictions).resolve(), Path(analysis).resolve()
    study, checksum = load_study(root)
    if study["role"] != "formal":
        raise ValueError("formal distribution requires a formal study")
    if report is None:
        raise ValueError("formal distribution requires a readable report and evidence-based interpretation")
    from .report import verify_report
    report = Path(report).resolve()
    readable = verify_report(report, analysis_sha256=file_hash(analysis / "manifest.json"), study_sha256=checksum)
    if active(root / ".controller.lock") or any(active(p) for p in (root / "jobs").glob("*/*/.lease")):
        raise ValueError("cannot stage an active study")
    seal = verify_seal(root / "selection/test-lock.json", verify_receipts=True)
    rows, reports = collect_rows(root, "test")
    prediction_evidence = verify_export(root, predictions, rows=rows, reports=reports)
    result = read_json(analysis / "manifest.json")
    if (result.get("schema") != "copuzzle.study_analysis.v1" or result.get("study_sha256") != checksum
            or result.get("split") != "test" or result.get("role") != "formal"
            or result.get("per_case_evidence_complete") is not True
            or result.get("prediction_evidence") != prediction_evidence
            or result.get("evaluation_jobs") != len(rows)):
        raise ValueError("analysis is not complete for this study/prediction evidence")
    if read_json(analysis / "rows.json") != rows or read_json(analysis / "costs.json") != attempt_costs(root):
        raise ValueError("analysis rows/costs do not match the current complete attempt history")
    from copuzzle.benchmark.spec import INSTANCES
    recomputed = summarize(rows, data_seeds=sorted({d["seed"] for d in study["datasets"]}),
                          fit_seeds=sorted({j["fit_seed"] for j in study["fit_jobs"]}),
                          families={s.task: s.family for s in INSTANCES})
    if read_json(analysis / "statistics.json") != recomputed:
        raise ValueError("analysis statistics do not reproduce the registered calculation")
    files = {}
    def add(prefix, base, relative):
        path = inside(base, relative)
        files[f"{prefix}/{relative}"] = path
    for name in ("study.json", "registration.json", "data_inventory.json", "selected_methods.json"):
        add("study", root, name)
    for name in study["source_snapshot"]:
        add("study", root, f"source/{name}")
    for path in (root / "selection").iterdir():
        if path.is_file():
            add("study", root, path.relative_to(root).as_posix())
    for checkpoint in seal["checkpoints"]:
        add("study", root, checkpoint["checkpoint"])
    for path in (root / "jobs").rglob("*"):
        if not path.is_file() or "source" in path.relative_to(root).parts:
            continue
        # Intermediate checkpoints are omitted unless used by a saved recovery.
        # All manifests and attempt/cost records remain available.
        if path.suffix in (".json", ".jsonl", ".log"):
            relative = path.relative_to(root).as_posix()
            add("study", root, relative)
            if path.name == "record.json":
                record = read_json(path)
                for key in ("checkpoint", "resume_checkpoint"):
                    if record.get(key):
                        add("study", root, record[key])
    for folder in ("controllers", "logs", "locations_history"):
        if (root / folder).exists():
            for path in (root / folder).rglob("*"):
                if path.is_file() and path.suffix in (".json", ".log", ".py"):
                    add("study", root, path.relative_to(root).as_posix())
    for name in ("analysis-registration.json", "analysis_plan.json"):
        if (root / name).exists():
            add("study", root, name)
    # locations.json is deliberately absent: relocate binds local paths after
    # extraction and verifies the data before any worker reads them.
    index = read_json(predictions / "index.json")
    prediction_files = {"index.json", "registration.json", "prediction_archive.py"}
    for entry in index["entries"]:
        if entry["prediction_status"] == "verified":
            prediction_files.update((entry["archive"], str(PurePosixPath(entry["archive"]).with_suffix(".json"))))
    for name in sorted(prediction_files):
        add("predictions", predictions, name)
    for name, digest in result["files"].items():
        if file_hash(inside(analysis, name)) != digest:
            raise ValueError("analysis output changed")
        add("analysis", analysis, name)
    add("analysis", analysis, "manifest.json")
    for name in readable["files"]:
        add("report", report, name)
    add("report", report, "manifest.json")
    return build_archives(files, output, identity={"kind": "formal study evidence and selected checkpoints",
        "study_sha256": checksum, "data_inventory_sha256": study["data_inventory_sha256"],
        "analysis_manifest_sha256": file_hash(analysis / "manifest.json"),
        "prediction_index_sha256": prediction_evidence["index_sha256"],
        "readable_report_manifest_sha256": file_hash(report / "manifest.json"),
        "omissions": "unselected intermediate checkpoints and identical per-fit source copies; current machine data path and lock files",
        "relocation": "run study.runner relocate --study <study> --data <data> after extraction"}, part_bytes=part_bytes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("data", "study", "physical", "verify"):
        p = commands.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        if name != "verify":
            p.add_argument("--part-bytes", type=int, default=1024**3)
        if name == "data":
            p.add_argument("--data", type=Path, required=True)
        if name == "study":
            p.add_argument("--study", type=Path, required=True)
            p.add_argument("--predictions", type=Path, required=True)
            p.add_argument("--analysis", type=Path, required=True)
            p.add_argument("--report", type=Path, required=True)
        if name == "physical":
            for flag in ("manifest", "inputs", "qualification", "script"):
                p.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify_archives(args.output)
    elif args.command == "data":
        result = stage_data(args.data, args.output, part_bytes=args.part_bytes)
    elif args.command == "physical":
        result = stage_physical(args.manifest, args.inputs, args.qualification, args.script, args.output, part_bytes=args.part_bytes)
    else:
        result = stage_study(args.study, args.predictions, args.analysis, args.output, report=args.report, part_bytes=args.part_bytes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
