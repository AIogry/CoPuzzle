"""Reproduce selected R/C predictions and bind per-case evidence to saved scores.

The registered evaluation reports already contain all G cases. This evaluator
side utility preserves R/C predictions as well, without modifying the frozen
worker snapshot or refitting any learner. Both planning budgets share one R/C
archive because these probes are independent of the G search budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from ..config import fingerprint
from ..runtime import _check_data, _predict, load_checkpoint, sources, state_digest, versions
from ..controls.evaluate import _predictions, source_sha256 as control_source_sha256
from ..controls.models import fit_control
from .analysis import collect_rows
from .evidence import verify_fit
from .registry import data_directory, dataset, job_for, load_study
from .storage import file_hash, inside, lease, read_json, write_new


def score_archive(path, bundle, report, *, split):
    """Check preserved predictions against the exact registered metric reports."""
    from copuzzle.benchmark.evaluation import score_predictions
    with np.load(path, allow_pickle=False) as file:
        expected = {"R_next_board", "R_legal", "C_prefix_boards"}
        if set(file.files) != expected:
            raise ValueError("prediction archive members differ")
        predictions = {"R": {"next_board": file["R_next_board"], "legal": file["R_legal"]},
                       "C": {"prefix_boards": file["C_prefix_boards"]}}
    for probe in ("R", "C"):
        scored = {"supported": True, **score_predictions(bundle, probe, predictions[probe], split=split)}
        if scored != report[probe]:
            raise ValueError(f"archived {probe} predictions do not reproduce saved scores")
    return predictions


def save_predictions(destination, predictions):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz") as temp:
        np.savez_compressed(temp.name, R_next_board=predictions["R"]["next_board"], R_legal=predictions["R"]["legal"],
                            C_prefix_boards=predictions["C"]["prefix_boards"])
        # Hard-link publication cannot silently replace an existing archive.
        import os
        os.link(temp.name, destination)


def _produce(root, study, checksum, job, bundle, split):
    from copuzzle.benchmark.data import load_probe, metadata
    lock = root / "selection/test-lock.json" if split == "test" else None
    if job["kind"] == "learned":
        fitted = job_for(study, "fit", job["fit_id"])
        record = verify_fit(root, study, checksum, fitted)
        path = inside(root, record["checkpoint"])
        agent, vocabulary, header = load_checkpoint(path)
        _check_data(bundle / "learner", header)
        if split == "test":
            from .access import authorize_test
            authorize_test(lock, path, bundle, header)
        before = state_digest(agent)
        predictions = {probe: _predict(agent, vocabulary, load_probe(bundle / "learner", probe, split), probe, 512)
                       for probe in ("R", "C")}
        if state_digest(agent) != before:
            raise ValueError("prediction export changed learner state")
        return predictions
    settings = job["settings"]
    if split == "test":
        from .access import authorize_control_test
        authorize_control_test(lock, bundle, settings, control_source_sha256())
    if settings["kind"] in ("oracle_dynamics", "optimal_oracle"):
        from copuzzle.benchmark.graph import partition
        meta = metadata(bundle / "learner")
        manifest = read_json(bundle / "manifest.json")
        p = partition(meta["task"], meta["regime"], manifest["seed"])
        class ExactModel:
            def predict(self, boards, actions):
                rows = [p.successor(p.graph.locate(board), int(action)) for board, action in zip(boards, actions)]
                return p.graph.states[[i for i, _ in rows]], np.asarray([legal for _, legal in rows], np.uint8)
        model = ExactModel()
    else:
        model, _ = fit_control(bundle / "learner", settings["kind"], coverage=settings["coverage"], coverage_seed=settings["coverage_seed"])
    return {probe: _predictions(model, load_probe(bundle / "learner", probe, split), probe) for probe in ("R", "C")}


def export(root, output, *, split="test", progress=None):
    invocation_started = time.perf_counter()
    root, output = Path(root).resolve(), Path(output).resolve()
    rows, reports = collect_rows(root, split)
    study, checksum = load_study(root)
    if sources() != study["learner_sources"] or versions() != study["dependency_versions"] or control_source_sha256() != study["control_source_sha256"]:
        raise ValueError("prediction exporter numerical implementation/runtime differs")
    output.mkdir(parents=True, exist_ok=True)
    with lease(output / ".export.lock", "export selected per-case predictions"):
        if (output / "index.json").exists():
            raise FileExistsError("prediction export already completed; verify the existing index")
        binding = {"schema": "copuzzle.prediction_archive.v1", "study_sha256": checksum, "split": split,
                   "exporter_sha256": file_hash(__file__), "analysis_source_sha256": file_hash(Path(__file__).with_name("analysis.py"))}
        if (output / "registration.json").exists():
            if read_json(output / "registration.json") != binding:
                raise ValueError("export resume identity changed")
        else:
            write_new(output / "registration.json", binding)
            (output / "prediction_archive.py").write_bytes(Path(__file__).read_bytes())
        entries, produced = [], {}
        for row in rows:
            entry = {"job_id": row["job_id"], "report_sha256": row["report_sha256"], "status": row["status"]}
            if row["status"] == "failed" or row["metrics"]["R"] is None:
                entry["prediction_status"] = "unavailable" if row["status"] == "failed" else "unsupported"
                entries.append(entry)
                continue
            job = job_for(study, row["phase"], row["job_id"])
            data = dataset(study, job["dataset_id"])
            bundle = data_directory(root) / data["path"]
            # Reference R/C predictions do not depend on rollout seed or G budget.
            identity = {"fit_id": job["fit_id"]} if job["kind"] == "learned" else {
                "dataset_id": data["id"], "kind": job["settings"]["kind"],
                "coverage": job["settings"]["coverage"], "coverage_seed": job["settings"]["coverage_seed"]}
            key = fingerprint(identity)
            path, receipt = output / f"{key}.npz", output / f"{key}.json"
            if key not in produced:
                if receipt.exists():
                    evidence = read_json(receipt)
                    if evidence["identity"] != identity or evidence["sha256"] != file_hash(path):
                        raise ValueError("resumed prediction archive changed")
                else:
                    started = time.perf_counter()
                    recovered = path.exists()
                    if not path.exists():
                        predictions = _produce(root, study, checksum, job, bundle, split)
                        save_predictions(path, predictions)
                    score_archive(path, bundle, reports[row["job_id"]], split=split)
                    evidence = {"identity": identity, "file": path.name, "sha256": file_hash(path),
                                "production_and_verification_seconds": time.perf_counter()-started,
                                "recovered_after_missing_receipt": recovered,
                                "prior_production_seconds_unknown": recovered}
                    write_new(receipt, evidence)
                produced[key] = evidence
            score_archive(path, bundle, reports[row["job_id"]], split=split)
            entry.update(prediction_status="verified", archive=path.name, archive_sha256=produced[key]["sha256"])
            entries.append(entry)
            if progress:
                progress({"verified_jobs": len(entries), "expected": len(rows), "archives": len(produced)})
        result = {**binding, "entries": entries, "unique_archives": len(produced),
                  "costs": list(produced.values()), "completed_invocation_elapsed_seconds": time.perf_counter()-invocation_started,
                  "note": "R/C inference repeated for verifiable per-case evidence; no refitting; G cases remain in original reports; invocation time includes revalidation, per-archive times exclude repeated checks"}
        write_new(output / "index.json", result)
        return result


def verify_export(root, output, *, split="test", rows=None, reports=None):
    root, output = Path(root), Path(output)
    if rows is None or reports is None:
        rows, reports = collect_rows(root, split)
    _, checksum = load_study(root)
    index = read_json(output / "index.json")
    binding = read_json(output / "registration.json")
    if (index.get("schema") != "copuzzle.prediction_archive.v1" or index.get("study_sha256") != checksum
            or index.get("split") != split or any(index.get(k) != v for k, v in binding.items())
            or file_hash(output / "prediction_archive.py") != binding["exporter_sha256"]):
        raise ValueError("prediction archive provenance differs")
    entries = index["entries"]
    if [e["job_id"] for e in entries] != [r["job_id"] for r in rows]:
        raise ValueError("prediction index omits/duplicates/reorders registered outcomes")
    study, _ = load_study(root)
    paths = set()
    for row, entry in zip(rows, entries):
        expected = "unavailable" if row["status"] == "failed" else "unsupported" if row["metrics"]["R"] is None else "verified"
        if (entry["prediction_status"] != expected or entry["report_sha256"] != row["report_sha256"]
                or entry["status"] != row["status"]):
            raise ValueError("prediction index outcome differs from the evaluation")
        if expected == "verified":
            path = inside(output, entry["archive"])
            if file_hash(path) != entry["archive_sha256"]:
                raise ValueError("prediction archive bytes changed")
            job = job_for(study, row["phase"], row["job_id"])
            bundle = data_directory(root) / dataset(study, job["dataset_id"])["path"]
            score_archive(path, bundle, reports[row["job_id"]], split=split)
            paths.add(path)
    if len(paths) != index["unique_archives"]:
        raise ValueError("unique prediction archive count differs")
    return {"valid": True, "index_sha256": file_hash(output / "index.json"),
            "verified_evaluation_jobs": len(entries), "unique_archives": len(paths)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify_export(args.study, args.output, split=args.split)))
    else:
        result = export(args.study, args.output, split=args.split, progress=lambda row: print(json.dumps(row), flush=True))
        print(json.dumps({"verified_evaluation_jobs": len(result["entries"]), "archives": result["unique_archives"]}))


if __name__ == "__main__":
    main()
