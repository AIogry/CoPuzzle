"""Complete formal matrices registered before fitting, with immutable identities."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import shutil

from ..config import Config, fingerprint
from ..data import load_training
from ..runtime import sources, versions
from ..controls.evaluate import source_sha256 as control_source_sha256
from .storage import file_hash, inside, read_json, write_new


STUDY_SCHEMA = "copuzzle.formal_study.v1"
TERMINAL = ("completed", "failed")


def compile_jobs(datasets, selected, views):
    fits, evaluations, controls = [], [], []
    for coverage in (1.0, 0.25, 0.5):
        for dataset in datasets:
            for seed in selected["fit_seeds"]:
                for method in ("gcbc", "gciql", "dynamics"):
                    config = replace(Config.from_dict(selected["method_configs"][method]), coverage=coverage,
                                     coverage_seed=selected["coverage_seed"]).to_dict()
                    view_id = f"{dataset['id']}__c{int(100 * coverage):03d}"
                    job_id = f"{view_id}__{method}__f{seed}"
                    row = {"id": job_id, "dataset_id": dataset["id"], "view_id": view_id,
                           "task": dataset["task"], "regime": dataset["regime"], "data_seed": dataset["seed"],
                           "fit_seed": seed, "method": method, "coverage": coverage, "primary": coverage == 1,
                           "config": config, "data_sha256": fingerprint(views[view_id]),
                           "steps": selected["training_steps"], "checkpoint_every": selected["checkpoint_every"],
                           "log_every": selected["log_every"], "budgets": selected["model_query_budgets"][method]}
                    fits.append(row)
                    for budget in row["budgets"]:
                        evaluations.append({"id": f"{job_id}__b{budget}", "fit_id": job_id,
                                            "dataset_id": dataset["id"], "kind": "learned", "budget": budget})
    for dataset in datasets:
        settings = [{"kind": "random", "model_query_limit": 0, "coverage": 1.0, "coverage_seed": 0, "seed": seed}
                    for seed in selected["fit_seeds"]]
        settings.append({"kind": "optimal_oracle", "model_query_limit": None, "coverage": 1.0, "coverage_seed": 0, "seed": 0})
        settings.extend({"kind": "oracle_dynamics", "model_query_limit": budget, "coverage": 1.0, "coverage_seed": 0, "seed": 0} for budget in (32, 128))
        kinds = ["lookup"]
        if dataset["task"].startswith("grid-"):
            kinds.append("fitted_xor")
        elif dataset["task"].startswith(("pancake-", "topspin-")):
            kinds.append("fitted_position")
        settings.extend({"kind": kind, "model_query_limit": budget, "coverage": coverage, "coverage_seed": selected["coverage_seed"], "seed": 0}
                        for kind in kinds for coverage in selected["coverage_fractions"] for budget in (32, 128))
        for value in settings:
            budget = "exempt" if value["model_query_limit"] is None else value["model_query_limit"]
            name = f"{dataset['id']}__{value['kind']}__c{int(100 * value['coverage']):03d}__b{budget}__r{value['seed']}"
            controls.append({"id": name, "dataset_id": dataset["id"], "kind": "control", "settings": value})
    return fits, evaluations, controls


def _validate_selected(selected):
    if selected.get("schema") != "copuzzle.selected_methods.v1" or selected.get("checkpoint_selection") != "last update only":
        raise ValueError("unsupported selected-method registry")
    if selected["data_seeds"] != [101, 202, 303] or selected["fit_seeds"] != [0, 1, 2] or selected["coverage_fractions"] != [0.25, 0.5, 1.0]:
        raise ValueError("formal seed/coverage scope differs")
    if selected["primary_fits"] != 324 or selected["coverage_ablation_fits"] != 648 or selected["total_learning_fits"] != 972:
        raise ValueError("formal fitting counts differ")
    if selected["backend"] != "cpu" or selected["max_workers"] != 2:
        raise ValueError("execution differs from the measured CPU registration")
    if selected["model_query_budgets"] != {"gcbc": [0], "gciql": [0], "dynamics": [32, 128]}:
        raise ValueError("model-query conditions differ")
    for method in ("gcbc", "gciql", "dynamics"):
        config = Config.from_dict(selected["method_configs"][method])
        if config.method != method or config.coverage != 1:
            raise ValueError("base method configuration differs")


def register(data_root, selected_path, output, progress=None):
    from copuzzle.benchmark.collection import validate_inventory
    output, data_root, selected_path = Path(output), Path(data_root).resolve(), Path(selected_path)
    if output.exists():
        raise FileExistsError(output)
    checked = validate_inventory(data_root)
    selected = read_json(selected_path)
    _validate_selected(selected)
    if selected["data_inventory_sha256"] != checked["inventory_sha256"] or selected["learner_sources"] != sources() or selected["dependency_versions"] != versions():
        raise ValueError("selected data, learner or dependency identity differs")
    inventory = read_json(data_root / "inventory.json")
    datasets = inventory["bundles"]
    views = {}
    for row in datasets:
        for coverage in selected["coverage_fractions"]:
            data = load_training(data_root / row["path"] / "learner", coverage=coverage, coverage_seed=selected["coverage_seed"])
            views[f"{row['id']}__c{int(100 * coverage):03d}"] = data.identity
        if progress:
            progress({"registered_training_views": len(views), "dataset": row["id"]})
    fits, evaluations, controls = compile_jobs(datasets, selected, views)
    output.mkdir(parents=True)
    shutil.copyfile(selected_path, output / "selected_methods.json")
    shutil.copyfile(data_root / "inventory.json", output / "data_inventory.json")
    shutil.copytree(data_root / "source/copuzzle", output / "source/copuzzle")
    package = Path(__file__).resolve().parents[1]
    for path in package.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        target = output / "source/copuzzle_baselines" / path.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    code = {path.relative_to(output / "source").as_posix(): file_hash(path)
            for path in sorted((output / "source").rglob("*.py"))}
    study = {"schema": STUDY_SCHEMA, "role": "formal", "selected_methods_sha256": file_hash(output / "selected_methods.json"),
             "data_inventory_sha256": checked["inventory_sha256"], "learner_sources": sources(),
             "dependency_versions": versions(), "control_source_sha256": control_source_sha256(),
             "source_snapshot": code, "backend": selected["backend"], "max_workers": selected["max_workers"],
             "retry_policy": "explicit retry of failed jobs only; stopped jobs resume latest valid checkpoint; no retries after test seal",
             "checkpoint_selection": "last registered update only", "datasets": datasets, "training_views": views,
             "fit_jobs": fits, "evaluation_jobs": evaluations, "control_jobs": controls}
    write_new(output / "study.json", study)
    write_new(output / "registration.json", {"schema": "copuzzle.study_registration.v1", "study_sha256": file_hash(output / "study.json"),
                                            "selected_methods_sha256": study["selected_methods_sha256"],
                                            "data_inventory_sha256": study["data_inventory_sha256"],
                                            "source_snapshot_sha256": fingerprint(code)})
    write_new(output / "locations.json", {"data_root": str(data_root)})
    return validate_study(output, verify_data=True, verify_snapshot=True)


def load_study(root):
    root = Path(root)
    study = read_json(root / "study.json")
    registration = read_json(root / "registration.json")
    if registration.get("schema") != "copuzzle.study_registration.v1" or file_hash(root / "study.json") != registration["study_sha256"]:
        raise ValueError("study differs from its immutable registration")
    if study.get("schema") != STUDY_SCHEMA or study.get("role") not in ("formal", "qualification"):
        raise ValueError("unsupported study identity")
    return study, registration["study_sha256"]


def data_directory(root, override=None):
    return Path(override or read_json(Path(root) / "locations.json")["data_root"]).resolve()


def validate_study(root, *, verify_data=False, verify_snapshot=False):
    root = Path(root)
    study, checksum = load_study(root)
    selected = read_json(root / "selected_methods.json")
    if file_hash(root / "selected_methods.json") != study["selected_methods_sha256"]:
        raise ValueError("selected method file changed")
    if study["role"] == "formal":
        _validate_selected(selected)
        if (study["learner_sources"] != selected["learner_sources"] or study["dependency_versions"] != selected["dependency_versions"]
                or study["data_inventory_sha256"] != selected["data_inventory_sha256"] or study["backend"] != selected["backend"]
                or study["max_workers"] != selected["max_workers"] or study["checkpoint_selection"] != "last registered update only"):
            raise ValueError("study differs from the selected source/runtime/data/checkpoint conditions")
        expected = compile_jobs(study["datasets"], selected, study["training_views"])
        if any(study[key] != value for key, value in zip(("fit_jobs", "evaluation_jobs", "control_jobs"), expected)):
            raise ValueError("formal job matrix differs from frozen scope")
        if len(study["fit_jobs"]) != 972 or len(study["evaluation_jobs"]) != 1296 or len(study["control_jobs"]) != 576:
            raise ValueError("formal job counts differ")
        datasets = study["datasets"]
        from copuzzle.benchmark.spec import INSTANCES, REGIMES
        combinations = {(item.task, regime, seed) for item in INSTANCES for regime in REGIMES for seed in (101, 202, 303)}
        if len(datasets) != 36 or {(d["task"], d["regime"], d["seed"]) for d in datasets} != combinations:
            raise ValueError("formal dataset cohort differs")
        for row in datasets:
            for coverage in selected["coverage_fractions"]:
                view = study["training_views"][f"{row['id']}__c{int(100*coverage):03d}"]
                expected_view = {"task": row["task"], "regime": row["regime"], "purpose": "benchmark",
                                 "protocol": "copuzzle.rule-composition.v1", "coverage": coverage,
                                 "coverage_seed": selected["coverage_seed"], "train_sha256": row["train_sha256"],
                                 "metadata_sha256": row["metadata_sha256"]}
                if any(view.get(key) != value for key, value in expected_view.items()):
                    raise ValueError("training view differs from registered data/coverage")
    for key in ("fit_jobs", "evaluation_jobs", "control_jobs"):
        ids = [row["id"] for row in study[key]]
        if len(ids) != len(set(ids)) or any("/" in name or ".." in name or "\\" in name for name in ids):
            raise ValueError("unsafe or duplicated job identity")
    if file_hash(root / "data_inventory.json") != study["data_inventory_sha256"]:
        raise ValueError("frozen data inventory copy changed")
    if verify_snapshot:
        for name, digest in study["source_snapshot"].items():
            if file_hash(inside(root / "source", name)) != digest:
                raise ValueError("worker source snapshot changed")
    if verify_data:
        if study["role"] == "formal":
            from copuzzle.benchmark.collection import validate_inventory
            verified = validate_inventory(data_directory(root))
            if verified["inventory_sha256"] != study["data_inventory_sha256"]:
                raise ValueError("relocated data do not match the registered inventory")
        else:
            from copuzzle.benchmark.generation import validate_bundle
            for row in study["datasets"]:
                bundle = data_directory(root) / row["path"]
                if file_hash(bundle / "manifest.json") != row["manifest_sha256"]:
                    raise ValueError("qualification data changed")
                validate_bundle(bundle)
    return {"valid": True, "role": study["role"], "study_sha256": checksum, "fits": len(study["fit_jobs"]),
            "primary_fits": sum(row.get("primary", False) for row in study["fit_jobs"]) if study["role"] == "formal" else 0,
            "learned_evaluations": len(study["evaluation_jobs"]), "control_evaluations": len(study["control_jobs"])}


def fit_context(study, study_sha256, job):
    return {"schema": "copuzzle.formal_fit_context.v1", "study_sha256": study_sha256, "job_id": job["id"],
            "config_sha256": fingerprint(job["config"]), "data_sha256": job["data_sha256"],
            "sources_sha256": fingerprint(study["learner_sources"]), "versions_sha256": fingerprint(study["dependency_versions"]),
            "fit_seed": job["fit_seed"], "steps": job["steps"], "backend": study["backend"]}


def dataset(study, dataset_id):
    return next(row for row in study["datasets"] if row["id"] == dataset_id)


def job_for(study, phase, job_id):
    key = "fit_jobs" if phase == "fit" else "control_jobs" if phase.startswith("control") else "evaluation_jobs"
    return next(row for row in study[key] if row["id"] == job_id)
