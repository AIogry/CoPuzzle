from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

import copuzzle
import copuzzle_baselines
from copuzzle.benchmark.generation import write_bundle
from copuzzle.benchmark.spec import INSTANCES, REGIMES
from copuzzle_baselines.config import Config, fingerprint
from copuzzle_baselines.controls.evaluate import source_sha256 as control_source_sha256
from copuzzle_baselines.data import load_training
from copuzzle_baselines.runtime import load_checkpoint, sources, state_digest, train, versions
from copuzzle_baselines.study.evidence import seal, verify_seal
from copuzzle_baselines.study.jobs import record_startup_failure, run_job
from copuzzle_baselines.study.registry import compile_jobs, load_study, validate_study
from copuzzle_baselines.study.runner import run, snapshot
from copuzzle_baselines.study.storage import BusyStudy, active, file_hash, lease, verified_record, write_new


def test_study_analysis_requires_terminal_evaluations_and_valid_reports(qualification):
    from copuzzle_baselines.study.analysis import collect_rows, attempt_costs
    root, _, _ = qualification
    with pytest.raises(ValueError, match="nonterminal"):
        collect_rows(root, "dev", include_controls=False)
    run_job(root, "fit", "tiny-gcbc")
    run_job(root, "dev", "tiny-gcbc-b0")
    rows, reports = collect_rows(root, "dev", include_controls=False)
    assert len(rows) == 1 and len(reports) == 1
    assert rows[0]["metrics"]["R"] is None
    assert rows[0]["metrics"]["C"] is None
    costs = attempt_costs(root)
    assert len(costs["attempts"]) == 2 and costs["known_process_work_seconds"] > 0
    report = root / rows[0]["report"]
    report.write_text('{}\n')
    with pytest.raises(ValueError, match="changed"):
        collect_rows(root, "dev", include_controls=False)


def test_study_analysis_keeps_failed_fits_and_evaluations(qualification):
    from copuzzle_baselines.study.analysis import collect_rows
    root, _, _ = qualification
    record_startup_failure(root, "fit", "tiny-gcbc", RuntimeError("fit failed before startup"))
    record_startup_failure(root, "dev", "tiny-gcbc-b0", RuntimeError("fitted model unavailable"))
    rows, reports = collect_rows(root, "dev", include_controls=False)
    assert len(rows) == 1 and not reports
    assert rows[0]["fit_status"] == rows[0]["status"] == "failed"
    assert rows[0]["metrics"] == {"R": None, "C": None, "G": 0.0, "G_normalized_cost": 1.0}
    assert rows[0]["expected_goal_cases"] == 3


def test_cost_history_counts_retries_once_and_rejects_history_changes(qualification, monkeypatch):
    from copuzzle_baselines.study.analysis import attempt_costs
    root, _, _ = qualification
    first = record_startup_failure(root, "fit", "tiny-gcbc", RuntimeError("startup failure"))
    assert run_job(root, "fit", "tiny-gcbc", retry_failed=True)["status"] == "completed"
    monkeypatch.chdir(root.parent)
    costs = attempt_costs(Path(root.name))
    assert len(costs["attempts"]) == 2
    assert {r["attempt"] for r in costs["attempts"]} == {1, 2}
    assert costs["known_process_work_seconds"] > 0
    assert not costs["unlinked_attempt_directories"]
    path = root / "jobs/fit/tiny-gcbc/attempts/0001/record.json"
    first["process_work_seconds"] = 9999
    path.write_text(json.dumps(first))
    with pytest.raises(ValueError, match="historical"):
        attempt_costs(root)


def test_prediction_archive_reproduces_selected_learners_and_controls(qualification, tmp_path):
    from copuzzle.benchmark.data import load_probe
    from copuzzle_baselines.study.prediction_archive import export, verify_export
    from copuzzle_baselines.study.analysis import analyze
    root, bundle, _ = qualification
    study, _ = load_study(root)
    fit = study["fit_jobs"][0]
    fit["method"] = "dynamics"
    fit["config"]["method"] = "dynamics"
    fit["budgets"] = [8, 16]
    study["evaluation_jobs"] = [{"id": f"tiny-b{budget}", "fit_id": fit["id"], "dataset_id": "tiny", "kind": "learned", "budget": budget}
                                for budget in (8, 16)]
    study["control_jobs"] = [{"id": kind, "dataset_id": "tiny", "kind": "control",
                              "settings": {"kind": kind, "model_query_limit": None if kind == "optimal_oracle" else 8,
                                           "coverage": 1.0, "coverage_seed": 0, "seed": 0}}
                             for kind in ("lookup", "fitted_xor", "oracle_dynamics", "optimal_oracle")]
    for probe in ("R", "C"):
        study["datasets"][0]["evaluation_cases"][f"{probe}_dev"] = len(load_probe(bundle / "learner", probe, "dev")["board"])
    (root / "study.json").write_text(json.dumps(study))
    (root / "registration.json").write_text(json.dumps({"schema": "copuzzle.study_registration.v1", "study_sha256": file_hash(root / "study.json")}))
    assert run_job(root, "fit", fit["id"])["status"] == "completed"
    for phase, jobs in (("dev", study["evaluation_jobs"]), ("control-dev", study["control_jobs"])):
        for job in jobs:
            assert run_job(root, phase, job["id"])["status"] == "completed"
    output = tmp_path / "predictions"
    index = export(root, output, split="dev")
    assert len(index["entries"]) == 6 and index["unique_archives"] == 5
    assert index["entries"][0]["archive"] == index["entries"][1]["archive"]
    assert verify_export(root, output, split="dev")["valid"]
    result = analyze(root, tmp_path / "analysis", split="dev", repetitions=20, predictions=output)
    assert result["per_case_evidence_complete"] and result["evaluation_jobs"] == 6
    with pytest.raises(FileExistsError):
        export(root, output, split="dev")
    path = output / index["entries"][0]["archive"]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        verify_export(root, output, split="dev")


@pytest.fixture
def qualification(tmp_path):
    data_root = tmp_path / "data"
    bundle = write_bundle(data_root / "tiny", "grid-3x3", "transition_holdout", 0, max_per_cell=1)
    data = load_training(bundle / "learner")
    config = Config(method="gcbc", hidden_dims=(4,), batch_size=4)
    root = tmp_path / "study"
    root.mkdir()
    write_new(root / "selected_methods.json", {"role": "qualification"})
    write_new(root / "data_inventory.json", {"role": "qualification"})
    write_new(root / "locations.json", {"data_root": str(data_root)})
    for package, module in (("copuzzle", copuzzle), ("copuzzle_baselines", copuzzle_baselines)):
        base = Path(module.__file__).parent
        for path in base.rglob("*.py"):
            if "__pycache__" not in path.parts:
                target = root / "source" / package / path.relative_to(base)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    source_files = {path.relative_to(root / "source").as_posix(): file_hash(path) for path in (root / "source").rglob("*.py")}
    job = {"id": "tiny-gcbc", "dataset_id": "tiny", "view_id": "tiny", "method": "gcbc",
           "config": config.to_dict(), "data_sha256": data.digest, "fit_seed": 0, "steps": 4,
           "checkpoint_every": 1, "log_every": 1, "budgets": [0], "primary": False}
    dataset = {"id": "tiny", "path": "tiny", "task": "grid-3x3", "regime": "transition_holdout", "seed": 0,
               "manifest_sha256": file_hash(bundle / "manifest.json"), "evaluation_cases": {"G_dev": 3, "G_test": 3}}
    study = {"schema": "copuzzle.formal_study.v1", "role": "qualification",
             "selected_methods_sha256": file_hash(root / "selected_methods.json"), "data_inventory_sha256": file_hash(root / "data_inventory.json"),
             "learner_sources": sources(), "dependency_versions": versions(), "control_source_sha256": control_source_sha256(),
             "source_snapshot": source_files, "backend": "cpu", "max_workers": 2, "checkpoint_selection": "last registered update only",
             "datasets": [dataset], "training_views": {"tiny": data.identity}, "fit_jobs": [job],
             "evaluation_jobs": [{"id": "tiny-gcbc-b0", "fit_id": job["id"], "dataset_id": "tiny", "kind": "learned", "budget": 0}], "control_jobs": []}
    write_new(root / "study.json", study)
    write_new(root / "registration.json", {"schema": "copuzzle.study_registration.v1", "study_sha256": file_hash(root / "study.json")})
    return root, bundle, config


def test_registered_matrix_has_every_primary_and_ablation_cell():
    datasets = [{"id": f"{s.task}-{r}-{seed}", "task": s.task, "regime": r, "seed": seed}
                for seed in (101, 202, 303) for s in INSTANCES for r in REGIMES]
    selected = {"fit_seeds": [0, 1, 2], "method_configs": {m: Config(method=m).to_dict() for m in ("gcbc", "gciql", "dynamics")},
                "coverage_seed": 0, "coverage_fractions": [0.25, 0.5, 1.0], "training_steps": 10000,
                "checkpoint_every": 1000, "log_every": 100, "model_query_budgets": {"gcbc": [0], "gciql": [0], "dynamics": [32, 128]}}
    views = {f"{d['id']}__c{int(100*c):03d}": {"dataset": d["id"], "coverage": c} for d in datasets for c in selected["coverage_fractions"]}
    fits, evaluations, controls = compile_jobs(datasets, selected, views)
    assert len(fits) == 972 and sum(row["primary"] for row in fits) == 324
    assert len(evaluations) == 1296 and len(controls) == 576
    assert len({r["id"] for r in fits}) == 972
    assert all(len([e for e in evaluations if e["fit_id"] == r["id"]]) == (2 if r["method"] == "dynamics" else 1) for r in fits)


def test_kernel_lease_is_authoritative_and_probe_does_not_rewrite(tmp_path):
    path = tmp_path / ".lock"
    with lease(path, "real worker"):
        before = path.read_bytes()
        assert active(path)
        with pytest.raises(BusyStudy):
            with lease(path):
                pass
        assert path.read_bytes() == before
    before = path.read_bytes()
    assert not active(path) and path.read_bytes() == before


def test_interrupted_fit_resumes_exact_state_and_keeps_attempt_history(qualification, tmp_path, monkeypatch):
    root, bundle, config = qualification
    import copuzzle_baselines.study.jobs as jobs
    original = jobs.train
    def interrupted(*args, **kwargs):
        kwargs["steps"] = 2
        original(*args, **kwargs)
        raise KeyboardInterrupt("controlled interruption")
    monkeypatch.setattr(jobs, "train", interrupted)
    first = run_job(root, "fit", "tiny-gcbc")
    assert first["status"] == "interrupted"
    first_bytes = (root / "jobs/fit/tiny-gcbc/attempts/0001/record.json").read_bytes()
    monkeypatch.setattr(jobs, "train", original)
    second = run_job(root, "fit", "tiny-gcbc")
    assert second["status"] == "completed" and second["execution_mode"] == "resume"
    assert second["checkpoint_evidence"]["step"] == 4
    assert (root / "jobs/fit/tiny-gcbc/attempts/0001/record.json").read_bytes() == first_bytes
    full = train(bundle / "learner", tmp_path / "uninterrupted", config=config, seed=0, steps=4)
    a, _, _ = load_checkpoint(root / second["checkpoint"])
    b, _, _ = load_checkpoint(tmp_path / "uninterrupted" / full["checkpoints"][-1]["file"])
    assert state_digest(a) == state_digest(b)
    assert snapshot(root, "fit")["primary_fits_completed"] == 0


def test_failed_fit_needs_explicit_retry_and_can_recover_final_checkpoint(qualification, monkeypatch):
    root, bundle, config = qualification
    import copuzzle_baselines.study.jobs as jobs
    original = jobs.train
    def after_checkpoint(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("receipt interrupted after final checkpoint")
    monkeypatch.setattr(jobs, "train", after_checkpoint)
    assert run_job(root, "fit", "tiny-gcbc")["status"] == "failed"
    with pytest.raises(ValueError, match="explicit"):
        run_job(root, "fit", "tiny-gcbc")
    monkeypatch.setattr(jobs, "train", lambda *args, **kwargs: pytest.fail("must recover, not refit"))
    result = run_job(root, "fit", "tiny-gcbc", retry_failed=True)
    assert result["status"] == "completed" and result["execution_mode"] == "recover_registered_final_checkpoint"


def test_startup_failure_is_recorded_without_run_manifest(qualification):
    root, _, _ = qualification
    result = record_startup_failure(root, "fit", "tiny-gcbc", RuntimeError("interpreter failed"))
    assert result["status"] == "failed" and result["failure_stage"] == "worker_startup"
    assert not list((root / "jobs").rglob("run.json"))
    assert snapshot(root, "fit")["counts"]["failed"] == 1


def test_completed_fit_is_not_rerun_and_corruption_is_not_hidden(qualification):
    root, _, _ = qualification
    result = run_job(root, "fit", "tiny-gcbc")
    assert run_job(root, "fit", "tiny-gcbc")["attempt"] == 1
    checkpoint = root / result["checkpoint"]
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt-container")
    with pytest.raises(ValueError, match="changed"):
        run_job(root, "fit", "tiny-gcbc")


def test_qualification_cannot_open_test_and_flags_alone_are_insufficient(qualification):
    root, _, _ = qualification
    run_job(root, "fit", "tiny-gcbc")
    with pytest.raises(ValueError, match="qualification"):
        seal(root)
    selection = root / "selection"
    selection.mkdir()
    write_new(selection / "test-lock.json", {"schema": "copuzzle.test_selection.v2", "selection_complete": True})
    with pytest.raises(ValueError, match="qualification"):
        verify_seal(selection / "test-lock.json")


def test_subprocess_controller_uses_snapshots_and_retains_all_goal_cases(qualification):
    root, _, _ = qualification
    fitted = run(root, "fit", workers=1)
    assert fitted["status"] == "completed"
    evaluated = run(root, "dev", workers=1)
    assert evaluated["status"] == "completed"
    record = verified_record(root, "dev", "tiny-gcbc-b0")
    report = json.loads((root / record["report"]).read_text())
    assert report["G"]["aggregate"]["cases"] == 3 and report["learner_state_unchanged"]
    assert validate_study(root, verify_data=True, verify_snapshot=True)["role"] == "qualification"


def test_stop_request_prevents_new_work(qualification):
    root, _, _ = qualification
    (root / "STOP").write_text("stop")
    with pytest.raises(ValueError, match="clear"):
        run(root, "fit", workers=1)
    assert snapshot(root, "fit")["counts"]["pending"] == 1


def test_simultaneous_worker_completion_does_not_abandon_pending_jobs(qualification, monkeypatch):
    root, _, _ = qualification
    study = json.loads((root / "study.json").read_text())
    template = study["fit_jobs"][0]
    study["fit_jobs"] = [{**deepcopy(template), "id": f"tiny-{i}", "fit_seed": i} for i in range(4)]
    (root / "study.json").write_text(json.dumps(study))
    (root / "registration.json").write_text(json.dumps({"schema": "copuzzle.study_registration.v1", "study_sha256": file_hash(root / "study.json")}))
    import copuzzle_baselines.study.runner as controller
    class ImmediateWorker:
        def __init__(self, command, **kwargs):
            job = command[command.index("--job") + 1]
            assert run_job(root, "fit", job)["status"] == "completed"
        def poll(self):return 0
        def wait(self):return 0
        def terminate(self):pass
    monkeypatch.setattr(controller.subprocess, "Popen", ImmediateWorker)
    result = run(root, "fit", workers=2)
    assert result["status"] == "completed"
    assert snapshot(root, "fit")["counts"]["completed"] == 4
