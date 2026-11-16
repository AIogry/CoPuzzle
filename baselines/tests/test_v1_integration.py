import hashlib
import json

import numpy as np
import pytest

from copuzzle.benchmark.evaluation import ModelBudget
from copuzzle.benchmark.generation import write_bundle
from copuzzle_baselines.config import Config, fingerprint
from copuzzle_baselines.controls.evaluate import evaluate_control
from copuzzle_baselines.controls.models import fit_control
from copuzzle_baselines.data import load_training
from copuzzle_baselines.planning import ModelPlanner
from copuzzle_baselines.runtime import evaluate, train
from copuzzle_baselines.study.access import authorize_test, validate_fit_context
from copuzzle_baselines.study.diagnostics import composition_diagnostics


@pytest.fixture
def v1_bundle(tmp_path):
    return write_bundle(tmp_path / "v1", "grid-3x3", "transition_holdout", 0, max_per_cell=2)


def test_batch_expansion_charges_all_candidates_even_on_early_goal():
    single_calls, batch_calls = [], []
    def single(board, action):
        single_calls.append(action)
        return board, False
    def batch(boards, actions):
        batch_calls.append(len(boards))
        return boards + 1, np.ones(len(boards), np.uint8)
    planner = ModelPlanner(single, 3, batch_predictor=batch)
    budget = ModelBudget(4)
    action = planner({"board": np.array([0], np.int32), "goal": np.array([1], np.int32), "steps_remaining": 2}, budget)
    assert action == 0 and batch_calls == [3] and not single_calls
    assert budget.used == 3 and budget.calls == 1


@pytest.mark.parametrize("method", ["gcbc", "gciql", "dynamics"])
def test_v1_train_only_adapter_and_frozen_evaluation(v1_bundle, tmp_path, method):
    data = load_training(v1_bundle / "learner")
    assert data.identity["protocol"] == "copuzzle.rule-composition.v1"
    cfg = Config(method=method, hidden_dims=(4,), batch_size=4)
    run = train(v1_bundle / "learner", tmp_path / method, config=cfg, seed=0, steps=2)
    checkpoint = tmp_path / method / run["checkpoints"][-1]["file"]
    report = evaluate(checkpoint, v1_bundle, model_query_limit=8 if method == "dynamics" else 0)
    assert report["learner_state_unchanged"] and report["G"]["aggregate"]["policy_error_rate"] == 0
    assert "tier_macro_success_rate" in report["G"]
    assert run["schedule"] == {"checkpoint_every": 1000, "log_every": 100}
    assert run["execution"]["backend"] == "cpu"
    if method == "dynamics":
        assert report["G"]["aggregate"]["model_queries"] > report["G"]["aggregate"]["model_calls"]
        assert report["C_context_diagnostic"]["cases_with_all_primitive_queries_observed"] == 6


@pytest.mark.parametrize("task,kind", [("grid-3x3", "fitted_xor"), ("pancake-6", "fitted_position")])
def test_fitted_structural_control_matches_exact_dynamics_with_same_planner(tmp_path, task, kind):
    bundle = write_bundle(tmp_path / "data", task, "state_holdout", 0, max_per_cell=1)
    fitted = evaluate_control(bundle, kind, model_query_limit=16)
    exact = evaluate_control(bundle, "oracle_dynamics", model_query_limit=16)
    assert fitted["R"]["balanced_joint_exact"] == fitted["C"]["length_macro_whole_prefix_exact"] == 1
    for a, b in zip(fitted["G"]["cases"], exact["G"]["cases"]):
        for key in ("success", "attempts", "illegal_attempts", "model_queries", "model_calls"):
            assert a[key] == b[key]
    assert not fitted["privileged"] and exact["privileged"]


def test_lookup_and_coverage_diagnostics_use_actual_selected_primitives(v1_bundle):
    model, data = fit_control(v1_bundle / "learner", "lookup")
    full = composition_diagnostics(v1_bundle, model.predict, split="dev", coverage=1.0, coverage_seed=0)
    assert full["teacher_forced_state_exact"] == 1 and full["cases_with_all_primitive_queries_observed"] == 6
    partial, _ = fit_control(v1_bundle / "learner", "lookup", coverage=0.25)
    quarter = composition_diagnostics(v1_bundle, partial.predict, split="dev", coverage=0.25, coverage_seed=0)
    assert quarter["by_training_support"]["unobserved"]["steps"] > 0
    assert quarter["cases_with_all_primitive_queries_observed"] < 6


def test_data_only_controls_do_not_query_rules(v1_bundle, monkeypatch):
    import copuzzle.benchmark.graph as module
    def forbidden(*args, **kwargs):
        raise AssertionError("fitting accessed true dynamics")
    monkeypatch.setattr(module, "partition", forbidden)
    for kind in ("lookup", "fitted_xor"):
        model, data = fit_control(v1_bundle / "learner", kind)
        raw = data.vocabulary.decode(data.states[:2])
        assert model.predict(raw, data.actions[:2])[0].shape == raw.shape


def test_test_access_binds_checkpoint_data_config_source_and_step(tmp_path):
    # Synthetic bytes/metadata only; no model is evaluated on final test data.
    checkpoint, bundle = tmp_path / "checkpoint.zip", tmp_path / "bundle"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    header = {"config": {"method": "dummy"}, "sources": {"file": "hash"}, "data": {"dataset": "fake"}, "fit_seed": 0, "step": 10}
    with pytest.raises(ValueError, match="sealed"):
        authorize_test(None, checkpoint, bundle, header)
    row = {"job_id": "synthetic-job", "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
           "bundle_manifest_sha256": hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
           "config_sha256": fingerprint(header["config"]), "sources_sha256": fingerprint(header["sources"]),
           "data_sha256": fingerprint(header["data"]), "fit_seed": 0, "step": 10}
    lock = tmp_path / "selection.json"
    lock.write_text(json.dumps({"schema": "copuzzle.test_selection.v1", "selection_complete": True, "checkpoints": [row]}))
    assert authorize_test(lock, checkpoint, bundle, header)["job_id"] == "synthetic-job"
    with pytest.raises(ValueError, match="binding"):
        authorize_test(lock, checkpoint, bundle, {**header, "step": 11})


def test_formal_context_rejects_development_data_and_changed_bindings():
    config, data, source, versions = {}, {"purpose": "benchmark"}, {}, {}
    context = {"schema": "copuzzle.formal_fit_context.v1", "study_sha256": "1" * 64, "job_id": "job",
               "config_sha256": fingerprint(config), "data_sha256": fingerprint(data), "sources_sha256": fingerprint(source),
               "versions_sha256": fingerprint(versions), "fit_seed": 0, "steps": 10000, "backend": "cpu"}
    validate_fit_context(context, config, data, source, versions, 0, 10000, "cpu")
    with pytest.raises(ValueError, match="benchmark"):
        validate_fit_context(context, config, {"purpose": "development"}, source, versions, 0, 10000, "cpu")
    with pytest.raises(ValueError, match="binding"):
        validate_fit_context(context, config, data, source, versions, 1, 10000, "cpu")
