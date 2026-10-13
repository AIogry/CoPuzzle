"""Protocol identity, public information, migration and transfer declarations."""
from copy import deepcopy
import hashlib
import json
import shutil

import numpy as np
import pytest

from copuzzle.games.lights_out import evaluation
from copuzzle.games.lights_out.evaluation import (
    OraclePolicy, RandomPolicy, evaluate_policy, generate_task_suite,
    load_task_suite, save_task_suite, validate_task_suite, with_structure,
)
from copuzzle.games.lights_out.rules import get_rule, graph_rule
from copuzzle.games.lights_out.structure import export_structure, save_structure


def make_suite(rule=None, **kwargs):
    rule = get_rule("grid-2x3") if rule is None else rule
    return generate_task_suite(rule, per_depth=2, depths=(1, 2), seed=19,
                               max_episode_steps=5, **kwargs)


def test_default_v0_generation_and_report_are_unchanged():
    suite = make_suite()
    assert suite == make_suite(observation_profile="legacy_hidden")
    assert suite["schema_version"] == "copuzzle.goal_tasks.v0"
    assert "structure" not in suite and "observation_profile" not in suite
    report = evaluate_policy(RandomPolicy(0), suite)
    assert set(report) == {
        "schema_version", "rule_id", "task_suite_sha256", "policy",
        "privileged_rule_access", "reveal_graph", "max_episode_steps",
        "summary", "by_optimal_steps", "cases",
    }
    assert report["schema_version"] == "copuzzle.evaluation.v0"
    assert report["reveal_graph"] is False
    adjacency_suite = make_suite(observation_profile="legacy_adjacency")
    assert adjacency_suite["reveal_graph"] is True
    assert evaluate_policy(RandomPolicy(0), adjacency_suite)["reveal_graph"] is True


def test_migration_preserves_cases_distances_and_budget_without_mutation():
    rule = get_rule("grid-2x3")
    old = make_suite(rule)
    before = deepcopy(old)
    new = with_structure(old, rule=rule)
    assert old == before
    assert new["schema_version"] == "copuzzle.goal_tasks.v1"
    assert new["observation_profile"] == "structure_v1"
    assert "reveal_graph" not in new
    for key in old.keys() - {"schema_version", "reveal_graph"}:
        assert new[key] == old[key]
    assert validate_task_suite(deepcopy(new)).rule_id == rule.rule_id
    assert new == make_suite(rule, observation_profile="structure_v1")
    new["cases"][0]["state"][0] ^= 1
    assert old == before


def test_portable_custom_graph_roundtrip_without_registry(tmp_path, monkeypatch):
    rule = graph_rule(np.zeros((6, 6), dtype=np.uint8), name="grid-2x3")
    suite = make_suite(rule, observation_profile="structure_v1")
    original = tmp_path / "original"
    path = save_task_suite(suite, original / "tasks.json")
    assert suite["structure"]["path"] is None
    raw = json.loads(path.read_text())
    assert raw["structure"]["path"] == "structure.npz"
    assert "_structure" not in raw
    portable = tmp_path / "portable"
    shutil.copytree(original, portable)
    shutil.rmtree(original)

    def missing_registry(*args, **kwargs):
        raise AssertionError("A portable custom graph must not use the named-rule registry")

    monkeypatch.setattr(evaluation, "get_rule", missing_registry)
    loaded = load_task_suite(portable / "tasks.json")
    assert loaded == raw
    reconstructed = validate_task_suite(loaded)
    assert reconstructed.rule_id == rule.rule_id
    seen = []
    oracle = OraclePolicy(reconstructed)

    def policy(observation):
        seen.append(set(observation))
        assert observation["adjacency"].dtype == np.uint8
        assert not observation["adjacency"].any()
        return oracle(observation)

    report = evaluate_policy(policy, deepcopy(loaded))
    assert seen and all(fields == {"board", "goal", "previous_event", "adjacency"} for fields in seen)
    assert report["summary"]["success_rate"] == 1
    assert report["structure_sha256"] == export_structure(rule).structure_sha256
    assert report["structure_file_sha256"] == raw["structure"]["file_sha256"]
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    assert report["task_suite_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_grid_policy_gets_complete_public_structure_and_no_task_answers():
    rule = get_rule("grid-2x3")
    suite = make_suite(rule, observation_profile="structure_v1")
    oracle = OraclePolicy(rule)
    fields = {"board", "goal", "previous_event", "grid_shape", "grid_coordinates"}

    def policy(observation):
        assert set(observation) == fields
        np.testing.assert_array_equal(observation["grid_shape"], [2, 3])
        np.testing.assert_array_equal(observation["grid_coordinates"],
                                      [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
        assert observation["grid_shape"].dtype == np.int32
        action = oracle(observation)
        # Observation edits must not corrupt future structure observations.
        observation["grid_shape"][:] = 0
        observation["grid_coordinates"][:] = -1
        return action

    report = evaluate_policy(policy, suite)
    assert report["actual_observation_fields"] == sorted(fields)
    assert report["summary"]["success_rate"] == 1
    assert report["observation_profile"] == "structure_v1"
    assert report["structure_file_sha256"] is None
    assert report["method_metadata"]["uses_analytic_rules"] is None


def test_save_upgrades_v0_and_reuses_existing_structure(tmp_path):
    rule = get_rule("grid-2x3")
    old = make_suite(rule)
    structure = export_structure(rule)
    artifact = save_structure(structure, tmp_path / "structure.npz")
    before = artifact.read_bytes(), artifact.with_suffix(".json").read_bytes()
    first = save_task_suite(old, tmp_path / "tasks.json", structure=structure)
    second = save_task_suite(load_task_suite(first), tmp_path / "again.json")
    assert json.loads(first.read_text()) == json.loads(second.read_text())
    assert before == (artifact.read_bytes(), artifact.with_suffix(".json").read_bytes())
    assert old["schema_version"] == "copuzzle.goal_tasks.v0"
    with pytest.raises(FileExistsError):
        save_task_suite(old, first, structure=structure)


def test_save_rejects_mismatched_existing_structure(tmp_path):
    save_structure(export_structure(get_rule("graph-path9")), tmp_path / "structure.npz")
    suite = make_suite(observation_profile="structure_v1")
    with pytest.raises(ValueError, match="different semantic hash"):
        save_task_suite(suite, tmp_path / "tasks.json")
    assert not (tmp_path / "tasks.json").exists()


def test_nested_artifact_path_and_collision_checks(tmp_path):
    suite = make_suite(observation_profile="structure_v1")
    path = save_task_suite(suite, tmp_path / "tasks.json", structure_path="public/board.npz")
    assert load_task_suite(path)["structure"]["path"] == "public/board.npz"
    with pytest.raises(ValueError, match="within"):
        save_task_suite(suite, tmp_path / "escape.json", structure_path="../escape.npz")
    with pytest.raises(ValueError, match="conflicts"):
        save_task_suite(suite, tmp_path / "structure.json")


@pytest.mark.parametrize("which", ["npz", "manifest"])
def test_loaded_artifact_file_hashes_are_verified(tmp_path, which):
    suite = make_suite(observation_profile="structure_v1")
    path = save_task_suite(suite, tmp_path / "tasks.json")
    artifact = tmp_path / ("structure.npz" if which == "npz" else "structure.json")
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash|sha256"):
        load_task_suite(path)


@pytest.mark.parametrize("field,value", [
    ("rule_id", "0" * 64), ("n_buttons", 9), ("family", "graph"),
    ("task_name", "renamed"), ("observation_profile", "legacy_hidden"),
])
def test_suite_cannot_misdeclare_its_bound_structure(field, value):
    suite = make_suite(observation_profile="structure_v1")
    suite[field] = value
    with pytest.raises(ValueError):
        validate_task_suite(suite)


def test_raw_v1_mapping_requires_binding_and_rule_identity_is_checked():
    suite = make_suite(observation_profile="structure_v1")
    with pytest.raises(ValueError, match="load_task_suite or with_structure"):
        validate_task_suite(dict(suite))
    with pytest.raises(ValueError, match="Supplied rule differs"):
        validate_task_suite(suite, get_rule("grid-3x3"))


@pytest.mark.parametrize("field", ["duplicate", "nonfinite"])
def test_v1_json_rejects_ambiguous_or_nonfinite_content(tmp_path, field):
    path = save_task_suite(make_suite(observation_profile="structure_v1"), tmp_path / "tasks.json")
    content = path.read_text()
    extra = '"seed": 999,' if field == "duplicate" else '"extra": NaN,'
    path.write_text("{" + extra + content[1:])
    with pytest.raises(ValueError, match="Duplicate|Nonfinite"):
        load_task_suite(path)


def test_v1_load_and_save_reject_escaping_sidecar_symlink(tmp_path):
    destination = tmp_path / "inside"
    suite = make_suite(observation_profile="structure_v1")
    path = save_task_suite(suite, destination / "tasks.json")
    sidecar = destination / "structure.json"
    outside = tmp_path / "outside.json"
    sidecar.rename(outside)
    sidecar.symlink_to(outside)
    with pytest.raises(ValueError, match="sidecar escapes"):
        load_task_suite(path)
    with pytest.raises(ValueError, match="sidecar escapes"):
        save_task_suite(suite, destination / "again.json")


def test_transfer_reports_declarations_without_claiming_parameter_enforcement():
    rule = get_rule("grid-2x3")
    suite = make_suite(rule, observation_profile="structure_v1")
    training = [get_rule("grid-3x3").rule_id]
    metadata = {"method_name": "a supplied model", "uses_solver": False}
    report = evaluate_policy(RandomPolicy(2), suite, evaluation_mode="frozen_transfer",
                             training_rule_ids=training, method_metadata=metadata)
    assert report["evaluation_mode"] == "frozen_transfer"
    assert report["training_rule_ids"] == training
    assert report["rule_id"] not in report["training_rule_ids"]
    assert report["parameter_freezing_enforced"] is False
    assert "not independently verified" in report["method_information_source"]
    assert report["method_metadata"]["uses_solver"] is False
    assert report["method_metadata"]["uses_environment_queries"] is False
    assert metadata == {"method_name": "a supplied model", "uses_solver": False}
    training.clear()
    assert len(report["training_rule_ids"]) == 1


@pytest.mark.parametrize("kwargs", [
    {"evaluation_mode": "online"}, {"training_rule_ids": "one rule"},
    {"training_rule_ids": [""]}, {"method_metadata": {"summary": "replace"}},
    {"method_metadata": {"structure_sha256": "replace"}},
    {"method_metadata": {"uses_solver": "no"}},
    {"method_metadata": {"something": float("nan")}},
])
def test_invalid_declarations_fail_before_policy_calls(kwargs):
    def must_not_run(observation):
        pytest.fail("Invalid declarations must fail before calling policy")

    with pytest.raises(ValueError):
        evaluate_policy(must_not_run, make_suite(), **kwargs)


def test_oracle_method_access_cannot_be_denied():
    rule = get_rule("grid-2x3")
    with pytest.raises(ValueError, match="Cannot deny"):
        evaluate_policy(OraclePolicy(rule), make_suite(rule), method_metadata={"uses_solver": False})
    report = evaluate_policy(OraclePolicy(rule), make_suite(rule, observation_profile="structure_v1"))
    assert report["method_metadata"] == {
        "uses_analytic_rules": True, "uses_solver": True, "uses_environment_queries": False,
    }


def test_legacy_suite_can_explicitly_record_new_declarations():
    suite = make_suite(observation_profile="legacy_adjacency")
    report = evaluate_policy(RandomPolicy(0), suite, method_metadata={"label": "declared"})
    assert report["schema_version"] == "copuzzle.evaluation.v1"
    assert report["observation_profile"] == "legacy_adjacency"
    assert report["structure_sha256"] is None
    assert "adjacency" in report["actual_observation_fields"]


def test_unsupported_factory_profile_rejected():
    with pytest.raises(ValueError, match="observation_profile"):
        make_suite(observation_profile="everything")
