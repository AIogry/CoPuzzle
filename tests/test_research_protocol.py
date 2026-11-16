"""Behavioral protocol checks, including deliberately leaky/tampered artifacts."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from copuzzle import get_rule
from copuzzle.cli import main
from copuzzle.research import INSTANCES, REGIMES, load_probe, load_training
from copuzzle.research.bundle import (_effect, _read_npz, materialize,
                                      validate_bundle, write_bundle)
from copuzzle.research.evaluation import (ModelBudget, QueryLimitExceeded,
                                          empirical_predictions, evaluate_policy,
                                          reference_report, score_predictions)
from copuzzle.research.graph import checked_diameter, exact_distances, graph, partition


@pytest.mark.parametrize("spec", INSTANCES, ids=lambda s: s.task)
def test_scope_exact_component_and_arbitrary_goal_diameter(spec):
    g = graph(spec.task)
    assert len(g.states) == spec.component_states
    assert checked_diameter(spec.task) == spec.diameter
    assert spec.easy[0] == 1
    assert spec.easy[1] + 1 == spec.medium[0]
    assert spec.medium[1] + 1 == spec.hard[0]
    assert spec.hard[1] == spec.diameter
    rule = get_rule(spec.task)
    for source in np.random.default_rng(4).choice(len(g.states), 24, replace=False):
        for action, successor in enumerate(g.successors[source]):
            if successor < 0:
                with pytest.raises(ValueError):
                    rule.transition(g.states[source], action)
            else:
                np.testing.assert_array_equal(rule.transition(g.states[source], action), g.states[successor])
    with pytest.raises(ValueError):
        g.states[0, 0] = 99


@pytest.mark.parametrize("spec", INSTANCES, ids=lambda s: s.task)
def test_reverse_and_parallel_edges_never_cross_splits(spec):
    p = partition(spec.task, "transition_holdout", 0)
    for source, row in enumerate(p.graph.successors):
        for action, successor in enumerate(row):
            if successor >= 0:
                reverse = p.graph.successors[successor] == source
                assert np.any(reverse)
                assert np.all(p.owners[successor, reverse] == p.owners[source, action])


@pytest.mark.parametrize("spec", INSTANCES, ids=lambda s: s.task)
def test_state_holdout_checks_both_ends_and_discards_cross_edges(spec):
    p = partition(spec.task, "state_holdout", 0)
    g = p.graph
    assert np.any(p.owners == -1)
    observed = []
    for split in range(3):
        queries = p.queries(split)
        states = set()
        for source, action in queries:
            successor, _ = p.successor(source, action)
            assert p.state_split[source] == p.state_split[successor] == split
            states.update((int(source), successor))
        observed.append(states)
    assert not observed[0] & (observed[1] | observed[2])
    assert not observed[1] & observed[2]
    for source, row in enumerate(g.successors):
        for action, successor in enumerate(row):
            if successor >= 0 and p.state_split[source] != p.state_split[successor]:
                assert p.owners[source, action] == -1


@pytest.mark.parametrize("task", [s.task for s in INSTANCES])
@pytest.mark.parametrize("regime", REGIMES)
def test_probe_difficulty_support_and_information_contract(task, regime):
    public, truth, meta, audit = materialize(task, regime, per_cell=2)
    p = partition(task, regime, 0)
    train = public["train"]
    observed = {tuple(board) for episode in train["observations"] for board in episode}
    empirical = {(tuple(boards[0]), int(action[0])) for boards, action in zip(train["observations"], train["actions"])}
    pairs = set()
    assert audit["composition_primitive_effect_overlap"] == 0
    for split in ("dev", "test"):
        assert set(public[f"R_{split}"]) == {"board", "action"}
        assert set(public[f"G_{split}"]) == {"board", "goal"}
        assert set(public[f"C_{split}"]) == {"board", "actions", "lengths"}
        for board, action in zip(public[f"R_{split}"]["board"], public[f"R_{split}"]["action"]):
            assert (tuple(board), int(action)) not in empirical
        for start, goal, d in zip(public[f"G_{split}"]["board"], public[f"G_{split}"]["goal"], truth[f"G_{split}_distance"]):
            pair = tuple(sorted((tuple(start), tuple(goal))))
            assert pair not in pairs
            pairs.add(pair)
            assert exact_distances(task, p.graph.locate(goal))[p.graph.locate(start)] == d
            if regime == "state_holdout":
                assert tuple(start) not in observed and tuple(goal) not in observed
        for i, (start, word, length) in enumerate(zip(*(public[f"C_{split}"][key] for key in ("board", "actions", "lengths")))):
            source, missing = p.graph.locate(start), 0
            for j, action in enumerate(word[:length]):
                missing += (tuple(p.graph.states[source]), int(action)) not in empirical
                source, legal = p.successor(source, action)
                assert legal
                np.testing.assert_array_equal(p.graph.states[source], truth[f"C_{split}_prefix_boards"][i, j])
            assert exact_distances(task, source)[p.graph.locate(start)] == length
            assert truth[f"C_{split}_unseen_transitions"][i] == missing
            if regime == "transition_holdout":
                assert missing == 0
            else:
                assert missing >= 2
                assert tuple(start) not in observed and tuple(p.graph.states[source]) not in observed
        assert meta["control_budget"] > max(truth[f"G_{split}_distance"])


@pytest.fixture
def bundle(tmp_path):
    return write_bundle(tmp_path / "bundle", "sliding-2x3", "transition_holdout", per_cell=2)


def _resign(path, name):
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["files"][name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_validator_recomputes_semantics_even_after_hash_is_updated(bundle):
    assert validate_bundle(bundle)["valid"]
    name = "learner/train.npz"
    arrays = _read_npz(bundle / name)
    arrays["observations"] = arrays["observations"].copy()
    arrays["observations"][0, 1] = 9
    np.savez_compressed(bundle / name, **arrays)
    _resign(bundle, name)
    with pytest.raises(ValueError, match="semantic or split"):
        validate_bundle(bundle)


def test_no_extra_oracle_fields_and_no_overwrite(bundle):
    with pytest.raises(FileExistsError):
        write_bundle(bundle, "sliding-2x3", "state_holdout")
    name = "learner/R_dev.npz"
    arrays = _read_npz(bundle / name)
    arrays["true_action_mask"] = np.ones((len(arrays["board"]), 4), dtype=np.uint8)
    np.savez_compressed(bundle / name, **arrays)
    _resign(bundle, name)
    with pytest.raises(ValueError, match="unexpected learner fields"):
        load_probe(bundle / "learner", "R")
    with pytest.raises(ValueError, match="semantic or split"):
        validate_bundle(bundle)


def test_learner_loading_without_evaluator_files_or_oracles(bundle, monkeypatch):
    import copuzzle.research.graph as graph_module
    # Import syntax may resolve to module; no learner API needs its functions.
    def forbidden(*args, **kwargs):
        raise AssertionError("learner attempted privileged computation")
    monkeypatch.setattr(graph_module, "graph", forbidden)
    for path in (bundle / "evaluator").iterdir():
        path.unlink()
    (bundle / "manifest.json").unlink()
    train = load_training(bundle / "learner")
    assert train["observations"].shape[1] == 2
    for probe in ("R", "C", "G"):
        assert len(load_probe(bundle / "learner", probe)["board"]) > 0


def test_balanced_rule_metrics_and_whole_prefix_not_only_final(bundle):
    truth = _read_npz(bundle / "evaluator/answers.npz")
    pred = {key: truth[f"R_dev_{key}"] for key in ("next_board", "legal")}
    score = score_predictions(bundle, "R", pred)
    assert score["balanced_joint_exact"] == score["joint_exact"] == 1
    assert score["by_legality"]["illegal"]["cases"] > 0
    prefix = truth["C_dev_prefix_boards"].copy()
    perfect = score_predictions(bundle, "C", {"prefix_boards": prefix})
    assert perfect["whole_prefix_exact"] == 1
    prefix[:, 0] = 99
    damaged = score_predictions(bundle, "C", {"prefix_boards": prefix})
    assert damaged["whole_prefix_exact"] == 0
    assert damaged["final_board_exact"] == 1
    with pytest.raises(ValueError, match="integer array"):
        score_predictions(bundle, "R", {**pred, "next_board": pred["next_board"].astype(float)})


def test_training_lookup_is_positive_control_for_supported_composition(bundle):
    result = empirical_predictions(bundle / "learner", "C")
    assert score_predictions(bundle, "C", result)["whole_prefix_exact"] == 1
    result = empirical_predictions(bundle / "learner", "R")
    assert score_predictions(bundle, "R", result)["by_legality"]["legal"]["joint_exact"] == 0


def test_sparse_heldout_legal_population_does_not_reject_dataset_seed():
    public, truth, _, audit = materialize("sliding-2x3", "state_holdout", seed=1, per_cell=8)
    for split, legal_count in (("dev", 6), ("test", 4)):
        assert int(truth[f"R_{split}_legal"].sum()) == legal_count
        assert len(public[f"R_{split}"]["board"]) == audit["query_counts"][split]
        assert len(public[f"G_{split}"]["board"]) == 24


def test_effect_audit_detects_aliases_and_cancellation():
    p = partition("grid-3x3", "transition_holdout", 0)
    assert _effect(p, [0, 1]) == _effect(p, [1, 0])
    assert _effect(p, [0, 0]) == _effect(p, [])
    assert _effect(p, [0, 1]) != _effect(p, [0])


def test_control_counts_illegal_attempts_and_failed_cases(bundle):
    p = partition("sliding-2x3", "transition_holdout", 0)
    received = []
    def factory():
        def invalid_policy(observation, budget):
            received.append(set(observation))
            source = p.graph.locate(observation["board"])
            return next(a for a in range(4) if not p.successor(source, a)[1])
        return invalid_policy
    report = evaluate_policy(bundle, factory)
    assert report["aggregate"]["success_rate"] == 0
    assert report["aggregate"]["illegal_attempt_rate"] == 1
    assert report["aggregate"]["mean_penalized_cost_all"] == 42
    assert report["aggregate"]["mean_success_excess"] is None
    assert all(keys == {"board", "goal", "steps_remaining", "previous_legal"} for keys in received)
    assert all(row["attempts"] == 42 for row in report["cases"])
    with pytest.raises(ValueError, match="out-of-range"):
        evaluate_policy(bundle, lambda: lambda obs, budget: 4)


def test_model_queries_are_counted_before_the_predictor_runs():
    calls = []
    budget = ModelBudget(1)
    def model(board, action):
        calls.append(action)
        return board
    budget.predict(model, [0, 1], 0)
    with pytest.raises(QueryLimitExceeded):
        budget.predict(model, [0, 1], 1)
    assert calls == [0] and budget.used == 1
    with pytest.raises(QueryLimitExceeded):
        ModelBudget(0).predict(model, [0], 0)


def test_reference_oracle_is_exact_and_labeled_privileged(bundle):
    report = reference_report(bundle)
    score = report["privileged_oracle_G"]["aggregate"]
    assert score["success_rate"] == 1
    assert score["mean_success_excess"] == score["illegal_attempt_rate"] == 0
    assert "not learned" in report["purpose"]


@pytest.mark.parametrize("value", [-1, True, 0.5])
def test_bad_seed_rejected_before_generation(value):
    with pytest.raises(ValueError):
        materialize("grid-3x3", "transition_holdout", seed=value, per_cell=1)


def test_cli_and_deterministic_files(tmp_path, capsys):
    assert main(["research", "spec"]) == 0
    assert "copuzzle.rule-composition.rc1" in capsys.readouterr().out
    first = tmp_path / "a"
    second = tmp_path / "b"
    args = ["research", "build", "--task", "grid-3x3", "--regime", "state_holdout", "--per-cell", "1"]
    assert main([*args, "--output", str(first)]) == 0
    assert main([*args, "--output", str(second)]) == 0
    for path in first.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()
    assert main(["research", "validate", str(first)]) == 0
    report = tmp_path / "references.json"
    assert main(["research", "references", str(first), "--output", str(report)]) == 0
    assert json.loads(report.read_text())["privileged_oracle_G"]["aggregate"]["success_rate"] == 1
