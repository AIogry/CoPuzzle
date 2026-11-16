"""V1 population, leakage, exact-distance and failure-accounting checks."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from copuzzle import get_rule
from copuzzle.benchmark.data import load_probe, load_training
from copuzzle.benchmark.evaluation import ModelBudget, evaluate_policy, score_predictions
from copuzzle.benchmark.generation import materialize, validate_bundle, write_bundle
from copuzzle.benchmark.graph import distances, full_distances, partition
from copuzzle.benchmark.population import Selection, _salt, population, priority
from copuzzle.research.bundle import _read_npz
from copuzzle.research.evaluation import QueryLimitExceeded
from copuzzle.research.graph import distances as slow_distances, graph
from copuzzle.research.spec import INSTANCES


@pytest.mark.parametrize("item", INSTANCES, ids=lambda item: item.task)
def test_vectorized_distances_match_native_oracles_and_restricted_bfs(item):
    rule, g = get_rule(item.task), graph(item.task)
    if item.family in ("pancake", "topspin"):
        from copuzzle.games.permutation.oracle import ExactOracle
        oracle = ExactOracle(rule)
    elif item.family in ("sliding", "hanoi"):
        from copuzzle.games.classic.oracle import ExactOracle
        oracle = ExactOracle(rule)
    else:
        from copuzzle.games.lights_out.oracle import solve
        oracle = None
    rng = np.random.default_rng(1)
    p = partition(item.task, "transition_holdout", 101)
    for goal in rng.choice(len(g.states), 4, replace=False):
        expected = slow_distances(g, int(goal))
        np.testing.assert_array_equal(full_distances(item.task, int(goal)), expected)
        np.testing.assert_array_equal(distances(g, int(goal), p.owners == 0), slow_distances(g, int(goal), p.owners == 0))
        bounded = distances(g, int(goal), p.owners == 0, max_depth=4)
        full_train = slow_distances(g, int(goal), p.owners == 0)
        np.testing.assert_array_equal(bounded, np.where(full_train > 4, -1, full_train))
        for source in rng.choice(len(g.states), 3, replace=False):
            result = len(solve(rule, g.states[source], g.states[goal])) if oracle is None else oracle.distance(g.states[source], g.states[goal])
            assert int(expected[source]) == result


def test_hash_selection_is_unique_and_independent_of_chunking():
    codes = np.arange(1000, dtype=np.uint64)
    a, b = Selection(32, 12), Selection(32, 12)
    a.add(codes)
    for chunk in np.array_split(codes[::-1], 19):
        b.add(chunk)
    assert a.count == b.count == 1000
    np.testing.assert_array_equal(a.codes, b.codes)
    assert len(set(priority(codes, 12).tolist())) == 1000


def test_small_populations_are_exact_and_small_cells_are_exhausted():
    task, regime, seed = "hanoi-5", "transition_holdout", 101
    p = partition(task, regime, seed)
    audit = population(task, regime, seed)
    actual = {key: set() for key in audit["cells"]}
    for goal in range(len(p.graph.states)):
        full = slow_distances(p.graph, goal)
        train = slow_distances(p.graph, goal, p.owners == 0)
        for start in range(goal + 1, len(p.graph.states)):
            code = goal * len(p.graph.states) + start
            label = "test" if int(priority([code], _salt(task, regime, seed, "shared_C_G_split"))[0] & np.uint64(1)) else "dev"
            d = int(full[start])
            tier = 0 if d <= 10 else 1 if d <= 20 else 2
            actual[f"G_{label}_{tier}"].add(code)
            if d in (2, 4, 8) and int(train[start]) == d:
                actual[f"C_{label}_{d}"].add(code)
    assert min(len(values) for key, values in actual.items() if key.startswith("C_")) == 6
    for key, values in actual.items():
        cell = audit["cells"][key]
        assert cell["eligible_pairs"] == len(values)
        assert cell["selected_cases"] == min(32, len(values))
        assert set(cell["pair_codes"]) <= values
        if len(values) <= 32:
            assert set(cell["pair_codes"]) == values
    audit["cells"]["C_dev_2"]["eligible_pairs"] = -1
    assert population(task, regime, seed)["cells"]["C_dev_2"]["eligible_pairs"] >= 0


def test_large_reference_bank_has_an_exact_declared_pair_universe():
    audit = population("topspin-8-4", "transition_holdout", 101, max_per_cell=4, bank_limit=8, exhaustive_limit=512)
    bank = set(audit["banks"]["0"]["state_indices"])
    n = 40320
    assert len(bank) == 8
    expected = 8 * (n - 8) + 8 * 7 // 2
    assert sum(cell["eligible_pairs"] for key, cell in audit["cells"].items() if key.startswith("G_")) == expected
    for cell in audit["cells"].values():
        for code in cell["pair_codes"]:
            a, b = divmod(code, n)
            assert a < b and (a in bank or b in bank)


@pytest.mark.parametrize("regime", ["transition_holdout", "state_holdout"])
def test_no_cross_probe_dev_test_pair_leakage_and_exact_composition(regime):
    public, truth, meta, audit = materialize("sliding-2x3", regime, 101, max_per_cell=8)
    p = partition("sliding-2x3", regime, 101)
    train = {tuple(s) for episode in public["train"]["observations"] for s in episode}
    dev = set(truth["C_dev_pair_codes"].tolist()) | set(truth["G_dev_pair_codes"].tolist())
    test = set(truth["C_test_pair_codes"].tolist()) | set(truth["G_test_pair_codes"].tolist())
    assert not dev & test
    for label in ("dev", "test"):
        c = public[f"C_{label}"]
        for i, (start, word, length) in enumerate(zip(c["board"], c["actions"], c["lengths"])):
            source = p.graph.locate(start)
            missing = 0
            for step, action in enumerate(word[:length]):
                missing += p.owners[source, p.action_map[action]] != 0
                source, legal = p.successor(source, action)
                assert legal
                np.testing.assert_array_equal(p.graph.states[source], truth[f"C_{label}_prefix_boards"][i, step])
            assert full_distances("sliding-2x3", source)[p.graph.locate(start)] == length
            assert missing == truth[f"C_{label}_unseen_transitions"][i]
            if regime == "transition_holdout":
                assert missing == 0
            else:
                assert missing >= 2 and tuple(start) not in train and tuple(p.graph.states[source]) not in train
    assert audit["composition_primitive_effect_overlap"] == 0


@pytest.fixture
def bundle(tmp_path):
    return write_bundle(tmp_path / "bundle", "grid-3x3", "transition_holdout", 0,
                        max_per_cell=32, bank_limit=1, exhaustive_limit=1)


def test_macro_metrics_do_not_overweight_large_cells(bundle):
    truth = _read_npz(bundle / "evaluator/answers.npz")
    c = load_probe(bundle / "learner", "C")
    counts = {int(k): int(np.sum(c["lengths"] == k)) for k in np.unique(c["lengths"])}
    assert len(set(counts.values())) > 1
    wrong_length = min(counts, key=counts.get)
    predictions = truth["C_dev_prefix_boards"].copy()
    predictions[c["lengths"] == wrong_length] = -99
    report = score_predictions(bundle, "C", {"prefix_boards": predictions})
    assert report["length_macro_whole_prefix_exact"] == pytest.approx(2 / 3)
    assert report["whole_prefix_exact"] == pytest.approx(1 - counts[wrong_length] / len(c["lengths"]))
    p = partition("grid-3x3", "transition_holdout", 0)
    tiers = truth["G_dev_tier"]
    counter = iter(range(len(tiers)))
    def factory():
        index = next(counter)
        def policy(obs, budget):
            if tiers[index] != 0:
                raise RuntimeError("controlled failure")
            source, goal = p.graph.locate(obs["board"]), p.graph.locate(obs["goal"])
            d = full_distances("grid-3x3", goal)
            return next(a for a in range(p.graph.n_actions) if d[p.successor(source, a)[0]] == d[source] - 1)
        return policy
    g = evaluate_policy(bundle, factory)
    assert g["tier_macro_success_rate"] == pytest.approx(1 / 3)
    assert g["aggregate"]["success_rate"] == pytest.approx(float(np.mean(tiers == 0)))
    assert len(g["cases"]) == len(tiers)


def test_failures_charge_full_budget_and_cancellation_propagates(bundle):
    def broken_factory():
        raise RuntimeError("startup failed")
    report = evaluate_policy(bundle, broken_factory)
    assert report["aggregate"]["policy_error_rate"] == 1
    assert report["aggregate"]["mean_penalized_cost_all"] == 18
    assert all(row["attempts"] == 0 and row["error"]["type"] == "RuntimeError" for row in report["cases"])
    invalid = evaluate_policy(bundle, lambda: lambda observation, budget: 100)
    assert invalid["aggregate"]["policy_error_rate"] == 1
    def interrupted():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        evaluate_policy(bundle, interrupted)


def test_batch_budget_counts_candidates_and_charges_before_failure():
    budget = ModelBudget(4)
    calls = []
    def model(boards, actions):
        calls.append(len(boards))
        return boards, np.ones(len(boards), bool)
    budget.predict_batch(model, np.zeros((3, 2), np.int32), np.zeros(3, np.int32))
    assert budget.used == 3 and budget.calls == 1 and calls == [3]
    with pytest.raises(QueryLimitExceeded):
        budget.predict_batch(model, np.zeros((2, 2), np.int32), np.zeros(2, np.int32))
    assert calls == [3] and budget.used == 3
    def fail(board, action):
        raise RuntimeError("model failed")
    with pytest.raises(RuntimeError):
        budget.predict(fail, [0, 1], 0)
    assert budget.used == 4 and budget.calls == 2


def test_strict_validation_recomputes_semantics_after_resigning(bundle):
    assert validate_bundle(bundle)["valid"]
    path = bundle / "learner/train.npz"
    arrays = _read_npz(path)
    arrays["observations"] = arrays["observations"].copy()
    arrays["observations"][0, 1] = -9
    np.savez_compressed(path, **arrays)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["files"]["learner/train.npz"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="semantic"):
        validate_bundle(bundle)


def test_learner_loads_without_evaluator_or_true_graph(bundle, monkeypatch):
    import copuzzle.benchmark.graph as module
    def forbidden(*args, **kwargs):
        raise AssertionError("learner requested a true graph")
    monkeypatch.setattr(module, "partition", forbidden)
    for path in (bundle / "evaluator").iterdir():
        path.unlink()
    (bundle / "manifest.json").unlink()
    assert len(load_training(bundle / "learner")["actions"]) > 0
    assert load_probe(bundle / "learner", "G")["board"].ndim == 2


def test_benchmark_role_requires_prespecified_settings(tmp_path):
    with pytest.raises(ValueError, match="prespecified"):
        write_bundle(tmp_path / "wrong", "grid-3x3", "transition_holdout", 0, purpose="benchmark")
    with pytest.raises(ValueError, match="prespecified"):
        write_bundle(tmp_path / "small", "grid-3x3", "transition_holdout", 101, purpose="benchmark", max_per_cell=1)
    assert not (tmp_path / "wrong").exists() and not (tmp_path / "small").exists()


def test_integer_validation_is_not_bypassed_by_caches():
    population("hanoi-5", "state_holdout", 1, max_per_cell=1)
    with pytest.raises(ValueError):
        population("hanoi-5", "state_holdout", True, max_per_cell=1)
    with pytest.raises(ValueError):
        full_distances("hanoi-5", -1)
