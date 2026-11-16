from copy import deepcopy

import numpy as np
import pytest

from copuzzle_baselines.study.analysis import audit_goals, audit_prediction_scores
from copuzzle_baselines.study.statistics import bootstrap_draws, estimate, scopes, seed_matrix, summarize


def test_paired_bootstrap_preserves_constant_differences():
    values = np.arange(54).reshape(6, 3, 3)/100
    weights = np.asarray([.1, .1, .2, .2, .2, .2])
    draws = bootstrap_draws(3, 3, repetitions=1000, seed=42)
    result = estimate((values+.125)-values, weights, draws)
    assert result["mean"] == pytest.approx(.125)
    assert result["interval"] == pytest.approx([.125, .125])
    assert np.array_equal(draws[0], bootstrap_draws(3, 3, repetitions=1000, seed=42)[0])


def test_family_macro_does_not_double_weight_lights_out():
    families = {"lo3": "lights_out", "lo4": "lights_out", "p": "pancake", "t": "topspin", "s": "sliding", "h": "hanoi"}
    tasks = list(families)
    name, members, weights = scopes(tasks, families, tasks)[-1]
    assert name == "core_family_macro" and members == tasks
    assert weights == pytest.approx([.1, .1, .2, .2, .2, .2])
    values = np.zeros((6, 3, 3)); values[:2] = 1
    result = estimate(values, weights, bootstrap_draws(3, 3, repetitions=10))
    assert result["mean"] == pytest.approx(.2)
    assert all(scope[0] != "core_family_macro" for scope in scopes(tasks[:2], families, tasks))


def test_seed_matrix_keeps_failed_cells_and_broadcasts_deterministic_controls():
    rows = [{"task": "x", "data_seed": seed, "fit_seed": fit, "metrics": {"G": float(fit != 1)}}
            for seed in (101, 202, 303) for fit in (0, 1, 2)]
    matrix = seed_matrix(rows, ["x"], [101, 202, 303], [0, 1, 2], "G")
    assert matrix.mean() == pytest.approx(2/3)
    with pytest.raises(ValueError, match="omits"):
        seed_matrix(rows[:-1], ["x"], [101, 202, 303], [0, 1, 2], "G")
    with pytest.raises(ValueError, match="duplicate"):
        seed_matrix(rows + rows[:1], ["x"], [101, 202, 303], [0, 1, 2], "G")
    controls = [{"task": "x", "data_seed": s, "fit_seed": None, "metrics": {"G": i/2}}
                for i, s in enumerate((101, 202, 303))]
    control = seed_matrix(controls, ["x"], [101, 202, 303], [0, 1, 2], "G")
    assert np.array_equal(control[0], [[0, 0, 0], [.5, .5, .5], [1, 1, 1]])


def test_summarize_keeps_unsupported_probes_out_of_rankings():
    rows = []
    for method, success in (("gcbc", .1), ("gciql", .3)):
        for data_seed in (101, 202, 303):
            for fit_seed in (0, 1, 2):
                rows.append({"task": "x", "regime": "state_holdout", "data_seed": data_seed, "fit_seed": fit_seed,
                             "method": method, "coverage": 1.0, "budget": 0,
                             "metrics": {"R": None, "C": None, "G": success, "G_normalized_cost": 1-success}})
    result = summarize(rows, data_seeds=[101, 202, 303], fit_seeds=[0, 1, 2], families={"x": "f"}, repetitions=20)
    assert {r["metric"] for r in result["estimates"]} == {"G", "G_normalized_cost"}
    differences = [r for r in result["paired_contrasts"] if r["metric"] == "G"]
    assert differences and all(r["interval"] == pytest.approx([-.2, -.2]) for r in differences)
    broken = deepcopy(rows); broken[0]["metrics"]["R"] = .5
    with pytest.raises(ValueError, match="support"):
        summarize(broken, data_seeds=[101, 202, 303], fit_seeds=[0, 1, 2], families={"x": "f"}, repetitions=20)


def test_goal_audit_recomputes_failure_inclusive_and_equal_tier_metrics():
    # Deliberately unequal tier sizes: pooled success=.5, equal-tier success=1/3.
    rows = []
    for i, tier in enumerate((0, 0, 1, 2)):
        success = tier == 0
        rows.append({"case_index": i, "tier": tier, "distance": 1, "success": success,
                     "outcome": "success" if success else "policy_error", "error": None if success else {"type": "ValueError"},
                     "attempts": 1 if success else 0, "illegal_attempts": 0, "penalized_cost": 1 if success else 10,
                     "success_excess": 0 if success else None, "empirically_reachable": True,
                     "model_queries": 0, "model_calls": 0, "policy_seconds": 0.0})
    def group(n, success):
        return {"cases": n, "success_rate": success, "mean_attempts_all": success,
                "mean_penalized_cost_all": 10-9*success, "illegal_attempt_rate": 0,
                "policy_error_rate": 1-success, "mean_success_excess": 0 if success else None,
                "model_queries": 0, "model_calls": 0, "policy_seconds": 0.0}
    goal = {"control_budget": 10, "model_query_limit_per_decision": 0, "cases": rows,
            "aggregate": group(4, .5), "by_tier": {"easy": group(2, 1), "medium": group(1, 0), "hard": group(1, 0)},
            "by_empirical_reachability": {"reachable": group(4, .5), "unreachable": {"cases": 0}},
            "tier_macro_success_rate": 1/3, "tier_macro_penalized_cost": 7}
    audit_goals(goal, expected_cases=4, horizon=10, budget=0)
    tampered = deepcopy(goal); tampered["tier_macro_success_rate"] = .5
    with pytest.raises(ValueError, match="equal-tier"):
        audit_goals(tampered, expected_cases=4, horizon=10, budget=0)
    tampered = deepcopy(goal); tampered["cases"][-1]["case_index"] = 0
    with pytest.raises(ValueError, match="duplicated"):
        audit_goals(tampered, expected_cases=4, horizon=10, budget=0)
    tampered = deepcopy(goal); tampered["cases"][-1]["penalized_cost"] = 0
    with pytest.raises(ValueError, match="failure-inclusive"):
        audit_goals(tampered, expected_cases=4, horizon=10, budget=0)


def test_prediction_audit_uses_legality_balancing_and_empty_group_nulls():
    report = {"R": {"cases": 10, "next_board_exact": .8, "legality_accuracy": .8, "joint_exact": .8,
                    "balanced_joint_exact": .5,
                    "by_legality": {"legal": {"cases": 8, "next_board_exact": 1, "legality_accuracy": 1, "joint_exact": 1},
                                    "illegal": {"cases": 2, "next_board_exact": 0, "legality_accuracy": 0, "joint_exact": 0}}}}
    audit_prediction_scores(report, "R", 10)
    report["R"]["balanced_joint_exact"] = .8
    with pytest.raises(ValueError, match="balanced"):
        audit_prediction_scores(report, "R", 10)
