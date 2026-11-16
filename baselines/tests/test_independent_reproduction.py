from copuzzle.benchmark.spec import INSTANCES, REGIMES
from copuzzle_baselines.config import Config
from copuzzle_baselines.study.registry import compile_jobs
from copuzzle_baselines.study.reproduce import comparable, selected_jobs


def test_independent_subset_covers_every_task_and_condition_without_score_selection():
    datasets = [{"id": f"{s.task}-{r}-{seed}", "task": s.task, "regime": r, "seed": seed}
                for seed in (101, 202, 303) for s in INSTANCES for r in REGIMES]
    selected = {"fit_seeds": [0, 1, 2], "method_configs": {m: Config(method=m).to_dict() for m in ("gcbc", "gciql", "dynamics")},
                "coverage_seed": 0, "coverage_fractions": [.25, .5, 1.], "training_steps": 10000,
                "checkpoint_every": 1000, "log_every": 100, "model_query_budgets": {"gcbc": [0], "gciql": [0], "dynamics": [32, 128]}}
    views = {f"{d['id']}__c{int(100*c):03d}": {"dataset": d["id"], "coverage": c}
             for d in datasets for c in selected["coverage_fractions"]}
    fits, evaluations, controls = compile_jobs(datasets, selected, views)
    study = {"datasets": datasets, "fit_jobs": fits, "evaluation_jobs": evaluations, "control_jobs": controls}
    jobs = selected_jobs(study)
    assert len(jobs) == 104 and sum(p == "test" for p, _ in jobs) == 48
    assert len({j["dataset_id"] for _, j in jobs}) == 12
    chosen_fits = {j["fit_id"] for p, j in jobs if p == "test"}
    assert len(chosen_fits) == 36
    assert {f["data_seed"] for f in fits if f["id"] in chosen_fits} == {101}
    assert {f["fit_seed"] for f in fits if f["id"] in chosen_fits} == {0}
    assert {f["coverage"] for f in fits if f["id"] in chosen_fits} == {1.0}


def test_reproduction_comparison_excludes_time_but_not_outcomes_or_budgets():
    original = {"evaluation_seconds": 4., "G": {"cases": [{"success": False, "attempts": 10, "model_queries": 32, "policy_seconds": 2.}], "model_query_limit": 32}}
    faster = {"evaluation_seconds": 2., "G": {"cases": [{"success": False, "attempts": 10, "model_queries": 32, "policy_seconds": 1.}], "model_query_limit": 32}}
    assert comparable(original) == comparable(faster)
    faster["G"]["cases"][0]["success"] = True
    assert comparable(original) != comparable(faster)
    faster["G"]["cases"][0]["success"] = False
    faster["G"]["model_query_limit"] = 128
    assert comparable(original) != comparable(faster)
