from copy import deepcopy

import pytest

from copuzzle.benchmark.spec import INSTANCES, REGIMES
from copuzzle_baselines.study.pilot import aggregate_selection


def matrix():
    jobs, results = [], []
    candidates = ["gcbc", "gciql_future", "gciql_mixed", "dynamics"]
    for item in INSTANCES:
        for regime in REGIMES:
            for candidate in candidates:
                row = {"id": f"{item.task}-{regime}-{candidate}", "task": item.task, "regime": regime, "candidate": candidate}
                jobs.append(row)
                results.append({**row, "status": "completed", "scores": {"0": {"G_tier_macro": 0.25}}})
    index = {"jobs": jobs, "candidates": candidates, "conditions": list(REGIMES),
             "data_inventory_sha256": "data", "plan_sha256": "plan"}
    return index, results


def test_pilot_selection_uses_all_cells_and_prefers_registered_tie_break():
    index, results = matrix()
    selected = aggregate_selection(index, results)
    assert selected["selected_candidates"]["gciql"] == "gciql_mixed"
    assert selected["completed_pilot_fits"] == 48 and selected["formal_fits_contributed"] == 0
    # Two Lights Out sizes together have the weight of one family, not two.
    for row in results:
        if row["candidate"] == "gciql_future":
            row["scores"]["0"]["G_tier_macro"] = 1.0 if row["task"].startswith("grid-") else 0.0
    selected = aggregate_selection(index, results)
    assert selected["scores"]["gciql_future"]["G_macro"] == pytest.approx(0.2)
    assert selected["selected_candidates"]["gciql"] == "gciql_mixed"


def test_incomplete_failed_or_duplicated_pilot_never_seals_selection():
    index, results = matrix()
    with pytest.raises(ValueError):
        aggregate_selection(index, results[:-1])
    failed = deepcopy(results)
    failed[0]["status"] = "failed"
    with pytest.raises(ValueError):
        aggregate_selection(index, failed)
    duplicate = deepcopy(results)
    duplicate[0] = duplicate[1]
    with pytest.raises(ValueError):
        aggregate_selection(index, duplicate)
