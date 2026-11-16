"""Paired uncertainty conditional on the frozen instances and evaluation cases."""
from __future__ import annotations

from itertools import combinations

import numpy as np


PRIMARY_METRICS = ("R", "C", "G", "G_normalized_cost")


def bootstrap_draws(data_count, fit_count, *, repetitions=10000, seed=0):
    """Reuse these indices across methods, tasks, budgets and coverage levels.

    A repeated outer cluster gets its own inner draw. Deterministic references
    are broadcast over inner indices for pairing, not treated as extra runs.
    """
    if data_count < 1 or fit_count < 1 or repetitions < 2:
        raise ValueError("bootstrap requires positive seed counts and >=2 draws")
    rng = np.random.default_rng(seed)
    outer = rng.integers(data_count, size=(repetitions, data_count))
    inner = rng.integers(fit_count, size=(repetitions, data_count, fit_count))
    return outer, inner


def estimate(values, weights, draws, *, confidence=0.95):
    """values has task × data-seed × fit-seed axes; weights sum to one."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if (values.ndim != 3 or weights.shape != (values.shape[0],)
            or not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1)
            or not 0 < confidence < 1):
        raise ValueError("invalid complete seed matrix or scope weights")
    outer, inner = draws
    if outer.shape[1:] != (values.shape[1],) or inner.shape != (*outer.shape, values.shape[2]):
        raise ValueError("resampling axes differ from registered seed axes")
    # Aggregate fixed tasks before resampling; never bootstrap game families.
    clusters = np.einsum("t,tdf->df", weights, values)
    sampled = clusters[outer[:, :, None], inner].mean(axis=(1, 2))
    alpha = (1 - confidence) / 2
    return {"mean": float(clusters.mean()),
            "interval": np.quantile(sampled, [alpha, 1-alpha]).tolist(),
            "confidence": confidence, "bootstrap_repetitions": len(outer),
            "data_seed_means": clusters.mean(axis=1).tolist(),
            "fit_values_by_data_seed": clusters.tolist()}


def series_key(row):
    return row["method"], row["coverage"], row["budget"]


def series_label(key):
    method, coverage, budget = key
    return f"{method}/c{int(coverage*100):03d}/b{budget if budget is not None else 'exempt'}"


def seed_matrix(rows, tasks, data_seeds, fit_seeds, metric):
    """Reject missing/duplicate cells; unsupported probes are handled upstream."""
    matrix = np.full((len(tasks), len(data_seeds), len(fit_seeds)), np.nan)
    for row in rows:
        if row["task"] not in tasks:
            continue
        t, d = tasks.index(row["task"]), data_seeds.index(row["data_seed"])
        slots = range(len(fit_seeds)) if row["fit_seed"] is None else [fit_seeds.index(row["fit_seed"])]
        for f in slots:
            if not np.isnan(matrix[t, d, f]):
                raise ValueError("duplicate statistical cell")
            value = row["metrics"][metric]
            if value is None or not np.isfinite(value):
                raise ValueError("unsupported/nonfinite statistical cell")
            matrix[t, d, f] = value
    if np.any(np.isnan(matrix)):
        raise ValueError("statistical matrix omits a registered seed")
    return matrix


def scopes(tasks, families, all_tasks):
    result = [(f"instance:{task}", [task], [1.0]) for task in tasks]
    for family in sorted({families[t] for t in tasks}):
        members = [t for t in tasks if families[t] == family]
        result.append((f"family:{family}", members, [1 / len(members)] * len(members)))
    if set(tasks) == set(all_tasks):
        names = {families[t] for t in tasks}
        weights = [1 / len(names) / sum(families[u] == families[t] for u in tasks) for t in tasks]
        result.append(("core_family_macro", tasks, weights))
    return result


def planned_contrasts(keys):
    """Fixed primary learned pairs, coverage/search effects and oracle gaps."""
    keys = set(keys)
    result = []
    learned = sorted((k for k in keys if k[0] in ("gcbc", "gciql", "dynamics") and k[1] == 1), key=series_label)
    for left, right in combinations(learned, 2):
        result.append(("primary_method_difference", left, right))
    for key in sorted(keys, key=series_label):
        method, coverage, budget = key
        if coverage in (0.25, 0.5) and (method, 1.0, budget) in keys:
            result.append(("coverage_full_minus_partial", (method, 1.0, budget), key))
        if budget == 32 and (method, coverage, 128) in keys:
            result.append(("search_128_minus_32", (method, coverage, 128), key))
        if method == "dynamics" and coverage == 1 and ("oracle_dynamics", 1.0, budget) in keys:
            result.append(("matched_oracle_minus_learned", ("oracle_dynamics", 1.0, budget), key))
        if method == "oracle_dynamics" and ("optimal_oracle", 1.0, None) in keys:
            result.append(("optimal_minus_bounded_oracle", ("optimal_oracle", 1.0, None), key))
    return result


def summarize(rows, *, data_seeds, fit_seeds, families, repetitions=10000, seed=0, confidence=0.95):
    """No missing/failed run is filtered here: failure policy is applied earlier."""
    draws = bootstrap_draws(len(data_seeds), len(fit_seeds), repetitions=repetitions, seed=seed)
    estimates, contrasts = [], []
    for regime in sorted({r["regime"] for r in rows}):
        groups = {}
        for row in rows:
            if row["regime"] == regime:
                groups.setdefault(series_key(row), []).append(row)
        for metric in PRIMARY_METRICS:
            matrices = {}
            for key, group in groups.items():
                supported = [r["metrics"][metric] is not None for r in group]
                if not any(supported):
                    continue
                if not all(supported):
                    raise ValueError("probe support changes across a registered series")
                tasks = sorted({r["task"] for r in group})
                for scope, members, weights in scopes(tasks, families, list(families)):
                    matrix = seed_matrix(group, members, data_seeds, fit_seeds, metric)
                    matrices[(key, scope)] = matrix, weights, members
                    estimates.append({"regime": regime, "metric": metric, "series": series_label(key),
                                      "scope": scope, "tasks": members,
                                      **estimate(matrix, weights, draws, confidence=confidence)})
            for kind, left, right in planned_contrasts(groups):
                for key, scope in list(matrices):
                    if key != left or (right, scope) not in matrices:
                        continue
                    a, weights, tasks = matrices[(left, scope)]
                    b, other_weights, other_tasks = matrices[(right, scope)]
                    if tasks != other_tasks or weights != other_weights:
                        raise ValueError("contrast scopes do not match")
                    contrasts.append({"kind": kind, "regime": regime, "metric": metric, "scope": scope,
                                      "left_minus_right": [series_label(left), series_label(right)], "tasks": tasks,
                                      **estimate(a-b, weights, draws, confidence=confidence)})
    return {"estimates": estimates, "paired_contrasts": contrasts,
            "resampling": {"seed": seed, "repetitions": repetitions, "confidence": confidence,
                           "data_seeds": data_seeds, "fit_seeds": fit_seeds,
                           "pairing": "shared outer data-seed and inner fit-seed draws across all series and fixed tasks",
                           "scope": "conditional on frozen tasks/cases; no case/family resampling",
                           "deterministic_references": "broadcast for pairing, not additional independent fits",
                           "limitation": "three outer seeds; descriptive percentile intervals, no significance or unseen-family claim"}}
