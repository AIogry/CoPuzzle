from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from copuzzle_baselines.config import Config, canonical
from copuzzle_baselines.data import Sampler, load_training


@pytest.mark.parametrize("kwargs", [{"discount": 1.1}, {"coverage": 0}, {"batch_size": True},
                                    {"hidden_dims": []}, {"method": "oracle"}, {"expectile": 0},
                                    {"learning_rate": float("nan")}, {"temperature": -1},
                                    {"method": "gcbc", "actor_future_probability": 0.5}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_unknown_config_fields_rejected():
    with pytest.raises(ValueError):
        Config.from_dict({"true_rule": "available"})


def test_coverage_is_nested_groups_inverse_edges_and_is_train_only(bundle):
    full = load_training(bundle / "learner")
    half = load_training(bundle / "learner", coverage=0.5)
    quarter = load_training(bundle / "learner", coverage=0.25)
    assert set(quarter.selected_rows) < set(half.selected_rows) < set(full.selected_rows)
    kept = set(half.selected_rows.tolist())
    native_states = full.vocabulary.decode(full.states)
    native_next = full.vocabulary.decode(full.next_states)
    groups = {}
    for i, (s, t, a, legal) in enumerate(zip(native_states, native_next, full.actions, full.legal)):
        key = ("legal", *sorted((tuple(s), tuple(t)))) if legal else ("illegal", tuple(s), int(a))
        groups.setdefault(key, []).append(i in kept)
    assert all(len(set(values)) == 1 for values in groups.values())
    for d in (full, half, quarter):
        pool = {tuple(row) for row in d.goal_pool}
        assert pool == {tuple(row) for row in np.concatenate((d.states, d.next_states))}
        assert d.identity["legality_supervision"].startswith("selected attempted action only")
        with pytest.raises(ValueError):
            d.states[0, 0] = 999


def test_unknown_symbols_do_not_refit_vocabulary(bundle):
    d = load_training(bundle / "learner")
    before = d.vocabulary.symbols.copy()
    unknown = np.full((2, d.width), 1000, dtype=np.int32)
    assert np.all(d.vocabulary.encode(unknown) == d.vocabulary.size)
    np.testing.assert_array_equal(d.vocabulary.symbols, before)
    with pytest.raises(ValueError):
        d.vocabulary.decode(np.asarray([d.vocabulary.size]))


def test_reading_does_not_depend_on_evaluator_files_or_queries(bundle, monkeypatch):
    import copuzzle
    import copuzzle.research.graph as graph_module
    def forbidden(*args, **kwargs):
        raise AssertionError("privileged rule query")
    monkeypatch.setattr(copuzzle, "get_rule", forbidden)
    monkeypatch.setattr(graph_module, "graph", forbidden)
    original = Path.open
    def guarded(path, *args, **kwargs):
        if "evaluator" in path.parts or path.name.startswith(("R_", "C_", "G_")) or path.name == "manifest.json":
            raise AssertionError(f"forbidden read: {path}")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    d = load_training(bundle / "learner")
    for method in ("gcbc", "gciql", "dynamics"):
        assert Sampler(d, Config(method=method), 0).sample()


def test_gcbc_uses_nontrivial_observed_successor_goals(bundle):
    d = load_training(bundle / "learner")
    sampler = Sampler(d, Config(method="gcbc"), 5)
    batch = sampler.sample(1024)
    observed = {(tuple(s), int(a)): t for s, a, t in zip(d.states, d.actions, d.next_states)}
    assert set(batch) == {"states", "actions", "actor_goals"}
    assert not np.any(np.all(batch["states"] == batch["actor_goals"], axis=1))
    for state, action, goal in zip(batch["states"], batch["actions"], batch["actor_goals"]):
        np.testing.assert_array_equal(goal, observed[tuple(state), int(action)])
    assert sampler.counts["rejected_attempts"] == 0


def test_iql_preserves_rejected_attempts_and_bootstraps_cutoffs(bundle):
    d = load_training(bundle / "learner")
    sampler = Sampler(d, Config(method="gciql", value_future_probability=1), 4)
    batch = sampler.sample(4096)
    no_change = np.all(batch["states"] == batch["next_states"], axis=1)
    assert no_change.any() and (~no_change).any()
    assert not np.any(np.all(batch["states"] == batch["value_goals"], axis=1))
    assert not np.any(np.all(batch["states"] == batch["actor_goals"], axis=1))
    assert np.all(batch["rewards"][no_change] == -1)
    assert np.all(batch["bootstrap"][no_change] == 1)
    assert np.all(batch["rewards"][~no_change] == 0)
    assert np.all(batch["bootstrap"][~no_change] == 0)
    assert sampler.counts["rejected_attempts"] == int(no_change.sum())
    assert not any("mask" in key for key in batch)


def test_sampler_resume_preserves_draws_counts_and_goals(bundle):
    d = load_training(bundle / "learner")
    c = Config()
    first, second = Sampler(d, c, 4), Sampler(d, c, 9)
    first.sample(10)
    snapshot = json.loads(canonical(first.state_dict()))
    second.restore(snapshot)
    a, b = first.sample(100), second.sample(100)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert canonical(first.state_dict()) == canonical(second.state_dict())
    invalid = deepcopy(snapshot)
    invalid["data_sha256"] = "different"
    before = canonical(second.state_dict())
    with pytest.raises(ValueError):
        second.restore(invalid)
    assert before == canonical(second.state_dict())


def test_dynamics_has_only_attempted_action_legality_labels(bundle):
    d = load_training(bundle / "learner")
    batch = Sampler(d, Config(method="dynamics"), 1).sample(1000)
    assert set(batch) == {"states", "actions", "next_states", "legal"}
    observed = {(tuple(s), int(a)): int(flag) for s, a, flag in zip(d.states, d.actions, d.legal)}
    assert batch["legal"].shape == (1000,)
    for s, a, legal in zip(batch["states"], batch["actions"], batch["legal"]):
        assert legal == observed[tuple(s), int(a)]
