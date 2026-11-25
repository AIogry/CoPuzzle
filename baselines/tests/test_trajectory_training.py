"""Multi-step training retains real episode boundaries and train-only goals."""
from copy import deepcopy
import importlib
import json

import numpy as np
import pytest

from copuzzle_baselines.config import Config, canonical
from copuzzle_baselines.trajectories import TrajectorySampler, load_trajectories


def archive(tmp_path, family, task, *, split="train"):
    data_api = importlib.import_module(f"copuzzle.games.{family}.data")
    rule = importlib.import_module(f"copuzzle.games.{family}.rules").get_rule(task)
    dataset = data_api.generate_dataset(rule, num_episodes=3, horizon=6, seed=17, split=split)
    path = dataset.save(tmp_path / f"{family}-{split}.npz")
    return path, dataset, rule


class IndependentButtons:
    name = "test-independent-buttons"
    rule_id = "test-independent-buttons-v1"
    n_buttons = 4

    def transition(self, state, action):
        result = state.copy()
        result[action] ^= 1
        return result


def scripted_archive(tmp_path, *, loops=False):
    """Real button trajectories with distinguishable episode state supports."""
    from copuzzle.games.lights_out.data import generate_dataset

    rule = IndependentButtons()
    dataset = generate_dataset(rule, num_episodes=2, horizon=4)
    dataset.states[:, 0] = [[0, 0, 0, 0], [0, 0, 0, 1]]
    dataset.actions[:] = [[0, 1, 1, 2] if loops else [0, 1, 2, 3], [0, 2, 0, 1]]
    dataset.events[:] = np.eye(4, dtype=np.uint8)[dataset.actions]
    for episode in range(2):
        for step in range(4):
            dataset.states[episode, step + 1] = rule.transition(
                dataset.states[episode, step], dataset.actions[episode, step])
    dataset.validate(rule)
    return dataset.save(tmp_path / "scripted.npz"), dataset


@pytest.mark.parametrize("family,task", [
    ("lights_out", "grid-3x3"), ("permutation", "pancake-4"),
    ("classic", "hanoi-3"),
])
def test_saved_archives_flatten_losslessly_without_rule_queries(tmp_path, monkeypatch, family, task):
    path, native, rule = archive(tmp_path, family, task)

    def forbidden(*args, **kwargs):
        raise AssertionError("offline adaptation queried a rule")

    monkeypatch.setattr(type(rule), "transition", forbidden)
    if hasattr(type(rule), "action_mask"):
        monkeypatch.setattr(type(rule), "action_mask", forbidden)
    data = load_trajectories(path, family=family)
    assert data.horizon == native.horizon
    assert len(data.states) == native.actions.size
    np.testing.assert_array_equal(data.vocabulary.decode(data.states), native.states[:, :-1].reshape(-1, data.width))
    np.testing.assert_array_equal(data.vocabulary.decode(data.next_states), native.states[:, 1:].reshape(-1, data.width))
    np.testing.assert_array_equal(data.actions, native.actions.ravel())
    assert data.trajectory_states.shape == native.states.shape
    for method in ("gcbc", "gciql", "dynamics"):
        batch = TrajectorySampler(data, Config(method=method), 4).sample(32)
        assert not any("mask" in key for key in batch)
    with pytest.raises(ValueError):
        data.trajectory_states[0, 0, 0] = 999


def test_repeated_queries_are_experience_not_duplicate_dataset_errors(tmp_path):
    path, native, _ = archive(tmp_path, "permutation", "pancake-2")
    data = load_trajectories(path, family="permutation")
    assert len(np.unique(np.concatenate((data.states, data.actions[:, None]), axis=1), axis=0)) < len(data.states)
    assert len(data.states) == native.actions.size


def test_future_goals_never_cross_resets_and_include_final_frame(tmp_path):
    path, native = scripted_archive(tmp_path)
    data = load_trajectories(path, family="lights_out")
    batch = TrajectorySampler(data, Config(method="gcbc"), 23).sample(2048)
    assert set(batch) == {"states", "actions", "actor_goals"}
    lookup = {(tuple(s), int(a)): (e, t) for e in range(2)
              for t, (s, a) in enumerate(zip(native.states[e, :-1], native.actions[e]))}
    final_hits, long_hits, last_steps = 0, 0, 0
    for state, action, goal in zip(data.vocabulary.decode(batch["states"]), batch["actions"],
                                   data.vocabulary.decode(batch["actor_goals"])):
        episode, step = lookup[tuple(state), int(action)]
        assert not np.array_equal(state, goal)
        assert np.any(np.all(native.states[episode, step + 1:] == goal, axis=1))
        final_hits += int(np.array_equal(native.states[episode, -1], goal))
        long_hits += int(not np.array_equal(native.states[episode, step + 1], goal))
        if step == native.horizon - 1:
            np.testing.assert_array_equal(goal, native.states[episode, -1])
            last_steps += 1
    assert final_hits > 0 and long_hits > 0 and last_steps > 0


def test_repeated_future_boards_keep_index_multiplicity(tmp_path):
    path, native = scripted_archive(tmp_path, loops=True)
    data = load_trajectories(path, family="lights_out")
    batch = TrajectorySampler(data, Config(method="gcbc"), 3).sample(12000)
    states = data.vocabulary.decode(batch["states"])
    goals = data.vocabulary.decode(batch["actor_goals"])
    starts = np.all(states == native.states[0, 0], axis=1)
    assert starts.sum() > 500
    # Four eligible future indices: B, C, B, D. Uniform unique boards would give 1/3.
    repeated = np.all(goals[starts] == native.states[0, 1], axis=1).mean()
    assert repeated == pytest.approx(0.5, abs=0.045)
    assert not np.any(np.all(states == goals, axis=1))

    value_batch = TrajectorySampler(data, Config(value_future_probability=1), 3).sample(2048)
    hit_now = np.all(value_batch["next_states"] == value_batch["value_goals"], axis=1)
    assert hit_now.any()
    # A target also occurring later must still terminate at the immediate hit.
    assert np.all(value_batch["rewards"][hit_now] == 0)
    assert np.all(value_batch["bootstrap"][hit_now] == 0)


def test_stationary_tail_uses_random_fallback_without_inventing_future_goals(tmp_path):
    from copuzzle.games.lights_out.data import generate_dataset

    class ButtonsWithNoop(IndependentButtons):
        def transition(self, state, action):
            return state.copy() if action == 3 else super().transition(state, action)

    rule = ButtonsWithNoop()
    native = generate_dataset(rule, num_episodes=2, horizon=3)
    native.states[:, 0] = [[0, 0, 0, 0], [0, 1, 0, 0]]
    native.actions[:] = [0, 3, 3]
    native.events[:] = np.eye(4, dtype=np.uint8)[native.actions]
    for episode in range(2):
        for step in range(3):
            native.states[episode, step + 1] = rule.transition(
                native.states[episode, step], native.actions[episode, step])
    native.validate(rule)
    data = load_trajectories(native.save(tmp_path / "stationary.npz"), family="lights_out")
    config = Config(actor_future_probability=1, value_future_probability=1)
    sampler = TrajectorySampler(data, config, 0)
    batch = sampler.sample(1024)
    stalled = batch["actions"] == 3
    assert stalled.any() and (~stalled).any()
    assert np.all(batch["rewards"][stalled] == -1)
    assert np.all(batch["bootstrap"][stalled] == 1)
    assert not np.any(np.all(batch["states"] == batch["value_goals"], axis=1))
    assert sampler.counts["value_future_choices"] == int((~stalled).sum())
    bc = TrajectorySampler(data, Config(method="gcbc"), 1).sample(128)
    assert np.all(bc["actions"] == 0)


def test_iql_random_goals_bootstrap_unsuccessful_collection_cutoffs(tmp_path):
    path, native = scripted_archive(tmp_path)
    data = load_trajectories(path, family="lights_out")
    config = Config(method="gciql", actor_future_probability=0, value_future_probability=0)
    batch = TrajectorySampler(data, config, 9).sample(2048)
    success = np.all(batch["next_states"] == batch["value_goals"], axis=1)
    np.testing.assert_array_equal(batch["rewards"], np.where(success, 0, -1))
    np.testing.assert_array_equal(batch["bootstrap"], ~success)
    assert np.any(np.any(batch["actor_goals"] != batch["value_goals"], axis=1))
    assert not np.any(np.all(batch["states"] == batch["value_goals"], axis=1))
    states = data.vocabulary.decode(batch["states"])
    final_rows = np.any(np.all(states[:, None] == native.states[:, -2][None], axis=-1), axis=1)
    assert np.any(final_rows & ~success)
    assert np.all(batch["bootstrap"][final_rows & ~success] == 1)


@pytest.mark.parametrize("method", ["gcbc", "gciql"])
def test_json_sampler_resume_is_exact_and_rejects_other_data(tmp_path, method):
    path, _ = scripted_archive(tmp_path)
    data = load_trajectories(path, family="lights_out")
    config = Config(method=method)
    first, second = TrajectorySampler(data, config, 1), TrajectorySampler(data, config, 2)
    first.sample(17)
    checkpoint = json.loads(canonical(first.state_dict()))
    second.restore(checkpoint)
    expected, actual = first.sample(256), second.sample(256)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert canonical(first.state_dict()) == canonical(second.state_dict())
    broken = deepcopy(checkpoint)
    broken["data_sha256"] = "different"
    before = canonical(second.state_dict())
    with pytest.raises(ValueError):
        second.restore(broken)
    assert canonical(second.state_dict()) == before


def test_nontrain_archives_and_partial_coverage_are_rejected(tmp_path):
    path, _, _ = archive(tmp_path, "permutation", "pancake-4", split="val")
    with pytest.raises(ValueError):
        load_trajectories(path, family="permutation")
    train_path, _, _ = archive(tmp_path, "permutation", "pancake-4")
    with pytest.raises(ValueError):
        load_trajectories(train_path, family="permutation", coverage=0.5)
    with pytest.raises(ValueError):
        load_trajectories(train_path, family="unknown")


@pytest.mark.parametrize("method", ["gcbc", "gciql"])
def test_real_training_resume_matches_uninterrupted_updates(tmp_path, bundle, method):
    from copuzzle_baselines.runtime import evaluate, load_checkpoint, state_digest, train

    path, _, _ = archive(tmp_path, "permutation", "pancake-4")
    config = Config(method=method, hidden_dims=(4,), batch_size=4)
    options = dict(config=config, seed=7, trajectory_family="permutation")
    full = train(path, tmp_path / "full", steps=3, **options)
    first = train(path, tmp_path / "first", steps=1, **options)
    checkpoint = tmp_path / "first" / first["checkpoints"][-1]["file"]
    resumed = train(path, tmp_path / "resumed", steps=3, resume=checkpoint, **options)
    a, _, ha = load_checkpoint(tmp_path / "full" / full["checkpoints"][-1]["file"])
    b, _, hb = load_checkpoint(tmp_path / "resumed" / resumed["checkpoints"][-1]["file"])
    assert state_digest(a) == state_digest(b)
    assert ha["sampler"] == hb["sampler"]
    assert resumed["start_step"] == 1 and resumed["completed_step"] == 3
    assert full["sampling_counts"] == resumed["sampling_counts"]
    with pytest.raises(ValueError, match="trajectory checkpoints"):
        evaluate(checkpoint, bundle)


def test_formal_study_context_rejected_before_creating_output(tmp_path):
    from copuzzle_baselines.runtime import train

    path, _, _ = archive(tmp_path, "permutation", "pancake-4")
    output = tmp_path / "not-a-formal-fit"
    with pytest.raises(ValueError, match="frozen formal study"):
        train(path, output, config=Config(), seed=0, steps=1,
              trajectory_family="permutation", fit_context={})
    assert not output.exists()


@pytest.mark.parametrize("source", [
    ["--trajectories", "train.npz"],
    ["--learner", "learner", "--trajectory-family", "permutation"],
    ["--learner", "learner", "--trajectories", "train.npz", "--trajectory-family", "permutation"],
])
def test_cli_requires_exactly_one_source_and_paired_trajectory_family(tmp_path, source, monkeypatch):
    from copuzzle_baselines.cli import main
    import copuzzle_baselines.runtime as runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid CLI source reached training")

    monkeypatch.setattr(runtime, "train", forbidden)
    output = tmp_path / "no-output"
    with pytest.raises(SystemExit) as exc:
        main(["train", *source, "--output", str(output)])
    assert exc.value.code == 2
    assert not output.exists()
