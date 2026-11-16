from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import zipfile

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copuzzle.research.evaluation import ModelBudget
from copuzzle_baselines.agents import Agent
from copuzzle_baselines.config import Config
from copuzzle_baselines.data import Sampler, load_training
from copuzzle_baselines.planning import ModelPlanner
from copuzzle_baselines.runtime import (evaluate, load_checkpoint, predict,
                                       state_digest, train)


def goal_batch():
    return {"states": np.array([[0, 0], [1, 0]], np.int32),
            "next_states": np.array([[0, 1], [1, 0]], np.int32),
            "actions": np.array([0, 1], np.int32),
            "value_goals": np.array([[0, 1], [0, 0]], np.int32),
            "actor_goals": np.array([[1, 1], [0, 1]], np.int32),
            "rewards": np.array([0, -1], np.float32), "bootstrap": np.array([0, 1], np.float32)}


def test_iql_loss_values_gradient_stops_and_post_update_targets():
    a = Agent(Config(method="gciql", hidden_dims=(4,), discount=0.5, target_rate=0.2), 2, 2, 3, 0)
    p = jax.tree_util.tree_map(jnp.zeros_like, a.params)
    t = jax.tree_util.tree_map(jnp.zeros_like, a.targets)
    for name, value in (("q1", 1.0), ("q2", 3.0), ("value", 2.0)):
        p[name]["output"]["bias"] = jnp.full_like(p[name]["output"]["bias"], value)
    for name, value in (("q1", 4.0), ("q2", 5.0)):
        t[name]["output"]["bias"] = jnp.full_like(t[name]["output"]["bias"], value)
    batch = a.prepare(goal_batch())
    _, metrics = a.loss(p, t, batch)
    assert float(metrics["value_loss"]) == pytest.approx(2.8)
    assert float(metrics["critic_loss"]) == pytest.approx(10.0)
    assert float(metrics["td_target_mean"]) == pytest.approx(0.0)
    assert float(metrics["actor_loss"]) == pytest.approx(math.exp(-3) * math.log(3))
    grads = jax.grad(lambda params: a.loss(params, t, batch)[1]["actor_loss"])(p)
    assert all(np.all(np.asarray(x) == 0) for key in ("value", "q1", "q2") for x in jax.tree_util.tree_leaves(grads[key]))
    targets_grad = jax.grad(lambda targets: a.loss(p, targets, batch)[0])(t)
    assert all(np.all(np.asarray(x) == 0) for x in jax.tree_util.tree_leaves(targets_grad))
    old_target = deepcopy(a.targets)
    a.update(goal_batch())
    for key in ("q1", "q2"):
        expected = jax.tree_util.tree_map(lambda new, old: 0.2 * new + 0.8 * old, a.params[key], old_target[key])
        for actual, wanted in zip(jax.tree_util.tree_leaves(a.targets[key]), jax.tree_util.tree_leaves(expected)):
            np.testing.assert_allclose(actual, wanted, rtol=1e-6, atol=1e-7)


def test_true_masks_and_inconsistent_goal_labels_are_rejected():
    a = Agent(Config(hidden_dims=(4,)), 2, 2, 3, 0)
    before = state_digest(a)
    batch = goal_batch()
    with pytest.raises(ValueError, match="true masks"):
        a.update({**batch, "action_mask": np.ones((2, 3), bool)})
    with pytest.raises(ValueError, match="bootstrap"):
        a.update({**batch, "bootstrap": np.zeros(2, np.float32)})
    assert state_digest(a) == before


def test_bc_uniform_actor_loss_and_unmasked_predictions():
    a = Agent(Config(method="gcbc", hidden_dims=(4,)), 2, 2, 3, 0)
    assert set(a.params) == {"actor"} and not a.targets
    p = jax.tree_util.tree_map(jnp.zeros_like, a.params)
    batch = {key: value for key, value in goal_batch().items() if key in {"states", "actor_goals", "actions"}}
    value, _ = a.loss(p, {}, a.prepare(batch))
    assert float(value) == pytest.approx(math.log(3))
    before = state_digest(a)
    assert a.actions(batch["states"], batch["actor_goals"]).shape == (2,)
    assert state_digest(a) == before


def test_dynamics_supervises_only_the_observed_action():
    a = Agent(Config(method="dynamics", hidden_dims=(4,)), 2, 2, 3, 0)
    batch = {"states": np.array([[0, 1]], np.int32), "next_states": np.array([[1, 0]], np.int32),
             "actions": np.array([0], np.int32), "legal": np.array([1], np.float32)}
    grads = jax.grad(lambda p: a.loss(p, {}, a.prepare(batch))[0])(a.params)
    # Action-ID feature columns for unattempted actions receive no target gradient.
    unused = np.asarray(grads["dynamics"]["hidden0"]["kernel"])[-2:]
    assert np.all(unused == 0)
    with pytest.raises(ValueError, match="exactly"):
        a.update({**batch, "next_action_mask": np.ones((1, 3), np.float32)})


def test_nonfinite_dynamics_is_not_silently_argmaxed_to_a_board():
    a = Agent(Config(method="dynamics", hidden_dims=(4,)), 2, 2, 3, 0)
    a.params["dynamics"]["output"]["bias"] = jnp.full_like(a.params["dynamics"]["output"]["bias"], jnp.nan)
    with pytest.raises(FloatingPointError, match="model logits"):
        a.predict(np.array([[0, 1]], np.int32), np.array([0], np.int32))


@pytest.mark.parametrize("method", ["gcbc", "gciql", "dynamics"])
def test_checkpoint_resume_is_exact_next_update_and_never_overwrites(bundle, tmp_path, method):
    c = Config(method=method, hidden_dims=(8,), batch_size=8)
    root = tmp_path / method
    full = train(bundle / "learner", root / "full", config=c, seed=3, steps=4, checkpoint_every=2)
    initial = train(bundle / "learner", root / "initial", config=c, seed=3, steps=2)
    checkpoint = root / "initial" / initial["checkpoints"][-1]["file"]
    resumed = train(bundle / "learner", root / "resume", config=c, seed=3, steps=4, resume=checkpoint)
    a, _, ha = load_checkpoint(root / "full" / full["checkpoints"][-1]["file"])
    b, _, hb = load_checkpoint(root / "resume" / resumed["checkpoints"][-1]["file"])
    assert state_digest(a) == state_digest(b)
    assert ha["sampler"] == hb["sampler"]
    assert resumed["start_step"] == 2 and resumed["completed_step"] == 4
    assert resumed["sampling_counts"] == full["sampling_counts"]
    with pytest.raises(FileExistsError):
        train(bundle / "learner", root / "full", config=c, seed=3, steps=4)
    with pytest.raises(ValueError, match="identical"):
        train(bundle / "learner", root / "bad", config=c, seed=4, steps=4, resume=checkpoint)
    assert not (root / "bad").exists()


def test_training_reads_no_probes_evaluator_or_rule(bundle, tmp_path, monkeypatch):
    import copuzzle
    import copuzzle.research.graph as graph_module
    def forbidden(*args, **kwargs):
        raise AssertionError("training queried an oracle")
    monkeypatch.setattr(copuzzle, "get_rule", forbidden)
    monkeypatch.setattr(graph_module, "graph", forbidden)
    original = Path.open
    def guarded(path, *args, **kwargs):
        if "evaluator" in path.parts or path.name.startswith(("R_", "C_", "G_")) or path.name == "manifest.json":
            raise AssertionError(f"training read held-out material: {path}")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    run = train(bundle / "learner", tmp_path / "train-only", config=Config(hidden_dims=(4,), batch_size=4), seed=0, steps=2)
    assert run["status"] == "completed"


def test_corrupt_or_misbound_checkpoint_is_rejected(bundle, tmp_path):
    root = tmp_path / "training"
    run = train(bundle / "learner", root, config=Config(method="gcbc", hidden_dims=(4,), batch_size=4), seed=0, steps=1)
    original = root / run["checkpoints"][-1]["file"]
    with zipfile.ZipFile(original) as stream:
        header = json.loads(stream.read("header.json"))
        state = stream.read("state.msgpack")
    corrupt = tmp_path / "bad.zip"
    header["sampler"]["data_sha256"] = "not-the-same-data"
    with zipfile.ZipFile(corrupt, "w") as stream:
        stream.writestr("header.json", json.dumps(header))
        stream.writestr("state.msgpack", state)
    with pytest.raises(ValueError, match="training identity"):
        load_checkpoint(corrupt)
    candidate = serialization.msgpack_restore(state)
    candidate["params"]["actor"]["output"]["bias"] = np.full(4, np.nan, np.float32)
    a, _, _ = load_checkpoint(original)
    before = state_digest(a)
    with pytest.raises(ValueError, match="nonfinite"):
        a.restore(candidate)
    assert before == state_digest(a)


def test_predictions_are_open_loop_and_preserve_padding(monkeypatch):
    from copuzzle_baselines.data import Vocabulary, immutable
    from copuzzle_baselines.runtime import _predict
    class ToyModel:
        config = Config(method="dynamics")
        def predict(self, boards, actions):
            return (boards + 1) % 3, np.ones(len(boards), np.uint8)
    vocabulary = Vocabulary(immutable(np.array([0, 1, 2], np.int32)), 1)
    inputs = {"board": np.array([[0], [1]], np.int32), "actions": np.array([[0, 0, 0], [0, -1, -1]], np.int32),
              "lengths": np.array([3, 1], np.int32)}
    result = _predict(ToyModel(), vocabulary, inputs, "C", 1)["prefix_boards"]
    assert result[:, :, 0].tolist() == [[1, 2, 0], [2, -1, -1]]


def test_planner_charges_every_query_and_never_queries_true_dynamics():
    calls = []
    def model(board, action):
        calls.append((int(board[0]), action))
        return np.asarray([int(board[0]) + action + 1], np.int32), True
    obs = {"board": np.array([0], np.int32), "goal": np.array([5], np.int32), "steps_remaining": 5}
    planner = ModelPlanner(model, 2)
    budget = ModelBudget(6)
    result = planner(obs, budget)
    assert result in [0, 1]
    assert len(calls) == budget.used <= 6
    with pytest.raises(ValueError, match="positive"):
        planner(obs, ModelBudget(0))
    assert len(calls) == budget.used


@pytest.mark.parametrize("method", ["gcbc", "gciql", "dynamics"])
def test_saved_models_evaluate_without_parameter_updates(bundle, tmp_path, method):
    root = tmp_path / "saved"
    run = train(bundle / "learner", root, config=Config(method=method, hidden_dims=(4,), batch_size=4), seed=0, steps=2)
    checkpoint = root / run["checkpoints"][-1]["file"]
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = evaluate(checkpoint, bundle, model_query_limit=4 if method == "dynamics" else 0)
    assert report["learner_state_unchanged"]
    assert report["G"]["aggregate"]["cases"] == 3
    assert report["R"]["supported"] == (method == "dynamics")
    assert before == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if method == "dynamics":
        assert report["G"]["aggregate"]["model_queries"] > 0
        result = predict(checkpoint, bundle / "learner", "C")
        assert result["prefix_boards"].ndim == 3
    else:
        with pytest.raises(ValueError, match="native R/C"):
            predict(checkpoint, bundle / "learner", "R")
