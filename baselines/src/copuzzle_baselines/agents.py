"""Categorical GCBC, goal-conditioned IQL and a generic learned transition model.

The IQL losses follow the paper/OGBench definitions; data and goal semantics
are specific to the declared CoPuzzle protocol. No true action masks are used.
"""
from __future__ import annotations

import math

from flax import linen as nn, serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import Config, canonical, integer


class MLP(nn.Module):
    hidden_dims: tuple[int, ...]
    outputs: int

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, width in enumerate(self.hidden_dims):
            x = nn.relu(nn.Dense(width, name=f"hidden{i}")(x))
        return nn.Dense(self.outputs, name="output")(x)


def selected(values, actions):
    return jnp.take_along_axis(values, actions[:, None], axis=1)[:, 0]


class Agent:
    """Fixed-width models with checked updates and atomic checkpoint restoration."""
    def __init__(self, config: Config, width: int, alphabet: int, n_actions: int, seed: int):
        self.config = config
        self.width = integer(width, "width", 1)
        self.alphabet = integer(alphabet, "alphabet", 1)
        self.n_actions = integer(n_actions, "n_actions", 1)
        self.seed = integer(seed, "fit seed")
        self.step = 0
        feature_width = self.width * (self.alphabet + 1)
        if config.method == "dynamics":
            outputs = {"dynamics": self.width * self.alphabet + 1}
            input_width = feature_width + self.n_actions
        else:
            outputs = {"actor": self.n_actions}
            if config.method == "gciql":
                outputs.update(value=1, q1=self.n_actions, q2=self.n_actions)
            input_width = 2 * feature_width
        self.modules = {name: MLP(config.hidden_dims, size) for name, size in outputs.items()}
        keys = jax.random.split(jax.random.PRNGKey(self.seed), len(outputs))
        self.params = {name: module.init(key, jnp.zeros((1, input_width), jnp.float32))["params"]
                       for key, (name, module) in zip(keys, self.modules.items())}
        self.targets = {name: jax.tree_util.tree_map(jnp.copy, self.params[name])
                        for name in ("q1", "q2") if name in self.params}
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        self._update = jax.jit(self._update_impl)
        self._actor = jax.jit(lambda params, s, g: self._apply("actor", params["actor"], self._goal_features(s, g))) if config.method != "dynamics" else None
        self._forward = jax.jit(self._forward_impl) if config.method == "dynamics" else None

    def _encode(self, boards):
        return jax.nn.one_hot(boards, self.alphabet + 1).reshape(len(boards), -1)

    def _goal_features(self, states, goals):
        return jnp.concatenate((self._encode(states), self._encode(goals)), axis=-1)

    def _apply(self, name, params, inputs):
        return self.modules[name].apply({"params": params}, inputs)

    def _dynamics_logits(self, params, states, actions):
        x = jnp.concatenate((self._encode(states), jax.nn.one_hot(actions, self.n_actions)), axis=-1)
        output = self._apply("dynamics", params["dynamics"], x)
        return output[:, :-1].reshape(-1, self.width, self.alphabet), output[:, -1]

    def loss(self, params, targets, batch):
        """Public pure loss for diagnostic tests; targets/advantages are stopped."""
        c = self.config
        if c.method == "dynamics":
            logits, legal_logits = self._dynamics_logits(params, batch["states"], batch["actions"])
            transition_loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["next_states"]).mean()
            legality_loss = optax.sigmoid_binary_cross_entropy(legal_logits, batch["legal"]).mean()
            loss = transition_loss + legality_loss
            return loss, {"loss": loss, "transition_loss": transition_loss, "legality_loss": legality_loss,
                          "next_state_exact": jnp.all(logits.argmax(-1) == batch["next_states"], axis=1).mean()}
        actor_x = self._goal_features(batch["states"], batch["actor_goals"])
        logits = self._apply("actor", params["actor"], actor_x)
        logp = selected(jax.nn.log_softmax(logits), batch["actions"])
        if c.method == "gcbc":
            loss = -logp.mean()
            return loss, {"loss": loss, "actor_loss": loss}
        value_x = self._goal_features(batch["states"], batch["value_goals"])
        following_x = self._goal_features(batch["next_states"], batch["value_goals"])
        target_q = jnp.minimum(*(selected(self._apply(q, targets[q], value_x), batch["actions"]) for q in ("q1", "q2")))
        value = self._apply("value", params["value"], value_x)[:, 0]
        error = jax.lax.stop_gradient(target_q) - value
        value_loss = (jnp.where(error >= 0, c.expectile, 1 - c.expectile) * error**2).mean()
        next_value = self._apply("value", params["value"], following_x)[:, 0]
        td_target = jax.lax.stop_gradient(batch["rewards"] + c.discount * batch["bootstrap"] * next_value)
        critic_loss = sum(((selected(self._apply(q, params[q], value_x), batch["actions"]) - td_target)**2).mean() for q in ("q1", "q2"))
        actor_q = jnp.minimum(*(selected(self._apply(q, params[q], actor_x), batch["actions"]) for q in ("q1", "q2")))
        actor_value = self._apply("value", params["value"], actor_x)[:, 0]
        advantage = jax.lax.stop_gradient(actor_q - actor_value)
        weights = jnp.exp(jnp.minimum(c.temperature * advantage, math.log(100.0)))
        actor_loss = -(weights * logp).mean()
        loss = value_loss + critic_loss + actor_loss
        return loss, {"loss": loss, "actor_loss": actor_loss, "value_loss": value_loss, "critic_loss": critic_loss,
                      "td_target_mean": td_target.mean(), "advantage_mean": advantage.mean(), "actor_weight_mean": weights.mean()}

    def _update_impl(self, params, targets, opt_state, batch):
        (_, metrics), gradients = jax.value_and_grad(self.loss, has_aux=True)(params, targets, batch)
        updates, following_opt = self.optimizer.update(gradients, opt_state, params)
        following = optax.apply_updates(params, updates)
        following_targets = {name: jax.tree_util.tree_map(lambda new, old: self.config.target_rate * new + (1 - self.config.target_rate) * old,
                                                         following[name], target) for name, target in targets.items()}
        leaves = jax.tree_util.tree_leaves((following, following_targets, following_opt, gradients))
        metrics["finite"] = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves])).astype(jnp.float32)
        metrics["gradient_norm"] = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(gradients)))
        return following, following_targets, following_opt, metrics

    def _boards(self, boards, name, *, unknown=False):
        a = np.asarray(boards)
        limit = self.alphabet + int(unknown)
        if a.ndim != 2 or a.shape[1] != self.width or not len(a) or a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a >= limit):
            raise ValueError(f"{name} must be nonempty integer encoded boards of width {self.width}")
        return a.astype(np.int32)

    def prepare(self, batch):
        expected = {"states", "actions", "actor_goals"}
        if self.config.method == "dynamics":
            expected = {"states", "actions", "next_states", "legal"}
        elif self.config.method == "gciql":
            expected |= {"next_states", "value_goals", "rewards", "bootstrap"}
        if set(batch) != expected:
            raise ValueError(f"batch fields must be exactly {sorted(expected)}; true masks are not accepted")
        n = len(batch["states"])
        result = {}
        for key, value in batch.items():
            if key in {"states", "next_states", "actor_goals", "value_goals"}:
                a = self._boards(value, key)
            else:
                a = np.asarray(value)
                if a.shape != (n,):
                    raise ValueError(f"{key} must have shape ({n},)")
                if key == "actions":
                    if a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a >= self.n_actions):
                        raise ValueError("actions must be valid integer action IDs")
                    a = a.astype(np.int32)
                else:
                    if a.dtype.kind not in "fiu" or not np.all(np.isfinite(a)):
                        raise ValueError(f"{key} must be finite numeric labels")
                    a = a.astype(np.float32)
            if len(a) != n:
                raise ValueError("unaligned batch fields")
            result[key] = a
        for key in ("actor_goals", "value_goals"):
            if key in result and np.any(np.all(result["states"] == result[key], axis=1)):
                raise ValueError("initially achieved goals do not provide action/value transition samples")
        if "legal" in result and np.any((result["legal"] != 0) & (result["legal"] != 1)):
            raise ValueError("observed attempted-action legality must be binary")
        if self.config.method == "gciql":
            success = np.all(result["next_states"] == result["value_goals"], axis=1)
            if not np.array_equal(result["rewards"], np.where(success, 0, -1)) or not np.array_equal(result["bootstrap"], ~success):
                raise ValueError("goal reward/bootstrap labels disagree with observed successor equality")
        return {key: jnp.asarray(value) for key, value in result.items()}

    def update(self, batch):
        prepared = self.prepare(batch)
        params, targets, opt_state, raw = self._update(self.params, self.targets, self.opt_state, prepared)
        metrics = {key: float(value) for key, value in raw.items()}
        if metrics["finite"] != 1 or not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("nonfinite update rejected without changing learner state")
        self.params, self.targets, self.opt_state = params, targets, opt_state
        self.step += 1
        return metrics

    def actions(self, states, goals):
        if self._actor is None:
            raise ValueError("dynamics has no native actor")
        states = self._boards(states, "states", unknown=True)
        goals = self._boards(goals, "goals", unknown=True)
        if states.shape != goals.shape:
            raise ValueError("states and goals must align")
        logits = np.asarray(self._actor(self.params, states, goals))
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("nonfinite policy logits")
        return logits.argmax(axis=-1).astype(np.int32)

    def _forward_impl(self, params, states, actions):
        logits, legal_logits = self._dynamics_logits(params, states, actions)
        finite = jnp.all(jnp.isfinite(logits)) & jnp.all(jnp.isfinite(legal_logits))
        return logits.argmax(-1), legal_logits >= 0, finite

    def predict(self, states, actions):
        if self._forward is None:
            raise ValueError("this policy has no native transition predictor")
        states = self._boards(states, "states", unknown=True)
        actions = np.asarray(actions)
        if actions.shape != (len(states),) or actions.dtype.kind not in "iu" or np.any(actions < 0) or np.any(actions >= self.n_actions):
            raise ValueError("invalid prediction actions")
        result, legal, finite = self._forward(self.params, states, actions.astype(np.int32))
        if not bool(finite):
            raise FloatingPointError("nonfinite model logits")
        return np.asarray(result, dtype=np.int32), np.asarray(legal, dtype=np.uint8)

    def state_dict(self):
        return {"schema": "copuzzle.baseline_agent.v1", "config": self.config.to_dict(), "width": self.width,
                "alphabet": self.alphabet, "n_actions": self.n_actions, "seed": self.seed, "step": self.step,
                "params": serialization.to_state_dict(self.params), "targets": serialization.to_state_dict(self.targets),
                "optimizer": serialization.to_state_dict(self.opt_state)}

    def restore(self, state):
        if set(state) != set(self.state_dict()) or state["schema"] != "copuzzle.baseline_agent.v1":
            raise ValueError("invalid agent checkpoint schema")
        if canonical(state["config"]) != canonical(self.config.to_dict()) or any(state[k] != getattr(self, k) for k in ("width", "alphabet", "n_actions", "seed")):
            raise ValueError("agent checkpoint configuration or identity differs")
        step = integer(state["step"], "checkpoint step")
        restored = []
        for name, original in (("params", self.params), ("targets", self.targets), ("optimizer", self.opt_state)):
            candidate = serialization.from_state_dict(original, state[name])
            old, structure = jax.tree_util.tree_flatten(original)
            new, new_structure = jax.tree_util.tree_flatten(candidate)
            if structure != new_structure or any(np.shape(a) != np.shape(b) or np.asarray(a).dtype != np.asarray(b).dtype for a, b in zip(old, new)):
                raise ValueError(f"checkpoint {name} shape/dtype mismatch")
            if any(not np.all(np.isfinite(x)) for x in new):
                raise ValueError(f"checkpoint {name} is nonfinite")
            restored.append(jax.tree_util.tree_map(jnp.asarray, candidate))
        self.params, self.targets, self.opt_state = restored
        self.step = step
