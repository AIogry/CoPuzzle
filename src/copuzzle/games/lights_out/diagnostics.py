"""Offline diagnostics for transition effects and their composition.

These tools consume recorded trajectories, never an environment or rule matrix.
EffectLookup is a fixed-instance effect table, not RL or a goal-solving policy.
Its compositional prediction assumes that each action has a state-independent
XOR effect. Fitting checks that assumption on the supplied observations only.
"""

from __future__ import annotations

import numpy as np

from .data import TrajectoryDataset


def _binary_array(value, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}D array")
    if array.dtype.kind not in "biuf" or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return array.astype(np.uint8, copy=False)


def _integer(value, name: str, minimum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validate_dataset(dataset: TrajectoryDataset) -> None:
    if not isinstance(dataset, TrajectoryDataset):
        raise TypeError("dataset must be a TrajectoryDataset")
    dataset.validate()


def prediction_metrics(predicted_next, states, next_states) -> dict[str, float | None]:
    """Compare binary predictions and targets, all shaped [batch, buttons].

    Exact accuracy averages over examples; bit accuracies average over bits.
    Changed/unchanged masks come from the true state transition. An empty bit
    category has accuracy None. With the same known initial state, next-state
    exact accuracy and XOR-effect exact accuracy are necessarily equal; they
    are two descriptions of the same prediction, not independent evidence.
    """
    predicted = _binary_array(predicted_next, "predicted_next", 2)
    states = _binary_array(states, "states", 2)
    targets = _binary_array(next_states, "next_states", 2)
    if predicted.shape != states.shape or targets.shape != states.shape:
        raise ValueError("predicted_next, states and next_states must have identical shapes")

    correct = predicted == targets
    effect = np.bitwise_xor(states, targets)
    predicted_effect = np.bitwise_xor(states, predicted)
    changed = effect.astype(bool)
    unchanged = ~changed
    return {
        "next_exact_accuracy": float(np.all(correct, axis=-1).mean()),
        "effect_exact_accuracy": float(np.all(predicted_effect == effect, axis=-1).mean()),
        "bit_accuracy": float(correct.mean()),
        "changed_bit_accuracy": float(correct[changed].mean()) if changed.any() else None,
        "unchanged_bit_accuracy": float(correct[unchanged].mean()) if unchanged.any() else None,
    }


def composition_examples(
    dataset: TrajectoryDataset,
    length: int = 2,
    num_examples: int = 128,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Sample recorded, contiguous event sequences without crossing episodes.

    Sampling is uniform over valid windows, with replacement. The result is
    initial states [B,N], events [B,K,N] and final next_states [B,N]. This is a
    sequence-prediction diagnostic; it neither creates nor guarantees unseen
    combinations. Callers must enforce any training/test separation separately.
    """
    _validate_dataset(dataset)
    length = _integer(length, "length", 1)
    num_examples = _integer(num_examples, "num_examples", 1)
    seed = _integer(seed, "seed", 0)
    if length > dataset.horizon:
        raise ValueError("length cannot exceed the dataset trajectory horizon")

    windows_per_episode = dataset.horizon - length + 1
    rng = np.random.default_rng(seed)
    window_ids = rng.integers(
        0, dataset.num_episodes * windows_per_episode, size=num_examples
    )
    episodes = window_ids // windows_per_episode
    starts = window_ids % windows_per_episode
    offsets = starts[:, None] + np.arange(length)[None, :]
    return {
        "states": dataset.states[episodes, starts].copy(),
        "events": dataset.events[episodes[:, None], offsets].copy(),
        "next_states": dataset.states[episodes, starts + length].copy(),
    }


class EffectLookup:
    """Learn one XOR effect per observed action from offline transitions.

    Only recorded states, actions and successors determine the effect table.
    The current trajectory schema uses one-hot events, which prediction also
    requires. Missing actions and contradictory effects fail explicitly. The
    table says nothing about graph transfer, learned embeddings, or a policy's
    ability to select operations to reach goals.
    """

    def __init__(self) -> None:
        self._effects: np.ndarray | None = None
        self._covered: np.ndarray | None = None

    def fit(self, dataset: TrajectoryDataset) -> EffectLookup:
        """Replace the table with effects supported by this dataset."""
        _validate_dataset(dataset)
        n_buttons = dataset.n_buttons
        observed_effects = np.bitwise_xor(
            dataset.states[:, :-1], dataset.states[:, 1:]
        ).reshape(-1, n_buttons)
        actions = dataset.actions.reshape(-1)
        effects = np.zeros((n_buttons, n_buttons), dtype=np.uint8)
        covered = np.zeros(n_buttons, dtype=bool)
        for action in np.unique(actions):
            candidates = observed_effects[actions == action]
            if np.any(candidates != candidates[0]):
                raise ValueError(f"Conflicting observed effects for action {int(action)}")
            effects[action] = candidates[0]
            covered[action] = True
        self._effects = effects
        self._covered = covered
        return self

    def _inputs(self, states, events, sequence: bool) -> tuple[np.ndarray, np.ndarray]:
        if self._effects is None or self._covered is None:
            raise RuntimeError("Call fit(dataset) before prediction")
        states = _binary_array(states, "states", 2)
        events = _binary_array(events, "events", 3 if sequence else 2)
        if states.shape[1] != self._effects.shape[1]:
            raise ValueError("states button count does not match the fitted dataset")
        if events.shape[0] != states.shape[0] or events.shape[-1] != states.shape[1]:
            raise ValueError("events batch and button dimensions must match states")
        if not np.all(events.sum(axis=-1) == 1):
            raise ValueError("events must be one-hot encodings of single-button actions")
        actions = events.argmax(axis=-1)
        missing = np.unique(actions[~self._covered[actions]])
        if missing.size:
            raise ValueError(f"Uncovered action(s) in fitted data: {missing.tolist()}")
        return states, actions

    def predict(self, states, events) -> np.ndarray:
        """Predict successors from states [B,N] and one-hot events [B,N]."""
        states, actions = self._inputs(states, events, sequence=False)
        return np.bitwise_xor(states, self._effects[actions])

    def predict_sequence(self, states, events) -> np.ndarray:
        """Predict endpoints for given one-hot event sequences [B,K,N]."""
        states, actions = self._inputs(states, events, sequence=True)
        combined_effect = np.bitwise_xor.reduce(self._effects[actions], axis=1)
        return np.bitwise_xor(states, combined_effect)
