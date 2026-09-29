"""Offline effect and sequence diagnostics, separate from goal solving."""

import numpy as np
import pytest

from copuzzle.data import generate_dataset
from copuzzle.diagnostics import EffectLookup, composition_examples, prediction_metrics
from copuzzle.rules import get_rule


def recorded_dataset(actions):
    """Record a chosen action schedule; only the returned data is fitted."""
    rule = get_rule("grid-3x3")
    actions = np.asarray(actions, dtype=np.int64)
    dataset = generate_dataset(
        rule, num_episodes=actions.shape[0], horizon=actions.shape[1], seed=14
    )
    dataset.actions[:] = actions
    dataset.events[:] = np.eye(rule.n_buttons, dtype=np.uint8)[actions]
    for episode in range(dataset.num_episodes):
        for step in range(dataset.horizon):
            dataset.states[episode, step + 1] = rule.transition(
                dataset.states[episode, step], int(actions[episode, step])
            )
    dataset.validate()
    return dataset


@pytest.fixture
def dataset():
    return recorded_dataset([list(range(9)), list(reversed(range(9)))])


def test_prediction_metrics_separate_exact_changed_and_unchanged_errors():
    states = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1]], dtype=np.uint8)
    targets = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=np.uint8)
    predicted = np.array([[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 1]], dtype=np.uint8)
    metrics = prediction_metrics(predicted, states, targets)
    assert metrics == pytest.approx({
        "next_exact_accuracy": 1 / 3,
        "effect_exact_accuracy": 1 / 3,
        "bit_accuracy": 10 / 12,
        "changed_bit_accuracy": 2 / 3,
        "unchanged_bit_accuracy": 8 / 9,
    })


def test_metrics_report_none_for_absent_bit_category():
    states = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    unchanged = prediction_metrics(states, states, states)
    assert unchanged["changed_bit_accuracy"] is None
    assert unchanged["unchanged_bit_accuracy"] == 1.0
    changed = prediction_metrics(1 - states, states, 1 - states)
    assert changed["changed_bit_accuracy"] == 1.0
    assert changed["unchanged_bit_accuracy"] is None
    assert changed["next_exact_accuracy"] == changed["effect_exact_accuracy"] == 1.0


@pytest.mark.parametrize("argument", ["predicted_next", "states", "next_states"])
@pytest.mark.parametrize(
    "invalid",
    [np.zeros((0, 3)), np.zeros((2, 0)), np.zeros(3), np.zeros((2, 1, 3)),
     np.full((2, 3), 0.5), np.full((2, 3), 2), np.full((2, 3), np.nan),
     np.full((2, 3), "0"), np.full((2, 3), 1 + 0j)],
)
def test_metrics_strictly_validate_all_inputs(argument, invalid):
    inputs = {name: np.zeros((2, 3), dtype=np.uint8)
              for name in ("predicted_next", "states", "next_states")}
    inputs[argument] = invalid
    with pytest.raises(ValueError):
        prediction_metrics(**inputs)


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        prediction_metrics(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((1, 3)))


def test_composition_windows_are_recorded_and_never_cross_episodes(dataset):
    length = 4
    examples = composition_examples(dataset, length=length, num_examples=100, seed=9)
    assert set(examples) == {"states", "events", "next_states"}
    assert examples["states"].shape == examples["next_states"].shape == (100, 9)
    assert examples["events"].shape == (100, length, 9)

    def key(state, events, successor):
        return state.tobytes(), events.tobytes(), successor.tobytes()

    recorded = {
        key(dataset.states[episode, start],
            dataset.events[episode, start:start + length],
            dataset.states[episode, start + length])
        for episode in range(dataset.num_episodes)
        for start in range(dataset.horizon - length + 1)
    }
    for state, events, successor in zip(
        examples["states"], examples["events"], examples["next_states"]
    ):
        assert key(state, events, successor) in recorded


def test_composition_can_sample_whole_trajectories_with_replacement(dataset):
    examples = composition_examples(dataset, length=dataset.horizon, num_examples=12, seed=7)
    for state, events, successor in zip(
        examples["states"], examples["events"], examples["next_states"]
    ):
        assert any(
            np.array_equal(state, dataset.states[episode, 0])
            and np.array_equal(events, dataset.events[episode])
            and np.array_equal(successor, dataset.states[episode, -1])
            for episode in range(dataset.num_episodes)
        )


def test_composition_sampling_is_seeded_and_output_is_isolated(dataset):
    original_states = dataset.states.copy()
    original_events = dataset.events.copy()
    first = composition_examples(dataset, seed=33)
    second = composition_examples(dataset, seed=33)
    for name in first:
        assert first[name].dtype == np.uint8
        np.testing.assert_array_equal(first[name], second[name])
        first[name].fill(0)
    np.testing.assert_array_equal(dataset.states, original_states)
    np.testing.assert_array_equal(dataset.events, original_events)


@pytest.mark.parametrize(
    "kwargs",
    [{"length": 0}, {"length": 10}, {"length": True}, {"length": 1.5},
     {"num_examples": 0}, {"num_examples": -1}, {"num_examples": True},
     {"seed": -1}, {"seed": True}, {"seed": 1.5}],
)
def test_composition_rejects_invalid_configuration(dataset, kwargs):
    with pytest.raises(ValueError):
        composition_examples(dataset, **kwargs)


def test_lookup_learns_effects_from_transitions_on_other_backgrounds(dataset):
    lookup = EffectLookup()
    assert lookup.fit(dataset) is lookup
    states = np.eye(9, dtype=np.uint8)
    events = np.eye(9, dtype=np.uint8)
    predicted = lookup.predict(states, events)
    rule = get_rule("grid-3x3")
    expected = np.stack([rule.transition(state, action) for action, state in enumerate(states)])
    np.testing.assert_array_equal(predicted, expected)
    assert predicted.dtype == np.uint8


def test_lookup_predicts_recorded_compositions_and_cancellation(dataset):
    lookup = EffectLookup().fit(dataset)
    examples = composition_examples(dataset, length=5, num_examples=24, seed=8)
    predicted = lookup.predict_sequence(examples["states"], examples["events"])
    np.testing.assert_array_equal(predicted, examples["next_states"])

    states = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
    repeated = np.eye(9, dtype=np.uint8)[[2, 2]][None]
    np.testing.assert_array_equal(lookup.predict_sequence(states, repeated), states)
    forward = np.eye(9, dtype=np.uint8)[[0, 4]][None]
    reverse = np.eye(9, dtype=np.uint8)[[4, 0]][None]
    np.testing.assert_array_equal(
        lookup.predict_sequence(states, forward), lookup.predict_sequence(states, reverse)
    )


def test_conflicting_effects_are_rejected(dataset):
    dataset.states[1, 1, 0] ^= 1
    with pytest.raises(ValueError, match="Conflicting observed effects for action"):
        EffectLookup().fit(dataset)


def test_uncovered_actions_fail_in_single_and_sequence_prediction():
    dataset = recorded_dataset([[0, 0, 0]])
    lookup = EffectLookup().fit(dataset)
    states = np.zeros((1, 9), dtype=np.uint8)
    known = np.eye(9, dtype=np.uint8)[[0]]
    unknown = np.eye(9, dtype=np.uint8)[[1]]
    lookup.predict(states, known)
    with pytest.raises(ValueError, match="Uncovered action"):
        lookup.predict(states, unknown)
    with pytest.raises(ValueError, match="Uncovered action"):
        lookup.predict_sequence(states, np.stack([known, unknown], axis=1))


def test_refit_replaces_previous_coverage(dataset):
    lookup = EffectLookup().fit(dataset)
    lookup.fit(recorded_dataset([[0, 0]]))
    with pytest.raises(ValueError, match="Uncovered action"):
        lookup.predict(np.zeros((1, 9), dtype=np.uint8), np.eye(9, dtype=np.uint8)[[1]])


def test_lookup_fit_and_outputs_do_not_alias_dataset(dataset):
    lookup = EffectLookup().fit(dataset)
    states = dataset.states[0, :-1].copy()
    events = dataset.events[0].copy()
    expected = dataset.states[0, 1:].copy()
    dataset.states.fill(0)
    predicted = lookup.predict(states, events)
    np.testing.assert_array_equal(predicted, expected)
    predicted.fill(0)
    np.testing.assert_array_equal(lookup.predict(states, events), expected)


def test_lookup_requires_fit_and_rejects_rule_as_training_input():
    lookup = EffectLookup()
    with pytest.raises(RuntimeError, match="fit"):
        lookup.predict(np.zeros((1, 9)), np.eye(9)[[0]])
    with pytest.raises(TypeError, match="TrajectoryDataset"):
        lookup.fit(get_rule("grid-3x3"))


@pytest.mark.parametrize(
    "events",
    [np.zeros((1, 9)), np.ones((1, 9)), np.full((1, 9), 0.5),
     np.eye(9, dtype=np.uint8)[[0, 1]], np.zeros((1, 2, 9)), np.zeros((0, 9))],
)
def test_lookup_rejects_invalid_single_events(dataset, events):
    lookup = EffectLookup().fit(dataset)
    with pytest.raises(ValueError):
        lookup.predict(np.zeros((1, 9), dtype=np.uint8), events)


def test_lookup_rejects_invalid_states_and_empty_sequences(dataset):
    lookup = EffectLookup().fit(dataset)
    with pytest.raises(ValueError):
        lookup.predict(np.zeros((1, 8)), np.eye(9)[[0]])
    with pytest.raises(ValueError):
        lookup.predict(np.full((1, 9), 0.5), np.eye(9)[[0]])
    with pytest.raises(ValueError):
        lookup.predict_sequence(np.zeros((1, 9)), np.zeros((1, 0, 9)))
    with pytest.raises(ValueError):
        lookup.predict_sequence(np.zeros((1, 9)), np.eye(9)[[0]])
