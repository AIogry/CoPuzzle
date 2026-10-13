"""Prediction inputs, sequence alignment and honest method reports."""
from copy import deepcopy

import numpy as np
import pytest

from copuzzle.games.lights_out.data import generate_dataset
from copuzzle.games.lights_out.diagnostics import EffectLookup
from copuzzle.games.lights_out.prediction import evaluate_predictor
from copuzzle.games.lights_out.rules import get_rule, graph_rule
from copuzzle.games.lights_out.structure import export_structure


class PublicRulePredictor:
    uses_analytic_rules = True
    uses_solver = False
    uses_environment_queries = False

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, observation, action):
        board = observation["board"].copy()
        board[action] ^= 1
        if "adjacency" in observation:
            assert set(observation) == {"board", "adjacency"}
            board ^= observation["adjacency"][:, action]
        else:
            assert set(observation) == {"board", "grid_shape", "grid_coordinates"}
            assert observation["grid_shape"].dtype == np.int32
            coords = observation["grid_coordinates"]
            neighbors = np.abs(coords - coords[action]).sum(axis=1) == 1
            board[neighbors] ^= 1
        return board


@pytest.mark.parametrize("rule", [get_rule("grid-2x3"), get_rule("graph-cycle9"),
                                      graph_rule(np.zeros((6, 6), dtype=np.uint8), name="grid-2x3")])
def test_public_rule_predictor_receives_only_allowed_inputs_and_scores_every_step(rule):
    dataset = generate_dataset(rule, num_episodes=3, horizon=5)
    model = PublicRulePredictor()
    report = evaluate_predictor(model, dataset, structure=export_structure(rule),
                                sequence_length=3, num_examples=20, seed=8)
    assert report["schema_version"] == "copuzzle.prediction.v1"
    assert report["predictor_calls"] == 60
    assert model.resets == 20
    assert set(report["by_step"]) == {"1", "2", "3"}
    for metrics in report["by_step"].values():
        assert metrics["next_exact_accuracy"] == 1.0
        assert metrics["effect_exact_accuracy"] == 1.0
        assert metrics["bit_accuracy"] == 1.0
    assert report["endpoint"] == report["by_step"]["3"]
    assert all(row["bit_errors_by_step"] == [0, 0, 0] for row in report["examples"])
    assert report["method_metadata"]["uses_analytic_rules"] is True
    assert report["parameter_freezing_enforced"] is False
    assert "not guaranteed" in report["sampling"]
    assert "goal" not in report["actual_observation_fields"]


def test_offline_effect_lookup_adapter_is_perfect_without_rule_in_fit():
    rule = get_rule("grid-2x3")
    training = generate_dataset(rule, num_episodes=20, horizon=8, seed=19)
    validation = generate_dataset(rule, num_episodes=3, horizon=6, seed=20, split="val")
    model = EffectLookup().fit(training)

    def predictor(observation, action):
        event = np.zeros((1, len(observation["board"])), dtype=np.uint8)
        event[0, action] = 1
        return model.predict(observation["board"][None, :], event)[0]

    report = evaluate_predictor(predictor, validation, structure=export_structure(rule),
                                sequence_length=4, num_examples=9,
                                method_metadata={"uses_analytic_rules": False})
    assert report["endpoint"]["next_exact_accuracy"] == 1
    assert report["method_metadata"]["uses_analytic_rules"] is False
    assert report["method_metadata"]["uses_solver"] is None


def test_later_steps_receive_predictions_and_target_sampling_stays_within_episode():
    rule = get_rule("grid-2x3")
    dataset = generate_dataset(rule, num_episodes=2, horizon=5, seed=7)
    seen = []

    class ZeroPredictor:
        def reset(self):
            seen.append([])

        def __call__(self, observation, action):
            seen[-1].append((observation["board"].copy(), action))
            return np.zeros(rule.n_buttons, dtype=np.uint8)

    report = evaluate_predictor(ZeroPredictor(), dataset, structure=export_structure(rule),
                                sequence_length=3, num_examples=7)
    for calls, row in zip(seen, report["examples"]):
        episode, start = row["episode"], row["start_step"]
        assert 0 <= start <= dataset.horizon - 3
        np.testing.assert_array_equal(calls[0][0], dataset.states[episode, start])
        assert all(not call[0].any() for call in calls[1:])
        assert [call[1] for call in calls] == dataset.actions[episode, start:start + 3].tolist()
        expected = np.count_nonzero(dataset.states[episode, start + 1:start + 4], axis=1)
        assert row["bit_errors_by_step"] == expected.tolist()
    assert report["endpoint"]["bit_accuracy"] < 1


def test_reproducible_sampling_and_independent_copies():
    rule = get_rule("graph-path9")
    dataset = generate_dataset(rule, num_episodes=2, horizon=4)
    original = deepcopy(dataset)
    structure = export_structure(rule)

    def modifies_inputs(observation, action):
        assert np.array_equal(observation["adjacency"], structure.adjacency)
        observation["adjacency"][:] = 0
        observation["board"][:] = 0
        return observation["board"]

    first = evaluate_predictor(modifies_inputs, dataset, structure=structure,
                               sequence_length=2, num_examples=12, seed=9)
    second = evaluate_predictor(modifies_inputs, dataset, structure=structure,
                                sequence_length=2, num_examples=12, seed=9)
    first.pop("inference_seconds")
    second.pop("inference_seconds")
    assert first == second
    np.testing.assert_array_equal(dataset.states, original.states)
    np.testing.assert_array_equal(dataset.actions, original.actions)
    np.testing.assert_array_equal(dataset.events, original.events)


@pytest.mark.parametrize("value", [np.zeros((1, 6)), np.zeros(5), [2] * 6,
                                     [float("nan")] * 6, ["0"] * 6])
def test_malformed_prediction_fails(value):
    rule = get_rule("grid-2x3")
    dataset = generate_dataset(rule, num_episodes=1, horizon=2)
    with pytest.raises(ValueError):
        evaluate_predictor(lambda obs, action: value, dataset, structure=export_structure(rule))


@pytest.mark.parametrize("kwargs", [
    {"sequence_length": 0}, {"sequence_length": True}, {"sequence_length": 3},
    {"num_examples": 0}, {"num_examples": 1.5}, {"seed": -1},
    {"method_metadata": {"endpoint": "replace"}},
    {"method_metadata": {"uses_solver": "unknown"}},
])
def test_invalid_configuration_fails_before_predictor(kwargs):
    rule = get_rule("grid-2x3")
    dataset = generate_dataset(rule, num_episodes=1, horizon=2)

    def must_not_run(*args):
        pytest.fail("Invalid configuration should fail before predictor calls")

    with pytest.raises(ValueError):
        evaluate_predictor(must_not_run, dataset, structure=export_structure(rule), **kwargs)


def test_incompatible_structure_rejected_before_prediction():
    rule = get_rule("grid-2x3")
    dataset = generate_dataset(rule, num_episodes=1, horizon=2)
    other = graph_rule(np.zeros((6, 6), dtype=np.uint8))
    with pytest.raises(ValueError, match="same ordered rule"):
        evaluate_predictor(PublicRulePredictor(), dataset, structure=export_structure(other))


def test_analytic_reference_cannot_claim_to_be_learned():
    rule = get_rule("grid-2x3")
    dataset = generate_dataset(rule, num_episodes=1, horizon=2)
    with pytest.raises(ValueError, match="Cannot deny"):
        evaluate_predictor(PublicRulePredictor(), dataset, structure=export_structure(rule),
                           method_metadata={"uses_analytic_rules": False})
