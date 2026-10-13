"""Data-to-policy transfer plumbing; these fixtures are not learned results.

A four-weight node scorer keeps the same parameters across all sizes. Only
inference RNG/cache/history change. No neural library or CoPuzzleLab is used.
"""
from copy import deepcopy

import numpy as np
import pytest

from copuzzle.games.lights_out.bundle import (
    BundleGoalSampler, load_offline_bundle, write_offline_bundle,
)
from copuzzle.games.lights_out.data import generate_dataset
from copuzzle.games.lights_out.evaluation import (
    evaluate_policy, generate_task_suite, load_task_suite, save_task_suite,
    with_structure,
)
from copuzzle.games.lights_out.rules import get_rule, graph_rule
from copuzzle.games.lights_out.structure import export_structure


class FixedParameterNodePolicy:
    """Untrained shared node scorer used only to exercise public inputs."""

    def __init__(self):
        self.weights = np.array([0.1, 0.2, 1.0, 0.05], dtype=np.float64)
        self.rng = np.random.default_rng(37)
        self.cache = {}
        self.episodes = []

    def scores(self, observation):
        board = np.asarray(observation['board'], dtype=np.float64)
        goal = np.asarray(observation['goal'], dtype=np.float64)
        size = len(board)
        assert goal.shape == board.shape == (size,)
        if 'adjacency' in observation:
            adjacency = observation['adjacency']
            assert adjacency.shape == (size, size)
            positional = adjacency.sum(axis=1) / max(size - 1, 1)
        else:
            shape = observation['grid_shape']
            coordinates = observation['grid_coordinates']
            assert shape.shape == (2,) and coordinates.shape == (size, 2)
            assert int(np.prod(shape)) == size
            positional = (coordinates / np.maximum(shape - 1, 1)).sum(axis=1)
        features = np.column_stack((board, goal, board != goal, positional))
        self.cache['features'] = features.copy()
        return features @ self.weights

    def reset(self):
        self.cache.clear()
        self.episodes.append([])

    def __call__(self, observation):
        scores = self.scores(observation)
        # RNG and cache may advance during frozen inference; weights must not.
        candidates = np.flatnonzero(scores == scores.max())
        draw = self.rng.random()
        action = int(candidates[min(int(draw * len(candidates)), len(candidates) - 1)])
        self.episodes[-1].append((deepcopy(observation), action))
        return action


def _nine_node_training_bundle(directory):
    directory.mkdir()
    rule = get_rule('grid-3x3')
    for split in ('train', 'val'):
        generate_dataset(rule, num_episodes=2, horizon=4, seed=13, split=split).save(directory / f'{split}.npz')
    write_offline_bundle(directory, train_path=directory / 'train.npz',
                         val_path=directory / 'val.npz', rule=rule)
    return load_offline_bundle(directory)


def _persisted_suite(rule, directory):
    # Two short cases are sufficient for interface acceptance, not performance.
    legacy = generate_task_suite(rule, per_depth=1, depths=(1, 2), seed=41, max_episode_steps=3)
    suite = with_structure(legacy, rule=rule)
    path = save_task_suite(suite, directory / 'tasks.json')
    return load_task_suite(path)


def test_one_node_policy_from_nine_node_bundle_to_all_larger_tasks(tmp_path):
    training = _nine_node_training_bundle(tmp_path / 'training')
    batch = BundleGoalSampler(training, seed=19).sample(3)
    assert batch['observations'].shape == batch['goals'].shape == (3, 9)
    policy = FixedParameterNodePolicy()
    policy_id, weights_id = id(policy), id(policy.weights)
    weights = policy.weights.copy()
    training_input = {
        'board': batch['observations'][0], 'goal': batch['goals'][0],
        'previous_event': batch['previous_events'][0], **batch['structure'],
    }
    assert policy.scores(training_input).shape == (9,)
    rng_before = deepcopy(policy.rng.bit_generator.state)
    tasks = [f'grid-4x{cols}' for cols in (4, 5, 6)]
    tasks += [f'graph-{topology}{size}' for topology in ('path', 'cycle') for size in (16, 20, 24)]
    for task in tasks:
        target = get_rule(task)
        expected_structure = export_structure(target)
        expected_fields = expected_structure.observation_fields()
        suite = _persisted_suite(target, tmp_path / task)
        first_episode = len(policy.episodes)
        result = evaluate_policy(
            policy, suite, evaluation_mode='frozen_transfer',
            training_rule_ids=[training.structure.rule_id],
            method_metadata={'test_fixture': 'untrained_fixed_parameter_node_scorer'},
        )
        assert id(policy) == policy_id and id(policy.weights) == weights_id
        np.testing.assert_array_equal(policy.weights, weights)
        assert result['evaluation_mode'] == 'frozen_transfer'
        assert result['training_rule_ids'] == [training.structure.rule_id]
        assert result['structure_sha256'] == expected_structure.structure_sha256
        assert result['rule_id'] == target.rule_id != training.structure.rule_id
        assert result['parameter_freezing_enforced'] is False  # Our caller checks it.
        histories = policy.episodes[first_episode:]
        assert len(histories) == len(suite['cases']) == 2
        for history, case in zip(histories, suite['cases']):
            assert history
            for observation, action in history:
                assert set(observation) == {'board', 'goal', 'previous_event'} | set(expected_fields)
                assert observation['board'].shape == (target.n_buttons,)
                np.testing.assert_array_equal(observation['goal'], case['goal'])
                for field, expected in expected_fields.items():
                    np.testing.assert_array_equal(observation[field], expected)
                assert 0 <= action < target.n_buttons
        assert policy.cache['features'].shape == (target.n_buttons, len(weights))
    assert policy.rng.bit_generator.state != rng_before
    assert len(policy.episodes) == 18


def test_fixed_width_policy_rejects_larger_input_without_reinitialization(tmp_path):
    training = _nine_node_training_bundle(tmp_path / 'training')
    batch = BundleGoalSampler(training, seed=11).sample(2)

    class FixedWidthPolicy:
        def __init__(self, size):
            self.weights = np.arange(size, dtype=np.float64)
            self.calls = 0

        def __call__(self, observation):
            self.calls += 1
            size = len(observation['board'])
            if size != len(self.weights):
                raise ValueError(f'fixed-width policy requires {len(self.weights)} nodes; got {size}')
            return int(np.argmax(self.weights))

    policy = FixedWidthPolicy(batch['observations'].shape[1])
    weights_id, saved = id(policy.weights), policy.weights.copy()
    suite = _persisted_suite(get_rule('grid-4x4'), tmp_path / 'target')
    with pytest.raises(ValueError, match='requires 9 nodes; got 16'):
        evaluate_policy(policy, suite, evaluation_mode='frozen_transfer',
                        training_rule_ids=[training.structure.rule_id])
    assert policy.calls == 1
    assert id(policy.weights) == weights_id
    np.testing.assert_array_equal(policy.weights, saved)


def test_grid_like_custom_graph_name_preserves_graph_structure(tmp_path, monkeypatch):
    from copuzzle.games.lights_out import evaluation

    graph = graph_rule(get_rule('graph-cycle6').adjacency, name='grid-2x3')
    public = export_structure(graph)
    assert public.family == 'graph'
    assert set(public.observation_fields()) == {'adjacency'}
    assert graph.rule_id != get_rule('grid-2x3').rule_id
    legacy = generate_task_suite(graph, per_depth=1, depths=(1,), seed=3, max_episode_steps=2)
    path = save_task_suite(with_structure(legacy, rule=graph), tmp_path / 'tasks.json')

    def forbidden_name_lookup(*args, **kwargs):
        raise AssertionError('A structure suite must not infer its family from its task name')

    monkeypatch.setattr(evaluation, 'get_rule', forbidden_name_lookup)
    suite = load_task_suite(path)
    assert suite['family'] == 'graph'
    policy = FixedParameterNodePolicy()
    result = evaluate_policy(policy, suite)
    assert result['structure_sha256'] == public.structure_sha256
    assert result['actual_observation_fields'] == ['adjacency', 'board', 'goal', 'previous_event']
    for history in policy.episodes:
        for observation, action in history:
            np.testing.assert_array_equal(observation['adjacency'], graph.adjacency)
            assert 'grid_shape' not in observation and 'grid_coordinates' not in observation
            assert 0 <= action < 6
