"""Offline permutation artifacts, replay, and goal relabeling contracts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from copuzzle.games.permutation.data import (
    GoalSampler, SCHEMA_VERSION, TrajectoryDataset, generate_dataset,
    load_dataset, relabel_transitions, sample_goal_transitions,
)
from copuzzle.games.permutation.rules import get_rule


def _assert_batches_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def _rewrite_archive(path, **arrays):
    """Recompute the checksum to exercise validation beneath the hash layer."""
    np.savez_compressed(path, **arrays)
    manifest_path = path.with_suffix('.json')
    manifest = json.loads(manifest_path.read_text())
    manifest['artifact']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize('task,n_tokens,n_actions', [
    ('pancake-2', 2, 1), ('pancake-5', 5, 4),
    ('topspin-6-3', 6, 3), ('topspin-8-4', 8, 3),
])
def test_generation_retains_successors_and_replays_every_operation(task, n_tokens, n_actions):
    rule = get_rule(task)
    dataset = generate_dataset(rule, num_episodes=4, horizon=9, seed=31)
    assert dataset.num_episodes == 4
    assert dataset.horizon == 9
    assert dataset.n_tokens == n_tokens
    assert dataset.n_actions == n_actions
    assert dataset.states.shape == (4, 10, n_tokens)
    assert dataset.events.shape == (4, 9, n_actions)
    assert dataset.states.dtype == np.int32
    assert dataset.actions.dtype == np.int64
    assert dataset.events.dtype == np.uint8
    assert dataset.manifest['schema_version'] == SCHEMA_VERSION
    assert dataset.manifest['rule_spec'] == rule.spec()
    for episode in range(4):
        for step in range(9):
            action = int(dataset.actions[episode, step])
            np.testing.assert_array_equal(
                dataset.states[episode, step + 1], rule.transition(dataset.states[episode, step], action))
            np.testing.assert_array_equal(dataset.events[episode, step], rule.event(action))
    dataset.validate(rule)


def test_deterministic_independent_split_streams():
    rule = get_rule('pancake-5')
    options = dict(num_episodes=14, horizon=11, seed=37)
    train = generate_dataset(rule, split='train', **options)
    repeat = generate_dataset(rule, split='train', **options)
    val = generate_dataset(rule, split='val', **options)
    for name in ('states', 'actions', 'events'):
        np.testing.assert_array_equal(getattr(train, name), getattr(repeat, name))
    assert train.manifest == repeat.manifest
    assert not np.array_equal(train.actions, val.actions)
    assert not np.array_equal(train.states[:, 0], val.states[:, 0])
    assert 'states may overlap' in train.manifest['split_semantics']
    # The initial-state stream is independent of the number of collected actions.
    longer = generate_dataset(rule, num_episodes=14, horizon=12, seed=37)
    np.testing.assert_array_equal(train.states[:, 0], longer.states[:, 0])


@pytest.mark.parametrize('options', [
    {'num_episodes': 0}, {'num_episodes': True}, {'num_episodes': 1.5},
    {'horizon': 0}, {'horizon': -1}, {'horizon': False},
    {'seed': -1}, {'seed': True}, {'split': ''}, {'split': ' '},
])
def test_invalid_generation_options(options):
    with pytest.raises(ValueError):
        generate_dataset(get_rule('pancake-4'), **options)


def test_structure_validation_is_separate_from_privileged_replay():
    rule = get_rule('topspin-6-3')
    dataset = generate_dataset(rule, num_episodes=1, horizon=3)
    dataset.states[0, -1, :2] = dataset.states[0, -1, :2][::-1]
    dataset.validate()
    with pytest.raises(ValueError, match='Rule transition mismatch'):
        dataset.validate(rule)
    with pytest.raises(ValueError, match='identity'):
        dataset.validate(get_rule('topspin-6-4'))


@pytest.mark.parametrize('corruption', [
    'missing_successor', 'float_states', 'binary_states', 'duplicate_token', 'zero_token',
    'out_of_range_token', 'float_actions', 'invalid_action', 'missing_event',
    'event_disagreement', 'event_width', 'manifest_actions', 'manifest_horizon',
    'manifest_rule_id', 'manifest_rule_spec', 'old_binary_schema', 'extra_field',
])
def test_strict_schema_validation(corruption):
    dataset = generate_dataset(get_rule('topspin-6-3'), num_episodes=2, horizon=4)
    if corruption == 'missing_successor':
        dataset.states = dataset.states[:, :-1]
    elif corruption == 'float_states':
        dataset.states = dataset.states.astype(np.float32)
    elif corruption == 'binary_states':
        dataset.states = (dataset.states % 2).astype(np.int32)
    elif corruption == 'duplicate_token':
        dataset.states[0, 0, 0] = dataset.states[0, 0, 1]
    elif corruption == 'zero_token':
        dataset.states[0, 0, 0] = 0
    elif corruption == 'out_of_range_token':
        dataset.states[0, 0, 0] = 7
    elif corruption == 'float_actions':
        dataset.actions = dataset.actions.astype(np.float64)
    elif corruption == 'invalid_action':
        dataset.actions[0, 0] = 3
    elif corruption == 'missing_event':
        dataset.events[0, 0] = 0
    elif corruption == 'event_disagreement':
        dataset.events[0, 0] = np.roll(dataset.events[0, 0], 1)
    elif corruption == 'event_width':
        dataset.events = np.pad(dataset.events, ((0, 0), (0, 0), (0, 3)))
    elif corruption == 'manifest_actions':
        dataset.manifest['n_actions'] = 6
    elif corruption == 'manifest_horizon':
        dataset.manifest['horizon'] = True
    elif corruption == 'manifest_rule_id':
        dataset.manifest['rule_id'] = 'fake-rule'
    elif corruption == 'manifest_rule_spec':
        dataset.manifest['rule_spec']['window'] = 4
    elif corruption == 'old_binary_schema':
        dataset.manifest['schema_version'] = 'copuzzle.trajectory.v1'
    elif corruption == 'extra_field':
        dataset.manifest['solver_solutions'] = []
    with pytest.raises(ValueError):
        dataset.validate()


def test_roundtrip_checks_identity_shapes_and_hash_without_executing_rules(tmp_path, monkeypatch):
    from copuzzle.games.permutation import rules

    rule = get_rule('topspin-8-4')
    dataset = generate_dataset(rule, num_episodes=3, horizon=7, split='val')
    path = dataset.save(tmp_path / 'validation.npz')
    manifest = json.loads(path.with_suffix('.json').read_text())
    assert manifest['artifact']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest['artifact']['arrays']['states'] == {'shape': [3, 8, 8], 'dtype': 'int32'}
    assert manifest['artifact']['arrays']['events'] == {'shape': [3, 7, 3], 'dtype': 'uint8'}

    def forbidden(*args, **kwargs):
        raise AssertionError('Offline loading and relabeling must not execute rules')

    monkeypatch.setattr(rules.PermutationRule, 'transition', forbidden)
    monkeypatch.setattr(rules, 'get_rule', forbidden)
    loaded = load_dataset(path)
    loaded.validate()
    assert loaded.manifest == dataset.manifest
    for name in ('states', 'actions', 'events'):
        np.testing.assert_array_equal(getattr(dataset, name), getattr(loaded, name))
    batch = GoalSampler(loaded).sample(12)
    relabel_transitions(loaded, batch['episode_indices'], batch['step_indices'], batch['goals'])


def test_save_refuses_both_existing_artifacts(tmp_path):
    dataset = generate_dataset(get_rule('pancake-3'), num_episodes=1, horizon=2)
    path = dataset.save(tmp_path / 'data.npz')
    archive, manifest = path.read_bytes(), path.with_suffix('.json').read_bytes()
    with pytest.raises(FileExistsError):
        dataset.save(path)
    assert path.read_bytes() == archive
    assert path.with_suffix('.json').read_bytes() == manifest
    path.unlink()
    with pytest.raises(FileExistsError):
        dataset.save(path)
    assert not path.exists()


def test_manifest_race_rolls_back_only_new_archive(tmp_path, monkeypatch):
    from copuzzle.games.permutation import data

    dataset = generate_dataset(get_rule('pancake-3'), num_episodes=1, horizon=2)
    path = tmp_path / 'data.npz'
    original = data.os.link

    def fail_manifest(source, target):
        if Path(target).suffix == '.json':
            Path(target).write_text('concurrent owner')
            raise FileExistsError('Another writer published a manifest')
        return original(source, target)

    monkeypatch.setattr(data.os, 'link', fail_manifest)
    with pytest.raises(FileExistsError):
        dataset.save(path)
    assert not path.exists()
    assert path.with_suffix('.json').read_text() == 'concurrent owner'
    assert set(tmp_path.iterdir()) == {path.with_suffix('.json')}


def test_tampered_archive_fails_checksum(tmp_path):
    path = generate_dataset(get_rule('pancake-3'), 1, 2).save(tmp_path / 'data.npz')
    with path.open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_dataset(path)


@pytest.mark.parametrize('corruption', ['filename', 'rule_id', 'rule_spec', 'shape', 'dtype', 'bool_shape'])
def test_tampered_metadata_is_rejected(tmp_path, corruption):
    path = generate_dataset(get_rule('pancake-3'), 1, 2).save(tmp_path / 'data.npz')
    meta_path = path.with_suffix('.json')
    manifest = json.loads(meta_path.read_text())
    if corruption == 'filename':
        manifest['artifact']['filename'] = 'other.npz'
    elif corruption == 'rule_id':
        manifest['dataset']['rule_id'] = 'wrong'
    elif corruption == 'rule_spec':
        manifest['dataset']['rule_spec']['action_encoding'] = 'wrong'
    elif corruption == 'shape':
        manifest['artifact']['arrays']['states']['shape'] = [1, 3, 4]
    elif corruption == 'dtype':
        manifest['artifact']['arrays']['states']['dtype'] = 'uint8'
    elif corruption == 'bool_shape':
        manifest['artifact']['arrays']['states']['shape'][0] = True
    meta_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_dataset(path)


@pytest.mark.parametrize('corruption', ['float_states', 'extra_array', 'invalid_token', 'event_width'])
def test_valid_checksum_does_not_allow_corrupt_arrays_or_silent_casts(tmp_path, corruption):
    dataset = generate_dataset(get_rule('topspin-6-3'), 2, 3)
    path = dataset.save(tmp_path / 'data.npz')
    arrays = {name: getattr(dataset, name).copy() for name in ('states', 'actions', 'events')}
    if corruption == 'float_states':
        arrays['states'] = arrays['states'].astype(np.float64)
        arrays['states'][0, 0, 0] += 0.25
    elif corruption == 'extra_array':
        arrays['solutions'] = np.ones(1)
    elif corruption == 'invalid_token':
        arrays['states'][0, 0, 0] = 0
    elif corruption == 'event_width':
        arrays['events'] = np.pad(arrays['events'], ((0, 0), (0, 0), (0, 3)))
    _rewrite_archive(path, **arrays)
    with pytest.raises(ValueError):
        load_dataset(path)


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = generate_dataset(get_rule('pancake-3'), 1, 2).save(tmp_path / 'data.npz')
    meta_path = path.with_suffix('.json')
    meta_path.write_text(meta_path.read_text().replace('"seed": 0', '"seed": 0, "seed": 1'))
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        load_dataset(path)


def test_future_goal_provenance_and_previous_event_width():
    dataset = generate_dataset(get_rule('topspin-8-4'), 4, 11, seed=71)
    batch = GoalSampler(dataset, seed=72).sample(256)
    assert batch['observations'].shape == (256, 8)
    assert batch['previous_events'].shape == (256, 3)
    assert batch['action_masks'].shape == (256, 3)
    assert np.all(batch['action_masks'] == 1)
    episodes, steps, goals = (batch[key] for key in ('episode_indices', 'step_indices', 'goal_indices'))
    assert np.all(goals > steps)
    assert np.all(goals <= dataset.horizon)
    assert np.all(np.any(batch['observations'] != batch['goals'], axis=1))
    np.testing.assert_array_equal(batch['observations'], dataset.states[episodes, steps])
    np.testing.assert_array_equal(batch['next_observations'], dataset.states[episodes, steps + 1])
    np.testing.assert_array_equal(batch['goals'], dataset.states[episodes, goals])
    for episode, step, previous in zip(episodes, steps, batch['previous_events']):
        expected = np.zeros(3, dtype=np.uint8) if step == 0 else dataset.events[episode, step - 1]
        np.testing.assert_array_equal(previous, expected)
    success = np.all(batch['next_observations'] == batch['goals'], axis=1)
    np.testing.assert_array_equal(batch['terminated'], success)
    np.testing.assert_array_equal(batch['rewards'], np.where(success, 0.0, -1.0))
    np.testing.assert_array_equal(batch['bootstrap_masks'], (~success).astype(np.float32))
    assert not np.any(batch['truncated'])
    assert set(steps) == set(range(dataset.horizon))
    assert np.any(goals == dataset.horizon)


def test_equal_future_boards_preserve_index_multiplicity_and_content_success():
    rule = get_rule('pancake-3')
    dataset = generate_dataset(rule, 1, 4)
    dataset.states[0, 0] = [1, 2, 3]
    dataset.actions[:] = 0
    dataset.events[:] = [1, 0]
    for step in range(4):
        dataset.states[0, step + 1] = rule.transition(dataset.states[0, step], 0)
    dataset.validate(rule)
    batch = GoalSampler(dataset, seed=83).sample(4000)
    assert np.all(batch['terminated'])
    at_zero = batch['step_indices'] == 0
    counts = np.bincount(batch['goal_indices'][at_zero], minlength=5)
    assert set(batch['goal_indices'][at_zero]) == {1, 3}
    assert 0.40 < counts[1] / (counts[1] + counts[3]) < 0.60


def test_explicit_goal_changes_labels_and_preserves_timeout_bootstrapping():
    dataset = generate_dataset(get_rule('pancake-3'), 1, 1)
    dataset.states[0] = [[1, 2, 3], [2, 1, 3]]
    dataset.actions[:] = 0
    dataset.events[:] = [1, 0]
    success = relabel_transitions(dataset, [0], [0], [[2, 1, 3]])
    timeout = relabel_transitions(dataset, [0], [0], [[3, 2, 1]])
    np.testing.assert_array_equal(success['observations'], timeout['observations'])
    np.testing.assert_array_equal(success['next_observations'], timeout['next_observations'])
    assert success['rewards'].tolist() == [0.0]
    assert success['terminated'].tolist() == [True]
    assert success['truncated'].tolist() == [False]
    assert success['bootstrap_masks'].tolist() == [0.0]
    assert timeout['rewards'].tolist() == [-1.0]
    assert timeout['terminated'].tolist() == [False]
    assert timeout['truncated'].tolist() == [True]
    assert timeout['bootstrap_masks'].tolist() == [1.0]


@pytest.mark.parametrize('episodes,steps,goals', [
    ([1], [0], [[1, 2, 3]]), ([0], [2], [[1, 2, 3]]),
    ([0], [-1], [[1, 2, 3]]), ([True], [0], [[1, 2, 3]]),
    ([0.0], [0], [[1, 2, 3]]), ([0], [0, 1], [[1, 2, 3]]),
    ([0], [0], [[0, 1, 2]]), ([0], [0], [[1, 1, 3]]),
    ([0], [0], [[1.0, 2.0, 3.0]]), ([0], [0], [[1, 2]]),
    ([0], [0], [[True, 2, 3]]),
    ([], [], []),
])
def test_relabel_rejects_invalid_input(episodes, steps, goals):
    dataset = generate_dataset(get_rule('pancake-3'), 1, 2)
    with pytest.raises(ValueError):
        relabel_transitions(dataset, episodes, steps, goals)


def test_relabel_rejects_already_successful_task():
    dataset = generate_dataset(get_rule('pancake-3'), 1, 2)
    with pytest.raises(ValueError, match='already successful'):
        relabel_transitions(dataset, [0], [0], dataset.states[:, 0])


def test_no_distinct_future_goal_is_an_explicit_error():
    dataset = generate_dataset(get_rule('pancake-3'), 1, 2)
    dataset.states[:] = [1, 2, 3]
    dataset.validate()  # No privileged transition checks at this boundary.
    with pytest.raises(ValueError, match='No transition'):
        GoalSampler(dataset)


def test_sampler_snapshots_input_and_returned_batches_do_not_alias(monkeypatch):
    dataset = generate_dataset(get_rule('topspin-6-3'), 3, 8)
    sampler, reference = GoalSampler(dataset, seed=91), GoalSampler(dataset, seed=91)
    dataset.states[:] = 0
    dataset.actions[:] = -1
    dataset.events[:] = 0
    dataset.manifest['rule_spec']['window'] = 99

    def forbidden(*args, **kwargs):
        raise AssertionError('Sampling batches must not revalidate complete trajectories')

    monkeypatch.setattr(TrajectoryDataset, 'validate', forbidden)
    batch = sampler.sample(32)
    _assert_batches_equal(batch, reference.sample(32))
    for values in batch.values():
        values[:] = 0
    _assert_batches_equal(sampler.sample(32), reference.sample(32))


def test_sampler_rng_advances_and_checkpoint_resumes_exactly():
    dataset = generate_dataset(get_rule('pancake-5'), 3, 8, seed=95)
    sampler = GoalSampler(dataset, seed=96)
    first = sampler.sample(64)
    checkpoint = json.loads(json.dumps(sampler.state_dict()))
    expected = sampler.sample(64)
    assert any(not np.array_equal(first[key], expected[key]) for key in first)
    restored = GoalSampler(dataset, seed=0)
    restored.load_state_dict(checkpoint)
    _assert_batches_equal(expected, restored.sample(64))
    altered = deepcopy(dataset)
    altered.states[0, 0] = altered.states[0, 0][::-1]
    with pytest.raises(ValueError, match='identity'):
        GoalSampler(altered).load_state_dict(checkpoint)
    checkpoint['rng_state'] = {'bit_generator': 'other'}
    with pytest.raises(ValueError, match='RNG state'):
        sampler.load_state_dict(checkpoint)


@pytest.mark.parametrize('size', [0, -1, True, 1.5])
def test_invalid_batch_sizes(size):
    sampler = GoalSampler(generate_dataset(get_rule('pancake-3'), 1, 2))
    with pytest.raises(ValueError):
        sampler.sample(size)


def test_convenience_sampling_matches_fresh_sampler():
    dataset = generate_dataset(get_rule('pancake-4'), 2, 4)
    _assert_batches_equal(sample_goal_transitions(dataset, 16, seed=99),
                          GoalSampler(dataset, seed=99).sample(16))
