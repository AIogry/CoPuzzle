"""Behavioral checks for standalone trajectory generation and persistence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from copuzzle.data import (TrajectoryDataset, GoalSampler, generate_dataset, load_dataset,
                          relabel_transitions, sample_goal_transitions)


class IndependentButtons:
    """Small deterministic rule protocol used to isolate data behavior."""

    name = 'test-independent-buttons'
    rule_id = 'test-rule-independent-v1'
    n_buttons = 3

    def transition(self, state, action):
        successor = np.asarray(state).copy()
        successor[action] ^= 1
        return successor


class UnchangedButtons(IndependentButtons):
    name = 'test-unchanged-buttons'
    rule_id = 'test-rule-unchanged-v1'

    def transition(self, state, action):
        return np.asarray(state).copy()


@pytest.fixture
def rule():
    return IndependentButtons()


def test_complete_trajectories_and_rule_validation(rule):
    dataset = generate_dataset(rule, num_episodes=4, horizon=7, seed=91)
    assert dataset.states.shape == (4, 8, 3)
    assert dataset.actions.shape == (4, 7)
    assert dataset.events.shape == (4, 7, 3)
    assert dataset.states.dtype == np.uint8
    assert dataset.actions.dtype == np.int64
    assert dataset.events.dtype == np.uint8
    for episode in range(4):
        for step in range(7):
            np.testing.assert_array_equal(
                dataset.states[episode, step + 1],
                rule.transition(dataset.states[episode, step], dataset.actions[episode, step]),
            )
    dataset.validate(rule)


def test_seed_and_split_streams_are_deterministic(rule):
    kwargs = dict(num_episodes=12, horizon=9, seed=57)
    train = generate_dataset(rule, split='train', **kwargs)
    repeated = generate_dataset(rule, split='train', **kwargs)
    validation = generate_dataset(rule, split='val', **kwargs)
    changed_seed = generate_dataset(rule, num_episodes=12, horizon=9, seed=58)
    for name in ('states', 'actions', 'events'):
        np.testing.assert_array_equal(getattr(train, name), getattr(repeated, name))
    assert train.manifest == repeated.manifest
    assert not np.array_equal(train.actions, validation.actions)
    assert not np.array_equal(train.states[:, 0], validation.states[:, 0])
    assert not np.array_equal(train.actions, changed_seed.actions)
    assert 'states may overlap' in train.manifest['split_semantics']


@pytest.mark.parametrize('kwargs', [
    {'num_episodes': 0}, {'num_episodes': 1.5}, {'num_episodes': True},
    {'horizon': 0}, {'horizon': -2}, {'seed': -1}, {'seed': True}, {'split': ''},
])
def test_generation_rejects_invalid_configuration(rule, kwargs):
    with pytest.raises(ValueError):
        generate_dataset(rule, **kwargs)


def test_validation_distinguishes_structure_from_rule_correctness(rule):
    dataset = generate_dataset(rule, num_episodes=1, horizon=3)
    dataset.states[0, -1, 0] ^= 1
    dataset.validate()  # Hidden-rule correctness cannot be inferred from schema.
    with pytest.raises(ValueError, match='Rule transition mismatch'):
        dataset.validate(rule)


@pytest.mark.parametrize('corruption', [
    'missing_successor', 'wrong_state_dtype', 'nonbinary_state', 'invalid_action',
    'event_not_one_hot', 'event_action_disagreement', 'wrong_event_shape',
    'manifest_horizon', 'hidden_adjacency',
])
def test_strict_schema_and_alignment_validation(rule, corruption):
    dataset = generate_dataset(rule, num_episodes=2, horizon=4)
    if corruption == 'missing_successor':
        dataset.states = dataset.states[:, :-1]
    elif corruption == 'wrong_state_dtype':
        dataset.states = dataset.states.astype(np.float32)
    elif corruption == 'nonbinary_state':
        dataset.states[0, 0, 0] = 2
    elif corruption == 'invalid_action':
        dataset.actions[0, 0] = rule.n_buttons
    elif corruption == 'event_not_one_hot':
        dataset.events[0, 0] = 0
    elif corruption == 'event_action_disagreement':
        dataset.events[0, 0] = np.roll(dataset.events[0, 0], 1)
    elif corruption == 'wrong_event_shape':
        dataset.events = dataset.events[:, :-1]
    elif corruption == 'manifest_horizon':
        dataset.manifest['horizon'] += 1
    elif corruption == 'hidden_adjacency':
        dataset.manifest['adjacency'] = np.eye(3).tolist()
    with pytest.raises(ValueError):
        dataset.validate()


def test_save_load_roundtrip_and_hidden_rule_boundary(rule, tmp_path):
    original = generate_dataset(rule, num_episodes=3, horizon=5, split='val')
    path = tmp_path / 'validation.npz'
    assert original.save(path) == path
    loaded = load_dataset(path)
    loaded.validate(rule)
    for name in ('states', 'actions', 'events'):
        np.testing.assert_array_equal(getattr(original, name), getattr(loaded, name))
    assert original.manifest == loaded.manifest
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {'states', 'actions', 'events'}
    manifest = json.loads(path.with_suffix('.json').read_text())
    assert manifest['artifact']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest['artifact']['filename'] == path.name
    assert set(manifest) == {'dataset', 'artifact'}
    for forbidden in ('adjacency', 'matrix', 'coordinates', 'M'):
        assert forbidden not in manifest['dataset']
    batch = sample_goal_transitions(loaded, 4)
    assert set(batch) == {
        'observations', 'next_observations', 'actions', 'events', 'previous_events',
        'goals', 'rewards', 'terminated', 'truncated', 'bootstrap_masks',
    }


def test_save_refuses_existing_archive_and_existing_manifest(rule, tmp_path):
    dataset = generate_dataset(rule, num_episodes=1, horizon=2)
    path = tmp_path / 'data.npz'
    dataset.save(path)
    old_archive, old_manifest = path.read_bytes(), path.with_suffix('.json').read_bytes()
    with pytest.raises(FileExistsError):
        dataset.save(path)
    assert path.read_bytes() == old_archive
    assert path.with_suffix('.json').read_bytes() == old_manifest
    path.unlink()
    with pytest.raises(FileExistsError):
        dataset.save(path)
    assert not path.exists()
    assert path.with_suffix('.json').read_bytes() == old_manifest


def test_failed_manifest_publication_removes_only_its_new_archive(rule, tmp_path, monkeypatch):
    import copuzzle.games.lights_out.data as module

    dataset = generate_dataset(rule, num_episodes=1, horizon=2)
    path = tmp_path / 'data.npz'
    original_link = module.os.link

    def fail_second_link(source, target):
        if Path(target).suffix == '.json':
            Path(target).write_text('concurrent owner', encoding='utf-8')
            raise FileExistsError('Manifest already published by another writer')
        return original_link(source, target)

    monkeypatch.setattr(module.os, 'link', fail_second_link)
    with pytest.raises(FileExistsError):
        dataset.save(path)
    assert not path.exists()
    assert path.with_suffix('.json').read_text() == 'concurrent owner'
    assert set(tmp_path.iterdir()) == {path.with_suffix('.json')}


def test_tampered_archive_fails_checksum(rule, tmp_path):
    path = tmp_path / 'data.npz'
    generate_dataset(rule, num_episodes=1, horizon=2).save(path)
    with path.open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_dataset(path)


def test_tampered_manifest_identity_is_rejected(rule, tmp_path):
    path = tmp_path / 'data.npz'
    generate_dataset(rule, num_episodes=1, horizon=2).save(path)
    meta_path = path.with_suffix('.json')
    manifest = json.loads(meta_path.read_text())
    manifest['artifact']['filename'] = 'other.npz'
    meta_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='filename'):
        load_dataset(path)


def test_npz_unknown_field_rejected_even_with_valid_checksum(rule, tmp_path):
    path = tmp_path / 'data.npz'
    dataset = generate_dataset(rule, num_episodes=1, horizon=2)
    dataset.save(path)
    np.savez(path, states=dataset.states, actions=dataset.actions,
             events=dataset.events, adjacency=np.eye(3))
    meta_path = path.with_suffix('.json')
    manifest = json.loads(meta_path.read_text())
    manifest['artifact']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='NPZ fields'):
        load_dataset(path)


def test_repeated_future_board_uses_content_success(rule):
    # Each trajectory repeatedly flips button 0. s1 == s3 while their indices
    # differ; every distinct future goal equals the immediate successor.
    dataset = generate_dataset(rule, num_episodes=2, horizon=3)
    dataset.actions[:] = 0
    dataset.events[:] = 0
    dataset.events[:, :, 0] = 1
    dataset.states[0, 0] = [0, 0, 0]
    dataset.states[1, 0] = [0, 1, 1]  # Its goals cannot occur in episode 0.
    for episode in range(2):
        for step in range(3):
            dataset.states[episode, step + 1] = rule.transition(dataset.states[episode, step], 0)
    dataset.validate(rule)
    batch = sample_goal_transitions(dataset, 128, seed=17)
    assert np.all(np.any(batch['observations'] != batch['goals'], axis=1))
    assert np.all(batch['terminated'])
    assert np.all(batch['rewards'] == 0)
    assert np.all(batch['bootstrap_masks'] == 0)
    np.testing.assert_array_equal(batch['next_observations'], batch['goals'])
    np.testing.assert_array_equal(batch['observations'][:, 1:], batch['goals'][:, 1:])
    assert np.any(np.all(batch['previous_events'] == 0, axis=1))
    assert np.any(batch['previous_events'][:, 0] == 1)


def test_future_goals_and_previous_events_align_with_same_episode(rule):
    dataset = generate_dataset(rule, num_episodes=1, horizon=3)
    dataset.actions[0] = [0, 1, 2]
    dataset.events[0] = np.eye(3, dtype=np.uint8)
    dataset.states[0] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
    dataset.validate(rule)
    batch = sample_goal_transitions(dataset, 128, seed=3)
    repeated = sample_goal_transitions(dataset, 128, seed=3)
    for key in batch:
        np.testing.assert_array_equal(batch[key], repeated[key])
    steps_seen = set()
    unsuccessful = 0
    for item in range(128):
        step = int(batch['actions'][item])
        steps_seen.add(step)
        np.testing.assert_array_equal(batch['observations'][item], dataset.states[0, step])
        np.testing.assert_array_equal(batch['next_observations'][item], dataset.states[0, step + 1])
        future = dataset.states[0, step + 1:]
        assert any(np.array_equal(batch['goals'][item], board) for board in future)
        expected_previous = np.zeros(3, dtype=np.uint8) if step == 0 else dataset.events[0, step - 1]
        np.testing.assert_array_equal(batch['previous_events'][item], expected_previous)
        success = np.array_equal(batch['next_observations'][item], batch['goals'][item])
        assert batch['terminated'][item] == success
        assert batch['rewards'][item] == (0 if success else -1)
        assert batch['bootstrap_masks'][item] == (0 if success else 1)
        assert batch['truncated'][item] == (step == 2 and not success)
        unsuccessful += int(not success)
    assert steps_seen == {0, 1, 2}
    assert unsuccessful > 0
    # With strict future goals, the last step's only eligible goal is s[L],
    # so it necessarily succeeds. Timeout behavior is tested by the env tests.
    assert not np.any(batch['truncated'])


def test_no_distinct_future_goals_is_an_explicit_error():
    dataset = generate_dataset(UnchangedButtons(), num_episodes=1, horizon=3)
    with pytest.raises(ValueError, match='No transition'):
        sample_goal_transitions(dataset, 4)


@pytest.mark.parametrize('size', [0, -1, True, 1.5])
def test_invalid_batch_size(rule, size):
    dataset = generate_dataset(rule, num_episodes=1, horizon=2)
    with pytest.raises(ValueError):
        sample_goal_transitions(dataset, size)


def test_explicit_last_step_truncation_keeps_bootstrapping(rule):
    dataset = generate_dataset(rule, num_episodes=1, horizon=2)
    dataset.actions[0] = [0, 1]
    dataset.events[0] = [[1, 0, 0], [0, 1, 0]]
    dataset.states[0] = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
    dataset.validate(rule)
    # The supplied goal need not occur later in this recorded trajectory.
    batch = relabel_transitions(dataset, [0], [1], [[0, 0, 1]])
    np.testing.assert_array_equal(batch['next_observations'], [[1, 1, 0]])
    np.testing.assert_array_equal(batch['previous_events'], [[1, 0, 0]])
    assert batch['rewards'].tolist() == [-1.0]
    assert batch['terminated'].tolist() == [False]
    assert batch['truncated'].tolist() == [True]
    assert batch['bootstrap_masks'].tolist() == [1.0]
    success = relabel_transitions(dataset, [0], [1], [[1, 1, 0]])
    assert success['rewards'].tolist() == [0.0]
    assert success['terminated'].tolist() == [True]
    assert success['truncated'].tolist() == [False]
    assert success['bootstrap_masks'].tolist() == [0.0]


def test_explicit_current_goal_is_rejected(rule):
    dataset = generate_dataset(rule, num_episodes=1, horizon=2)
    with pytest.raises(ValueError, match='already successful'):
        relabel_transitions(dataset, [0], [0], dataset.states[:, 0])


@pytest.mark.parametrize('episodes,steps,goals', [
    ([1], [0], [[0, 0, 0]]),
    ([0], [2], [[0, 0, 0]]),
    ([0], [-1], [[0, 0, 0]]),
    ([0.0], [0], [[0, 0, 0]]),
    ([True], [0], [[0, 0, 0]]),
    ([0], [0, 1], [[0, 0, 0]]),
    ([0], [0], [[0, 2, 0]]),
    ([0], [0], [[0, 0]]),
    ([], [], []),
])
def test_explicit_relabel_rejects_invalid_inputs(rule, episodes, steps, goals):
    dataset = generate_dataset(rule, num_episodes=1, horizon=2)
    with pytest.raises(ValueError):
        relabel_transitions(dataset, episodes, steps, goals)


@pytest.mark.parametrize('task_id', [
    'grid-3x3', 'grid-4x4', 'grid-4x5', 'grid-4x6', 'graph-path9', 'graph-cycle9',
])
def test_public_rule_integration(task_id):
    from copuzzle.rules import get_rule

    rule = get_rule(task_id)
    dataset = generate_dataset(rule, num_episodes=2, horizon=30, seed=14)
    dataset.validate(rule)
    assert dataset.states.shape == (2, 31, rule.n_buttons)
    batch = sample_goal_transitions(dataset, batch_size=8, seed=15)
    assert batch['observations'].shape == (8, rule.n_buttons)
    assert np.all(np.any(batch['observations'] != batch['goals'], axis=1))
    for state, action, successor in zip(batch['observations'], batch['actions'], batch['next_observations']):
        np.testing.assert_array_equal(rule.transition(state, int(action)), successor)


def test_goal_sampler_advances_rng_and_reproduces_the_same_sequence(rule):
    dataset = generate_dataset(rule, num_episodes=4, horizon=8, seed=33)
    first = GoalSampler(dataset, seed=34)
    repeated = GoalSampler(dataset, seed=34)
    batch_one, batch_two = first.sample(64), first.sample(64)
    assert any(not np.array_equal(batch_one[key], batch_two[key]) for key in batch_one)
    for actual, expected in ((batch_one, repeated.sample(64)), (batch_two, repeated.sample(64))):
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])


def test_goal_sampler_snapshot_is_independent_of_source_mutation(rule):
    dataset = generate_dataset(rule, num_episodes=3, horizon=8, seed=35)
    sampler, reference = GoalSampler(dataset, seed=36), GoalSampler(dataset, seed=36)
    # In-place corruption and replacement of source arrays/metadata are both
    # allowed after construction and cannot change an existing sampler.
    dataset.states[:] = 9
    dataset.actions[:] = -1
    dataset.events = np.zeros((1, 1, 1), dtype=np.uint8)
    dataset.manifest['array_schema']['states'] = 'corrupted'
    dataset.manifest['horizon'] = -1
    for _ in range(2):
        actual, expected = sampler.sample(32), reference.sample(32)
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
    with pytest.raises(ValueError):
        dataset.validate()


def test_goal_sampler_batches_do_not_validate_or_alias_snapshot(rule, monkeypatch):
    dataset = generate_dataset(rule, num_episodes=3, horizon=8, seed=37)
    sampler, reference = GoalSampler(dataset, seed=38), GoalSampler(dataset, seed=38)

    def unexpected_validation(self, rule=None):
        raise AssertionError('sample must not validate the complete dataset')

    monkeypatch.setattr(TrajectoryDataset, 'validate', unexpected_validation)
    batch = sampler.sample(32)
    reference.sample(32)
    for value in batch.values():
        value[:] = 0
    actual, expected = sampler.sample(32), reference.sample(32)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_convenience_sampling_matches_fresh_sampler(rule):
    dataset = generate_dataset(rule, num_episodes=2, horizon=5, seed=39)
    batch = sample_goal_transitions(dataset, 12, seed=40)
    expected = GoalSampler(dataset, seed=40).sample(12)
    for key in batch:
        np.testing.assert_array_equal(batch[key], expected[key])
