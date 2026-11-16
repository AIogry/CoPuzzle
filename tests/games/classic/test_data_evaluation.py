"""Exercise offline artifacts, state-dependent masks, and evaluation boundaries."""
from copy import deepcopy
import json

import numpy as np
import pytest

from copuzzle.games.classic.data import (
    GoalSampler, generate_dataset, load_dataset, relabel_transitions,
)
from copuzzle.games.classic.evaluation import (
    OraclePolicy, RandomPolicy, audit_overlap, evaluate_policy,
    generate_task_suite, load_task_suite, save_task_suite, validate_task_suite,
)
from copuzzle.games.classic.oracle import ExactOracle
from copuzzle.games.classic.rules import get_rule


@pytest.mark.parametrize('task', ['sliding-2x3', 'hanoi-3', 'hanoi-5'])
def test_generation_reproducible_legal_and_complete(task, tmp_path):
    rule = get_rule(task)
    data = generate_dataset(rule, 6, 12, seed=19)
    duplicate = generate_dataset(rule, 6, 12, seed=19)
    validation = generate_dataset(rule, 6, 12, seed=19, split='val')
    for field in ('states', 'actions', 'events', 'action_masks'):
        np.testing.assert_array_equal(getattr(data, field), getattr(duplicate, field))
    assert not np.array_equal(data.states, validation.states)
    assert data.states.shape == (6, 13, rule.state_size)
    assert data.action_masks.shape == (6, 13, rule.n_actions)
    assert np.all(data.action_masks.sum(axis=-1) < rule.n_actions)
    assert np.all(np.take_along_axis(data.action_masks[:, :-1], data.actions[..., None], axis=-1))
    path = data.save(tmp_path / 'data.npz')
    loaded = load_dataset(path)
    loaded.validate(rule)
    with pytest.raises(FileExistsError):
        data.save(path)


def test_loading_and_relabeling_never_query_rules_or_oracles(tmp_path, monkeypatch):
    rule = get_rule('hanoi-3')
    data = generate_dataset(rule, 5, 12)
    path = data.save(tmp_path / 'data.npz')
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline sampling must not query dynamics or distances')
    for method in ('transition', 'action_mask', 'is_reachable', 'sample_state'):
        monkeypatch.setattr(type(rule), method, forbidden)
    for method in ('distance', 'solve', 'states_at_distance'):
        monkeypatch.setattr(ExactOracle, method, forbidden)
    loaded = load_dataset(path)
    sampler = GoalSampler(loaded, seed=42)
    batch = sampler.sample(40)
    for index, (e, t, g) in enumerate(zip(batch['episode_indices'], batch['step_indices'], batch['goal_indices'])):
        assert t < g <= data.horizon
        np.testing.assert_array_equal(batch['observations'][index], data.states[e, t])
        np.testing.assert_array_equal(batch['next_observations'][index], data.states[e, t + 1])
        np.testing.assert_array_equal(batch['goals'][index], data.states[e, g])
        np.testing.assert_array_equal(batch['action_masks'][index], data.action_masks[e, t])
        np.testing.assert_array_equal(batch['next_action_masks'][index], data.action_masks[e, t + 1])
        success = np.array_equal(data.states[e, t + 1], data.states[e, g])
        assert batch['rewards'][index] == (0 if success else -1)
        assert batch['bootstrap_masks'][index] == (0 if success else 1)
    assert not {'distance', 'optimal_steps', 'solution', 'rule'}.intersection(batch)
    state = sampler.state_dict()
    expected = sampler.sample(20)
    sampler.load_state_dict(state)
    actual = sampler.sample(20)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_sampler_snapshot_and_returned_arrays_are_independent():
    data = generate_dataset(get_rule('sliding-2x3'), 3, 8)
    reference = GoalSampler(data, seed=7)
    sampler = GoalSampler(data, seed=7)
    data.states[:] = 0
    actual = sampler.sample(20)
    expected = reference.sample(20)
    for field in actual:
        np.testing.assert_array_equal(actual[field], expected[field])
        actual[field][...] = 0
    for field, array in sampler.sample(20).items():
        assert array.size > 0
    other = GoalSampler(generate_dataset(get_rule('sliding-2x3'), 3, 8, seed=99))
    with pytest.raises(ValueError, match='identity'):
        other.load_state_dict(sampler.state_dict())


def test_explicit_relabel_timeout_bootstraps_and_current_goal_rejected():
    data = generate_dataset(get_rule('hanoi-3'), 2, 3)
    current, successor = data.states[0, -2:]
    goal = next(np.asarray(v, np.int32) for v in np.ndindex(3, 3, 3)
                if not np.array_equal(v, current) and not np.array_equal(v, successor))
    batch = relabel_transitions(data, [0], [2], [goal])
    assert batch['rewards'].tolist() == [-1]
    assert batch['truncated'].tolist() == [True]
    assert batch['terminated'].tolist() == [False]
    assert batch['bootstrap_masks'].tolist() == [1]
    with pytest.raises(ValueError, match='already successful'):
        relabel_transitions(data, [0], [2], [current])
    with pytest.raises(ValueError, match='booleans'):
        relabel_transitions(data, [True], [2], [goal])


@pytest.mark.parametrize('corruption', ['dtype', 'state', 'event', 'mask_shape', 'mask_binary', 'illegal_action', 'identity'])
def test_rejects_static_corruption(corruption):
    data = generate_dataset(get_rule('sliding-2x3'), 2, 3)
    if corruption == 'dtype':
        data.states = data.states.astype(np.float64)
    elif corruption == 'state':
        data.states[0, 0, 0] = 99
    elif corruption == 'event':
        data.events[0, 0] = 0
    elif corruption == 'mask_shape':
        data.action_masks = data.action_masks[:, :-1]
    elif corruption == 'mask_binary':
        data.action_masks[0, 0, 0] = 2
    elif corruption == 'illegal_action':
        action = int(np.flatnonzero(data.action_masks[0, 0] == 0)[0])
        data.actions[0, 0] = action
        data.events[0, 0] = np.eye(data.n_actions, dtype=np.uint8)[action]
    else:
        data.manifest['rule_id'] += 'forged'
    with pytest.raises(ValueError):
        data.validate()


def test_optional_rule_audit_detects_forged_successors_and_masks():
    rule = get_rule('hanoi-3')
    data = generate_dataset(rule, 2, 4)
    original = deepcopy(data)
    data.states[0, -1] = (data.states[0, -1] + 1) % 3
    data.validate()  # representation-only validation intentionally cannot infer dynamics
    with pytest.raises(ValueError):
        data.validate(rule)
    original.action_masks[0, -1] = 1
    original.validate()
    with pytest.raises(ValueError, match='mask mismatch'):
        original.validate(rule)


def test_archive_integrity_and_array_metadata(tmp_path):
    data = generate_dataset(get_rule('hanoi-3'), 2, 3)
    path = data.save(tmp_path / 'data.npz')
    meta_path = path.with_suffix('.json')
    meta = json.loads(meta_path.read_text())
    meta['artifact']['arrays']['action_masks']['dtype'] = 'int64'
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='shape/dtype'):
        load_dataset(path)
    meta['artifact']['arrays']['action_masks']['dtype'] = 'uint8'
    meta_path.write_text(json.dumps(meta))
    with path.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        load_dataset(path)


@pytest.mark.parametrize('task,depths', [('sliding-2x3', (1, 3, 6)), ('hanoi-3', (1, 3, 7))])
def test_exact_suites_pair_exclusion_and_privileged_reference(task, depths, tmp_path):
    rule = get_rule(task)
    dev = generate_task_suite(rule, depths=depths, tasks_per_depth=4)
    test = generate_task_suite(rule, depths=depths, tasks_per_depth=4, exclude_pairs=dev['cases'])
    pairs = lambda suite: {(tuple(c['state']), tuple(c['goal'])) for c in suite['cases']}
    assert not pairs(dev) & pairs(test)
    assert len({tuple(c['goal']) for c in dev['cases']}) > 1
    save_task_suite(test, tmp_path / 'tasks.json')
    report = evaluate_policy(OraclePolicy(rule), load_task_suite(tmp_path / 'tasks.json'))
    assert report['summary']['success_rate'] == 1
    assert report['summary']['mean_excess_steps_on_success'] == 0
    assert report['privileged_rule_access'] is True
    data = generate_dataset(rule, 4, 12)
    overlap = audit_overlap(test, data)
    assert 0 <= overlap['forward_pair_seen_fraction'] <= 1
    with pytest.raises(ValueError):
        generate_task_suite(rule, depths=(99,), tasks_per_depth=1, max_episode_steps=100)
    with pytest.raises(ValueError):
        generate_task_suite(rule, depths=(1,), tasks_per_depth=10000)


def test_evaluation_copies_observations_and_never_supplies_answers():
    rule = get_rule('hanoi-3')
    suite = generate_task_suite(rule, depths=(1, 2), tasks_per_depth=2)
    before = deepcopy(suite)
    base = OraclePolicy(rule)
    def policy(obs):
        assert set(obs) == {'board', 'goal', 'action_mask', 'previous_event'}
        action = base(obs)
        for value in obs.values():
            value[:] = 99
        return action
    report = evaluate_policy(policy, suite)
    assert report['summary']['success_rate'] == 1
    assert report['privileged_rule_access'] is None
    assert suite == before
    random = RandomPolicy(rule.n_actions, seed=0)
    for _ in range(50):
        assert random({'action_mask': np.asarray([0, 0, 1, 0, 0, 0], np.uint8)}) == 2


def test_rejects_forged_exact_distance_even_after_rehash():
    from copuzzle.games.classic.evaluation import _suite_hash
    suite = generate_task_suite(get_rule('hanoi-3'), depths=(1, 2), tasks_per_depth=2)
    suite['cases'][0]['optimal_steps'] = 2
    suite['task_suite_sha256'] = _suite_hash(suite)
    with pytest.raises(ValueError, match='distance'):
        validate_task_suite(suite)
