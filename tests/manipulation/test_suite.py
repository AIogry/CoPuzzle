"""Native task witnesses, honest split boundaries, and portable strict JSON."""
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from copuzzle.manipulation import suite as module
from copuzzle.manipulation.suite import (
    DEFAULT_CONTROL_STEPS, MAX_CASES_PER_DEPTH, generate_suite, load_suite,
    save_suite, split_overlap, suite_sha256, validate_suite,
)


@pytest.fixture
def suite():
    return generate_suite()


def rebind(value):
    value['sha256'] = suite_sha256(value)
    return value


def test_default_suite_has_balanced_goal_cases_and_fixed_family_budgets(suite):
    assert validate_suite(suite) is None
    assert len(suite['tasks']) == 27
    assert 'upper bound' in suite['reference_semantics']
    assert 'not a shortest distance' in suite['reference_semantics']
    groups = Counter((case['task_id'], case['split'], case['reference_length']) for case in suite['tasks'])
    assert set(groups.values()) == {1} and len(groups) == 27
    assert len({case['id'] for case in suite['tasks']}) == 27
    assert len({case['reset_seed'] for case in suite['tasks']}) == 27
    for case in suite['tasks']:
        assert len(case['reference_operations']) == case['reference_length']
        assert case['max_control_steps'] == DEFAULT_CONTROL_STEPS[case['task_id']]
        assert case['state'] != case['goal']
        assert not {'distance', 'shortest_distance', 'operation_prefix', 'task_mode'} & set(case)


def test_generation_is_deterministic_and_does_not_share_mutable_results():
    first = generate_suite(seed=72, cases_per_depth=2, depths=(4, 1, 2))
    assert first['depths'] == [1, 2, 4]
    second = generate_suite(seed=72, cases_per_depth=2, depths=(1, 2, 4))
    assert first == second
    assert first['sha256'] != generate_suite(seed=73, cases_per_depth=2, depths=(1, 2, 4))['sha256']
    first['tasks'][0]['state'][0] ^= 1
    assert generate_suite(seed=72, cases_per_depth=2, depths=(1, 2, 4)) == second


def test_topspin_uses_twenty_fixed_positions_and_three_explicit_operations():
    value = generate_suite(task_ids=('topspin-20-4',), depths=(1,))
    state = list(range(1, 21))
    goals = [list(range(4, 0, -1)) + list(range(5, 21)), list(range(2, 21)) + [1],
             [20] + list(range(1, 20))]
    for case, operation, goal in zip(value['tasks'], ('reverse', 'shift_left', 'shift_right'), goals):
        case['state'], case['goal'], case['reference_operations'] = state.copy(), goal, [operation]
    rebind(value)
    validate_suite(value)
    assert all(len(case['goal']) == 20 for case in value['tasks'])
    assert value['tasks'][1]['goal'] != value['tasks'][1]['state']


def test_witness_length_is_allowed_to_exceed_actual_logical_solution_length():
    from copuzzle.games.lights_out.rules import grid_rule

    value = generate_suite(task_ids=('grid-3x3',), depths=(3,), seed=39)
    case = value['tasks'][0]
    case['state'] = [0] * 9
    case['reference_operations'] = [0, 1, 0]
    # XOR commutativity cancels the two button-0 presses, yet no path state repeats.
    case['goal'] = grid_rule(3, 3).transition(case['state'], 1).tolist()
    rebind(value)
    assert case['reference_length'] == 3
    validate_suite(value)


def test_report_shows_shared_states_without_claiming_unseen_state_generalization():
    value = generate_suite(task_ids=('pancake-4',), depths=(1,))
    for case, length in zip(value['tasks'], (2, 3, 4)):
        case['state'] = [1, 2, 3, 4]
        case['goal'] = list(reversed(case['state'][:length])) + case['state'][length:]
        case['reference_operations'] = [length]
    rebind(value)
    report = split_overlap(value)
    assert report['unordered_pairs_disjoint']
    assert report['unseen_state_split'] is False
    assert len(report['comparisons']) == 3
    for comparison in report['comparisons']:
        assert comparison['unordered_pair_overlap'] == 0
        assert comparison['initial_state_overlap'] == 1
        assert comparison['goal_state_overlap'] == 0
        assert comparison['endpoint_state_overlap'] == comparison['witness_state_overlap'] == 1
        assert comparison['shared_initial_states'] == [[1, 2, 3, 4]]


def test_reverse_endpoint_pair_is_not_allowed_across_splits():
    value = generate_suite(task_ids=('pancake-4',), depths=(1,))
    train, dev = value['tasks'][:2]
    dev['state'], dev['goal'] = train['goal'].copy(), train['state'].copy()
    dev['reference_operations'] = train['reference_operations'].copy()
    with pytest.raises(ValueError, match='unordered'):
        validate_suite(value)
    with pytest.raises(ValueError, match='unordered'):
        suite_sha256(value)


def test_duplicate_pair_with_another_witness_length_is_also_rejected():
    from copuzzle.games.lights_out.rules import grid_rule

    value = generate_suite(task_ids=('grid-3x3',), depths=(1, 3))
    first, longer = value['tasks'][:2]
    state = [0] * 9
    goal = grid_rule(3, 3).transition(state, 1).tolist()
    for case, operations in ((first, [1]), (longer, [0, 1, 0])):
        case['state'], case['goal'], case['reference_operations'] = state.copy(), goal.copy(), operations
    with pytest.raises(ValueError, match='unordered'):
        validate_suite(value)


def test_duplicate_path_and_duplicate_id_are_rejected():
    value = generate_suite(task_ids=('pancake-4',), depths=(1,))
    first, other = value['tasks'][:2]
    for name in ('state', 'goal', 'reference_operations'):
        other[name] = first[name].copy()
    with pytest.raises(ValueError, match='Duplicate reference witness path'):
        validate_suite(value)
    value = generate_suite()
    value['tasks'][1]['id'] = value['tasks'][0]['id']
    with pytest.raises(ValueError, match='Duplicate task id'):
        validate_suite(value)


@pytest.mark.parametrize('operations', [[0, 0, 1], [1, 0, 0]])
def test_repeated_state_or_premature_final_goal_is_rejected(operations):
    from copuzzle.games.lights_out.rules import grid_rule

    value = generate_suite(task_ids=('grid-3x3',), depths=(3,))
    case = value['tasks'][0]
    case['state'], case['reference_operations'] = [0] * 9, operations
    state = case['state']
    for operation in operations:
        state = grid_rule(3, 3).transition(state, operation).tolist()
    case['goal'] = state
    with pytest.raises(ValueError, match='revisits a state'):
        validate_suite(value)


def test_capacity_is_checked_and_dense_feasible_allocations_terminate():
    value = generate_suite(task_ids=('pancake-4',), depths=(1,), cases_per_depth=12)
    assert len(value['tasks']) == 36
    with pytest.raises(ValueError, match='capacity exceeded.*requested 39, available 36'):
        generate_suite(task_ids=('pancake-4',), depths=(1,), cases_per_depth=13)
    dense = generate_suite(task_ids=('pancake-4',), depths=(1, 2, 3, 4), cases_per_depth=12)
    assert len(dense['tasks']) == 144
    validate_suite(dense)
    assert split_overlap(dense)['unordered_pairs_disjoint']


def test_random_sampling_has_a_finite_failure_bound(monkeypatch):
    attempts = 0

    def none(*args):
        nonlocal attempts
        attempts += 1
        return None

    monkeypatch.setattr(module, '_random_candidate', none)
    with pytest.raises(ValueError, match='Bounded sampling exhausted'):
        generate_suite(task_ids=('topspin-20-4',), depths=(1,))
    assert attempts == module.MAX_SAMPLING_ATTEMPTS


def test_short_timeout_budget_can_be_rebound_without_altering_reference(suite):
    original = deepcopy(suite['tasks'][0])
    suite['tasks'][0]['max_control_steps'] = 1
    with pytest.raises(ValueError, match='checksum'):
        validate_suite(suite)
    rebind(suite)
    validate_suite(suite)
    assert suite['tasks'][0]['reference_operations'] == original['reference_operations']
    assert suite['tasks'][0]['goal'] == original['goal']


@pytest.mark.parametrize('arguments', [
    {'seed': True}, {'seed': -1}, {'seed': 1.0}, {'seed': float('nan')}, {'seed': 2**63},
    {'cases_per_depth': True}, {'cases_per_depth': 0}, {'cases_per_depth': 1.0},
    {'cases_per_depth': MAX_CASES_PER_DEPTH + 1}, {'depths': ()}, {'depths': (True,)},
    {'depths': (1.0,)}, {'depths': (0,)}, {'depths': (5,)}, {'depths': (1, 1)},
    {'task_ids': ()}, {'task_ids': ('topspin-8-4',)}, {'task_ids': ('pancake-6',)},
    {'task_ids': (True,)}, {'task_ids': ('pancake-4', 'pancake-4')},
])
def test_invalid_generation_arguments(arguments):
    with pytest.raises(ValueError):
        generate_suite(**arguments)


def test_unknown_generation_parameter_is_not_silently_ignored():
    with pytest.raises(TypeError):
        generate_suite(unseen_states=True)


@pytest.mark.parametrize('mutation', [
    lambda s: s.update(unknown=True),
    lambda s: s.__setitem__('schema_version', 'copuzzle.manipulation_suite.v0'),
    lambda s: s.__setitem__('seed', True),
    lambda s: s.__setitem__('cases_per_depth', 1.0),
    lambda s: s.__setitem__('depths', [1, True, 3]),
    lambda s: s.__setitem__('depths', [3, 2, 1]),
    lambda s: s.__setitem__('task_ids', ['pancake-4']),
    lambda s: s.__setitem__('reference_semantics', 'exact shortest distance'),
    lambda s: s.__setitem__('split_semantics', 'unseen states'),
    lambda s: s.__setitem__('sha256', 'a' * 64),
    lambda s: s['tasks'].pop(),
    lambda s: s['tasks'][0].update(distance=1),
    lambda s: s['tasks'][0].__setitem__('id', '../../bad'),
    lambda s: s['tasks'][0].__setitem__('task_id', 'topspin-8-4'),
    lambda s: s['tasks'][0].__setitem__('split', 'validation'),
    lambda s: s['tasks'][0].__setitem__('reset_seed', False),
    lambda s: s['tasks'][0].__setitem__('reference_length', True),
    lambda s: s['tasks'][0].__setitem__('max_control_steps', True),
    lambda s: s['tasks'][0].__setitem__('max_control_steps', 0),
    lambda s: s['tasks'][0].__setitem__('max_control_steps', 1501),
    lambda s: s['tasks'][0]['state'].__setitem__(0, True),
    lambda s: s['tasks'][0]['goal'].__setitem__(0, float('nan')),
    lambda s: s['tasks'][0]['state'].__setitem__(0, 2),
    lambda s: s['tasks'][0].__setitem__('state', tuple(s['tasks'][0]['state'])),
    lambda s: s['tasks'][0]['goal'].__setitem__(0, 1 - s['tasks'][0]['goal'][0]),
    lambda s: s['tasks'][0]['reference_operations'].__setitem__(0, True),
    lambda s: s['tasks'][0]['reference_operations'].__setitem__(0, 9),
    lambda s: s['tasks'][9]['reference_operations'].__setitem__(0, 1),
    lambda s: s['tasks'][18]['reference_operations'].__setitem__(0, 0),
    lambda s: s['tasks'][18]['reference_operations'].__setitem__(0, 'reverse_four'),
])
def test_strict_suite_validation_rejects_corruption(suite, mutation):
    mutation(suite)
    with pytest.raises(ValueError):
        validate_suite(suite)


def test_save_load_roundtrip_and_refuse_overwrite_or_symlink(suite, tmp_path):
    path = save_suite(suite, tmp_path / 'tasks.json')
    assert path.is_absolute() and load_suite(path) == suite
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_suite(suite, path)
    assert path.read_bytes() == original
    broken = tmp_path / 'broken.json'
    broken.symlink_to(tmp_path / 'nonexistent.json')
    with pytest.raises(FileExistsError):
        save_suite(suite, broken)
    assert broken.is_symlink() and not (tmp_path / 'nonexistent.json').exists()


def test_json_loader_rejects_duplicate_keys_nonfinite_and_hash_changes(suite, tmp_path):
    path = tmp_path / 'tasks.json'
    original = json.dumps(suite)
    for payload in (original.replace('"seed": 0', '"seed": 0, "seed": 0', 1),
                    original.replace('"seed": 0', '"seed": NaN', 1),
                    original.replace('"seed": 0', '"seed": Infinity', 1),
                    original.replace('"seed": 0', '"seed": 1', 1)):
        path.write_text(payload)
        with pytest.raises(ValueError):
            load_suite(path)


def test_atomic_publication_cleans_up_without_replacing_concurrent_writer(suite, tmp_path, monkeypatch):
    def racing_link(source, target):
        Path(target).write_text('other writer')
        raise FileExistsError('concurrent writer')

    monkeypatch.setattr(module.os, 'link', racing_link)
    destination = tmp_path / 'tasks.json'
    with pytest.raises(FileExistsError):
        save_suite(suite, destination)
    assert destination.read_text() == 'other writer'
    assert sorted(path.name for path in tmp_path.iterdir()) == ['tasks.json']


def test_module_needs_no_mujoco_and_seed_is_process_hash_independent(tmp_path):
    program = '''
import builtins,json,sys
original=builtins.__import__
def checked(name,*args,**kwargs):
    if name=='mujoco' or name.startswith('mujoco.'):
        raise AssertionError('suite imported physics')
    return original(name,*args,**kwargs)
builtins.__import__=checked
from copuzzle.manipulation.suite import generate_suite,save_suite,load_suite,split_overlap
s=generate_suite(seed=91,depths=(1,2,3,4));save_suite(s,sys.argv[1]);assert load_suite(sys.argv[1])==s
assert split_overlap(s)['unseen_state_split'] is False
assert 'mujoco' not in sys.modules
print(s['sha256'])
'''
    results = []
    for index, hash_seed in enumerate(('0', '8751')):
        environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
        run = subprocess.run([sys.executable, '-c', program, str(tmp_path / f'suite-{index}.json')],
                             env=environment, capture_output=True, text=True, check=True)
        results.append(run.stdout.strip())
    assert results[0] == results[1]
