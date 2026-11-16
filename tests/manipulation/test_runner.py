"""Cross-family collector semantics and small real-physics checks."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from copuzzle.manipulation.runner import measure_operations, run_collection
from copuzzle.manipulation.suite import generate_suite, suite_sha256


TASKS = ('grid-3x3', 'pancake-4', 'topspin-20-4')


def _suite(task_ids=TASKS, horizon=None):
    result = generate_suite(seed=91, depths=(1,), task_ids=task_ids)
    if horizon is not None:
        for task in result['tasks']:
            task['max_control_steps'] = horizon
        result['sha256'] = suite_sha256(result)
    return result


@pytest.mark.parametrize('task_id', TASKS)
def test_real_zero_collection_cap_is_not_environment_timeout(task_id, tmp_path, capsys):
    pytest.importorskip('mujoco')
    from copuzzle.manipulation.data import load_transitions
    from copuzzle.manipulation.runner import load_native
    suite = _suite((task_id,))
    manifest = run_collection(suite, tmp_path / 'cutoff', policy='zero', splits=('train',), max_steps=2)
    assert manifest['status'] == 'complete'
    assert manifest['selected_task_ids'] == manifest['attempted_task_ids'] == [row['id'] for row in manifest['episodes']]
    row = manifest['episodes'][0]
    assert row['outcome'] == 'collection_cutoff' and row['collection_cutoff']
    assert not row['terminated'] and not row['truncated'] and row['bootstrap_mask'] == 1
    assert row['control_steps'] == 2 and row['verified_replay']
    assert row['replay']['max_absolute_observation_error'] == 0
    assert row['measured_operations'] == [] and row['measured_atomic_cost'] == 0
    episode = load_native(task_id, tmp_path / 'cutoff' / row['trajectory'])
    task = suite['tasks'][0]
    assert episode.configuration['max_episode_steps'] == task['max_control_steps'] > 2
    assert episode.metadata['reset_options'] == {'state': task['state'], 'goal': task['goal']}
    for name, checksum in manifest['files'].items():
        assert hashlib.sha256((tmp_path / 'cutoff' / name).read_bytes()).hexdigest() == checksum
    dataset = load_transitions(tmp_path / 'cutoff', task_id=task_id)
    assert len(dataset) == 2 and dataset.collection_cutoff[-1]
    assert dataset.bootstrap_masks.tolist() == [1., 1.]
    assert dataset.actions.shape[1] == (7 if task_id == 'pancake-4' else 3)
    assert capsys.readouterr().out


@pytest.mark.parametrize('task_id', TASKS)
def test_real_short_horizon_is_a_native_timeout(task_id, tmp_path):
    pytest.importorskip('mujoco')
    suite = _suite((task_id,), horizon=2)
    report = run_collection(suite, tmp_path / 'timeout', policy='zero', splits=('train',))
    row = report['episodes'][0]
    assert row['outcome'] == 'timeout' and row['truncated']
    assert not row['terminated'] and not row['collection_cutoff']
    assert row['native_configuration']['max_episode_steps'] == 2
    assert report['summary']['attempts'] == report['summary']['failures'] == report['summary']['timeouts'] == 1
    assert report['summary']['control_steps'] == 2
    assert row['verified_replay']


def test_external_factory_and_policy_only_receive_the_allowed_interface(tmp_path):
    pytest.importorskip('mujoco')
    from copuzzle.manipulation.runner import load_native
    suite = _suite(('grid-3x3',))
    calls, seen = [], []
    class MutatingPolicy:
        def __call__(self, observation):
            assert isinstance(observation, dict)
            assert not {'reference_operations', 'reference_length', 'suite', 'task', 'reset_seed', 'split', 'id'} & observation.keys()
            seen.append(observation['goal'].copy())
            for value in observation.values():
                value[:] = 99
            return np.zeros(3, dtype=np.float32)
    def factory(task_id):
        calls.append(task_id)
        return MutatingPolicy()
    manifest = run_collection(suite, tmp_path / 'external', policy_factory=factory,
                              splits=('train',), max_steps=2)
    assert calls == ['grid-3x3'] and len(seen) == 2
    assert manifest['policy_type'] == 'external'
    assert manifest['policy_configuration']['reference_witness_access'] is False
    row = manifest['episodes'][0]
    assert row['policy_type'] == 'external'
    episode = load_native('grid-3x3', tmp_path / 'external' / row['trajectory'])
    assert episode.metadata['user']['reference_witness_access'] is False
    np.testing.assert_array_equal(episode.observations['board'][0], suite['tasks'][0]['state'])
    assert np.all(episode.observations['goal'] == np.asarray(suite['tasks'][0]['goal']))


def test_unsaveable_external_failure_stays_in_denominator_and_keeps_real_cost(tmp_path):
    pytest.importorskip('mujoco')
    from copuzzle.manipulation.data import load_transitions
    class FailAfterTwo:
        def __init__(self): self.calls = 0
        def __call__(self, observation):
            self.calls += 1
            if self.calls == 3: raise RuntimeError('injected failure after two native steps')
            return np.zeros(3, np.float32)
    manifest = run_collection(_suite(('grid-3x3',)), tmp_path / 'failed',
                              policy_factory=lambda task_id: FailAfterTwo(), splits=('train',))
    row = manifest['episodes'][0]
    assert manifest['status'] == 'complete'
    assert row['trajectory'] is None and row['outcome'] == 'collection_error'
    assert not row['goal_success'] and row['control_steps'] == 2
    assert row['control_seconds'] == pytest.approx(.1)
    assert manifest['summary']['attempts'] == manifest['summary']['failures'] == manifest['summary']['errors'] == 1
    assert manifest['summary']['mean_control_steps_all_attempts'] == 2
    with pytest.raises(ValueError, match='no trajectory'):
        load_transitions(tmp_path / 'failed', task_id='grid-3x3')


def test_pancake_native_policy_failure_prefix_remains_a_loadable_failure(tmp_path):
    pytest.importorskip('mujoco')
    from copuzzle.manipulation.data import load_transitions
    class FailAfterTwo:
        def __init__(self): self.calls = 0
        def __call__(self, observation):
            self.calls += 1
            if self.calls == 3: raise RuntimeError('deliberate controller failure')
            return np.zeros(7, np.float32)
    manifest = run_collection(_suite(('pancake-4',)), tmp_path / 'partial',
        policy_factory=lambda task_id: FailAfterTwo(), splits=('train',))
    row = manifest['episodes'][0]
    assert row['trajectory'] and row['outcome'] == 'policy_failed' and row['error'] is None
    assert row['controller_error']['message'] == 'deliberate controller failure'
    assert row['native_stop'] == 'policy_error' and row['control_steps'] == 2
    assert row['collection_cutoff'] and row['verified_replay']
    data = load_transitions(tmp_path / 'partial', task_id='pancake-4')
    assert len(data) == 2 and data.collection_cutoff[-1] and data.bootstrap_masks[-1] == 1


def _pancake_evidence():
    boards = np.array([[1, 2, 3, 4], [3, 2, 1, 4], [2, 1, 3, 4],
                       [2, 1, 3, 4], [2, 1, 3, 4]], dtype=np.int32)
    target_z = .10 + .004 / 2 + .02 * np.arange(3, -1, -1)
    positions = np.zeros((5, 4, 3))
    for index, board in enumerate(boards):
        positions[index, board - 1, 2] = target_z
    positions[1, :, 2] += .08  # A tempting but floating decoded permutation.
    orientations = np.zeros((5, 4, 4)); orientations[:, :, 0] = 1
    tool_positions = np.tile([-.1, 0., .2], (5, 1)); tool_positions[3, :2] = 0
    contacts = np.zeros((5, 4), np.uint8); contacts[2, 0] = 1
    forces = np.zeros((5, 4)); forces[2, 0] = .5
    return SimpleNamespace(observations={
        'board': boards, 'board_valid': np.ones((5, 1), np.uint8), 'token_positions': positions,
        'token_orientations': orientations, 'token_velocities': np.zeros((5, 4, 6)),
        'tool_contacts': contacts, 'tool_contact_forces': forces, 'end_effector_position': tool_positions},
        configuration={'scene': {'stack_center_xy': [0., 0.], 'support_top_m': .1,
            'symmetric_boss_height_m': .004, 'stack_pitch_m': .02, 'withdraw_distance_m': .05}})


def test_measured_pancake_operations_ignore_floating_held_and_unwithdrawn_boards():
    evidence = _pancake_evidence()
    report = measure_operations('pancake-4', evidence)
    assert report['measured_operations'] == [2] and report['measured_atomic_cost'] == 1
    assert report['operation_events'][0]['control_step'] == 3
    evidence.observations['board_valid'][:] = 0
    unavailable = measure_operations('pancake-4', evidence)
    assert unavailable['measured_operations'] is None and unavailable['measured_atomic_cost'] is None


def test_unknown_topspin_event_keeps_unknown_total_cost():
    episode = SimpleNamespace(infos=[{'event_log': []}, {'event_log': [
        {'operation': 'shift_right', 'atomic_cost': 1},
        {'operation': 'unclassified_measured_rearrangement', 'atomic_cost': None}]}])
    report = measure_operations('topspin-20-4', episode)
    assert report['measured_atomic_cost'] is None
    assert report['known_atomic_cost'] == 1 and report['unknown_atomic_cost_events'] == 1
    assert report['measured_operations'][-1] == 'unclassified_measured_rearrangement'


def test_measured_lights_out_count_comes_from_events_not_reference_length():
    report = measure_operations('grid-3x3', SimpleNamespace(event_buttons=np.array([4, 4, 0])))
    assert report['measured_operations'] == [4, 4, 0] and report['measured_atomic_cost'] == 3


def test_invalid_selection_and_existing_output_do_not_overwrite(tmp_path):
    suite = _suite(('grid-3x3',))
    for kwargs in ({'max_steps': 0}, {'max_tasks': True}, {'task_ids': ['pancake-4']}, {'policy': 'unknown'}):
        with pytest.raises(ValueError): run_collection(suite, tmp_path / 'invalid', **kwargs)
        assert not (tmp_path / 'invalid').exists()
    output = tmp_path / 'existing'; output.mkdir(); (output / 'keep').write_text('preserve')
    with pytest.raises(FileExistsError):run_collection(suite, output)
    assert (output / 'keep').read_text() == 'preserve'


def test_real_one_press_reference_reaches_the_original_task_goal(tmp_path):
    pytest.importorskip('mujoco')
    suite = _suite(('grid-3x3',))
    report = run_collection(suite, tmp_path / 'one-press', splits=('train',))
    row = report['episodes'][0]
    assert row['goal_success'] and row['terminated'] and row['outcome'] == 'success'
    assert row['measured_operations'] == suite['tasks'][0]['reference_operations']
    assert row['measured_atomic_cost'] == 1 and row['verified_replay']
    assert report['summary']['success_rate'] == 1
    assert report['observed_overlap']['by_task']['grid-3x3']['split_counts']['train']['states'] == 2
