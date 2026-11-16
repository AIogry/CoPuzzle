"""Continuous data stays bound to original tasks, physics and episode endings."""
import json
from pathlib import Path

import numpy as np
import pytest

from copuzzle.manipulation.data import load_transitions
from copuzzle.manipulation.suite import generate_suite, suite_sha256
from copuzzle.manipulation.runner import run_collection, load_native


@pytest.fixture(scope='module')
def collection(tmp_path_factory):
    pytest.importorskip('mujoco')
    root = tmp_path_factory.mktemp('continuous-data') / 'collection'
    suite = generate_suite(seed=43, depths=(1, 2))
    run_collection(suite, root, splits=('train',), policy='zero', max_steps=2)
    return root


@pytest.mark.parametrize('task_id,width', [('grid-3x3', 3), ('topspin-20-4', 3), ('pancake-4', 7)])
def test_native_actions_features_and_cutoffs_survive_batching(collection, task_id, width):
    data = load_transitions(collection, task_id=task_id)
    assert data.actions.dtype == np.float32 and data.actions.shape == (4, width)
    assert data.episode_offsets.tolist() == [0, 2, 4]
    assert data.collection_cutoff.tolist() == [False, True, False, True]
    assert not data.terminated.any() and not data.truncated.any()
    assert np.all(data.bootstrap_masks == 1)
    manifest = json.loads((collection / 'manifest.json').read_text())
    rows = [row for row in manifest['episodes'] if row['task_id'] == task_id]
    for index, row in enumerate(rows):
        episode = load_native(task_id, collection / row['trajectory'])
        start, stop = data.episode_offsets[index:index + 2]
        assert np.array_equal(data.actions[start:stop], episode.actions)
        assert np.array_equal(data.goals[start:stop], episode.observations['goal'][:-1])
        for spec in data.feature_spec:
            if spec['name'] not in episode.observations:
                continue  # Lights Out static public structure is added separately.
            expected = episode.observations[spec['name']][1:].reshape(2, -1).astype(np.float32)
            assert np.array_equal(data.next_observations[start:stop, spec['start']:spec['stop']], expected)
    assert 'goal' not in {item['name'] for item in data.feature_spec}
    batch = data.sample(100, rng=np.random.default_rng(2))
    assert not {'reference_operations', 'controller_steps', 'phases', 'infos', 'metadata'} & set(batch)
    assert batch['episode_index'].max() == 1 and batch['step_index'].max() == 1
    indices = data.episode_offsets[batch['episode_index']] + batch['step_index']
    assert np.array_equal(batch['next_observations'], data.next_observations[indices])
    assert np.array_equal(batch['episode_end'], batch['step_index'] == 1)


def test_true_environment_timeout_is_not_collector_cutoff(tmp_path):
    pytest.importorskip('mujoco')
    suite = generate_suite(seed=9, depths=(1,), task_ids=('grid-3x3',))
    for task in suite['tasks']:
        task['max_control_steps'] = 2
    suite['sha256'] = suite_sha256(suite)
    root = tmp_path / 'timeout'
    run_collection(suite, root, splits=('train',), policy='zero')
    data = load_transitions(root, task_id='grid-3x3')
    assert data.truncated.tolist() == [False, True]
    assert not data.collection_cutoff.any()
    assert data.bootstrap_masks.tolist() == [1., 1.]
    finite = load_transitions(root, task_id='grid-3x3', bootstrap_mode='finite_horizon')
    assert finite.bootstrap_masks.tolist() == [1., 0.]
    assert finite.provenance['bootstrap_mode'] == 'finite_horizon'


def test_success_retains_native_goal_and_masks_terminal_bootstrap(tmp_path):
    pytest.importorskip('mujoco')
    suite = generate_suite(seed=1, depths=(1,), task_ids=('grid-3x3',))
    root = tmp_path / 'success'
    run_collection(suite, root, splits=('train',), policy='reference')
    data = load_transitions(root, task_id='grid-3x3')
    assert data.terminated[-1] and not data.truncated.any()
    assert data.bootstrap_masks[-1] == 0 and np.all(data.bootstrap_masks[:-1] == 1)
    assert data.rewards[-1] == 0
    assert np.all(data.goals == next(task['goal'] for task in suite['tasks'] if task['split'] == 'train'))


def test_missing_failed_episode_is_never_silently_discarded(collection, tmp_path):
    import shutil
    root = tmp_path / 'with-error'
    shutil.copytree(collection, root)
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    row = next(row for row in manifest['episodes'] if row['task_id'] == 'pancake-4')
    row.update(trajectory=None, outcome='collection_error', error='failure before capture')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='refusing to drop'):
        load_transitions(root, task_id='pancake-4')


def test_manifest_cannot_convert_cutoff_into_success(collection, tmp_path):
    import shutil
    root = tmp_path / 'forged'
    shutil.copytree(collection, root)
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    row = next(row for row in manifest['episodes'] if row['task_id'] == 'topspin-20-4')
    row['goal_success'] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Goal success'):
        load_transitions(root, task_id='topspin-20-4')


def test_rejects_changed_native_archive_before_loading(collection, tmp_path):
    import shutil
    root = tmp_path / 'corrupt'
    shutil.copytree(collection, root)
    manifest = json.loads((root / 'manifest.json').read_text())
    archive = root / manifest['episodes'][0]['trajectory']
    archive.write_bytes(archive.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        load_transitions(root, task_id='grid-3x3')
