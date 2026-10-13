"""Exercise published CLI workflows through saved artifacts."""
import json

import numpy as np
import pytest

from copuzzle.cli import main
from copuzzle.games.lights_out.bundle import load_offline_bundle
from copuzzle.games.lights_out.evaluation import load_task_suite


@pytest.mark.parametrize('task', ['grid-4x5', 'graph-cycle20'])
def test_structured_workflow(tmp_path, task):
    folder = tmp_path / task
    main(['generate', '--task', task, '--output', str(folder), '--episodes', '3',
          '--val-episodes', '2', '--horizon', '5', '--observation-profile', 'structure_v1'])
    main(['validate', str(folder), '--check-rule'])
    bundle = load_offline_bundle(folder)
    main(['tasks', '--task', task, '--output', str(folder / 'tasks.json'),
          '--per-depth', '2', '--depths', '1,2', '--observation-profile', 'structure_v1'])
    suite = load_task_suite(folder / 'tasks.json')
    assert suite['structure']['structure_sha256'] == bundle.structure.structure_sha256
    main(['evaluate', '--tasks', str(folder / 'tasks.json'), '--policy', 'oracle',
          '--output', str(folder / 'oracle.json')])
    report = json.loads((folder / 'oracle.json').read_text())
    assert report['summary']['success_rate'] == 1.0
    assert report['observation_profile'] == 'structure_v1'


def test_custom_graph_from_cli(tmp_path):
    adjacency = tmp_path / 'adjacency.json'
    adjacency.write_text(json.dumps([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]))
    structure = tmp_path / 'custom' / 'structure.npz'
    main(['structure', '--adjacency', str(adjacency), '--name', 'star-four', '--output', str(structure)])
    output = tmp_path / 'data'
    main(['generate', '--structure', str(structure), '--output', str(output),
          '--episodes', '2', '--val-episodes', '1', '--horizon', '3',
          '--observation-profile', 'structure_v1'])
    main(['tasks', '--structure', str(structure), '--output', str(output / 'tasks.json'),
          '--depths', '1', '--per-depth', '2', '--observation-profile', 'structure_v1'])
    main(['evaluate', '--tasks', str(output / 'tasks.json'), '--policy', 'oracle',
          '--output', str(output / 'oracle.json')])
    assert json.loads((output / 'oracle.json').read_text())['summary']['success_rate'] == 1.0
    np.testing.assert_array_equal(load_offline_bundle(output).structure.adjacency, json.loads(adjacency.read_text()))


def test_migrate_preserves_bytes_and_cases(tmp_path):
    source, target = tmp_path / 'old', tmp_path / 'new'
    main(['generate', '--output', str(source), '--episodes', '2', '--val-episodes', '1', '--horizon', '4'])
    main(['tasks', '--output', str(source / 'tasks.json'), '--depths', '1,2', '--per-depth', '2'])
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    main(['bundle', '--train', str(source / 'train.npz'), '--val', str(source / 'val.npz'),
          '--output', str(target)])
    main(['migrate-tasks', '--tasks', str(source / 'tasks.json'), '--output', str(target / 'tasks.json')])
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert load_task_suite(source / 'tasks.json')['cases'] == load_task_suite(target / 'tasks.json')['cases']
    assert load_offline_bundle(target).dataset.states.shape == (2, 5, 9)
    with pytest.raises(SystemExit):
        main(['tasks', '--output', str(tmp_path / 'bad.json'), '--reveal-graph',
              '--observation-profile', 'structure_v1'])


def test_validate_bundle_audits_validation_transitions(tmp_path):
    import hashlib
    from copuzzle.games.lights_out.data import load_dataset

    root = tmp_path / 'bundle'
    main(['generate', '--output', str(root), '--episodes', '1', '--val-episodes', '1',
          '--horizon', '2', '--observation-profile', 'structure_v1'])
    path = root / 'val.npz'
    data = load_dataset(path)
    data.states[0, 0, 0] ^= 1
    np.savez_compressed(path, states=data.states, actions=data.actions, events=data.events)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    sidecar = json.loads(path.with_suffix('.json').read_text())
    sidecar['artifact']['sha256'] = sha(path)
    path.with_suffix('.json').write_text(json.dumps(sidecar))
    manifest = json.loads((root / 'bundle.json').read_text())
    manifest['splits']['val']['file_sha256'] = sha(path)
    manifest['splits']['val']['manifest_sha256'] = sha(path.with_suffix('.json'))
    (root / 'bundle.json').write_text(json.dumps(manifest))
    # Hashes and array contracts are internally consistent; the recorded rule is not.
    main(['validate', str(root)])
    with pytest.raises(SystemExit) as error:
        main(['validate', str(root), '--check-rule'])
    assert error.value.code == 2
