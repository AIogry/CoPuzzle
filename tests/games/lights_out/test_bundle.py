"""Public structure bundles preserve legacy bytes and learner boundaries."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from copuzzle.games.lights_out import bundle as bundle_module
from copuzzle.games.lights_out.bundle import (
    BUNDLE_SCHEMA, BundleGoalSampler, create_offline_bundle,
    load_offline_bundle, write_offline_bundle,
)
from copuzzle.games.lights_out.data import GoalSampler, generate_dataset
from copuzzle.games.lights_out.rules import get_rule, graph_rule


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(path.read_text())


def _write_json(path, value):
    path.write_text(json.dumps(value))


def _pair(directory, rule, *, common_name=False):
    paths = {}
    for split in ('train', 'val'):
        path = directory / split / 'data.npz' if common_name else directory / f'{split}.npz'
        generate_dataset(rule, num_episodes=3, horizon=5, seed=17, split=split).save(path)
        paths[split] = path
    return paths


@pytest.fixture
def prepared(tmp_path):
    rule = get_rule('grid-2x3')
    paths = _pair(tmp_path / 'old', rule)
    manifest = create_offline_bundle(tmp_path / 'bundle', train_path=paths['train'],
                                     val_path=paths['val'], rule=rule)
    return rule, paths, manifest


@pytest.mark.parametrize('task', ['grid-4x5', 'graph-path9', 'custom'])
@pytest.mark.parametrize('common_name', [False, True])
def test_migration_keeps_original_and_copied_trajectory_bytes(tmp_path, task, common_name):
    rule = (graph_rule([[0, 1, 0], [1, 0, 1], [0, 1, 0]], name='my-lab-graph')
            if task == 'custom' else get_rule(task))
    paths = _pair(tmp_path / 'old', rule, common_name=common_name)
    originals = {path: path.read_bytes() for source in paths.values()
                 for path in (source, source.with_suffix('.json'))}
    manifest_path = create_offline_bundle(tmp_path / 'bundle', train_path=paths['train'],
                                         val_path=paths['val'], rule=rule)
    manifest = _json(manifest_path)
    assert manifest['schema_version'] == BUNDLE_SCHEMA
    assert manifest['observation_profile'] == 'structure_v1'
    for split, source in paths.items():
        loaded = load_offline_bundle(manifest_path.parent if split == 'train' else manifest_path, split=split)
        target = manifest_path.parent / manifest['splits'][split]['path']
        if not common_name:
            assert target == manifest_path.parent / f'{split}.npz'
        assert target.read_bytes() == originals[source]
        assert target.with_suffix('.json').read_bytes() == originals[source.with_suffix('.json')]
        loaded.dataset.validate(rule)
        assert loaded.dataset.manifest['split'] == split
        assert loaded.structure.rule_id == rule.rule_id
        assert loaded.structure.task_name == rule.name
        assert loaded.provenance['bundle_sha256'] == _hash(manifest_path)
        assert loaded.provenance['structure_sha256'] == loaded.structure.structure_sha256
        assert loaded.provenance['dataset']['file_sha256'] == _hash(target)
        fields = loaded.structure.observation_fields()
        assert set(fields) == ({'grid_shape', 'grid_coordinates'} if task.startswith('grid') else {'adjacency'})
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_in_place_binding_does_not_resave_trajectories(tmp_path):
    rule = get_rule('graph-cycle9')
    paths = _pair(tmp_path, rule)
    before = {path: path.read_bytes() for source in paths.values()
              for path in (source, source.with_suffix('.json'))}
    result = write_offline_bundle(tmp_path, train_path=paths['train'], val_path=paths['val'], rule=rule)
    assert result == tmp_path / 'bundle.json'
    assert load_offline_bundle(result).structure.family == 'graph'
    assert all(path.read_bytes() == content for path, content in before.items())
    with pytest.raises(FileExistsError):
        write_offline_bundle(tmp_path, train_path=paths['train'], val_path=paths['val'], rule=rule)


def test_loading_and_sampling_do_not_construct_or_query_rules(prepared, monkeypatch):
    from copuzzle.games.lights_out import env, oracle, rules

    def forbidden(*args, **kwargs):
        raise AssertionError('learner data loading queried a rule/environment/solver')

    _, _, manifest = prepared
    monkeypatch.setattr(rules.PuzzleRule, '__post_init__', forbidden)
    for name in ('get_rule', 'grid_rule', 'graph_rule'):
        monkeypatch.setattr(rules, name, forbidden)
    for name in ('solve', 'is_reachable'):
        monkeypatch.setattr(oracle, name, forbidden)
    monkeypatch.setattr(env.PuzzleEnv, '__init__', forbidden)
    loaded = load_offline_bundle(manifest)
    assert BundleGoalSampler(loaded, seed=3).sample(7)['observations'].shape == (7, 6)


def test_sampler_keeps_existing_labels_and_isolates_snapshots(prepared):
    _, _, manifest = prepared
    loaded = load_offline_bundle(manifest)
    sampler = BundleGoalSampler(loaded, seed=23)
    reference = GoalSampler(loaded.dataset, seed=23)
    public = deepcopy(loaded.structure.observation_fields())
    first, expected = sampler.sample(13), reference.sample(13)
    assert set(first) == set(expected) | {'structure'}
    for key in expected:
        np.testing.assert_array_equal(first[key], expected[key])
    for key, value in public.items():
        np.testing.assert_array_equal(first['structure'][key], value)
        assert first['structure'][key].shape == value.shape  # No batch expansion.
        first['structure'][key][...] = 99
    external = sampler.structure
    for value in external.values():
        value[...] = 88
    loaded.dataset.states[...] = 2
    loaded.dataset.events[...] = 2
    loaded.dataset.manifest['task_name'] = 'mutated'
    second, expected_second = sampler.sample(13), reference.sample(13)
    for key in expected_second:
        np.testing.assert_array_equal(second[key], expected_second[key])
    for key, value in public.items():
        np.testing.assert_array_equal(second['structure'][key], value)


@pytest.mark.parametrize('artifact', ['train.npz', 'train.json', 'val.npz', 'val.json',
                                      'structure.npz', 'structure.json'])
def test_any_referenced_file_tampering_is_rejected(prepared, artifact):
    _, _, manifest = prepared
    path = manifest.parent / artifact
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        load_offline_bundle(manifest, split='train')


@pytest.mark.parametrize('artifact', ['val.npz', 'val.json', 'structure.json'])
def test_missing_unselected_or_structure_file_is_rejected(prepared, artifact):
    _, _, manifest = prepared
    (manifest.parent / artifact).unlink()
    with pytest.raises(ValueError, match='Missing bundle file'):
        load_offline_bundle(manifest)


@pytest.mark.parametrize('change', ['schema', 'profile', 'extra_field', 'missing_split',
                                    'task_name', 'rule_id', 'n_buttons', 'family', 'structure_id'])
def test_manifest_contract_and_identity_are_checked(prepared, change):
    _, _, path = prepared
    manifest = _json(path)
    if change == 'schema':
        manifest['schema_version'] = 'future'
    elif change == 'profile':
        manifest['observation_profile'] = 'legacy_hidden'
    elif change == 'extra_field':
        manifest['toggle_matrix'] = []
    elif change == 'missing_split':
        del manifest['splits']['val']
    elif change == 'n_buttons':
        manifest['n_buttons'] = True
    elif change == 'structure_id':
        manifest['structure']['structure_sha256'] = '0' * 64
    else:
        manifest[change] = 'incorrect'
    _write_json(path, manifest)
    with pytest.raises(ValueError):
        load_offline_bundle(path)


def test_duplicate_json_keys_fail(prepared):
    _, _, path = prepared
    text = path.read_text()
    path.write_text(text.replace('{', '{"schema_version":"duplicate",', 1))
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        load_offline_bundle(path)


@pytest.mark.parametrize('reference', ['../outside.npz', '/tmp/outside.npz',
                                       'train/../train.npz', './train.npz', 'train\\file.npz'])
def test_path_references_cannot_escape_or_alias(prepared, reference):
    _, _, path = prepared
    manifest = _json(path)
    manifest['splits']['train']['path'] = reference
    _write_json(path, manifest)
    with pytest.raises(ValueError):
        load_offline_bundle(path)


def test_symlink_and_duplicate_references_are_rejected(prepared):
    _, _, path = prepared
    manifest = _json(path)
    (path.parent / 'linked.npz').symlink_to(path.parent / 'train.npz')
    changed = deepcopy(manifest)
    changed['splits']['train']['path'] = 'linked.npz'
    _write_json(path, changed)
    with pytest.raises(ValueError, match='symlink'):
        load_offline_bundle(path)
    changed = deepcopy(manifest)
    changed['splits']['val'] = deepcopy(changed['splits']['train'])
    _write_json(path, changed)
    with pytest.raises(ValueError, match='distinct files'):
        load_offline_bundle(path)


@pytest.mark.parametrize('field,value', [('split', 'train'), ('task_name', 'wrong'),
                                        ('rule_id', 'wrong')])
def test_split_metadata_must_match_even_with_updated_hashes(prepared, field, value):
    _, _, path = prepared
    metadata_path = path.parent / 'val.json'
    metadata = _json(metadata_path)
    metadata['dataset'][field] = value
    _write_json(metadata_path, metadata)
    manifest = _json(path)
    manifest['splits']['val']['manifest_sha256'] = _hash(metadata_path)
    _write_json(path, manifest)
    with pytest.raises(ValueError, match='does not match'):
        load_offline_bundle(path)


def test_preparation_audits_true_transitions_and_cleans_new_directory(tmp_path):
    rule = get_rule('grid-2x3')
    paths = _pair(tmp_path / 'old', rule)
    dataset = generate_dataset(rule, num_episodes=2, horizon=3, seed=91, split='val')
    dataset.states[0, -1, 0] ^= 1
    corrupted = tmp_path / 'bad.npz'
    dataset.save(corrupted)  # Structurally valid but not a true transition.
    with pytest.raises(ValueError, match='Rule transition mismatch'):
        create_offline_bundle(tmp_path / 'new', train_path=paths['train'], val_path=corrupted, rule=rule)
    assert not (tmp_path / 'new').exists()
    assert paths['train'].exists() and corrupted.exists()


def test_copy_failure_cleans_only_new_destination(prepared, monkeypatch):
    rule, sources, existing = prepared
    before = existing.read_bytes()
    with pytest.raises(FileExistsError):
        create_offline_bundle(existing.parent, train_path=sources['train'], val_path=sources['val'], rule=rule)
    assert existing.read_bytes() == before

    def fail(*args, **kwargs):
        raise OSError('copy failed')

    monkeypatch.setattr(bundle_module.shutil, 'copyfile', fail)
    target = existing.parent.parent / 'failed'
    with pytest.raises(OSError, match='copy failed'):
        create_offline_bundle(target, train_path=sources['train'], val_path=sources['val'], rule=rule)
    assert not target.exists()
    assert all(path.exists() for path in sources.values())


def test_partial_publication_failure_preserves_existing_data(tmp_path, monkeypatch):
    rule = get_rule('grid-2x3')
    paths = _pair(tmp_path, rule)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    original = bundle_module.os.link

    def fail_bundle_manifest(source, destination, **kwargs):
        if Path(destination) == tmp_path / 'bundle.json':
            raise OSError('bundle publication failed')
        return original(source, destination, **kwargs)

    monkeypatch.setattr(bundle_module.os, 'link', fail_bundle_manifest)
    with pytest.raises(OSError, match='publication failed'):
        write_offline_bundle(tmp_path, train_path=paths['train'], val_path=paths['val'], rule=rule)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_in_place_sources_must_be_inside_directory(prepared, tmp_path):
    rule, sources, _ = prepared
    destination = tmp_path / 'empty'
    destination.mkdir()
    with pytest.raises(ValueError, match='inside'):
        write_offline_bundle(destination, train_path=sources['train'], val_path=sources['val'], rule=rule)
    assert not list(destination.iterdir())


def test_invalid_split_and_incomplete_directory_fail(prepared, tmp_path):
    _, _, manifest = prepared
    with pytest.raises(ValueError, match='split'):
        load_offline_bundle(manifest, split='test')
    incomplete = tmp_path / 'incomplete'
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError):
        load_offline_bundle(incomplete)
