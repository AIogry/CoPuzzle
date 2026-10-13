"""Public structure, profile compatibility, and untrusted-file validation."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from copuzzle.games.lights_out.env import PuzzleEnv
from copuzzle.games.lights_out.rules import PuzzleRule, get_rule, graph_rule, grid_rule
from copuzzle.games.lights_out.structure import (
    PublicStructure, STRUCTURE_SCHEMA, export_structure, load_structure,
    rule_from_structure, save_structure,
)


def _manifest(path):
    return json.loads(path.with_suffix('.json').read_text())


def _write_manifest(path, manifest):
    path.with_suffix('.json').write_text(json.dumps(manifest))


def _rewrite_arrays(path, fields):
    np.savez_compressed(path, **fields)
    manifest = _manifest(path)
    manifest['npz_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_manifest(path, manifest)


def _custom_graph():
    adjacency = np.zeros((5, 5), dtype=np.uint8)
    for left, right in ((0, 1), (0, 3), (1, 2), (1, 4)):
        adjacency[left, right] = adjacency[right, left] = 1
    return graph_rule(adjacency, name='lab tree with spaces', coordinates=np.arange(10).reshape(5, 2))


def test_rectangular_grid_metadata_axes_and_legacy_identity():
    rule = grid_rule(2, 3)
    assert rule.grid_shape == (2, 3)
    np.testing.assert_array_equal(rule.coordinates, [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
    structure = export_structure(rule)
    assert structure.family == 'grid'
    np.testing.assert_array_equal(structure.grid_shape, [2, 3])
    np.testing.assert_array_equal(structure.grid_coordinates, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    expected = hashlib.sha256(b'copuzzle/undirected-self-toggle/v1\0' + b'6\0' + rule.adjacency.tobytes()).hexdigest()
    assert rule.rule_id == expected
    assert rule.rule_id == graph_rule(rule.adjacency).rule_id
    assert structure.grid_shape.dtype == structure.grid_coordinates.dtype == np.int32


def test_family_is_metadata_not_name_or_display_coordinates():
    grid = grid_rule(2, 3)
    misleading_name = graph_rule(grid.adjacency, name='grid-2x3', coordinates=grid.coordinates)
    assert export_structure(misleading_name).family == 'graph'
    renamed_grid = PuzzleRule('unusual-grid-name', grid.adjacency, grid_shape=(2, 3))
    assert export_structure(renamed_grid).family == 'grid'
    assert renamed_grid.rule_id == misleading_name.rule_id
    assert export_structure(grid).structure_sha256 != export_structure(misleading_name).structure_sha256


@pytest.mark.parametrize('shape', [(0, 6), (2, 2), (True, 6), (2.0, 3), (2,), [2, 3, 1], '23'])
def test_bad_grid_shape_metadata_rejected(shape):
    with pytest.raises(ValueError):
        PuzzleRule('bad', grid_rule(2, 3).adjacency, grid_shape=shape)


def test_grid_shape_rejects_another_adjacency_even_same_button_count():
    with pytest.raises(ValueError, match='row-major'):
        PuzzleRule('bad', get_rule('graph-path6').adjacency, grid_shape=(2, 3))


@pytest.mark.parametrize('rule', [grid_rule(2, 3), grid_rule(4, 5), _custom_graph()])
def test_structure_env_spaces_reset_step_and_independent_arrays(rule):
    env = PuzzleEnv(rule, observation_profile='structure_v1')
    check_env(env, skip_render_check=True)
    initial, info = env.reset(seed=4)
    fields = export_structure(rule).observation_fields()
    assert set(initial) == {'board', 'goal', 'previous_event'} | set(fields)
    assert env.observation_space.contains(initial)
    assert set(info) == {'is_success', 'steps', 'rule_id'}
    for name, array in fields.items():
        np.testing.assert_array_equal(initial[name], array)
        assert env.observation_space[name].dtype == array.dtype
        initial[name].fill(77)
    after, *_ = env.step(0)
    assert env.observation_space.contains(after)
    for name, array in fields.items():
        np.testing.assert_array_equal(after[name], array)
    assert env.observation_profile == 'structure_v1'


@pytest.mark.parametrize('profile,flag,expected', [
    (None, None, 'legacy_hidden'), (None, False, 'legacy_hidden'),
    (None, True, 'legacy_adjacency'), ('legacy_hidden', None, 'legacy_hidden'),
    ('legacy_hidden', False, 'legacy_hidden'), ('legacy_adjacency', None, 'legacy_adjacency'),
    ('legacy_adjacency', True, 'legacy_adjacency'), ('structure_v1', None, 'structure_v1'),
])
def test_profile_resolution(profile, flag, expected):
    rule = grid_rule(2, 3)
    env = PuzzleEnv(rule, reveal_graph=flag, observation_profile=profile)
    observation, _ = env.reset(seed=7)
    assert env.observation_profile == expected
    if expected == 'legacy_hidden':
        assert set(observation) == {'board', 'goal', 'previous_event'}
    elif expected == 'legacy_adjacency':
        assert set(observation) == {'board', 'goal', 'previous_event', 'adjacency'}
        np.testing.assert_array_equal(observation['adjacency'], rule.adjacency)
    else:
        assert 'grid_shape' in observation and 'adjacency' not in observation


@pytest.mark.parametrize('profile,flag', [
    ('legacy_hidden', True), ('legacy_adjacency', False),
    ('structure_v1', True), ('structure_v1', False), ('unknown', None),
])
def test_conflicting_or_unknown_profiles_rejected(profile, flag):
    with pytest.raises(ValueError):
        PuzzleEnv(grid_rule(2, 3), reveal_graph=flag, observation_profile=profile)


@pytest.mark.parametrize('rule', [grid_rule(2, 3), grid_rule(3, 2), _custom_graph()])
def test_structure_roundtrip_and_explicit_rule_reconstruction(tmp_path, rule):
    original = export_structure(rule)
    path = tmp_path / 'structure.npz'
    assert save_structure(original, path) == path
    loaded = load_structure(path)
    assert type(loaded) is PublicStructure
    for name in ('task_name', 'family', 'n_buttons', 'rule_id', 'structure_sha256'):
        assert getattr(loaded, name) == getattr(original, name)
    for name, array in original.observation_fields().items():
        np.testing.assert_array_equal(loaded.observation_fields()[name], array)
    assert not hasattr(loaded, 'transition') and not hasattr(loaded, 'solve')
    rebuilt = rule_from_structure(loaded)
    assert rebuilt.name == rule.name and rebuilt.rule_id == rule.rule_id
    assert rebuilt.grid_shape == rule.grid_shape
    np.testing.assert_array_equal(rebuilt.toggle_matrix, rule.toggle_matrix)
    if rule.grid_shape is not None:
        np.testing.assert_array_equal(rebuilt.coordinates, rule.coordinates)
    else:
        assert rebuilt.coordinates is None  # Display geometry is not public structure.
    metadata = _manifest(path)
    assert metadata['schema'] == STRUCTURE_SCHEMA
    assert metadata['npz_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata['structure_sha256'] == original.structure_sha256
    assert metadata['npz_sha256'] != metadata['structure_sha256']


def test_loading_is_pure_data_without_executable_rule_construction(tmp_path, monkeypatch):
    import copuzzle.games.lights_out.rules as rules
    import copuzzle.games.lights_out.oracle as oracle
    import copuzzle.games.lights_out.env as env
    path = save_structure(export_structure(_custom_graph()), tmp_path / 'structure.npz')
    def forbidden(*args, **kwargs):
        raise AssertionError('Executable task helper called during pure data loading')
    for module, names in ((rules, ('PuzzleRule', 'grid_rule', 'graph_rule', 'get_rule')),
                          (oracle, ('solve', 'rank', 'is_reachable')), (env, ('PuzzleEnv',))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    assert load_structure(path).task_name == 'lab tree with spaces'


@pytest.mark.parametrize('rule', [grid_rule(2, 3), _custom_graph()])
def test_public_structure_and_observations_do_not_alias(rule):
    public = export_structure(rule)
    original = public.observation_fields()
    for array in public.observation_fields().values():
        array.fill(0)
    for name, array in original.items():
        np.testing.assert_array_equal(public.observation_fields()[name], array)
        with pytest.raises(ValueError):
            getattr(public, name).setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        public.task_name = 'changed'
    assert public.rule_id == rule.rule_id


def test_structure_hash_covers_semantics_and_order_ignores_display_geometry():
    rule = _custom_graph()
    public = export_structure(rule)
    assert export_structure(graph_rule(rule.adjacency, rule.name, np.zeros((5, 3)))).structure_sha256 == public.structure_sha256
    assert export_structure(graph_rule(rule.adjacency, 'renamed')).structure_sha256 != public.structure_sha256
    permutation = [1, 0, 2, 3, 4]
    permuted = graph_rule(rule.adjacency[np.ix_(permutation, permutation)], rule.name)
    assert export_structure(permuted).structure_sha256 != public.structure_sha256
    assert export_structure(grid_rule(2, 3)).structure_sha256 != export_structure(grid_rule(3, 2)).structure_sha256


@pytest.mark.parametrize('which', ['npz', 'json', 'both'])
def test_save_refuses_either_existing_file(tmp_path, which):
    path = tmp_path / 'structure.npz'
    for suffix in (['.npz', '.json'] if which == 'both' else ['.' + which]):
        path.with_suffix(suffix).write_bytes(b'existing')
    with pytest.raises(FileExistsError):
        save_structure(export_structure(grid_rule(2, 3)), path)
    for suffix in (['.npz', '.json'] if which == 'both' else ['.' + which]):
        assert path.with_suffix(suffix).read_bytes() == b'existing'
    assert not list(tmp_path.glob('.structure-*'))


def test_failed_manifest_publication_removes_only_own_archive(tmp_path, monkeypatch):
    import copuzzle.games.lights_out.structure as module
    original_link = module.os.link
    def publish(source, target):
        if Path(target).suffix == '.json':
            Path(target).write_text('racing writer')
            raise FileExistsError('racing writer')
        return original_link(source, target)
    monkeypatch.setattr(module.os, 'link', publish)
    path = tmp_path / 'structure.npz'
    with pytest.raises(FileExistsError):
        save_structure(export_structure(grid_rule(2, 3)), path)
    assert not path.exists()
    assert path.with_suffix('.json').read_text() == 'racing writer'
    assert not list(tmp_path.glob('.structure-*'))


@pytest.mark.parametrize('suffix', ['.json', '.npy', '', '.NPZ'])
def test_npz_extension_required(tmp_path, suffix):
    with pytest.raises(ValueError):
        save_structure(export_structure(grid_rule(2, 3)), tmp_path / ('structure' + suffix))
    with pytest.raises(ValueError):
        load_structure(tmp_path / ('structure' + suffix))


def test_npz_file_hash_detects_modification(tmp_path):
    path = save_structure(export_structure(grid_rule(2, 3)), tmp_path / 'structure.npz')
    path.write_bytes(path.read_bytes() + b'extra')
    with pytest.raises(ValueError, match='file hash'):
        load_structure(path)


@pytest.mark.parametrize('field,value', [
    ('schema', 'copuzzle.structure.v9'), ('n_buttons', True), ('n_buttons', 6.0),
    ('n_buttons', -1), ('task_name', ''), ('family', 'lights_out'),
    ('rule_id', '0' * 64), ('structure_sha256', '0' * 64),
    ('npz_sha256', 'not-sha'), ('semantics', {}), ('extra', 1),
])
def test_invalid_manifest_rejected(tmp_path, field, value):
    path = save_structure(export_structure(grid_rule(2, 3)), tmp_path / 'structure.npz')
    manifest = _manifest(path)
    manifest[field] = value
    _write_manifest(path, manifest)
    with pytest.raises(ValueError):
        load_structure(path)


def test_manifest_missing_and_duplicate_fields_rejected(tmp_path):
    path = save_structure(export_structure(grid_rule(2, 3)), tmp_path / 'structure.npz')
    manifest = _manifest(path)
    del manifest['rule_id']
    _write_manifest(path, manifest)
    with pytest.raises(ValueError, match='fields'):
        load_structure(path)
    path.with_suffix('.json').write_text('{"schema":"one","schema":"two"}')
    with pytest.raises(ValueError, match='duplicate'):
        load_structure(path)


@pytest.mark.parametrize('mutation', ['axes', 'duplicate', 'missing', 'float', 'object', 'extra', 'count'])
def test_grid_archive_rejects_bad_coordinates_and_types(tmp_path, mutation):
    public = export_structure(grid_rule(2, 3))
    path = save_structure(public, tmp_path / 'structure.npz')
    fields = public.observation_fields()
    if mutation == 'axes':
        fields['grid_coordinates'] = fields['grid_coordinates'][:, ::-1]
    elif mutation == 'duplicate':
        fields['grid_coordinates'][1] = fields['grid_coordinates'][0]
    elif mutation == 'missing':
        fields.pop('grid_coordinates')
    elif mutation == 'float':
        fields['grid_coordinates'] = fields['grid_coordinates'].astype(np.float32)
    elif mutation == 'object':
        fields['grid_coordinates'] = fields['grid_coordinates'].astype(object)
    elif mutation == 'extra':
        fields['solution'] = np.zeros(6, dtype=np.uint8)
    elif mutation == 'count':
        fields['grid_shape'] = np.array([2, 4], dtype=np.int32)
    _rewrite_arrays(path, fields)
    with pytest.raises(ValueError):
        load_structure(path)


@pytest.mark.parametrize('mutation', ['self-loop', 'directed', 'nonbinary', 'float'])
def test_graph_archive_constraints_checked_even_if_file_hash_is_updated(tmp_path, mutation):
    public = export_structure(_custom_graph())
    path = save_structure(public, tmp_path / 'structure.npz')
    fields = public.observation_fields()
    if mutation == 'self-loop':
        fields['adjacency'][0, 0] = 1
    elif mutation == 'directed':
        fields['adjacency'][0, 1] = 0
    elif mutation == 'nonbinary':
        fields['adjacency'][0, 1] = fields['adjacency'][1, 0] = 2
    elif mutation == 'float':
        fields['adjacency'] = fields['adjacency'].astype(np.float64)
    _rewrite_arrays(path, fields)
    with pytest.raises(ValueError):
        load_structure(path)


def test_graph_with_valid_constraints_but_wrong_rule_identity_is_rejected(tmp_path):
    public = export_structure(_custom_graph())
    path = save_structure(public, tmp_path / 'structure.npz')
    fields = public.observation_fields()
    fields['adjacency'][2, 3] = fields['adjacency'][3, 2] = 1
    _rewrite_arrays(path, fields)
    with pytest.raises(ValueError, match='rule_id'):
        load_structure(path)


def test_huge_claimed_numpy_shape_rejected_before_loading_array(tmp_path, monkeypatch):
    public = export_structure(_custom_graph())
    path = save_structure(public, tmp_path / 'structure.npz')
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {'descr': '|u1', 'fortran_order': False, 'shape': (10**18, 10**18)})
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('adjacency.npy', header.getvalue())
    manifest = _manifest(path)
    manifest['npz_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_manifest(path, manifest)
    def forbidden(*args, **kwargs):
        raise AssertionError('np.load must not run before safe header validation')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(ValueError, match='header'):
        load_structure(path)


def test_array_metadata_types_are_strict(tmp_path):
    path = save_structure(export_structure(grid_rule(2, 3)), tmp_path / 'structure.npz')
    manifest = _manifest(path)
    manifest['arrays']['grid_coordinates']['shape'] = [6.0, 2.0]
    _write_manifest(path, manifest)
    with pytest.raises(ValueError, match='metadata'):
        load_structure(path)
