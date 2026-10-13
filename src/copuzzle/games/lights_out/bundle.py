"""Bind unchanged operation trajectories to a checked public structure.

Preparation audits real transitions with an explicitly supplied rule. Loading
and goal sampling read pure data; they never construct a rule or query a solver.
A bundle binds both train and validation artifacts, but it does not assert that
states in those splits are disjoint.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from .data import GoalSampler, TrajectoryDataset, load_dataset
from .structure import PublicStructure, export_structure, load_structure, save_structure

BUNDLE_SCHEMA = 'copuzzle.offline_bundle.v1'
OBSERVATION_PROFILE = 'structure_v1'
_BUNDLE_FIELDS = {
    'schema_version', 'observation_profile', 'task_name', 'family', 'n_buttons',
    'rule_id', 'structure', 'splits',
}
_REFERENCE_FIELDS = {'path', 'file_sha256', 'manifest_sha256'}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _checksum(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in '0123456789abcdef' for char in value)):
        raise ValueError('Invalid SHA256 checksum')
    return value


def _path(directory, relative):
    """Accept canonical in-bundle relative references without symlink traversal."""
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Bundle references must be relative POSIX paths')
    parts = PurePosixPath(relative)
    if parts.is_absolute() or '..' in parts.parts or str(parts) != relative or relative == '.':
        raise ValueError('Bundle reference escapes its directory or is not canonical')
    path = directory
    for part in parts.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('Bundle references must not traverse symlinks')
    if not path.is_file():
        raise ValueError(f'Missing bundle file: {relative}')
    return path


def _reference(path, directory):
    path = Path(path).expanduser().absolute()
    try:
        relative = path.relative_to(directory).as_posix()
    except ValueError as error:
        raise ValueError('Trajectory files must be inside the bundle directory') from error
    path = _path(directory, relative)
    if path.suffix != '.npz':
        raise ValueError('Bundle archive references must end in .npz')
    manifest = _path(directory, path.with_suffix('.json').relative_to(directory).as_posix())
    return {'path': relative, 'file_sha256': _sha256(path),
            'manifest_sha256': _sha256(manifest)}


def _verify_reference(directory, reference, *, structure=False):
    fields = _REFERENCE_FIELDS | ({'structure_sha256'} if structure else set())
    if not isinstance(reference, dict) or set(reference) != fields:
        raise ValueError('Unexpected or missing bundle reference fields')
    path = _path(directory, reference['path'])
    if path.suffix != '.npz':
        raise ValueError('Bundle archive references must end in .npz')
    manifest = _path(directory, path.with_suffix('.json').relative_to(directory).as_posix())
    if _sha256(path) != _checksum(reference['file_sha256']):
        raise ValueError(f'Bundle archive checksum mismatch: {reference["path"]}')
    if _sha256(manifest) != _checksum(reference['manifest_sha256']):
        raise ValueError(f'Bundle manifest checksum mismatch: {manifest.name}')
    if structure:
        _checksum(reference['structure_sha256'])
    return path


def _check_dataset(dataset, structure, split):
    dataset.validate()
    expected = {'task_name': structure.task_name, 'rule_id': structure.rule_id,
                'n_buttons': structure.n_buttons, 'split': split}
    for name, value in expected.items():
        if dataset.manifest[name] != value:
            raise ValueError(f'Bundle dataset {name} does not match its structure/split')


@dataclass(frozen=True)
class OfflineBundle:
    """One selected trajectory split, its public structure, and source identity.

    The dataset retains the legacy mutable API. ``BundleGoalSampler`` snapshots
    it on construction, so subsequent caller edits cannot affect sampling.
    Provenance describes artifacts; it is not additional model input.
    """
    dataset: TrajectoryDataset
    structure: PublicStructure
    provenance: dict


def load_offline_bundle(path, split='train') -> OfflineBundle:
    """Load a complete bundle without a rule, environment, or solver.

    Both splits and their manifests are checked even when selecting only one.
    Each archive must have its sibling JSON. All references remain inside the
    bundle directory, and incomplete, duplicate-key, or altered artifacts fail.
    """
    if split not in ('train', 'val'):
        raise ValueError("split must be 'train' or 'val'")
    path = Path(path).expanduser().absolute()
    if path.is_dir():
        path = path / 'bundle.json'
    if path.name != 'bundle.json' or path.is_symlink():
        raise ValueError('Expected a bundle directory or its bundle.json')
    directory = path.parent.resolve()
    with path.open(encoding='utf-8') as stream:
        manifest = json.load(stream, object_pairs_hook=_object)
    if not isinstance(manifest, dict) or set(manifest) != _BUNDLE_FIELDS:
        raise ValueError('Unexpected or missing offline bundle fields')
    if manifest['schema_version'] != BUNDLE_SCHEMA or manifest['observation_profile'] != OBSERVATION_PROFILE:
        raise ValueError('Unsupported offline bundle schema or observation profile')
    if type(manifest['n_buttons']) is not int or manifest['n_buttons'] < 1:
        raise ValueError('Invalid bundle node count')
    if not isinstance(manifest['splits'], dict) or set(manifest['splits']) != {'train', 'val'}:
        raise ValueError('A complete bundle requires train and val splits')
    structure_path = _verify_reference(directory, manifest['structure'], structure=True)
    paths = {name: _verify_reference(directory, reference)
             for name, reference in manifest['splits'].items()}
    all_files = [structure_path, structure_path.with_suffix('.json')]
    for archive in paths.values():
        all_files.extend((archive, archive.with_suffix('.json')))
    if len(set(all_files)) != len(all_files):
        raise ValueError('Bundle references must identify distinct files')
    structure = load_structure(structure_path)
    for name in ('task_name', 'family', 'n_buttons', 'rule_id'):
        if manifest[name] != getattr(structure, name):
            raise ValueError(f'Bundle {name} does not match its public structure')
    if manifest['structure']['structure_sha256'] != structure.structure_sha256:
        raise ValueError('Bundle structure identity mismatch')
    selected = None
    for name, archive in paths.items():
        dataset = load_dataset(archive)
        _check_dataset(dataset, structure, name)
        if name == split:
            selected = dataset
    provenance = {
        'schema_version': BUNDLE_SCHEMA, 'observation_profile': OBSERVATION_PROFILE,
        'bundle_path': str(directory / 'bundle.json'), 'bundle_sha256': _sha256(path),
        'split': split, 'task_name': structure.task_name, 'family': structure.family,
        'n_buttons': structure.n_buttons, 'rule_id': structure.rule_id,
        'structure_sha256': structure.structure_sha256,
        'dataset': deepcopy(manifest['splits'][split]),
        'structure': deepcopy(manifest['structure']),
    }
    return OfflineBundle(selected, structure, provenance)


def write_offline_bundle(directory, *, train_path, val_path, rule) -> Path:
    """Bind existing in-directory train/val files without rewriting their bytes.

    This preparation-only function audits every transition against ``rule``.
    Trajectories must already exist inside ``directory``. New structure files
    and bundle.json are published exclusively; failure leaves trajectories as-is.
    A reader requires bundle.json and all checked references before using data.
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError('The bundle directory must already exist')
    targets = [directory / name for name in ('structure.npz', 'structure.json', 'bundle.json')]
    if any(os.path.lexists(target) for target in targets):
        raise FileExistsError('Refusing to overwrite a structure or bundle artifact')
    structure = export_structure(rule)
    references = {}
    for split, source in (('train', train_path), ('val', val_path)):
        before = _reference(source, directory)
        archive = directory / before['path']
        dataset = load_dataset(archive)
        _check_dataset(dataset, structure, split)
        dataset.validate(rule)
        if _reference(archive, directory) != before:
            raise ValueError('Trajectory changed during bundle preparation')
        references[split] = before
    if references['train']['path'] == references['val']['path']:
        raise ValueError('Train and validation must use distinct artifacts')
    staging = Path(tempfile.mkdtemp(prefix='.bundle-', dir=directory))
    published = []
    try:
        archive = save_structure(structure, staging / 'structure.npz')
        structure_reference = _reference(archive, staging)
        structure_reference['structure_sha256'] = structure.structure_sha256
        manifest = {
            'schema_version': BUNDLE_SCHEMA, 'observation_profile': OBSERVATION_PROFILE,
            'task_name': structure.task_name, 'family': structure.family,
            'n_buttons': structure.n_buttons, 'rule_id': structure.rule_id,
            'structure': structure_reference, 'splits': references,
        }
        with (staging / 'bundle.json').open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        for target in targets:
            temporary = staging / target.name
            stat = temporary.stat()
            os.link(temporary, target)
            published.append((target, stat.st_dev, stat.st_ino))
        return directory / 'bundle.json'
    except BaseException:
        for target, device, inode in reversed(published):
            try:
                stat = target.lstat()
                if (stat.st_dev, stat.st_ino) == (device, inode):
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        shutil.rmtree(staging)


def create_offline_bundle(output_directory, *, train_path, val_path, rule) -> Path:
    """Copy old artifacts into a new directory and bind their public structure.

    Original NPZ/JSON bytes remain untouched. Conventional train.npz/val.npz
    names are retained at the root; other names are kept in train/val subfolders
    so each unchanged legacy manifest still names its own archive correctly.
    An incomplete directory fails loading; preparation failure removes only the
    directory newly created by this call. Existing destinations are rejected.
    """
    directory = Path(output_directory).expanduser().absolute()
    sources = {name: Path(path).expanduser().absolute()
               for name, path in (('train', train_path), ('val', val_path))}
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    identity = directory.stat()
    try:
        conventional = all(source.name == f'{name}.npz' for name, source in sources.items())
        copied = {}
        for name, source in sources.items():
            if source.suffix != '.npz':
                raise ValueError('Dataset path must have the .npz suffix')
            target = directory / source.name if conventional else directory / name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            shutil.copyfile(source.with_suffix('.json'), target.with_suffix('.json'))
            copied[name] = target
        return write_offline_bundle(directory, train_path=copied['train'], val_path=copied['val'], rule=rule)
    except BaseException:
        try:
            current = directory.lstat()
            if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                shutil.rmtree(directory)
        except FileNotFoundError:
            pass
        raise


class BundleGoalSampler:
    """Existing goal labels plus one static structure dictionary per batch.

    Structure is not broadcast to B copies. Both data and public arrays are
    snapshotted at construction. Each returned dictionary owns its structure
    copies, so user mutations cannot affect later samples or environment rules.
    Do not pass goal fields into a dynamics predictor: a goal can be s_next.
    """
    def __init__(self, bundle: OfflineBundle, seed=0):
        if not isinstance(bundle, OfflineBundle):
            raise TypeError('bundle must be an OfflineBundle')
        _check_dataset(bundle.dataset, bundle.structure, bundle.provenance['split'])
        self._sampler = GoalSampler(bundle.dataset, seed=seed)
        self._structure = deepcopy(bundle.structure.observation_fields())
        self.provenance = deepcopy(bundle.provenance)

    @property
    def structure(self):
        return deepcopy(self._structure)

    def sample(self, batch_size):
        batch = self._sampler.sample(batch_size)
        batch['structure'] = self.structure
        return batch


__all__ = ['BUNDLE_SCHEMA', 'OfflineBundle', 'create_offline_bundle',
           'write_offline_bundle', 'load_offline_bundle', 'BundleGoalSampler']
