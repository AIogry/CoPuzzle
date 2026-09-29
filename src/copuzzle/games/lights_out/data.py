"""Offline operation trajectories with explicit successors and checked artifacts.

The archive contains observations of executed operations, not the hidden rule.
Different splits use independent random streams; they may still share states.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .rules import PuzzleRule

SCHEMA_VERSION = 'copuzzle.trajectory.v1'
GENERATOR_VERSION = 'copuzzle.uniform_operations.v1'
_SPLIT_SEMANTICS = 'Independent RNG streams; states may overlap between splits.'
_ARRAY_SCHEMA = {
    'states': 'uint8[E,L+1,N]',
    'actions': 'int64[E,L]',
    'events': 'uint8[E,L,N]',
}
_MANIFEST_KEYS = {
    'schema_version', 'generator_version', 'seed', 'split', 'rule_id',
    'task_name', 'num_episodes', 'horizon', 'n_buttons', 'array_schema',
    'initial_state_distribution', 'behavior_policy', 'rng_algorithm',
    'seed_derivation', 'split_semantics', 'numpy_version',
}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError('seed must be a nonnegative integer')
    return int(value)


def _stream(seed: int, split: str, purpose: str) -> np.random.Generator:
    # Stable across Python processes: never use Python's randomized hash().
    domain = json.dumps([SCHEMA_VERSION, GENERATOR_VERSION, seed, split, purpose],
                        ensure_ascii=True, separators=(',', ':')).encode('ascii')
    entropy = int.from_bytes(hashlib.sha256(domain).digest(), 'big')
    return np.random.Generator(np.random.PCG64(entropy))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_path(path: str | Path) -> Path:
    path = Path(path).expanduser().absolute()
    if path.suffix != '.npz':
        raise ValueError('Dataset path must have the .npz suffix')
    return path


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


@dataclass
class TrajectoryDataset:
    """E complete trajectories, each with L actions and L+1 board states.

    ``events[e,t]`` is the one-hot operation that was applied between
    ``states[e,t]`` and ``states[e,t+1]``. No rule matrix is stored.
    ``validate()`` checks schema and event/action alignment. Passing a rule
    additionally checks every state transition using that explicitly given rule.
    """

    states: np.ndarray
    actions: np.ndarray
    events: np.ndarray
    manifest: dict[str, Any]

    @property
    def num_episodes(self) -> int:
        return int(self.states.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])

    @property
    def n_buttons(self) -> int:
        return int(self.states.shape[2])

    def validate(self, rule: 'PuzzleRule | None' = None) -> None:
        for name, array, dtype, ndim in (
            ('states', self.states, np.uint8, 3),
            ('actions', self.actions, np.int64, 2),
            ('events', self.events, np.uint8, 3),
        ):
            if not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype) or array.ndim != ndim:
                raise ValueError(f'{name} must be a {ndim}D {np.dtype(dtype)} NumPy array')
        episodes, states_per_episode, buttons = self.states.shape
        horizon = self.actions.shape[1]
        if episodes < 1 or buttons < 1 or horizon < 1:
            raise ValueError('Dataset dimensions must be positive')
        if states_per_episode != horizon + 1:
            raise ValueError('states must retain L+1 states for L actions')
        if self.actions.shape != (episodes, horizon):
            raise ValueError('actions shape does not align with states')
        if self.events.shape != (episodes, horizon, buttons):
            raise ValueError('events shape does not align with states/actions')
        if not np.all(self.states <= 1) or not np.all(self.events <= 1):
            raise ValueError('states and events must contain binary 0/1 values')
        if not np.all((self.actions >= 0) & (self.actions < buttons)):
            raise ValueError('actions must contain valid button indices')
        if not np.all(self.events.sum(axis=-1) == 1):
            raise ValueError('Each operation must have exactly one event')
        if not np.array_equal(np.argmax(self.events, axis=-1), self.actions):
            raise ValueError('events must be one-hot encodings of actions')

        meta = self.manifest
        if not isinstance(meta, dict) or set(meta) != _MANIFEST_KEYS:
            raise ValueError('Unexpected or missing dataset manifest fields')
        expected = {
            'schema_version': SCHEMA_VERSION,
            'generator_version': GENERATOR_VERSION,
            'num_episodes': episodes, 'horizon': horizon, 'n_buttons': buttons,
            'array_schema': _ARRAY_SCHEMA,
            'initial_state_distribution': 'independent_uniform_binary',
            'behavior_policy': 'independent_uniform_button',
            'rng_algorithm': 'numpy.PCG64',
            'seed_derivation': 'sha256(schema,generator,seed,split,purpose)',
            'split_semantics': _SPLIT_SEMANTICS,
        }
        for key, expected_value in expected.items():
            if meta[key] != expected_value:
                raise ValueError(f'Invalid manifest {key}')
        for name in ('num_episodes', 'horizon', 'n_buttons'):
            _positive_int(meta[name], name)
        _seed(meta['seed'])
        for name in ('split', 'rule_id', 'task_name', 'numpy_version'):
            if not isinstance(meta[name], str) or not meta[name].strip():
                raise ValueError(f'Manifest {name} must be a nonempty string')
        if rule is not None:
            if (rule.n_buttons != buttons or rule.rule_id != meta['rule_id']
                    or rule.name != meta['task_name']):
                raise ValueError('Supplied rule does not match the dataset identity')
            for episode in range(episodes):
                for step in range(horizon):
                    expected_state = rule.transition(
                        self.states[episode, step].copy(), int(self.actions[episode, step]))
                    if not np.array_equal(expected_state, self.states[episode, step + 1]):
                        raise ValueError(f'Rule transition mismatch at episode {episode}, step {step}')

    def save(self, path: str | Path) -> Path:
        """Publish a new NPZ and a sibling JSON manifest without overwriting.

        Both files are prepared in the destination directory. Exclusive hard
        links publish complete files. If the JSON cannot be published, the NPZ
        created by this call is removed. Readers require both files and SHA256.
        """
        self.validate()
        path = _archive_path(path)
        manifest_path = path.with_suffix('.json')
        if os.path.lexists(path) or os.path.lexists(manifest_path):
            raise FileExistsError('Refusing to overwrite the dataset or manifest')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: list[Path] = []
        published_archive = False
        archive_identity = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent,
                                             prefix=f'.{path.name}.', suffix='.tmp',
                                             delete=False) as stream:
                temporary_archive = Path(stream.name)
                temporary.append(temporary_archive)
                np.savez_compressed(stream, states=self.states, actions=self.actions, events=self.events)
                stream.flush()
                os.fsync(stream.fileno())
            manifest = {
                'dataset': self.manifest,
                'artifact': {'filename': path.name, 'sha256': _sha256(temporary_archive)},
            }
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                             prefix=f'.{manifest_path.name}.', suffix='.tmp',
                                             delete=False) as stream:
                temporary_manifest = Path(stream.name)
                temporary.append(temporary_manifest)
                json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            stat = temporary_archive.stat()
            archive_identity = (stat.st_dev, stat.st_ino)
            os.link(temporary_archive, path)
            published_archive = True
            os.link(temporary_manifest, manifest_path)
        except BaseException:
            if published_archive:
                try:
                    current = path.lstat()
                    if (current.st_dev, current.st_ino) == archive_identity:
                        path.unlink()
                except FileNotFoundError:
                    pass
            raise
        finally:
            for item in temporary:
                item.unlink(missing_ok=True)
        return path


def generate_dataset(rule: 'PuzzleRule', num_episodes: int = 1000, horizon: int = 30,
                     seed: int = 0, split: str = 'train') -> TrajectoryDataset:
    """Generate independent uniform-operation trajectories for one fixed rule.

    Split labels select stable independent random streams. This is a trajectory
    split, not a claim that boards in another split were never seen in training.
    """
    episodes = _positive_int(num_episodes, 'num_episodes')
    horizon = _positive_int(horizon, 'horizon')
    buttons = _positive_int(rule.n_buttons, 'rule.n_buttons')
    seed = _seed(seed)
    if not isinstance(split, str) or not split.strip():
        raise ValueError('split must be a nonempty string')
    initial_rng = _stream(seed, split, 'initial_states')
    action_rng = _stream(seed, split, 'actions')
    states = np.empty((episodes, horizon + 1, buttons), dtype=np.uint8)
    states[:, 0] = initial_rng.integers(0, 2, size=(episodes, buttons), dtype=np.uint8)
    actions = action_rng.integers(0, buttons, size=(episodes, horizon), dtype=np.int64)
    events = np.eye(buttons, dtype=np.uint8)[actions]
    for episode in range(episodes):
        for step in range(horizon):
            successor = np.asarray(rule.transition(states[episode, step].copy(), int(actions[episode, step])))
            if successor.shape != (buttons,) or not np.all((successor == 0) | (successor == 1)):
                raise ValueError('Rule returned an invalid binary successor')
            states[episode, step + 1] = successor
    dataset = TrajectoryDataset(states, actions, events, {
        'schema_version': SCHEMA_VERSION, 'generator_version': GENERATOR_VERSION,
        'seed': seed, 'split': split, 'rule_id': rule.rule_id, 'task_name': rule.name,
        'num_episodes': episodes, 'horizon': horizon, 'n_buttons': buttons,
        'array_schema': dict(_ARRAY_SCHEMA),
        'initial_state_distribution': 'independent_uniform_binary',
        'behavior_policy': 'independent_uniform_button', 'rng_algorithm': 'numpy.PCG64',
        'seed_derivation': 'sha256(schema,generator,seed,split,purpose)',
        'split_semantics': _SPLIT_SEMANTICS, 'numpy_version': np.__version__,
    })
    dataset.validate(rule)
    return dataset


def load_dataset(path: str | Path) -> TrajectoryDataset:
    """Load a checked artifact, without pickle or access to its hidden rule."""
    path = _archive_path(path)
    with path.with_suffix('.json').open(encoding='utf-8') as stream:
        manifest = json.load(stream, object_pairs_hook=_no_duplicate_keys)
    if not isinstance(manifest, dict) or set(manifest) != {'dataset', 'artifact'}:
        raise ValueError('Invalid artifact manifest schema')
    artifact = manifest['artifact']
    if not isinstance(artifact, dict) or set(artifact) != {'filename', 'sha256'}:
        raise ValueError('Invalid artifact metadata')
    checksum = artifact['sha256']
    if artifact['filename'] != path.name:
        raise ValueError('Manifest filename does not match the NPZ file')
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
        raise ValueError('Invalid SHA256 checksum in manifest')
    if _sha256(path) != checksum:
        raise ValueError('Dataset SHA256 checksum mismatch')
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 3 or set(archive.files) != {'states', 'actions', 'events'}:
            raise ValueError('Unexpected or missing NPZ fields')
        dataset = TrajectoryDataset(archive['states'], archive['actions'], archive['events'], manifest['dataset'])
    dataset.validate()
    return dataset


def relabel_transitions(dataset: TrajectoryDataset, episode_indices, step_indices,
                        goals) -> dict[str, np.ndarray]:
    """Attach explicit goal boards to transitions without consulting a rule.

    Supplied goals are not promised to be reachable. Current-state goals are
    rejected: these would start an already successful, terminal task. Success
    is measured after the action, independently of a goal's dataset index.
    A last step that does not succeed is truncated and retains bootstrapping.
    """
    dataset.validate()
    episodes, steps = np.asarray(episode_indices), np.asarray(step_indices)
    if (episodes.ndim != 1 or steps.ndim != 1 or episodes.shape != steps.shape
            or len(episodes) == 0 or episodes.dtype.kind not in 'iu'
            or steps.dtype.kind not in 'iu'):
        raise ValueError('episode_indices and step_indices must be nonempty, aligned integer vectors')
    if not np.all((episodes >= 0) & (episodes < dataset.num_episodes)):
        raise ValueError('episode index is outside the dataset')
    if not np.all((steps >= 0) & (steps < dataset.horizon)):
        raise ValueError('step index is outside the trajectory')
    episodes, steps = episodes.astype(np.int64), steps.astype(np.int64)
    goals = np.asarray(goals)
    if (goals.shape != (len(episodes), dataset.n_buttons) or goals.dtype.kind not in 'buif'
            or not np.all((goals == 0) | (goals == 1))):
        raise ValueError('goals must be aligned binary board vectors')
    goals = goals.astype(np.uint8, copy=True)
    if np.any(np.all(dataset.states[episodes, steps] == goals, axis=1)):
        raise ValueError('A goal must differ from the current state; the task is already successful')
    return _relabel_checked(dataset, episodes, steps, goals)


def _relabel_checked(dataset: TrajectoryDataset, episodes: np.ndarray,
                     steps: np.ndarray, goals: np.ndarray) -> dict[str, np.ndarray]:
    """Build a batch from already validated data, indices, and distinct goals."""
    observations = dataset.states[episodes, steps]
    next_observations = dataset.states[episodes, steps + 1]
    previous_events = np.zeros((len(episodes), dataset.n_buttons), dtype=np.uint8)
    has_previous = steps > 0
    previous_events[has_previous] = dataset.events[episodes[has_previous], steps[has_previous] - 1]
    success = np.all(next_observations == goals, axis=1)
    return {
        'observations': observations,
        'next_observations': next_observations,
        'actions': dataset.actions[episodes, steps],
        'events': dataset.events[episodes, steps],
        'previous_events': previous_events,
        'goals': goals,
        'rewards': np.where(success, 0.0, -1.0).astype(np.float32),
        'terminated': success,
        'truncated': (steps == dataset.horizon - 1) & ~success,
        'bootstrap_masks': (~success).astype(np.float32),
    }


class GoalSampler:
    """Cache future goals once and own a persistent RNG for repeated batches.

    Construction takes a deep snapshot, validates it, and marks its arrays
    read-only. Later edits to the caller's dataset cannot change sampling or
    invalidate the cached indices. Returned batch arrays are independent copies.
    Construct one sampler per training/validation stream and repeatedly call
    ``sample``. Initialization costs O(E*L*L*N); batches do not scan trajectories
    or revalidate the complete dataset.

    Transitions are uniform among those with a strictly later board different
    from the current board. A goal is uniform among its eligible future indices,
    including the final state. Repeated equal boards retain their multiplicity.
    The last step's only eligible future goal is its successor, so sampled last
    steps always succeed. Use ``relabel_transitions`` for explicit goals and
    timeout cases; time truncation retains bootstrapping.
    """

    def __init__(self, dataset: TrajectoryDataset, seed: int = 0):
        seed = _seed(seed)
        if not isinstance(dataset, TrajectoryDataset):
            raise TypeError('dataset must be a TrajectoryDataset')
        self._dataset = deepcopy(dataset)
        self._dataset.validate()
        for array in (self._dataset.states, self._dataset.actions, self._dataset.events):
            array.setflags(write=False)
        episodes, steps, futures = [], [], []
        for episode in range(self._dataset.num_episodes):
            for step in range(self._dataset.horizon):
                different = np.any(
                    self._dataset.states[episode, step + 1:] != self._dataset.states[episode, step], axis=1)
                future = np.flatnonzero(different) + step + 1
                if len(future):
                    episodes.append(episode)
                    steps.append(step)
                    futures.append(future)
        if not futures:
            raise ValueError('No transition has a later goal different from its current state')
        self._episodes = np.asarray(episodes, dtype=np.int64)
        self._steps = np.asarray(steps, dtype=np.int64)
        self._future_offsets = np.concatenate((
            np.zeros(1, dtype=np.int64),
            np.cumsum([len(future) for future in futures], dtype=np.int64),
        ))
        self._future_indices = np.concatenate(futures).astype(np.int64, copy=False)
        for array in (self._episodes, self._steps, self._future_offsets, self._future_indices):
            array.setflags(write=False)
        self._rng = np.random.Generator(np.random.PCG64(seed))

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Draw the next batch and advance this sampler's RNG state."""
        batch_size = _positive_int(batch_size, 'batch_size')
        picks = self._rng.integers(len(self._episodes), size=batch_size)
        lower = self._future_offsets[picks]
        lengths = self._future_offsets[picks + 1] - lower
        positions = lower + self._rng.integers(lengths)
        episodes, steps = self._episodes[picks], self._steps[picks]
        goal_steps = self._future_indices[positions]
        goals = self._dataset.states[episodes, goal_steps]
        return _relabel_checked(self._dataset, episodes, steps, goals)


def sample_goal_transitions(dataset: TrajectoryDataset, batch_size: int,
                            seed: int = 0) -> dict[str, np.ndarray]:
    """One-shot reproducible convenience wrapper around ``GoalSampler``.

    Repeating this call with the same data and seed returns the same batch and
    rebuilds the cache. Training loops should create one ``GoalSampler`` and
    call its ``sample`` method so the RNG advances and preprocessing is reused.
    """
    _positive_int(batch_size, 'batch_size')
    return GoalSampler(dataset, seed=seed).sample(batch_size)


__all__ = ['TrajectoryDataset', 'GoalSampler', 'generate_dataset', 'load_dataset',
           'relabel_transitions', 'sample_goal_transitions']
