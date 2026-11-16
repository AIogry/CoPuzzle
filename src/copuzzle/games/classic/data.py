"""Offline operation trajectories with explicit successors and checked artifacts.

State tokens and operation dimensions are independent: Sliding Puzzle has four direction actions; Hanoi has six directed peg moves.
The legal action set depends on the current state. Loading and goal relabeling never query
a transition rule or a shortest-path oracle. Splits have independent random
streams, but may contain overlapping states.
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
    from .rules import ClassicRule

SCHEMA_VERSION = 'copuzzle.classic_trajectory.v1'
GENERATOR_VERSION = 'copuzzle.classic_uniform_legal_operations.v1'
_SPLIT_SEMANTICS = 'Independent RNG streams; states may overlap between splits.'
_ARRAY_SCHEMA = {
    'states': 'int32[E,L+1,N]',
    'actions': 'int64[E,L]',
    'events': 'uint8[E,L,A]',
    'action_masks': 'uint8[E,L+1,A]',
}
_MANIFEST_KEYS = {
    'schema_version', 'generator_version', 'seed', 'split', 'rule_id',
    'task_name', 'rule_spec', 'num_episodes', 'horizon', 'state_size', 'n_actions', 'array_schema',
    'family', 'initial_state_distribution', 'behavior_policy', 'rng_algorithm',
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


def _contains_boolean(value: Any) -> bool:
    # NumPy would silently coerce [True, 2, 3] into integer token values.
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    return isinstance(value, (bool, np.bool_))


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


def _validate_states(values: np.ndarray, state_size: int, family: str, name: str) -> None:
    """Pure static representation checks; never query dynamics or an oracle."""
    if family == 'sliding':
        if not np.all(np.sort(values, axis=-1) == np.arange(state_size)):
            raise ValueError(f'{name} must contain permutations of tiles 0 through N-1')
    elif family == 'hanoi':
        if not np.all((values >= 0) & (values < 3)):
            raise ValueError(f'{name} must contain peg assignments 0, 1 or 2')
    else:
        raise ValueError('Unsupported classic game family')


def _validate_rule_metadata(meta: dict[str, Any], state_size: int, n_actions: int) -> None:
    # This pure schema helper checks canonical action definitions and identity;
    # it does not evaluate any transition, reachability, distance, or solution.
    from .rules import validate_rule_spec

    spec = validate_rule_spec(meta['rule_spec'])
    if (spec['state_size'] != state_size or spec['n_actions'] != n_actions
            or spec['rule_id'] != meta['rule_id'] or spec['name'] != meta['task_name'] or spec['family'] != meta['family']):
        raise ValueError('Dataset identity does not match its rule_spec')


def _array_metadata(dataset: 'TrajectoryDataset') -> dict[str, Any]:
    return {
        name: {'shape': list(getattr(dataset, name).shape),
               'dtype': str(getattr(dataset, name).dtype)}
        for name in ('states', 'actions', 'events', 'action_masks')
    }


def _validate_array_metadata(value: Any, dataset: 'TrajectoryDataset') -> None:
    expected = _array_metadata(dataset)
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError('Invalid artifact array metadata')
    for name, description in value.items():
        if (not isinstance(description, dict) or set(description) != {'shape', 'dtype'}
                or not isinstance(description['shape'], list)
                or any(type(size) is not int for size in description['shape'])
                or description != expected[name]):
            raise ValueError('Artifact array shape/dtype metadata mismatch')


@dataclass
class TrajectoryDataset:
    """E complete trajectories, each with L actions and L+1 board states.

    ``events[e,t]`` is the one-hot operation that was applied between
    ``states[e,t]`` and ``states[e,t+1]``. State and action widths differ.
    ``validate()`` checks schema and event/action alignment. Passing a rule
    additionally checks every state transition using that explicitly given rule.
    """

    states: np.ndarray
    actions: np.ndarray
    events: np.ndarray
    action_masks: np.ndarray
    manifest: dict[str, Any]

    @property
    def num_episodes(self) -> int:
        return int(self.states.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])

    @property
    def state_size(self) -> int:
        return int(self.states.shape[2])

    @property
    def n_actions(self) -> int:
        return int(self.events.shape[2])

    def validate(self, rule: 'ClassicRule | None' = None) -> None:
        for name, array, dtype, ndim in (
            ('states', self.states, np.int32, 3),
            ('actions', self.actions, np.int64, 2),
            ('events', self.events, np.uint8, 3),
            ('action_masks', self.action_masks, np.uint8, 3),
        ):
            if not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype) or array.ndim != ndim:
                raise ValueError(f'{name} must be a {ndim}D {np.dtype(dtype)} NumPy array')
        episodes, states_per_episode, tokens = self.states.shape
        n_actions = self.events.shape[2]
        horizon = self.actions.shape[1]
        if episodes < 1 or tokens < 2 or n_actions < 1 or horizon < 1:
            raise ValueError('Dataset dimensions must be positive')
        if states_per_episode != horizon + 1:
            raise ValueError('states must retain L+1 states for L actions')
        if self.actions.shape != (episodes, horizon):
            raise ValueError('actions shape does not align with states')
        if self.events.shape != (episodes, horizon, n_actions):
            raise ValueError('events shape does not align with states/actions')
        if self.action_masks.shape != (episodes, horizon + 1, n_actions):
            raise ValueError('action_masks shape does not align with states/actions')
        if not np.all(self.action_masks <= 1) or not np.all(self.action_masks.any(axis=-1)):
            raise ValueError('action_masks must be binary with at least one legal action per state')
        if not isinstance(self.manifest, dict):
            raise ValueError('Dataset manifest must be a dictionary')
        _validate_states(self.states, tokens, self.manifest.get('family'), 'states')
        if not np.all(self.events <= 1):
            raise ValueError('events must contain binary 0/1 values')
        if not np.all((self.actions >= 0) & (self.actions < n_actions)):
            raise ValueError('actions must contain valid operation indices')
        if not np.all(np.take_along_axis(self.action_masks[:, :-1], self.actions[..., None], axis=-1)):
            raise ValueError('Stored actions must be legal according to stored action_masks')
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
            'num_episodes': episodes, 'horizon': horizon, 'state_size': tokens, 'n_actions': n_actions,
            'array_schema': _ARRAY_SCHEMA,
            'initial_state_distribution': 'independent_uniform_valid_representation',
            'behavior_policy': 'uniform_legal_action',
            'rng_algorithm': 'numpy.PCG64',
            'seed_derivation': 'sha256(schema,generator,seed,split,purpose)',
            'split_semantics': _SPLIT_SEMANTICS,
        }
        for key, expected_value in expected.items():
            if meta[key] != expected_value:
                raise ValueError(f'Invalid manifest {key}')
        for name in ('num_episodes', 'horizon', 'state_size', 'n_actions'):
            _positive_int(meta[name], name)
        _seed(meta['seed'])
        for name in ('split', 'rule_id', 'task_name', 'numpy_version'):
            if not isinstance(meta[name], str) or not meta[name].strip():
                raise ValueError(f'Manifest {name} must be a nonempty string')
        _validate_rule_metadata(meta, tokens, n_actions)
        if rule is not None:
            if (rule.state_size != tokens or rule.n_actions != n_actions
                    or rule.rule_id != meta['rule_id'] or rule.name != meta['task_name']
                    or rule.spec() != meta['rule_spec']):
                raise ValueError('Supplied rule does not match the dataset identity')
            for episode in range(episodes):
                for step in range(horizon + 1):
                    if not np.array_equal(rule.action_mask(self.states[episode, step].copy()), self.action_masks[episode, step]):
                        raise ValueError(f'Rule action mask mismatch at episode {episode}, step {step}')
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
                np.savez_compressed(stream, states=self.states, actions=self.actions, events=self.events, action_masks=self.action_masks)
                stream.flush()
                os.fsync(stream.fileno())
            manifest = {
                'dataset': self.manifest,
                'artifact': {'filename': path.name, 'sha256': _sha256(temporary_archive),
                             'arrays': _array_metadata(self)},
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


def generate_dataset(rule: 'ClassicRule', num_episodes: int = 1000, horizon: int = 30,
                     seed: int = 0, split: str = 'train') -> TrajectoryDataset:
    """Generate independent uniform-legal-action trajectories for one fixed rule.

    Split labels select stable independent random streams. This is a trajectory
    split, not a claim that boards in another split were never seen in training.
    """
    episodes = _positive_int(num_episodes, 'num_episodes')
    horizon = _positive_int(horizon, 'horizon')
    tokens = _positive_int(rule.state_size, 'rule.state_size')
    n_actions = _positive_int(rule.n_actions, 'rule.n_actions')
    seed = _seed(seed)
    if not isinstance(split, str) or not split.strip():
        raise ValueError('split must be a nonempty string')
    initial_rng = _stream(seed, split, 'initial_states')
    action_rng = _stream(seed, split, 'actions')
    states = np.empty((episodes, horizon + 1, tokens), dtype=np.int32)
    actions = np.empty((episodes, horizon), dtype=np.int64)
    action_masks = np.empty((episodes, horizon + 1, n_actions), dtype=np.uint8)
    for episode in range(episodes):
        states[episode, 0] = rule.sample_state(initial_rng)
        for step in range(horizon):
            mask = rule.action_mask(states[episode, step].copy())
            action_masks[episode, step] = mask
            action = int(action_rng.choice(np.flatnonzero(mask)))
            actions[episode, step] = action
            successor = np.asarray(rule.transition(states[episode, step].copy(), action))
            if successor.shape != (tokens,) or successor.dtype != np.dtype(np.int32):
                raise ValueError('Rule returned an invalid int32 successor')
            _validate_states(successor, tokens, rule.family, 'successor')
            states[episode, step + 1] = successor
        action_masks[episode, horizon] = rule.action_mask(states[episode, horizon].copy())
    events = np.eye(n_actions, dtype=np.uint8)[actions]
    dataset = TrajectoryDataset(states, actions, events, action_masks, {
        'schema_version': SCHEMA_VERSION, 'generator_version': GENERATOR_VERSION,
        'seed': seed, 'split': split, 'rule_id': rule.rule_id, 'task_name': rule.name, 'family': rule.family,
        'rule_spec': deepcopy(rule.spec()),
        'num_episodes': episodes, 'horizon': horizon, 'state_size': tokens, 'n_actions': n_actions,
        'array_schema': dict(_ARRAY_SCHEMA),
        'initial_state_distribution': 'independent_uniform_valid_representation',
        'behavior_policy': 'uniform_legal_action', 'rng_algorithm': 'numpy.PCG64',
        'seed_derivation': 'sha256(schema,generator,seed,split,purpose)',
        'split_semantics': _SPLIT_SEMANTICS, 'numpy_version': np.__version__,
    })
    dataset.validate(rule)
    return dataset


def load_dataset(path: str | Path) -> TrajectoryDataset:
    """Load a checked artifact, without pickle or executing its rule."""
    path = _archive_path(path)
    with path.with_suffix('.json').open(encoding='utf-8') as stream:
        manifest = json.load(stream, object_pairs_hook=_no_duplicate_keys)
    if not isinstance(manifest, dict) or set(manifest) != {'dataset', 'artifact'}:
        raise ValueError('Invalid artifact manifest schema')
    artifact = manifest['artifact']
    if not isinstance(artifact, dict) or set(artifact) != {'filename', 'sha256', 'arrays'}:
        raise ValueError('Invalid artifact metadata')
    checksum = artifact['sha256']
    if artifact['filename'] != path.name:
        raise ValueError('Manifest filename does not match the NPZ file')
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
        raise ValueError('Invalid SHA256 checksum in manifest')
    if _sha256(path) != checksum:
        raise ValueError('Dataset SHA256 checksum mismatch')
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 4 or set(archive.files) != {'states', 'actions', 'events', 'action_masks'}:
            raise ValueError('Unexpected or missing NPZ fields')
        dataset = TrajectoryDataset(archive['states'], archive['actions'], archive['events'], archive['action_masks'], manifest['dataset'])
    dataset.validate()
    _validate_array_metadata(artifact['arrays'], dataset)
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
    if _contains_boolean(episode_indices) or _contains_boolean(step_indices):
        raise ValueError('episode_indices and step_indices must contain integers, not booleans')
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
    if _contains_boolean(goals):
        raise ValueError('goals must contain integer tokens, not booleans')
    goals = np.asarray(goals)
    if goals.shape != (len(episodes), dataset.state_size) or goals.dtype.kind not in 'iu':
        raise ValueError('goals must be aligned integer state vectors')
    _validate_states(goals, dataset.state_size, dataset.manifest['family'], 'goals')
    goals = goals.astype(np.int32, copy=True)
    if np.any(np.all(dataset.states[episodes, steps] == goals, axis=1)):
        raise ValueError('A goal must differ from the current state; the task is already successful')
    return _relabel_checked(dataset, episodes, steps, goals)


def _relabel_checked(dataset: TrajectoryDataset, episodes: np.ndarray,
                     steps: np.ndarray, goals: np.ndarray) -> dict[str, np.ndarray]:
    """Build a batch from already validated data, indices, and distinct goals."""
    observations = dataset.states[episodes, steps]
    next_observations = dataset.states[episodes, steps + 1]
    previous_events = np.zeros((len(episodes), dataset.n_actions), dtype=np.uint8)
    has_previous = steps > 0
    previous_events[has_previous] = dataset.events[episodes[has_previous], steps[has_previous] - 1]
    success = np.all(next_observations == goals, axis=1)
    return {
        'observations': observations,
        'next_observations': next_observations,
        'actions': dataset.actions[episodes, steps],
        'events': dataset.events[episodes, steps],
        'previous_events': previous_events,
        'goals': goals.copy(),
        'rewards': np.where(success, 0.0, -1.0).astype(np.float32),
        'terminated': success,
        'truncated': (steps == dataset.horizon - 1) & ~success,
        'bootstrap_masks': (~success).astype(np.float32),
        'action_masks': dataset.action_masks[episodes, steps],
        'next_action_masks': dataset.action_masks[episodes, steps + 1],
        'episode_indices': episodes.copy(),
        'step_indices': steps.copy(),
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
        for array in (self._dataset.states, self._dataset.actions, self._dataset.events, self._dataset.action_masks):
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
        digest = hashlib.sha256(json.dumps(self._dataset.manifest, sort_keys=True,
                                          separators=(',', ':')).encode('utf-8'))
        for array in (self._dataset.states, self._dataset.actions, self._dataset.events, self._dataset.action_masks):
            digest.update(array.tobytes(order='C'))
        self._dataset_sha256 = digest.hexdigest()

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
        batch = _relabel_checked(self._dataset, episodes, steps, goals)
        batch['goal_indices'] = goal_steps.copy()
        return batch

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable RNG checkpoint bound to this dataset."""
        return {
            'schema_version': 'copuzzle.classic_goal_sampler.v1',
            'dataset_sha256': self._dataset_sha256,
            'rng_state': deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a checkpoint only if it belongs to the same dataset snapshot."""
        if (not isinstance(state, dict)
                or set(state) != {'schema_version', 'dataset_sha256', 'rng_state'}
                or state['schema_version'] != 'copuzzle.classic_goal_sampler.v1'
                or state['dataset_sha256'] != self._dataset_sha256):
            raise ValueError('Sampler checkpoint schema or dataset identity mismatch')
        candidate = np.random.PCG64()
        try:
            candidate.state = deepcopy(state['rng_state'])
        except (TypeError, ValueError, KeyError, OverflowError) as error:
            raise ValueError('Invalid sampler RNG state') from error
        self._rng = np.random.Generator(candidate)


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
           'relabel_transitions', 'sample_goal_transitions', 'dataset_identity', 'dataset_coverage']


def dataset_identity(dataset: TrajectoryDataset) -> dict[str, Any]:
    """Portable content identity, independent of ZIP timestamps or compression."""
    dataset.validate()
    digest = hashlib.sha256(json.dumps(dataset.manifest, sort_keys=True,
                                      separators=(',', ':')).encode('utf-8'))
    for name in ('states', 'actions', 'events', 'action_masks'):
        digest.update(getattr(dataset, name).tobytes(order='C'))
    return {'schema_version': SCHEMA_VERSION, 'rule_id': dataset.manifest['rule_id'],
            'split': dataset.manifest['split'], 'content_sha256': digest.hexdigest()}


def dataset_coverage(dataset: TrajectoryDataset) -> dict[str, Any]:
    """Measure observed coverage without querying a rule or claiming novelty."""
    dataset.validate()
    return {
        'episodes': dataset.num_episodes, 'horizon': dataset.horizon,
        'transitions': int(dataset.actions.size), 'state_size': dataset.state_size,
        'n_actions': dataset.n_actions,
        'unique_states': int(len(np.unique(dataset.states.reshape(-1, dataset.state_size), axis=0))),
        'action_counts': np.bincount(dataset.actions.ravel(), minlength=dataset.n_actions).tolist(),
        'legal_action_count_min': int(dataset.action_masks.sum(axis=-1).min()),
        'legal_action_count_max': int(dataset.action_masks.sum(axis=-1).max()),
        'identity': dataset_identity(dataset),
    }
