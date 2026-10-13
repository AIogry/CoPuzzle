"""Complete continuous-control episodes and goal relabeling for physical puzzles.

Artifacts are independent of the abstract one-button trajectory format. Loading
uses NumPy and public structure data only; it never imports a physics engine.
Press counts describe measured events, not intended controller button targets.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ..structure import PublicStructure

PHYSICS_TRAJECTORY_SCHEMA = 'copuzzle.physics_trajectory.v1'
_DYNAMIC = {
    'board': (np.uint8, ('n',)), 'goal': (np.uint8, ('n',)),
    'joint_positions': (np.float64, (6,)), 'joint_velocities': (np.float64, (6,)),
    'end_effector_position': (np.float64, (3,)),
    'end_effector_orientation': (np.float64, (4,)),
    'button_positions': (np.float64, ('n', 3)),
    'button_displacements': (np.float64, ('n',)),
    'button_velocities': (np.float64, ('n',)),
    'button_armed': (np.uint8, ('n',)),
    'previous_event_counts': (np.int32, ('n',)),
    'steps_remaining': (np.int32, (1,)),
}
_ARRAYS = ('actions', 'event_counts', 'event_steps', 'event_substeps', 'event_buttons',
           'rewards', 'terminated', 'truncated', 'collection_cutoff', 'button_rest_positions')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('JSON metadata keys must be strings')
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError('JSON metadata must be finite')
        return value
    raise ValueError('Metadata must contain JSON values only')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _array(value, dtype, shape, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError(f'{name} must be a {np.dtype(dtype)} array with shape {shape}')
    if value.dtype.kind == 'f' and not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be finite')


def _substeps(configuration):
    try:
        physics = float(configuration['physics_timestep'])
        control = float(configuration['control_timestep'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Configuration must declare physics/control timesteps') from error
    if not np.isfinite(physics) or not np.isfinite(control) or physics <= 0 or control <= 0:
        raise ValueError('Physics/control timesteps must be finite and positive')
    count = round(control / physics)
    if count < 1 or abs(control / physics - count) > 1e-8:
        raise ValueError('Control timestep must be a positive multiple of physics timestep')
    if 'substeps' in configuration and configuration['substeps'] != count:
        raise ValueError('Configuration substeps disagree with timesteps')
    return count


@dataclass
class PhysicsTrajectory:
    """One variable-length episode with T actions and T+1 dynamic observations.

    ``button_rest_positions`` and ``structure`` are static and saved once.
    ``collection_cutoff`` means the recorder stopped before the environment did.
    Metadata and configuration are provenance, not automatic learner inputs.
    """
    observations: dict[str, np.ndarray]
    actions: np.ndarray
    event_counts: np.ndarray
    event_steps: np.ndarray
    event_substeps: np.ndarray
    event_buttons: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    collection_cutoff: np.ndarray
    button_rest_positions: np.ndarray
    structure: PublicStructure
    configuration: dict
    metadata: dict

    @property
    def num_steps(self):
        return len(self.actions)

    @property
    def n_buttons(self):
        return self.structure.n_buttons

    def validate(self):
        if not isinstance(self.structure, PublicStructure):
            raise ValueError('structure must be a PublicStructure')
        if not isinstance(self.actions, np.ndarray) or self.actions.ndim != 2:
            raise ValueError('actions must be a two-dimensional NumPy array')
        steps, buttons = self.num_steps, self.n_buttons
        if steps < 1:
            raise ValueError('An episode must contain at least one transition')
        if not isinstance(self.observations, dict) or set(self.observations) != set(_DYNAMIC):
            raise ValueError('Unexpected or missing physical observation fields')
        for name, (dtype, dimensions) in _DYNAMIC.items():
            shape = (steps + 1,) + tuple(buttons if dim == 'n' else dim for dim in dimensions)
            _array(self.observations[name], dtype, shape, name)
        for name in ('board', 'goal', 'button_armed'):
            if np.any(self.observations[name] > 1):
                raise ValueError(f'{name} must be binary')
        _array(self.actions, np.float32, (steps, 3), 'actions')
        if np.any(np.abs(self.actions) > 1):
            raise ValueError('Actions must be within [-1, 1]')
        _array(self.button_rest_positions, np.float64, (buttons, 3), 'button_rest_positions')
        _array(self.event_counts, np.int32, (steps, buttons), 'event_counts')
        if np.any(self.event_counts < 0):
            raise ValueError('Event counts must be nonnegative')
        previous = self.observations['previous_event_counts']
        if np.any(previous[0]) or not np.array_equal(previous[1:], self.event_counts):
            raise ValueError('previous_event_counts must align with actual preceding transitions')
        goal = self.observations['goal']
        board = self.observations['board']
        if not np.all(goal == goal[0]):
            raise ValueError('The collection goal must remain fixed within an episode')
        if np.any(np.all(board[:-1] == goal[:-1], axis=1)):
            raise ValueError('An episode cannot continue after its goal was reached')
        _array(self.rewards, np.float32, (steps,), 'rewards')
        for name in ('terminated', 'truncated', 'collection_cutoff'):
            _array(getattr(self, name), np.bool_, (steps,), name)
        end_count = self.terminated.astype(int) + self.truncated + self.collection_cutoff
        if np.any(end_count[:-1]) or end_count[-1] != 1:
            raise ValueError('Exactly one ending condition is required, at the final step only')
        success = np.all(board[1:] == goal[1:], axis=1)
        if not np.array_equal(self.terminated, success):
            raise ValueError('terminated must mean post-action board equals goal')
        if not np.array_equal(self.rewards, np.where(success, 0.0, -1.0).astype(np.float32)):
            raise ValueError('Rewards must be 0 on success and -1 otherwise')
        if not isinstance(self.configuration, dict) or not isinstance(self.metadata, dict):
            raise ValueError('Configuration and metadata must be JSON dictionaries')
        configuration, metadata = _json_value(self.configuration), _json_value(self.metadata)
        required = {'reset_seed', 'reset_options', 'configuration_sha256', 'model_sha256', 'user'}
        if set(metadata) != required or not isinstance(metadata['user'], dict) or not isinstance(metadata['reset_options'], dict):
            raise ValueError('Unexpected or missing episode provenance fields')
        _integer(metadata['reset_seed'], 'reset_seed')
        if metadata['configuration_sha256'] != hashlib.sha256(_canonical(configuration)).hexdigest():
            raise ValueError('Configuration checksum mismatch')
        if not _hash_string(metadata['model_sha256']):
            raise ValueError('Invalid model checksum')
        if configuration.get('model_sha256', metadata['model_sha256']) != metadata['model_sha256']:
            raise ValueError('Model identity differs from configuration')
        if configuration.get('schema_version') != 'copuzzle.physics_config.v1':
            raise ValueError('Unsupported physical configuration schema')
        for name in ('task_name', 'family', 'n_buttons', 'rule_id', 'structure_sha256'):
            if configuration.get(name) != getattr(self.structure, name):
                raise ValueError(f'Configuration and public structure disagree on {name}')
        if not np.array_equal(np.asarray(configuration.get('button_rest_positions')), self.button_rest_positions):
            raise ValueError('Configuration and physical button layout disagree')
        horizon = _integer(configuration.get('max_episode_steps'), 'max_episode_steps', 1)
        remaining = self.observations['steps_remaining'][:, 0]
        if steps > horizon or not np.array_equal(remaining, horizon - np.arange(steps + 1)):
            raise ValueError('steps_remaining must track the configured episode horizon')
        if configuration.get('finite_horizon_observation') != 'steps_remaining':
            raise ValueError('Configuration must declare steps_remaining')
        if not np.array_equal(self.truncated, (remaining[1:] == 0) & ~success):
            raise ValueError('truncated must mean the environment horizon expired without success')
        substeps = _substeps(configuration)
        if not isinstance(self.event_steps, np.ndarray) or self.event_steps.ndim != 1:
            raise ValueError('event_steps must be a one-dimensional NumPy array')
        count = len(self.event_steps)
        for name, dtype in (('event_steps', np.int64), ('event_substeps', np.int32), ('event_buttons', np.int32)):
            _array(getattr(self, name), dtype, (count,), name)
        if (np.any(self.event_steps < 0) or np.any(self.event_steps >= steps)
                or np.any(self.event_substeps < 0) or np.any(self.event_substeps >= substeps)
                or np.any(self.event_buttons < 0) or np.any(self.event_buttons >= buttons)):
            raise ValueError('Event log indices are outside the episode/substep/button range')
        events = list(zip(self.event_steps.tolist(), self.event_substeps.tolist(), self.event_buttons.tolist()))
        if events != sorted(set(events)):
            raise ValueError('Event log must be sorted and contain no duplicate events')
        measured = np.zeros((steps, buttons), dtype=np.int32)
        np.add.at(measured, (self.event_steps, self.event_buttons), 1)
        if not np.array_equal(measured, self.event_counts):
            raise ValueError('Event log and event counts differ')
        # Check the public Lights Out mechanism using arrays, never an executable rule.
        if self.structure.family == 'graph':
            adjacency = self.structure.adjacency
        else:
            coordinates = self.structure.grid_coordinates
            adjacency = (np.abs(coordinates[:, None] - coordinates[None, :]).sum(axis=-1) == 1)
        toggle = np.asarray(adjacency, dtype=np.int32) + np.eye(buttons, dtype=np.int32)
        effect = ((self.event_counts % 2) @ toggle.T % 2).astype(np.uint8)
        if not np.array_equal(board[1:], board[:-1] ^ effect):
            raise ValueError('Board transitions do not match the measured event counts')

    def _arrays(self):
        arrays = {f'observation__{name}': value for name, value in self.observations.items()}
        arrays.update({name: getattr(self, name) for name in _ARRAYS})
        arrays.update({f'structure__{name}': value for name, value in self.structure.observation_fields().items()})
        return arrays

    def save(self, path):
        """Publish a checked NPZ plus sibling JSON without replacing either."""
        self.validate()
        path = Path(path).expanduser().absolute()
        if path.suffix != '.npz':
            raise ValueError('Episode path must end in .npz')
        sidecar = path.with_suffix('.json')
        if os.path.lexists(path) or os.path.lexists(sidecar):
            raise FileExistsError('Refusing to replace an episode or its manifest')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary, published = [], []
        try:
            arrays = self._arrays()
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
                archive = Path(stream.name)
                temporary.append(archive)
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            payload = {
                'schema_version': PHYSICS_TRAJECTORY_SCHEMA,
                'artifact': {'filename': path.name, 'sha256': _sha(archive)},
                'num_steps': self.num_steps,
                'structure': {name: getattr(self.structure, name) for name in
                              ('task_name', 'family', 'n_buttons', 'rule_id', 'structure_sha256')},
                'configuration': _json_value(self.configuration), 'metadata': _json_value(self.metadata),
                'arrays': {name: {'dtype': array.dtype.name, 'shape': list(array.shape)} for name, array in arrays.items()},
            }
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, suffix='.tmp', delete=False) as stream:
                manifest = Path(stream.name)
                temporary.append(manifest)
                json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            for source, target in ((archive, path), (manifest, sidecar)):
                stat = source.stat()
                os.link(source, target)
                published.append((target, stat.st_dev, stat.st_ino))
            return path
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
            for item in temporary:
                item.unlink(missing_ok=True)


def load_physics_trajectory(path):
    """Read and validate a self-contained episode without importing MuJoCo."""
    path = Path(path).expanduser().absolute()
    if path.suffix != '.npz':
        raise ValueError('Episode path must end in .npz')
    with path.with_suffix('.json').open(encoding='utf-8') as stream:
        manifest = json.load(stream, object_pairs_hook=_unique_object)
    expected = {'schema_version', 'artifact', 'num_steps', 'structure', 'configuration', 'metadata', 'arrays'}
    if not isinstance(manifest, dict) or set(manifest) != expected or manifest['schema_version'] != PHYSICS_TRAJECTORY_SCHEMA:
        raise ValueError('Unsupported or incomplete physical trajectory manifest')
    artifact = manifest['artifact']
    if (not isinstance(artifact, dict) or set(artifact) != {'filename', 'sha256'}
            or artifact['filename'] != path.name or not _hash_string(artifact['sha256'])
            or _sha(path) != artifact['sha256']):
        raise ValueError('Episode archive checksum or filename mismatch')
    meta = manifest['structure']
    if not isinstance(meta, dict) or set(meta) != {'task_name', 'family', 'n_buttons', 'rule_id', 'structure_sha256'}:
        raise ValueError('Invalid episode structure metadata')
    structure_names = {'grid_shape', 'grid_coordinates'} if meta['family'] == 'grid' else {'adjacency'}
    expected_arrays = {f'observation__{name}' for name in _DYNAMIC} | set(_ARRAYS) | {f'structure__{name}' for name in structure_names}
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(expected_arrays) or set(archive.files) != expected_arrays:
            raise ValueError('Unexpected, missing, or duplicate episode arrays')
        arrays = {name: archive[name] for name in archive.files}
    if manifest['arrays'] != {name: {'dtype': value.dtype.name, 'shape': list(value.shape)} for name, value in arrays.items()}:
        raise ValueError('Episode array declarations do not match stored arrays')
    structure = PublicStructure(meta['task_name'], meta['family'], meta['n_buttons'], meta['rule_id'],
                                **{name: arrays[f'structure__{name}'] for name in structure_names})
    if structure.structure_sha256 != meta['structure_sha256']:
        raise ValueError('Episode public structure identity mismatch')
    episode = PhysicsTrajectory(
        observations={name: arrays[f'observation__{name}'] for name in _DYNAMIC},
        **{name: arrays[name] for name in _ARRAYS}, structure=structure,
        configuration=manifest['configuration'], metadata=manifest['metadata'],
    )
    episode.validate()
    if type(manifest['num_steps']) is not int or manifest['num_steps'] != episode.num_steps:
        raise ValueError('Episode length does not match its manifest')
    return episode


def record_episode(env, policy, *, seed, options=None, max_steps=None, metadata=None):
    """Record one rollout; the policy only sees public observations.

    Float32 actions are the exact values submitted to env.step. The recorder
    retains the final observation and distinguishes its own cutoff from an
    environment timeout. ``max_steps=None`` uses the environment horizon.
    It does not close the caller-owned environment.
    """
    seed = _integer(seed, 'seed')
    options = {} if options is None else deepcopy(options)
    if not isinstance(options, dict) or (metadata is not None and not isinstance(metadata, dict)):
        raise ValueError('options and metadata must be dictionaries')
    structure = env.public_structure
    configuration = _json_value(env.configuration())
    max_steps = _integer(configuration['max_episode_steps'] if max_steps is None else max_steps, 'max_steps', 1)
    provenance = dict(reset_seed=seed, reset_options=_json_value(options), user=_json_value(metadata or {}),
                      configuration_sha256=env.configuration_sha256, model_sha256=env.model_sha256)
    observation, info = env.reset(seed=seed, options=options)
    if info.get('is_success') or np.array_equal(observation['board'], observation['goal']):
        raise ValueError('Cannot record a transition from an already successful reset')
    if callable(getattr(policy, 'reset', None)):
        policy.reset()
    static = {**structure.observation_fields(), 'button_rest_positions': np.array(observation['button_rest_positions'], copy=True)}
    history = {name: [] for name in _DYNAMIC}

    def capture(current):
        if set(current) != set(_DYNAMIC) | set(static):
            raise ValueError('Environment returned unexpected physical observation fields')
        for name, value in static.items():
            if not np.array_equal(current[name], value):
                raise ValueError('Static structure/layout changed within the episode')
        for name in history:
            history[name].append(np.array(current[name], copy=True))

    capture(observation)
    actions, counts, rewards, terminal, timeouts, cutoffs, events = [], [], [], [], [], [], []
    for step in range(max_steps):
        action = np.asarray(policy(deepcopy(observation)), dtype=np.float32)
        if action.shape != (3,) or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1):
            raise ValueError('Policy must return three finite continuous actions in [-1, 1]')
        action = action.copy()
        following, reward, terminated, truncated, info = env.step(action.copy())
        actions.append(action)
        counts.append(np.array(following['previous_event_counts'], copy=True))
        rewards.append(reward)
        terminal.append(terminated)
        timeouts.append(truncated)
        cutoffs.append(step == max_steps - 1 and not (terminated or truncated))
        if not isinstance(info.get('event_log'), list):
            raise ValueError('Step info must contain the measured event_log list')
        for event in info['event_log']:
            if not isinstance(event, dict) or set(event) != {'substep', 'button'}:
                raise ValueError('Invalid measured press event')
            events.append((step, _integer(event['substep'], 'substep'), _integer(event['button'], 'button')))
        capture(following)
        observation = following
        if terminated or truncated:
            break
    episode = PhysicsTrajectory(
        observations={name: np.stack(values) for name, values in history.items()},
        actions=np.stack(actions), event_counts=np.stack(counts),
        event_steps=np.array([event[0] for event in events], dtype=np.int64),
        event_substeps=np.array([event[1] for event in events], dtype=np.int32),
        event_buttons=np.array([event[2] for event in events], dtype=np.int32),
        rewards=np.array(rewards, dtype=np.float32), terminated=np.array(terminal, dtype=bool),
        truncated=np.array(timeouts, dtype=bool), collection_cutoff=np.array(cutoffs, dtype=bool),
        button_rest_positions=static['button_rest_positions'], structure=structure,
        configuration=configuration, metadata=provenance,
    )
    episode.validate()
    return episode


def _relabel_checked(episode, indices, goals):
    observations = {name: value[indices].copy() for name, value in episode.observations.items()}
    following = {name: value[indices + 1].copy() for name, value in episode.observations.items()}
    observations['goal'], following['goal'] = goals.copy(), goals.copy()
    success = np.all(following['board'] == goals, axis=1)
    timeout = (following['steps_remaining'][:, 0] == 0) & ~success
    return dict(
        observations=observations, next_observations=following, goals=goals.copy(),
        actions=episode.actions[indices].copy(), event_counts=episode.event_counts[indices].copy(),
        rewards=np.where(success, 0.0, -1.0).astype(np.float32), terminated=success,
        truncated=timeout,
        collection_cutoff=(episode.collection_cutoff[indices] | episode.terminated[indices]) & ~success & ~timeout,
        bootstrap_masks=(~success).astype(np.float32), structure=episode.structure.observation_fields(),
        button_rest_positions=episode.button_rest_positions.copy(),
    )


def relabel_physics_transitions(episode, step_indices, goals):
    """Explicit board-only goals; arbitrary supplied goals need not be reachable.

    A collection ending at a different original goal becomes a collection cutoff
    under relabeling, unless the environment horizon also expired (a timeout).
    Only success at the new goal removes bootstrapping: this uses the continuing
    task time-limit convention. A strictly finite-horizon learner may additionally
    mask timeouts using the returned truncated flag.
    """
    episode.validate()
    indices, goals = np.asarray(step_indices), np.asarray(goals)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= episode.num_steps)):
        raise ValueError('step_indices must be nonempty legal integer indices')
    if goals.shape != (len(indices), episode.n_buttons) or goals.dtype.kind not in 'biuf' or not np.all((goals == 0) | (goals == 1)):
        raise ValueError('goals must be aligned binary board vectors')
    goals = goals.astype(np.uint8, copy=True)
    if np.any(np.all(episode.observations['board'][indices] == goals, axis=1)):
        raise ValueError('A relabeled goal must differ from the current board')
    return _relabel_checked(episode, indices, goals)


class PhysicsGoalSampler:
    """Persistent future-board sampler over a private episode snapshot.

    No-event transitions remain eligible if a later board differs. Transitions
    with no different future board cannot be sampled; whole no-press episodes
    require explicit goals instead. Repeated future indices retain their weight.
    Goals contain only board bits, never target robot poses.
    A future goal may equal s_next, so predictors must exclude the goal input.
    """
    def __init__(self, episode, seed=0):
        if not isinstance(episode, PhysicsTrajectory):
            raise TypeError('episode must be a PhysicsTrajectory')
        self._episode = deepcopy(episode)
        self._episode.validate()
        self._rng = np.random.default_rng(_integer(seed, 'seed'))
        board = self._episode.observations['board']
        self._futures = {}
        for step in range(self._episode.num_steps):
            candidates = np.flatnonzero(np.any(board[step + 1:] != board[step], axis=1)) + step + 1
            if len(candidates):
                self._futures[step] = candidates
        if not self._futures:
            raise ValueError('No transition has a future board different from its current board')
        self._indices = np.array(list(self._futures), dtype=np.int64)

    def sample(self, batch_size):
        size = _integer(batch_size, 'batch_size', 1)
        indices = self._indices[self._rng.integers(len(self._indices), size=size)]
        goals = np.stack([self._episode.observations['board'][self._rng.choice(self._futures[int(step)])]
                          for step in indices])
        return _relabel_checked(self._episode, indices, goals)


__all__ = ['PHYSICS_TRAJECTORY_SCHEMA', 'PhysicsTrajectory', 'record_episode',
           'load_physics_trajectory', 'PhysicsGoalSampler', 'relabel_physics_transitions']
