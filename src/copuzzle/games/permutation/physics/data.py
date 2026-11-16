"""Checked continuous-action fixture trajectories, replay, and future goals.

Loading and sampling need only NumPy.  They never construct a simulator,
execute a transition rule, or invoke a shortest-path oracle.  Ordered measured
events are audited with explicit permutation formulas.  An in-flight board is
the last committed permutation, not a claim that its tokens occupy valid slots.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile

import numpy as np


PHYSICS_TRAJECTORY_SCHEMA = 'copuzzle.permutation_physics_trajectory.v1'
_OBSERVATIONS = {
    'board': (np.int32, ('n',)), 'goal': (np.int32, ('n',)),
    'board_valid': (np.uint8, (1,)), 'mechanism_phase': (np.int32, (1,)),
    'active_operation': (np.int32, (1,)), 'rack_start': (np.float64, (1,)),
    'joint_positions': (np.float64, (6,)), 'joint_velocities': (np.float64, (6,)),
    'end_effector_position': (np.float64, (3,)),
    'end_effector_orientation': (np.float64, (4,)),
    'token_positions': (np.float64, ('n', 3)),
    'token_orientations': (np.float64, ('n', 4)),
    'token_velocities': (np.float64, ('n', 6)),
    'slot_positions': (np.float64, ('n', 3)),
    'rack_positions': (np.float64, ('a', 3)),
    'rack_rest_positions': (np.float64, ('a', 3)),
    'rack_displacements': (np.float64, ('a',)),
    'rack_velocities': (np.float64, ('a',)),
    'carrier_angles': (np.float64, ('a',)),
    'carrier_velocities': (np.float64, ('a',)),
    'previous_event_counts': (np.int32, ('a',)),
    'steps_remaining': (np.int32, (1,)),
}
_ARRAYS = (
    'actions', 'event_counts', 'event_steps', 'event_substeps', 'event_operations',
    'event_started_steps', 'event_started_substeps', 'event_durations',
    'rewards', 'terminated', 'truncated', 'collection_cutoff',
)
_EVENT_KEYS = {'substep', 'operation', 'started_step', 'started_substep', 'duration_seconds'}
_INFO_KEYS = {'is_success', 'steps', 'elapsed_time', 'rule_id', 'event_log',
              'attempt_log', 'failure_reason'}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('JSON keys must be strings')
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        if type(value) is float and not np.isfinite(value):
            raise ValueError('JSON numbers must be finite')
        return value
    raise ValueError('Metadata and info must contain JSON values only')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _array(value, dtype, shape, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError(f'{name} must be a {np.dtype(dtype)} array with shape {shape}')
    if value.dtype.kind == 'f' and not np.isfinite(value).all():
        raise ValueError(f'{name} must contain finite values')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value)


def _permutations(array, n, name):
    if not np.all(np.sort(array, axis=-1) == np.arange(1, n + 1)):
        raise ValueError(f'{name} must contain permutations of tokens 1..N')


def _ready(observations):
    """Public-state success eligibility, shared by validation and relabeling.

    Token velocities use world-frame [angular, linear] coordinates, as exposed
    by the fixture environment. Robot motion is not part of the board goal.
    """
    return ((observations['mechanism_phase'][..., 0] == 0)
            & (observations['board_valid'][..., 0] == 1)
            & (np.max(np.abs(observations['token_velocities']), axis=(-2, -1)) < .01)
            & (np.max(np.abs(observations['carrier_velocities']), axis=-1) < .1)
            & (np.max(np.abs(observations['rack_velocities']), axis=-1) < .01))


def _apply_event(board, operation, spec):
    result = board.copy()
    if spec['family'] == 'pancake':
        result[:operation + 2] = result[:operation + 2][::-1]
    elif operation < 2:
        result = np.roll(result, -1 if operation == 0 else 1)
    else:
        result[:spec['window']] = result[:spec['window']][::-1]
    return result


@dataclass
class PhysicalPermutationTrajectory:
    """One complete recorded episode, with T actions and T+1 observations.

    ``infos`` also retains all T+1 info dictionaries, including failed attempts
    and mechanism faults. A collection cutoff is distinct from both goal
    termination and environment timeout. Events refer to zero-based control
    intervals and physics substeps; durations run between actual latch and
    completion callbacks, after their respective physics integrations.
    """
    observations: dict[str, np.ndarray]
    actions: np.ndarray
    event_counts: np.ndarray
    event_steps: np.ndarray
    event_substeps: np.ndarray
    event_operations: np.ndarray
    event_started_steps: np.ndarray
    event_started_substeps: np.ndarray
    event_durations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    collection_cutoff: np.ndarray
    infos: list[dict]
    configuration: dict
    metadata: dict

    @property
    def num_steps(self):
        return len(self.actions)

    @property
    def n_tokens(self):
        return self.configuration['rule_spec']['n_tokens']

    @property
    def n_actions(self):
        return self.configuration['rule_spec']['n_actions']

    def events_at(self, step):
        indices = np.flatnonzero(self.event_steps == step)
        return [{'substep': int(self.event_substeps[i]),
                 'operation': int(self.event_operations[i]),
                 'started_step': int(self.event_started_steps[i]),
                 'started_substep': int(self.event_started_substeps[i]),
                 'duration_seconds': float(self.event_durations[i])} for i in indices]

    def validate(self):
        from ..rules import validate_rule_spec  # Static identity validation only.

        if not isinstance(self.configuration, dict) or not isinstance(self.metadata, dict):
            raise ValueError('configuration and metadata must be dictionaries')
        configuration, metadata = _json_value(self.configuration), _json_value(self.metadata)
        if configuration.get('schema_version') != 'copuzzle.permutation_physics_config.v1':
            raise ValueError('Unsupported permutation physics configuration')
        spec = validate_rule_spec(configuration.get('rule_spec'))
        if configuration.get('task_name') != spec['name']:
            raise ValueError('Configuration task_name differs from rule_spec')
        n, a = spec['n_tokens'], spec['n_actions']
        if not isinstance(self.actions, np.ndarray) or self.actions.ndim != 2 or len(self.actions) < 1:
            raise ValueError('An episode requires at least one continuous action')
        steps = self.num_steps
        if not isinstance(self.observations, dict) or set(self.observations) != set(_OBSERVATIONS):
            raise ValueError('Unexpected or missing physical observation fields')
        dimensions = {'n': n, 'a': a}
        for name, (dtype, dims) in _OBSERVATIONS.items():
            _array(self.observations[name], dtype,
                   (steps + 1,) + tuple(dimensions.get(d, d) for d in dims), name)
        obs = self.observations
        _permutations(obs['board'], n, 'board')
        _permutations(obs['goal'], n, 'goal')
        if not np.all(obs['goal'] == obs['goal'][0]):
            raise ValueError('The collection goal must remain fixed')
        phase, active, valid = (obs['mechanism_phase'][:, 0], obs['active_operation'][:, 0],
                                obs['board_valid'][:, 0])
        if np.any(valid > 1) or np.any((phase < 0) | (phase > 3)):
            raise ValueError('Invalid mechanism phase or board_valid flag')
        if (np.any((active < -1) | (active >= a)) or np.any(active[phase == 0] != -1)
                or np.any(active[np.isin(phase, [1, 2])] < 0)):
            raise ValueError('active_operation disagrees with mechanism phase')
        if np.any(valid[np.isin(phase, [1, 3])]):
            raise ValueError('Carrying or fault phases cannot declare a valid committed board')
        if phase[0] != 0 or not valid[0]:
            raise ValueError('Reset must start at a valid docked fixture state')
        for field in ('slot_positions', 'rack_rest_positions'):
            if not np.all(obs[field] == obs[field][0]):
                raise ValueError(f'Static field {field} changed within the episode')
        for field in ('token_orientations', 'end_effector_orientation'):
            if not np.allclose(np.linalg.norm(obs[field], axis=-1), 1, atol=1e-6, rtol=0):
                raise ValueError(f'{field} must contain unit quaternions')
        slots = obs['slot_positions'][0]
        if np.unique(slots, axis=0).shape[0] != n:
            raise ValueError('Physical slot positions must be distinct')
        scene = configuration.get('scene', {})
        if (isinstance(scene, dict) and 'slot_positions' in scene
                and not np.array_equal(np.asarray(scene['slot_positions']), slots)):
            raise ValueError('Recorded slot positions differ from scene configuration')
        # Audit claimed valid observations from token geometry, without physics.
        for index in np.flatnonzero(valid):
            distances = np.linalg.norm(obs['token_positions'][index, :, None] - slots[None], axis=-1)
            assignment = distances.argmin(axis=1)
            if (len(set(assignment.tolist())) != n
                    or np.max(distances[np.arange(n), assignment]) >= .0025):
                raise ValueError('board_valid disagrees with measured token slot geometry')
            board = np.zeros(n, dtype=np.int32)
            board[assignment] = np.arange(1, n + 1)
            if not np.array_equal(board, obs['board'][index]):
                raise ValueError('Committed board disagrees with measured token identities')
        _array(self.actions, np.float32, (steps, 3), 'actions')
        if np.any(np.abs(self.actions) > 1):
            raise ValueError('Continuous actions must be in [-1, 1]')
        _array(self.event_counts, np.int32, (steps, a), 'event_counts')
        if np.any(self.event_counts < 0):
            raise ValueError('Event counts must be nonnegative')
        if np.any(obs['previous_event_counts'][0]) or not np.array_equal(
                obs['previous_event_counts'][1:], self.event_counts):
            raise ValueError('previous_event_counts disagree with preceding measured events')
        _array(self.rewards, np.float32, (steps,), 'rewards')
        for field in ('terminated', 'truncated', 'collection_cutoff'):
            _array(getattr(self, field), np.bool_, (steps,), field)
        ends = self.terminated.astype(int) + self.truncated.astype(int) + self.collection_cutoff.astype(int)
        if np.any(ends[:-1]) or ends[-1] != 1:
            raise ValueError('Exactly one end condition is required, at the final transition only')
        eligible = _ready(obs)
        success = eligible & np.all(obs['board'] == obs['goal'], axis=-1)
        if np.any(success[:-1]):
            raise ValueError('A recorded episode cannot continue from a successful state')
        if not np.array_equal(self.terminated, success[1:]):
            raise ValueError('terminated requires the target board and a valid stationary ready fixture')
        if not np.array_equal(self.rewards, np.where(success[1:], 0., -1.).astype(np.float32)):
            raise ValueError('Rewards must be zero on success and minus one otherwise')
        horizon = _integer(configuration.get('max_episode_steps'), 'max_episode_steps', 1)
        remaining = obs['steps_remaining'][:, 0]
        if steps > horizon or not np.array_equal(remaining, horizon - np.arange(steps + 1)):
            raise ValueError('steps_remaining does not track the configured horizon')
        if not np.array_equal(self.truncated, (remaining[1:] == 0) & ~success[1:]):
            raise ValueError('truncated must mean horizon expiry without goal success')
        for key in ('physics_timestep', 'control_timestep'):
            value = configuration.get(key)
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError('Configuration timesteps must be finite and positive')
        physics, control = configuration['physics_timestep'], configuration['control_timestep']
        substeps = _integer(configuration.get('substeps'), 'substeps', 1)
        if not np.isclose(control / physics, substeps, atol=1e-8, rtol=0):
            raise ValueError('Configuration substeps disagree with timesteps')
        if not isinstance(self.event_steps, np.ndarray) or self.event_steps.ndim != 1:
            raise ValueError('event_steps must be a one-dimensional array')
        count = len(self.event_steps)
        for name, dtype in [('event_steps', np.int64), ('event_substeps', np.int32),
                            ('event_operations', np.int32), ('event_started_steps', np.int64),
                            ('event_started_substeps', np.int32), ('event_durations', np.float64)]:
            _array(getattr(self, name), dtype, (count,), name)
        if (np.any(self.event_steps < 0) or np.any(self.event_steps >= steps)
                or np.any(self.event_started_steps < 0) or np.any(self.event_started_steps > self.event_steps)
                or np.any(self.event_substeps < 0) or np.any(self.event_substeps >= substeps)
                or np.any(self.event_started_substeps < 0) or np.any(self.event_started_substeps >= substeps)
                or np.any(self.event_operations < 0) or np.any(self.event_operations >= a)):
            raise ValueError('Measured event indices are outside their declared ranges')
        stamps = self.event_steps * substeps + self.event_substeps
        starts = self.event_started_steps * substeps + self.event_started_substeps
        if np.any(np.diff(stamps) < 0) or np.any(starts >= stamps):
            raise ValueError('Measured events must be chronological and complete after starting')
        triples = list(zip(self.event_steps.tolist(), self.event_substeps.tolist(), self.event_operations.tolist()))
        if len(set(triples)) != count:
            raise ValueError('Duplicate measured operation event')
        expected_durations = (stamps - starts) * physics
        if not np.allclose(self.event_durations, expected_durations, atol=1e-8, rtol=0):
            raise ValueError('Measured event durations disagree with start/completion substeps')
        measured = np.zeros((steps, a), dtype=np.int32)
        np.add.at(measured, (self.event_steps, self.event_operations), 1)
        if not np.array_equal(measured, self.event_counts):
            raise ValueError('Event counts and ordered event log differ')
        board = obs['board'][0].copy()
        cursor = 0
        for step in range(steps):
            while cursor < count and self.event_steps[cursor] == step:
                board = _apply_event(board, int(self.event_operations[cursor]), spec)
                cursor += 1
            if not np.array_equal(board, obs['board'][step + 1]):
                raise ValueError(f'Committed board does not match ordered measured operations at step {step}')
        required_metadata = {'reset_seed', 'reset_options', 'configuration_sha256', 'model_sha256',
                             'sources', 'source_sha256', 'collector', 'user'}
        if set(metadata) != required_metadata:
            raise ValueError('Unexpected or missing trajectory provenance')
        _integer(metadata['reset_seed'], 'reset_seed')
        if not isinstance(metadata['reset_options'], dict) or set(metadata['reset_options']) - {'state', 'goal'}:
            raise ValueError('reset_options may contain only state and goal')
        for key, field in [('state', 'board'), ('goal', 'goal')]:
            if key in metadata['reset_options']:
                raw = metadata['reset_options'][key]
                if (not isinstance(raw, list) or len(raw) != n or any(type(item) is not int for item in raw)
                        or not np.array_equal(np.asarray(raw), obs[field][0])):
                    raise ValueError(f'Reset {key} differs from the first observation')
        if metadata['configuration_sha256'] != hashlib.sha256(_canonical(configuration)).hexdigest():
            raise ValueError('Configuration checksum mismatch')
        if not _hash(metadata['model_sha256']) or configuration.get('model_sha256') != metadata['model_sha256']:
            raise ValueError('Model identity mismatch')
        sources = metadata['sources']
        if not isinstance(sources, dict) or not sources or any(not _hash(value) for value in sources.values()):
            raise ValueError('sources must map source names to SHA256 digests')
        if metadata['source_sha256'] != hashlib.sha256(_canonical(sources)).hexdigest():
            raise ValueError('Source identity checksum mismatch')
        if not isinstance(metadata['user'], dict) or not isinstance(metadata['collector'], dict):
            raise ValueError('Collector declarations and user metadata must be dictionaries')
        collector = metadata['collector']
        if set(collector) != {'policy', 'max_steps', 'stop_reason'} or not isinstance(collector['policy'], str):
            raise ValueError('Invalid collector declarations')
        if _integer(collector['max_steps'], 'collector max_steps', 1) < steps:
            raise ValueError('Episode exceeds the declared collection budget')
        expected_stop = 'environment_terminated' if self.terminated[-1] else (
            'environment_truncated' if self.truncated[-1] else None)
        if ((expected_stop is not None and collector['stop_reason'] != expected_stop)
                or (expected_stop is None and collector['stop_reason'] not in ('max_steps', 'policy_completed'))
                or (collector['stop_reason'] == 'max_steps' and steps != collector['max_steps'])):
            raise ValueError('Collector stop reason disagrees with recorded ending')
        if not isinstance(self.infos, list) or len(self.infos) != steps + 1:
            raise ValueError('infos must preserve T+1 info dictionaries')
        for index, original in enumerate(self.infos):
            info = _json_value(original)
            if not isinstance(info, dict) or set(info) != _INFO_KEYS:
                raise ValueError('Unexpected or missing physical info fields')
            if (type(info['is_success']) is not bool or info['is_success'] != bool(success[index])
                    or _integer(info['steps'], 'info steps') != index or info['rule_id'] != spec['rule_id']):
                raise ValueError('Info success/step/rule fields disagree with the episode')
            if type(info['elapsed_time']) not in (int, float) or not np.isclose(
                    info['elapsed_time'], index * control, atol=1e-7, rtol=0):
                raise ValueError('Info elapsed time disagrees with the control clock')
            if not isinstance(info['event_log'], list):
                raise ValueError('Info event_log must be a list')
            for event in info['event_log']:
                if not isinstance(event, dict) or set(event) != _EVENT_KEYS:
                    raise ValueError('Invalid info event fields')
                for key in _EVENT_KEYS - {'duration_seconds'}:
                    _integer(event[key], key)
                if type(event['duration_seconds']) not in (int, float):
                    raise ValueError('Event duration must be a numeric number of seconds')
            if info['event_log'] != ([] if index == 0 else self.events_at(index - 1)):
                raise ValueError('Info event log differs from measured event arrays')
            if not isinstance(info['attempt_log'], list):
                raise ValueError('attempt_log must be a list')
            for attempt in info['attempt_log']:
                if (not isinstance(attempt, dict) or set(attempt) != {'substep', 'operation', 'result'}
                        or _integer(attempt['substep'], 'attempt substep') >= substeps
                        or _integer(attempt['operation'], 'attempt operation') >= a
                        or attempt['result'] != 'aborted_and_returned' or index == 0):
                    raise ValueError('Invalid failed operation attempt log')
            if info['failure_reason'] is not None and not isinstance(info['failure_reason'], str):
                raise ValueError('failure_reason must be text or null')

    def _arrays(self):
        arrays = {f'observation__{name}': value for name, value in self.observations.items()}
        arrays.update({name: getattr(self, name) for name in _ARRAYS})
        return arrays

    def save(self, path):
        """Atomically publish each complete NPZ/JSON file without replacing either."""
        self.validate()
        path = Path(path).expanduser().absolute()
        if path.suffix != '.npz':
            raise ValueError('Episode path must end in .npz')
        sidecar = path.with_suffix('.json')
        if os.path.lexists(path) or os.path.lexists(sidecar):
            raise FileExistsError('Refusing to overwrite an episode or its manifest')
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
            manifest_value = {
                'schema_version': PHYSICS_TRAJECTORY_SCHEMA,
                'artifact': {'filename': path.name, 'sha256': _sha(archive)},
                'num_steps': self.num_steps, 'configuration': _json_value(self.configuration),
                'metadata': _json_value(self.metadata), 'infos': _json_value(self.infos),
                'arrays': {name: {'dtype': value.dtype.name, 'shape': list(value.shape)}
                           for name, value in arrays.items()},
            }
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                             suffix='.tmp', delete=False) as stream:
                manifest = Path(stream.name)
                temporary.append(manifest)
                json.dump(manifest_value, stream, indent=2, sort_keys=True, allow_nan=False)
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
    """Load and fully validate an episode without importing a physics engine."""
    path = Path(path).expanduser().absolute()
    if path.suffix != '.npz':
        raise ValueError('Episode path must end in .npz')
    with path.with_suffix('.json').open(encoding='utf-8') as stream:
        manifest = json.load(stream, object_pairs_hook=_unique_object)
    required = {'schema_version', 'artifact', 'num_steps', 'configuration', 'metadata', 'infos', 'arrays'}
    if not isinstance(manifest, dict) or set(manifest) != required or manifest['schema_version'] != PHYSICS_TRAJECTORY_SCHEMA:
        raise ValueError('Unsupported or incomplete permutation physics trajectory manifest')
    artifact = manifest['artifact']
    if (not isinstance(artifact, dict) or set(artifact) != {'filename', 'sha256'}
            or artifact['filename'] != path.name or not _hash(artifact['sha256'])
            or artifact['sha256'] != _sha(path)):
        raise ValueError('Episode archive checksum or filename mismatch')
    expected_arrays = {f'observation__{name}' for name in _OBSERVATIONS} | set(_ARRAYS)
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(expected_arrays) or set(archive.files) != expected_arrays:
            raise ValueError('Unexpected, missing, or duplicate episode arrays')
        arrays = {name: archive[name] for name in archive.files}
    declarations = {name: {'dtype': value.dtype.name, 'shape': list(value.shape)} for name, value in arrays.items()}
    declared = manifest['arrays']
    valid_declarations = isinstance(declared, dict) and set(declared) == set(declarations)
    if valid_declarations:
        for description in declared.values():
            if (not isinstance(description, dict) or set(description) != {'dtype', 'shape'}
                    or not isinstance(description['dtype'], str) or not isinstance(description['shape'], list)
                    or any(type(size) is not int for size in description['shape'])):
                valid_declarations = False
                break
    if not valid_declarations or declared != declarations:
        raise ValueError('Stored arrays disagree with declared shape/dtype metadata')
    episode = PhysicalPermutationTrajectory(
        observations={name: arrays[f'observation__{name}'] for name in _OBSERVATIONS},
        **{name: arrays[name] for name in _ARRAYS}, infos=manifest['infos'],
        configuration=manifest['configuration'], metadata=manifest['metadata'])
    episode.validate()
    if type(manifest['num_steps']) is not int or manifest['num_steps'] != episode.num_steps:
        raise ValueError('Episode length differs from its manifest')
    return episode


def _source_hashes(env, policy):
    sources = {'data': _sha(Path(__file__))}
    candidates = [('environment', type(env)), ('policy', policy if inspect.isfunction(policy) else type(policy))]
    if hasattr(env, 'controller'):
        candidates.append(('controller', type(env.controller)))
    # The scene module has already been imported by the caller's environment.
    module = inspect.getmodule(type(env))
    if module is not None and hasattr(module, 'build_scene'):
        candidates.append(('scene', module.build_scene))
    for label, candidate in candidates:
        try:
            path = inspect.getsourcefile(candidate)
        except (TypeError, OSError):
            path = None
        if path is not None and Path(path).is_file():
            sources[label] = _sha(path)
    return sources


def record_trajectory(env, policy, *, seed, options=None, max_steps=None, metadata=None):
    """Record exact float32 actions, all successors, and measured operation logs.

    Policy completion ends collection, even when the environment's goal differs.
    The caller owns the environment. No physics package is imported here.
    """
    seed = _integer(seed, 'seed')
    options = {} if options is None else deepcopy(options)
    if not isinstance(options, dict) or (metadata is not None and not isinstance(metadata, dict)):
        raise ValueError('options and metadata must be dictionaries')
    configuration = _json_value(env.configuration())
    limit = _integer(configuration['max_episode_steps'] if max_steps is None else max_steps, 'max_steps', 1)
    source = _source_hashes(env, policy)
    name = getattr(policy, '__qualname__', type(policy).__qualname__)
    module = getattr(policy, '__module__', type(policy).__module__)
    provenance = {
        'reset_seed': seed, 'reset_options': _json_value(options), 'user': _json_value(metadata or {}),
        'configuration_sha256': env.configuration_sha256, 'model_sha256': configuration['model_sha256'],
        'sources': source, 'source_sha256': hashlib.sha256(_canonical(source)).hexdigest(),
        'collector': {'policy': f'{module}.{name}', 'max_steps': limit, 'stop_reason': None},
    }
    observation, info = env.reset(seed=seed, options=options)
    if info.get('is_success') or (_ready(observation) and np.array_equal(observation['board'], observation['goal'])):
        raise ValueError('Cannot record from an already successful reset')
    if callable(getattr(policy, 'reset', None)):
        policy.reset()
    history = {name: [] for name in _OBSERVATIONS}
    infos = [_json_value(info)]

    def capture(current):
        if set(current) != set(history):
            raise ValueError('Environment returned unexpected observation fields')
        for field in history:
            history[field].append(np.array(current[field], copy=True))

    capture(observation)
    actions, counts, rewards, terminals, timeouts, events = [], [], [], [], [], []
    stop_reason = 'max_steps'
    for step in range(limit):
        if bool(getattr(policy, 'completed', False)):
            stop_reason = 'policy_completed'
            break
        action = np.asarray(policy(deepcopy(observation)), dtype=np.float32)
        if bool(getattr(policy, 'completed', False)):
            stop_reason = 'policy_completed'
            break
        if action.shape != (3,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('Policy must return three finite actions in [-1, 1]')
        following, reward, terminated, truncated, info = env.step(action.copy())
        actions.append(action.copy())
        counts.append(np.array(following['previous_event_counts'], copy=True))
        rewards.append(reward)
        terminals.append(terminated)
        timeouts.append(truncated)
        if not isinstance(info.get('event_log'), list):
            raise ValueError('Step info must contain a measured event_log list')
        for event in info['event_log']:
            if not isinstance(event, dict) or set(event) != _EVENT_KEYS:
                raise ValueError('Unexpected or missing measured operation event fields')
            events.append((step, _integer(event['substep'], 'substep'),
                           _integer(event['operation'], 'operation'),
                           _integer(event['started_step'], 'started_step'),
                           _integer(event['started_substep'], 'started_substep'), event['duration_seconds']))
        infos.append(_json_value(info))
        capture(following)
        observation = following
        if terminated or truncated:
            stop_reason = 'environment_terminated' if terminated else 'environment_truncated'
            break
    if not actions:
        raise ValueError('Policy completed before any transition was collected')
    cutoffs = np.zeros(len(actions), dtype=bool)
    cutoffs[-1] = not (terminals[-1] or timeouts[-1])
    provenance['collector']['stop_reason'] = stop_reason
    episode = PhysicalPermutationTrajectory(
        observations={name: np.stack(values) for name, values in history.items()},
        actions=np.stack(actions), event_counts=np.stack(counts),
        event_steps=np.array([event[0] for event in events], dtype=np.int64),
        event_substeps=np.array([event[1] for event in events], dtype=np.int32),
        event_operations=np.array([event[2] for event in events], dtype=np.int32),
        event_started_steps=np.array([event[3] for event in events], dtype=np.int64),
        event_started_substeps=np.array([event[4] for event in events], dtype=np.int32),
        event_durations=np.array([event[5] for event in events], dtype=np.float64),
        rewards=np.array(rewards, dtype=np.float32), terminated=np.array(terminals, dtype=bool),
        truncated=np.array(timeouts, dtype=bool), collection_cutoff=cutoffs,
        infos=infos, configuration=configuration, metadata=provenance)
    episode.validate()
    return episode


def _compare_json(actual, expected, *, atol):
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(_compare_json(actual[k], v, atol=atol) for k, v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_compare_json(a, b, atol=atol) for a, b in zip(actual, expected)))
    if type(expected) is float:
        return type(actual) in (float, int) and bool(np.isclose(actual, expected, atol=atol, rtol=0))
    return type(actual) is type(expected) and actual == expected


def replay_trajectory(env, episode, *, atol=1e-8):
    """Replay recorded actions and compare every observation/info/reward/end flag."""
    episode.validate()
    if isinstance(atol, (bool, np.bool_)) or not np.isscalar(atol) or not np.isfinite(atol) or atol < 0:
        raise ValueError('atol must be finite and nonnegative')
    if (env.configuration_sha256 != episode.metadata['configuration_sha256']
            or _json_value(env.configuration()) != episode.configuration):
        raise ValueError('Replay requires the recorded environment configuration')
    observation, info = env.reset(seed=episode.metadata['reset_seed'], options=episode.metadata['reset_options'])
    maximum = 0.0

    def compare(actual, actual_info, index):
        nonlocal maximum
        if set(actual) != set(episode.observations):
            raise ValueError('Replay observation fields differ')
        for name, history in episode.observations.items():
            expected, value = history[index], actual[name]
            if not isinstance(value, np.ndarray) or value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f'Replay shape/dtype mismatch at observation {index}: {name}')
            if expected.dtype.kind == 'f':
                maximum = max(maximum, float(np.max(np.abs(value - expected))))
                matches = np.allclose(value, expected, atol=atol, rtol=0)
            else:
                matches = np.array_equal(value, expected)
            if not matches:
                raise ValueError(f'Replay mismatch at observation {index}: {name}')
        if not _compare_json(_json_value(actual_info), episode.infos[index], atol=atol):
            raise ValueError(f'Replay info/event/attempt mismatch at observation {index}')

    compare(observation, info, 0)
    for step, action in enumerate(episode.actions):
        observation, reward, terminated, truncated, info = env.step(action.copy())
        compare(observation, info, step + 1)
        if (reward != episode.rewards[step] or terminated != episode.terminated[step]
                or truncated != episode.truncated[step]):
            raise ValueError(f'Replay reward/end mismatch at control step {step}')
    return {'control_steps': episode.num_steps, 'operation_events': len(episode.event_steps),
            'max_absolute_observation_error': maximum, 'atol': float(atol),
            'configuration_sha256': env.configuration_sha256}


def _relabel_checked(episode, indices, goals, goal_indices=None):
    observation = {name: values[indices].copy() for name, values in episode.observations.items()}
    following = {name: values[indices + 1].copy() for name, values in episode.observations.items()}
    observation['goal'], following['goal'] = goals.copy(), goals.copy()
    success = _ready(following) & np.all(following['board'] == goals, axis=-1)
    timeout = (following['steps_remaining'][:, 0] == 0) & ~success
    # The original successful episode ended collection for a different goal.
    cutoff = (episode.collection_cutoff[indices] | episode.terminated[indices]) & ~success & ~timeout
    return {
        'observations': observation, 'next_observations': following, 'goals': goals.copy(),
        'actions': episode.actions[indices].copy(), 'event_counts': episode.event_counts[indices].copy(),
        'rewards': np.where(success, 0., -1.).astype(np.float32), 'terminated': success,
        'truncated': timeout, 'collection_cutoff': cutoff,
        'bootstrap_masks': (~success).astype(np.float32),
        'raw_terminated': episode.terminated[indices].copy(),
        'raw_truncated': episode.truncated[indices].copy(),
        'raw_collection_cutoff': episode.collection_cutoff[indices].copy(),
        'step_indices': indices.copy(),
        'goal_observation_indices': (np.full(len(indices), -1, dtype=np.int64)
                                     if goal_indices is None else goal_indices.copy()),
    }


def relabel_physics_transitions(episode, step_indices, goals):
    """Relabel board goals using physical readiness, without reachability claims.

    A last-committed board may equal the new goal during motion; success still
    requires measured validity, a ready phase, and stationary fixtures. Only
    success removes bootstrap under the continuing-task time-limit convention.
    """
    episode.validate()
    indices, values = np.asarray(step_indices), np.asarray(goals)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= episode.num_steps)):
        raise ValueError('step_indices must contain nonempty legal integer indices')
    if values.shape != (len(indices), episode.n_tokens) or values.dtype.kind not in 'iu':
        raise ValueError('goals must be aligned integer permutation vectors')
    _permutations(values, episode.n_tokens, 'goals')
    values = values.astype(np.int32, copy=True)
    ready = _ready(episode.observations)[indices]
    if np.any(ready & np.all(episode.observations['board'][indices] == values, axis=-1)):
        raise ValueError('A relabeled goal is already achieved at a sampled current state')
    return _relabel_checked(episode, indices.astype(np.int64), values)


class PhysicsGoalSampler:
    """Persistent sampler using only stable future board observations as goals.

    Return-to-dock transitions remain eligible even when their committed board
    already equals the future goal. Sampling retains robot state and ignores
    target robot poses. Repeated stable future observations retain their weight.
    """
    def __init__(self, episode, seed=0):
        if not isinstance(episode, PhysicalPermutationTrajectory):
            raise TypeError('episode must be a PhysicalPermutationTrajectory')
        self._episode = deepcopy(episode)
        self._episode.validate()
        self._rng = np.random.default_rng(_integer(seed, 'seed'))
        self._ready = _ready(self._episode.observations)
        self._stable = np.flatnonzero(self._ready)
        eligible = []
        for step in range(self._episode.num_steps):
            if len(self._candidates(step)):
                eligible.append(step)
        self._indices = np.asarray(eligible, dtype=np.int64)
        if not len(self._indices):
            raise ValueError('No transition has an unachieved stable future board goal')

    def _candidates(self, step):
        candidates = self._stable[np.searchsorted(self._stable, step + 1):]
        if self._ready[step]:
            board = self._episode.observations['board']
            candidates = candidates[np.any(board[candidates] != board[step], axis=-1)]
        return candidates

    def sample(self, batch_size):
        size = _integer(batch_size, 'batch_size', 1)
        indices = self._rng.choice(self._indices, size=size)
        future = np.array([self._rng.choice(self._candidates(int(step))) for step in indices], dtype=np.int64)
        goals = self._episode.observations['board'][future]
        return _relabel_checked(self._episode, indices, goals, future)


PhysicsTrajectory = PhysicalPermutationTrajectory
record_episode = record_trajectory

__all__ = ['PHYSICS_TRAJECTORY_SCHEMA', 'PhysicalPermutationTrajectory', 'PhysicsTrajectory',
           'record_trajectory', 'record_episode', 'load_physics_trajectory', 'replay_trajectory',
           'PhysicsGoalSampler', 'relabel_physics_transitions']
