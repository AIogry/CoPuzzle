"""Portable 7D-action Pancake contact episodes and deterministic action replay.

This schema is separate from the archived three-button permutation fixture.
Loading requires NumPy only and does not construct MuJoCo or a puzzle oracle.
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


SCHEMA = 'copuzzle.pancake_contact_trajectory.v1'
OPERATION_SCHEMA = 'copuzzle.pancake_operation_trajectory.v1'
_FIELDS = {
    'board': ('int32', (4,)), 'goal': ('int32', (4,)), 'board_valid': ('uint8', (1,)),
    'token_positions': ('float64', (4, 3)), 'token_orientations': ('float64', (4, 4)),
    'token_velocities': ('float64', (4, 6)), 'tool_contacts': ('uint8', (4,)),
    'tool_contact_forces': ('float64', (4,)), 'joint_positions': ('float64', (6,)),
    'arm_joint_targets': ('float64', (6,)),
    'joint_velocities': ('float64', (6,)), 'end_effector_position': ('float64', (3,)),
    'end_effector_orientation': ('float64', (4,)), 'tool_target_position': ('float64', (3,)),
    'tool_target_orientation': ('float64', (4,)), 'jaw_position': ('float64', (1,)),
    'jaw_velocity': ('float64', (1,)), 'jaw_target': ('float64', (1,)),
    'steps_remaining': ('int32', (1,)),
}
_ARRAYS = ('actions', 'rewards', 'terminated', 'truncated', 'collection_cutoff')
_OPERATION_FIELDS = {
    'operation_prefix': ('int32', (1,)), 'operation_stage': ('int32', (1,)),
    'operation_selected': ('uint8', (4,)), 'operation_initial_board': ('int32', (4,)),
    'operation_initial_positions': ('float64', (4, 3)),
    'operation_initial_orientations': ('float64', (4, 4)),
    'operation_peak_lift': ('float64', (4,)), 'operation_peak_displacement': ('float64', (4,)),
    'operation_minimum_normal_dot': ('float64', (4,)), 'operation_violation': ('uint8', (1,)),
}


def _fields_for_config(config):
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a dictionary')
    schema = config.get('schema_version')
    if schema == 'copuzzle.pancake_manipulation_config.v1':
        return _FIELDS
    if schema == 'copuzzle.pancake_operation_config.v1':
        return {**_FIELDS, **_OPERATION_FIELDS}
    raise ValueError('Unsupported native Pancake configuration')


def _normal_vectors(quaternions):
    w, x, y, z = np.asarray(quaternions).T
    return np.column_stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))


def _check_operation_trace(obs, config, goal_ready):
    """Recompute diagnostic evidence from positions, orientations, and contacts.

    Collector phase labels do not participate. No simulator or executable
    puzzle transition is needed, and flip-one is never a logical operation.
    """
    prefix = _integer(config.get('operation_prefix'), 'operation_prefix', 1)
    if prefix > 4 or config.get('task_mode') != 'operation':
        raise ValueError('Invalid operation diagnostic configuration')
    protocol = config.get('operation_protocol')
    expected_protocol = {'lift_height_m': .06, 'contact_force_min_n': .05,
        'inversion_cosine_max': -.90, 'unselected_motion_limit_m': .01,
        'unselected_normal_cosine_min': .98}
    if protocol != expected_protocol:
        raise ValueError('Unsupported operation evidence protocol')
    initial_board = obs['board'][0]
    goal = initial_board.copy()
    goal[:prefix] = goal[:prefix][::-1]
    if not np.array_equal(obs['goal'][0], goal):
        raise ValueError('Diagnostic goal differs from the requested unsigned prefix reversal')
    selected = np.isin(np.arange(1, 5), initial_board[:prefix])
    static = {'operation_initial_board': initial_board,
              'operation_initial_positions': obs['token_positions'][0],
              'operation_initial_orientations': obs['token_orientations'][0],
              'operation_selected': selected.astype(np.uint8),
              'operation_prefix': np.array([prefix], dtype=np.int32)}
    for name, value in static.items():
        if not np.all(obs[name] == value):
            raise ValueError(f'Diagnostic static field disagrees with initial evidence: {name}')
    stage, violation = 0, False
    peak_lift, peak_motion, minimum_dot = np.zeros(4), np.zeros(4), np.ones(4)
    initial_normals = _normal_vectors(obs['token_orientations'][0])
    success = np.zeros(len(goal_ready), dtype=bool)
    for index in range(len(goal_ready)):
        if index:
            delta = obs['token_positions'][index] - obs['token_positions'][0]
            lift, motion = delta[:, 2], np.linalg.norm(delta, axis=1)
            normal_dot = np.clip(np.sum(_normal_vectors(obs['token_orientations'][index]) * initial_normals,
                                        axis=1), -1, 1)
            peak_lift, peak_motion = np.maximum(peak_lift, lift), np.maximum(peak_motion, motion)
            minimum_dot = np.minimum(minimum_dot, normal_dot)
            violation |= bool(np.any(motion[~selected] > protocol['unselected_motion_limit_m'])
                              or np.any(normal_dot[~selected] < protocol['unselected_normal_cosine_min']))
            lifted = np.all(lift[selected] >= protocol['lift_height_m'])
            grasped = (np.any(obs['tool_contacts'][index, selected])
                       and np.sum(obs['tool_contact_forces'][index, selected]) > protocol['contact_force_min_n'])
            inverted = np.all(normal_dot[selected] <= protocol['inversion_cosine_max'])
            if stage == 0 and lifted and grasped:
                stage = 1
            if stage >= 1 and lifted and inverted:
                stage = 2
            if stage >= 2 and inverted and not violation and goal_ready[index]:
                stage = 3
        if (obs['operation_stage'][index, 0] != stage
                or obs['operation_violation'][index, 0] != int(violation)):
            raise ValueError('Operation stage/violation differs from measured execution evidence')
        for field, expected in [('operation_peak_lift', peak_lift), ('operation_peak_displacement', peak_motion),
                                ('operation_minimum_normal_dot', minimum_dot)]:
            if not np.allclose(obs[field][index], expected, atol=1e-12, rtol=0):
                raise ValueError(f'Operation history differs from measured evidence: {field}')
        success[index] = stage == 3 and goal_ready[index]
    return success


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError('JSON keys must be strings')
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is float and not np.isfinite(value):
            raise ValueError('JSON numbers must be finite')
        return value
    raise ValueError('Expected JSON-compatible values')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _array(value, dtype, shape, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError(f'{name} must have dtype {dtype} and shape {shape}')
    if value.dtype.kind == 'f' and not np.isfinite(value).all():
        raise ValueError(f'{name} contains non-finite values')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


@dataclass
class PancakeContactTrajectory:
    observations: dict[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    collection_cutoff: np.ndarray
    infos: list[dict]
    phases: list[str | None]
    configuration: dict
    metadata: dict

    @property
    def num_steps(self):
        return len(self.actions)

    @property
    def schema_version(self):
        return (OPERATION_SCHEMA if self.configuration.get('schema_version') ==
                'copuzzle.pancake_operation_config.v1' else SCHEMA)

    def validate(self):
        if not isinstance(self.actions, np.ndarray) or self.actions.ndim != 2 or len(self.actions) < 1:
            raise ValueError('An episode needs at least one transition')
        t = self.num_steps
        config, meta = json_value(self.configuration), json_value(self.metadata)
        fields = _fields_for_config(config)
        operation_mode = self.schema_version == OPERATION_SCHEMA
        if not isinstance(self.observations, dict) or set(self.observations) != set(fields):
            raise ValueError('Unexpected observation fields')
        for name, (dtype, shape) in fields.items():
            _array(self.observations[name], dtype, (t + 1, *shape), name)
        obs = self.observations
        for name in ('board', 'goal'):
            if not np.all(np.sort(obs[name], axis=-1) == np.arange(1, 5)):
                raise ValueError(f'{name} contains invalid permutations')
        if not np.all(obs['goal'] == obs['goal'][0]):
            raise ValueError('Collection goal changed within the episode')
        for name in ('board_valid', 'tool_contacts'):
            if np.any(obs[name] > 1):
                raise ValueError(f'{name} must be binary')
        if np.any(obs['tool_contact_forces'] < 0) or np.any((obs['jaw_target'] < 0) | (obs['jaw_target'] > 1)):
            raise ValueError('Invalid force or jaw target')
        for name in ('token_orientations', 'end_effector_orientation', 'tool_target_orientation'):
            if not np.allclose(np.linalg.norm(obs[name], axis=-1), 1., atol=1e-6, rtol=0):
                raise ValueError('Orientations must be unit quaternions')
        _array(self.actions, 'float32', (t, 7), 'actions')
        if np.any(np.abs(self.actions[:, :6]) > 1) or np.any((self.actions[:, 6] < 0) | (self.actions[:, 6] > 1)):
            raise ValueError('Actions lie outside the 7D continuous action space')
        _array(self.rewards, 'float32', (t,), 'rewards')
        for name in ('terminated', 'truncated', 'collection_cutoff'):
            _array(getattr(self, name), 'bool', (t,), name)
        endings = self.terminated.astype(int) + self.truncated.astype(int) + self.collection_cutoff.astype(int)
        if np.any(endings[:-1]) or endings[-1] != 1:
            raise ValueError('One final ending condition is required')
        required = {'seed', 'reset_options', 'configuration_sha256', 'sources', 'source_sha256',
                    'max_steps', 'stop_reason', 'policy_error', 'policy', 'user'}
        if not isinstance(meta, dict) or set(meta) != required:
            raise ValueError('Unexpected provenance fields')
        _integer(meta['seed'], 'seed')
        if meta['configuration_sha256'] != hashlib.sha256(canonical(config)).hexdigest():
            raise ValueError('Configuration checksum mismatch')
        sources = meta['sources']
        if (not isinstance(sources, dict) or not sources or any(not isinstance(v, str) or len(v) != 64
                or any(c not in '0123456789abcdef' for c in v) for v in sources.values())):
            raise ValueError('Source hashes must be SHA256 digests')
        if meta['source_sha256'] != hashlib.sha256(canonical(sources)).hexdigest():
            raise ValueError('Source manifest checksum mismatch')
        if not isinstance(meta['reset_options'], dict) or set(meta['reset_options']) - {'state', 'goal'}:
            raise ValueError('Invalid reset options')
        for key, field in (('state', 'board'), ('goal', 'goal')):
            if key in meta['reset_options']:
                value = meta['reset_options'][key]
                if (not isinstance(value, list) or any(type(x) is not int for x in value)
                        or not np.array_equal(value, obs[field][0])):
                    raise ValueError('Reset options disagree with initial observation')
        horizon = _integer(config.get('max_episode_steps'), 'max_episode_steps', 1)
        if t > horizon or not np.array_equal(obs['steps_remaining'][:, 0], horizon - np.arange(t + 1)):
            raise ValueError('steps_remaining disagrees with horizon')
        scene = config['scene']
        xy, clearance = np.asarray(scene['stack_center_xy']), scene['withdraw_distance_m']
        stable = (obs['board_valid'][:, 0].astype(bool)
                  & (np.max(np.abs(obs['token_velocities']), axis=(1, 2)) < .015)
                  & ~np.any(obs['tool_contacts'], axis=1)
                  & (np.linalg.norm(obs['end_effector_position'][:, :2] - xy, axis=-1) > clearance))
        success = stable & np.all(obs['board'] == obs['goal'], axis=-1)
        if operation_mode:
            success = _check_operation_trace(obs, config, success)
        if np.any(success[:-1]) or not np.array_equal(success[1:], self.terminated):
            raise ValueError('Termination requires a stable target stack and released, withdrawn tool')
        if not np.array_equal(self.rewards, np.where(success[1:], 0, -1).astype(np.float32)):
            raise ValueError('Rewards disagree with native success')
        timeout = (obs['steps_remaining'][1:, 0] == 0) & ~success[1:]
        if not np.array_equal(timeout, self.truncated):
            raise ValueError('Timeout flags disagree with horizon')
        # Validate decoded claims from physical positions, not intended actions.
        target_z = scene['support_top_m'] + scene['symmetric_boss_height_m'] / 2 + scene['stack_pitch_m'] * np.arange(3, -1, -1)
        for i in np.flatnonzero(obs['board_valid'][:, 0]):
            positions = obs['token_positions'][i]
            order = np.argsort(positions[:, 2])[::-1]
            if (np.max(np.linalg.norm(positions[:, :2] - xy, axis=-1)) >= .005
                    or np.max(np.abs(positions[order, 2] - target_z)) >= .0025
                    or np.min(np.abs(1 - 2 * np.sum(obs['token_orientations'][i, :, 1:3] ** 2, axis=1))) <= .98
                    or not np.array_equal(order + 1, obs['board'][i])):
                raise ValueError('Valid board claim differs from physical stack geometry')
        if not isinstance(self.infos, list) or len(self.infos) != t + 1:
            raise ValueError('infos must retain T+1 simulator reports')
        for i, info in enumerate(self.infos):
            json_value(info)
            if (not isinstance(info, dict) or type(info.get('is_success')) is not bool
                    or info['is_success'] != bool(success[i]) or _integer(info.get('steps'), 'info steps') != i
                    or info.get('no_token_equalities') is not True
                    or not np.isclose(info.get('elapsed_time', np.nan), i * config['control_timestep'], atol=1e-7, rtol=0)):
                raise ValueError('Simulator info disagrees with the contact episode')
            if operation_mode and (info.get('task_mode') != 'operation'
                    or _integer(info.get('operation_prefix'), 'info operation_prefix', 1) != config['operation_prefix']
                    or type(info.get('execution_success')) is not bool
                    or info['execution_success'] != bool(success[i])
                    or _integer(info.get('operation_stage'), 'info operation_stage') != int(obs['operation_stage'][i, 0])
                    or type(info.get('permutation_changed')) is not bool
                    or info['permutation_changed'] != (not np.array_equal(obs['board'][i], obs['board'][0]))):
                raise ValueError('Execution diagnostic info disagrees with measured operation trace')
        if not isinstance(self.phases, list) or len(self.phases) != t or any(v is not None and not isinstance(v, str) for v in self.phases):
            raise ValueError('phases must contain T optional collector annotations')
        if _integer(meta['max_steps'], 'max_steps', 1) < t or not isinstance(meta['user'], dict):
            raise ValueError('Invalid collector metadata')
        if (not isinstance(meta['policy'], str) or not meta['policy']
                or (meta['stop_reason'] == 'max_steps' and t != meta['max_steps'])
                or (meta['stop_reason'] == 'policy_error' and not isinstance(meta['policy_error'], str))
                or (meta['stop_reason'] != 'policy_error' and meta['policy_error'] is not None)):
            raise ValueError('Invalid policy or collection stop declaration')
        expected = 'success' if self.terminated[-1] else 'timeout' if self.truncated[-1] else None
        if ((expected and meta['stop_reason'] != expected)
                or (not expected and meta['stop_reason'] not in ('max_steps', 'policy_completed', 'policy_error'))):
            raise ValueError('Collector ending reason differs from episode flags')

    def arrays(self):
        result = {f'observation__{name}': value for name, value in self.observations.items()}
        result.update({name: getattr(self, name) for name in _ARRAYS})
        return result

    def save(self, path):
        self.validate()
        path = Path(path).expanduser().absolute()
        if path.suffix != '.npz':
            raise ValueError('Trajectory path must have .npz suffix')
        sidecar = path.with_suffix('.json')
        if os.path.lexists(path) or os.path.lexists(sidecar):
            raise FileExistsError('Refusing to overwrite contact trajectory artifacts')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary, published = [], []
        try:
            arrays = self.arrays()
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                archive = Path(stream.name)
                temporary.append(archive)
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            manifest = {'schema_version': self.schema_version, 'artifact': {'filename': path.name, 'sha256': sha(archive)},
                'arrays': {k: {'dtype': v.dtype.name, 'shape': list(v.shape)} for k, v in arrays.items()},
                'num_steps': self.num_steps, 'configuration': json_value(self.configuration),
                'metadata': json_value(self.metadata), 'infos': json_value(self.infos), 'phases': self.phases}
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=path.parent, delete=False) as stream:
                description = Path(stream.name)
                temporary.append(description)
                json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            for source, destination in ((archive, path), (description, sidecar)):
                stat = source.stat()
                os.link(source, destination)
                published.append((destination, stat.st_dev, stat.st_ino))
            return path
        except BaseException:
            for destination, device, inode in reversed(published):
                try:
                    stat = destination.lstat()
                    if (stat.st_dev, stat.st_ino) == (device, inode):
                        destination.unlink()
                except FileNotFoundError:
                    pass
            raise
        finally:
            for item in temporary:
                item.unlink(missing_ok=True)


def load_trajectory(path):
    path = Path(path).expanduser().absolute()
    if path.suffix != '.npz':
        raise ValueError('Trajectory path must have .npz suffix')
    manifest = json.loads(path.with_suffix('.json').read_text(), object_pairs_hook=_unique)
    required = {'schema_version', 'artifact', 'arrays', 'num_steps', 'configuration', 'metadata', 'infos', 'phases'}
    if (not isinstance(manifest, dict) or set(manifest) != required
            or manifest['schema_version'] not in (SCHEMA, OPERATION_SCHEMA)):
        raise ValueError('Unsupported contact trajectory manifest')
    fields = _fields_for_config(manifest['configuration'])
    operation_mode = manifest['configuration']['schema_version'] == 'copuzzle.pancake_operation_config.v1'
    if manifest['schema_version'] != (OPERATION_SCHEMA if operation_mode else SCHEMA):
        raise ValueError('Trajectory schema and task mode differ')
    if manifest['artifact'] != {'filename': path.name, 'sha256': sha(path)}:
        raise ValueError('Artifact name/checksum mismatch')
    expected = {f'observation__{k}' for k in fields} | set(_ARRAYS)
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(expected) or set(archive.files) != expected:
            raise ValueError('Unexpected or duplicate episode arrays')
        arrays = {k: archive[k] for k in archive.files}
    descriptions = {k: {'dtype': v.dtype.name, 'shape': list(v.shape)} for k, v in arrays.items()}
    declared = manifest['arrays']
    valid_declarations = isinstance(declared, dict) and set(declared) == set(descriptions)
    if valid_declarations:
        for item in declared.values():
            if (not isinstance(item, dict) or set(item) != {'dtype', 'shape'}
                    or not isinstance(item['dtype'], str) or not isinstance(item['shape'], list)
                    or any(type(size) is not int for size in item['shape'])):
                valid_declarations = False
                break
    if not valid_declarations or declared != descriptions:
        raise ValueError('Array declarations mismatch')
    result = PancakeContactTrajectory({k: arrays[f'observation__{k}'] for k in fields},
        **{k: arrays[k] for k in _ARRAYS}, infos=manifest['infos'], phases=manifest['phases'],
        configuration=manifest['configuration'], metadata=manifest['metadata'])
    result.validate()
    if type(manifest['num_steps']) is not int or manifest['num_steps'] != result.num_steps:
        raise ValueError('Episode length mismatch')
    return result


def record_trajectory(env, policy, *, seed=0, options=None, max_steps=None, metadata=None,
                      on_policy_error='raise'):
    if on_policy_error not in ('raise', 'record'):
        raise ValueError("on_policy_error must be 'raise' or 'record'")
    seed = _integer(seed, 'seed')
    options = json_value({} if options is None else options)
    user = json_value({} if metadata is None else metadata)
    if not isinstance(options, dict) or not isinstance(user, dict):
        raise ValueError('options and metadata must be dictionaries')
    config = json_value(env.configuration())
    limit = _integer(max_steps if max_steps is not None else config['max_episode_steps'], 'max_steps', 1)
    source_paths = sorted(Path(__file__).parent.glob('*.py'))
    sources = {p.name: sha(p) for p in source_paths}
    shared = Path(__file__).parents[2] / 'lights_out' / 'physics' / 'control.py'
    if shared.exists():
        sources['shared_arm_control.py'] = sha(shared)
    policy_type = policy if inspect.isfunction(policy) else type(policy)
    try:
        policy_path = inspect.getsourcefile(policy_type)
    except (TypeError, OSError):
        policy_path = None
    if policy_path and Path(policy_path).is_file():
        sources['collector_policy.py'] = sha(policy_path)
    meta = {'seed': seed, 'reset_options': options, 'configuration_sha256': env.configuration_sha256,
            'sources': sources, 'source_sha256': hashlib.sha256(canonical(sources)).hexdigest(),
            'max_steps': limit, 'stop_reason': 'max_steps', 'policy_error': None,
            'policy': f'{policy_type.__module__}.{policy_type.__qualname__}', 'user': user}
    obs, info = env.reset(seed=seed, options=options)
    if info['is_success']:
        raise ValueError('Cannot record an already successful reset')
    if callable(getattr(policy, 'reset', None)):
        policy.reset()
    histories = {k: [np.array(v, copy=True)] for k, v in obs.items()}
    actions, rewards, terminals, timeouts, infos, phases = [], [], [], [], [deepcopy(info)], []
    for _ in range(limit):
        if getattr(policy, 'completed', False):
            meta['stop_reason'] = 'policy_completed'
            break
        try:
            action = np.asarray(policy(deepcopy(obs)), dtype=np.float32)
        except RuntimeError as error:
            if on_policy_error == 'raise':
                raise
            meta['stop_reason'], meta['policy_error'] = 'policy_error', str(error)
            break
        if getattr(policy, 'completed', False):
            meta['stop_reason'] = 'policy_completed'
            break
        obs, reward, terminal, timeout, info = env.step(action.copy())
        if set(obs) != set(histories):
            raise ValueError('Observation fields changed during collection')
        actions.append(action.copy())
        rewards.append(reward)
        terminals.append(terminal)
        timeouts.append(timeout)
        infos.append(deepcopy(info))
        phases.append(getattr(policy, 'phase', None))
        for k, v in obs.items():
            histories[k].append(np.array(v, copy=True))
        if terminal or timeout:
            meta['stop_reason'] = 'success' if terminal else 'timeout'
            break
    if not actions:
        raise ValueError('No transition was collected')
    cutoff = np.zeros(len(actions), dtype=bool)
    cutoff[-1] = not (terminals[-1] or timeouts[-1])
    result = PancakeContactTrajectory({k: np.stack(v) for k, v in histories.items()},
        np.stack(actions), np.asarray(rewards, dtype=np.float32), np.asarray(terminals, dtype=bool),
        np.asarray(timeouts, dtype=bool), cutoff, infos, phases, config, meta)
    result.validate()
    return result


def replay_trajectory(env, episode, *, atol=1e-8):
    episode.validate()
    if isinstance(atol, bool) or not np.isscalar(atol) or not np.isfinite(atol) or atol < 0:
        raise ValueError('atol must be nonnegative and finite')
    if env.configuration() != episode.configuration or env.configuration_sha256 != episode.metadata['configuration_sha256']:
        raise ValueError('Replay configuration differs from recording')
    obs, info = env.reset(seed=episode.metadata['seed'], options=episode.metadata['reset_options'])
    maximum = 0.

    def compare(actual, info, index):
        nonlocal maximum
        if set(actual) != set(episode.observations):
            raise ValueError('Replay observation fields differ')
        for k, history in episode.observations.items():
            value, expected = actual[k], history[index]
            if value.dtype != expected.dtype or value.shape != expected.shape:
                raise ValueError(f'Replay shape/dtype mismatch: {k}')
            if value.dtype.kind == 'f':
                maximum = max(maximum, float(np.max(np.abs(value - expected))))
                equal = np.allclose(value, expected, atol=atol, rtol=0)
            else:
                equal = np.array_equal(value, expected)
            if not equal:
                raise ValueError(f'Replay observation mismatch at {index}: {k}')
        expected_info = episode.infos[index]
        if set(info) != set(expected_info):
            raise ValueError('Replay info fields differ')
        for k, expected in expected_info.items():
            if isinstance(expected, float):
                equal = np.isclose(info[k], expected, atol=atol, rtol=0)
            else:
                equal = info[k] == expected
            if not equal:
                raise ValueError(f'Replay info mismatch at {index}: {k}')

    compare(obs, info, 0)
    for i, action in enumerate(episode.actions):
        obs, reward, terminal, timeout, info = env.step(action.copy())
        compare(obs, info, i + 1)
        if reward != episode.rewards[i] or terminal != episode.terminated[i] or timeout != episode.truncated[i]:
            raise ValueError('Replay reward or ending mismatch')
    return {'control_steps': episode.num_steps, 'max_absolute_observation_error': maximum,
            'configuration_sha256': env.configuration_sha256}


__all__ = ['SCHEMA', 'OPERATION_SCHEMA', 'PancakeContactTrajectory', 'record_trajectory',
           'load_trajectory', 'replay_trajectory']
