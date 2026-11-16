"""Continuous, original-goal transition batches from checked physical collections.

This is an offline data boundary, not an adapter for CoPuzzleLab's discrete
learners. No simulator, planner, hindsight goal or reference action is used to
construct learner features. All recorded outcomes remain in the selected split.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = 'copuzzle.manipulation_transitions.v1'
TASK_IDS = ('grid-3x3', 'pancake-4', 'topspin-20-4')
SPLITS = ('train', 'dev', 'test')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _json(path):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique, parse_constant=invalid)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _local(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Collection paths must be relative and contained in the collection')
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing or external collection file: {name}')
    return path


def _public_observations(task_id, episode):
    observations = dict(episode.observations)
    if task_id == 'grid-3x3':
        static = {**episode.structure.observation_fields(),
                  'button_rest_positions': episode.button_rest_positions}
        observations.update({name: np.broadcast_to(value, (len(episode.actions) + 1,) + value.shape)
                             for name, value in static.items()})
    return observations


def _check_profile(task_id, config):
    valid = {
        'grid-3x3': (config.get('task_name') == 'grid-3x3'
                      and config.get('scene', {}).get('facility_profile') == 'button_board_v2'),
        'pancake-4': config.get('schema_version') == 'copuzzle.pancake_manipulation_config.v1',
        'topspin-20-4': (config.get('schema_version') == 'copuzzle.topspin20_contact_config.v2'
                          and config.get('facility_profile') == 'engaged_v2'),
    }
    if not valid[task_id]:
        raise ValueError('Unsupported facility or diagnostic mode; use a direct goal-task collection')


def _float_features(value):
    if not np.issubdtype(value.dtype, np.number) and value.dtype != np.dtype(bool):
        raise ValueError('Public observation features must be numeric')
    with np.errstate(over='ignore', invalid='ignore'):
        result = value.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Features cannot be represented as finite float32 values')
    return result


@dataclass
class ContinuousDataset:
    observations: np.ndarray
    next_observations: np.ndarray
    goals: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    collection_cutoff: np.ndarray
    bootstrap_masks: np.ndarray
    episode_offsets: np.ndarray
    episode_ids: tuple[str, ...]
    feature_spec: list[dict]
    configuration: dict
    provenance: dict

    def __len__(self):
        return len(self.actions)

    @property
    def episode_end(self):
        return self.terminated | self.truncated | self.collection_cutoff

    def sample(self, batch_size, *, rng):
        """Sample stored transitions; next states never come from another episode.

        Episode boundaries always remain available. Bootstrap masks follow the
        explicitly recorded mode selected by load_transitions.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if not isinstance(rng, np.random.Generator):
            raise TypeError('Pass an explicit numpy.random.Generator')
        indices = rng.integers(len(self), size=batch_size)
        names = ('observations', 'next_observations', 'goals', 'actions', 'rewards',
                 'terminated', 'truncated', 'collection_cutoff', 'bootstrap_masks')
        batch = {name: getattr(self, name)[indices].copy() for name in names}
        episodes = np.searchsorted(self.episode_offsets[1:], indices, side='right')
        batch.update(episode_end=self.episode_end[indices], episode_index=episodes,
                     step_index=indices - self.episode_offsets[episodes])
        return batch


def load_transitions(collection, *, task_id, split='train', bootstrap_mode='terminated_only'):
    """Load one task/split without silently discarding unsuccessful trajectories.

    Native headers and every declared collection file are checked before use.
    A collection error with no trajectory is rejected for this split: the caller
    must address that incomplete data explicitly instead of training on a hidden
    success-only subset. Reference plans and diagnostic info never enter batches.
    """
    from .suite import load_suite, suite_sha256
    from .runner import load_native

    if bootstrap_mode not in ('terminated_only', 'finite_horizon'):
        raise ValueError('bootstrap_mode must be terminated_only or finite_horizon')
    if task_id not in TASK_IDS or split not in SPLITS:
        raise ValueError('Unsupported task_id or split')
    supplied = Path(collection).expanduser().resolve()
    manifest_path = supplied / 'manifest.json' if supplied.is_dir() else supplied
    root = manifest_path.parent
    manifest = _json(manifest_path)
    if (manifest.get('schema_version') != 'copuzzle.manipulation_collection.v1'
            or manifest.get('status') != 'complete'):
        raise ValueError('Require a completed physical collection manifest')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Missing collection file checksums')
    for name, expected in files.items():
        if not isinstance(expected, str) or _sha(_local(root, name)) != expected:
            raise ValueError(f'Collection checksum mismatch: {name}')
    suite_file = manifest.get('suite_file')
    if suite_file not in files:
        raise ValueError('Suite must be bound by the collection file manifest')
    suite = load_suite(_local(root, suite_file))
    if suite_sha256(suite) != manifest.get('suite_sha256'):
        raise ValueError('Collection suite identity mismatch')
    tasks = {task['id']: task for task in suite['tasks']}
    episodes = manifest.get('episodes')
    if not isinstance(episodes, list) or any(not isinstance(row, dict) for row in episodes):
        raise ValueError('Invalid collection episodes')
    ids = [row.get('id') for row in episodes]
    if any(not isinstance(value, str) for value in ids) or len(ids) != len(set(ids)):
        raise ValueError('Invalid or duplicate episode IDs')
    selection = manifest.get('selection', {})
    for name, choices in (('splits', SPLITS), ('task_ids', TASK_IDS)):
        values = selection.get(name)
        if (not isinstance(values, list) or not values or any(value not in choices for value in values)
                or len(set(values)) != len(values)):
            raise ValueError('Invalid collection task selection')
    maximum = selection.get('max_tasks')
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError('Invalid collection max_tasks')
    expected_ids = [task['id'] for task in suite['tasks']
                    if task['split'] in selection['splits'] and task['task_id'] in selection['task_ids']]
    if maximum is not None:
        expected_ids = expected_ids[:maximum]
    if (ids != expected_ids or manifest.get('selected_task_ids') != expected_ids
            or manifest.get('attempted_task_ids') != expected_ids):
        raise ValueError('Missing or reordered attempted task; collection coverage differs from selection')
    summary = manifest.get('summary', {})
    if summary.get('attempts') != len(expected_ids):
        raise ValueError('Collection summary denominator differs from attempted tasks')
    for row in episodes:
        task = tasks.get(row['id'])
        if task is None or any(row.get(key) != task[key] for key in ('task_id', 'split', 'state', 'goal', 'reset_seed')):
            raise ValueError('Episode/task binding mismatch')
    selected = [row for row in episodes if row['task_id'] == task_id and row['split'] == split]
    if not selected:
        raise ValueError('Collection has no episodes for the requested task/split')
    arrays = {key: [] for key in ('observations', 'next_observations', 'goals', 'actions', 'rewards',
                                  'terminated', 'truncated', 'collection_cutoff', 'bootstrap_masks')}
    offsets, ids, provenance_rows = [0], [], []
    feature_spec = configuration = None
    for row in selected:
        if row.get('trajectory') is None:
            raise ValueError(f"Episode {row['id']} has no trajectory; refusing to drop a collection failure")
        if row.get('error') or str(row.get('outcome', '')).endswith('_error'):
            raise ValueError(f"Episode {row['id']} has an unresolved collection/replay error")
        name = row['trajectory']
        path = _local(root, name)
        if name not in files or str(Path(name).with_suffix('.json')) not in files:
            raise ValueError('Native trajectory and sidecar must both be bound by checksums')
        episode = load_native(task_id, path)
        config = episode.configuration
        _check_profile(task_id, config)
        if configuration is None:
            configuration = deepcopy(config)
        elif configuration != config:
            raise ValueError('One continuous dataset requires the same full environment configuration')
        if row.get('configuration_sha256') != episode.metadata.get('configuration_sha256'):
            raise ValueError('Episode configuration binding mismatch')
        observation = _public_observations(task_id, episode)
        count = len(episode.actions)
        task = tasks[row['id']]
        seed_key = 'seed' if task_id == 'pancake-4' else 'reset_seed'
        options = episode.metadata.get('reset_options')
        if (episode.metadata.get(seed_key) != task['reset_seed']
                or options != {'state': task['state'], 'goal': task['goal']}
                or not np.array_equal(observation['board'][0], task['state'])
                or not np.all(observation['goal'] == np.asarray(task['goal']))):
            raise ValueError('Native reset/goal differs from its declared task')
        horizon = config['max_episode_steps']
        if horizon != task['max_control_steps'] or count > horizon or row.get('control_steps') != count:
            raise ValueError('Horizon/control count mismatch')
        if not np.array_equal(observation['steps_remaining'][:, 0], horizon - np.arange(count + 1)):
            raise ValueError('Steps remaining must match the recorded control horizon')
        terminated, truncated = episode.terminated.copy(), episode.truncated.copy()
        if np.any(truncated != ((np.arange(1, count + 1) == horizon) & ~terminated)):
            raise ValueError('Timeout flag disagrees with the physical horizon')
        cutoff = np.zeros(count, dtype=bool)
        cutoff[-1] = not (terminated[-1] or truncated[-1])
        if hasattr(episode, 'collection_cutoff') and not np.array_equal(cutoff, episode.collection_cutoff):
            raise ValueError('Native collection cutoff differs from its episode ending')
        if any(row.get(key) != bool(value[-1]) for key, value in
               (('terminated', terminated), ('truncated', truncated), ('collection_cutoff', cutoff))):
            raise ValueError('Collection ending differs from the native trajectory')
        if row.get('goal_success') != bool(terminated[-1]):
            raise ValueError('Goal success must be a true native environment termination')
        specification, flat, offset = [], [], 0
        for key in sorted(set(observation) - {'goal'}):
            value = observation[key]
            if len(value) != count + 1:
                raise ValueError('Observation history must contain T+1 frames')
            width = int(np.prod(value.shape[1:]))
            specification.append({'name': key, 'shape': list(value.shape[1:]),
                                  'dtype': str(value.dtype), 'start': offset, 'stop': offset + width})
            flat.append(_float_features(value).reshape(count + 1, width))
            offset += width
        if feature_spec is None:
            feature_spec = specification
        elif feature_spec != specification:
            raise ValueError('Public observation feature layout changed')
        states = np.concatenate(flat, axis=1)
        goals = _float_features(observation['goal'][:-1]).reshape(count, -1)
        width = 7 if task_id == 'pancake-4' else 3
        if episode.actions.shape != (count, width) or episode.actions.dtype != np.float32:
            raise ValueError('Native continuous action contract mismatch')
        mask = ~(terminated | truncated) if bootstrap_mode == 'finite_horizon' else ~terminated
        values = dict(observations=states[:-1], next_observations=states[1:], goals=goals,
                      actions=episode.actions.copy(), rewards=episode.rewards.copy(),
                      terminated=terminated, truncated=truncated, collection_cutoff=cutoff,
                      bootstrap_masks=mask.astype(np.float32))
        for key, value in values.items():
            arrays[key].append(value)
        ids.append(row['id']); offsets.append(offsets[-1] + count)
        provenance_rows.append({'id': row['id'], 'trajectory': name, 'sha256': files[name],
                                'outcome': row['outcome']})
    return ContinuousDataset(**{key: np.concatenate(values) for key, values in arrays.items()},
        episode_offsets=np.asarray(offsets, dtype=np.int64), episode_ids=tuple(ids),
        feature_spec=feature_spec, configuration=configuration,
        provenance={'schema_version': SCHEMA, 'collection_sha256': _sha(manifest_path),
                    'configuration_sha256': selected[0]['configuration_sha256'],
                    'suite_sha256': manifest['suite_sha256'], 'task_id': task_id, 'split': split,
                    'goal_semantics': 'original recorded goal; no hindsight relabeling',
                    'bootstrap_mode': bootstrap_mode,
                    'feature_profile': 'all public numeric observations except goal, sorted by name',
                    'action_semantics': 'original float32 continuous commands; no discrete conversion',
                    'episodes': provenance_rows})
