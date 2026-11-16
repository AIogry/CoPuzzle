"""Run explicit physical tasks through their unchanged native trajectory APIs.

Reference controllers alone receive a task's witness. External factories receive
only a task ID, and policies receive copied public observations. Failed attempts
remain in summaries, including native-recorder errors that return no trajectory.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile

import numpy as np

TASK_IDS = ('grid-3x3', 'pancake-4', 'topspin-20-4')
SPLITS = ('train', 'dev', 'test')
SCHEMA = 'copuzzle.manipulation_collection.v1'


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    """Publish a complete finite JSON document without replacing another file."""
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(f'Refusing to overwrite {path}')
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _check_task_id(task_id):
    if task_id not in TASK_IDS:
        raise ValueError(f'Unsupported physical task: {task_id}')
    return task_id


def make_environment(task):
    """Build the fixed current physical profile with the suite's actual horizon.

    ``max_control_steps`` is the environment budget. A collection cap is passed
    separately to the native recorder and never silently changes this horizon.
    """
    task_id = _check_task_id(task['task_id'])
    horizon = _positive(task['max_control_steps'], 'max_control_steps')
    if task_id == 'grid-3x3':
        from copuzzle.games.lights_out.physics.env import PhysicalPuzzleEnv
        from copuzzle.games.lights_out.rules import get_rule
        return PhysicalPuzzleEnv(get_rule(task_id), max_episode_steps=horizon,
                                 facility_profile='button_board_v2')
    if task_id == 'pancake-4':
        from copuzzle.games.pancake.physics.env import PancakeManipulationEnv
        return PancakeManipulationEnv(max_episode_steps=horizon, task_mode='goal')
    from copuzzle.games.topspin.physics.env import TopSpinPhysicalEnv
    return TopSpinPhysicalEnv(max_episode_steps=horizon, facility_profile='engaged_v2')


def _native_api(task_id):
    _check_task_id(task_id)
    if task_id == 'grid-3x3':
        from copuzzle.games.lights_out.physics.data import load_physics_trajectory, record_episode
        from copuzzle.games.lights_out.physics.replay import replay_trajectory
        return record_episode, load_physics_trajectory, replay_trajectory
    if task_id == 'pancake-4':
        from copuzzle.games.pancake.physics.data import load_trajectory, record_trajectory, replay_trajectory
        return record_trajectory, load_trajectory, replay_trajectory
    from copuzzle.games.topspin.physics.trace import load_trace, record_trace, replay_trace
    return record_trace, load_trace, replay_trace


def _check_native_identity(task_id, episode):
    config = episode.configuration
    if task_id == 'grid-3x3':
        matches = config.get('task_name') == task_id and config.get('n_buttons') == 9
    elif task_id == 'pancake-4':
        matches = config.get('rule_spec', {}).get('name') == task_id
    else:
        scene = config.get('scene', {})
        matches = scene.get('tokens') == 20 and scene.get('turntable_tokens') == 4
    if not matches:
        raise ValueError('Native trajectory belongs to a different physical task')


def load_native(task_id, path):
    """Load and validate one family's original archive without conversion."""
    episode = _native_api(task_id)[1](path)
    _check_native_identity(task_id, episode)
    return episode


def replay_native(task_id, episode):
    """Reconstruct the recorded profile and parameters, then replay its actions."""
    _check_task_id(task_id)
    episode.validate()
    _check_native_identity(task_id, episode)
    config = episode.configuration
    common = {'max_episode_steps': config['max_episode_steps'],
              'physics_dt': config['physics_timestep'], 'control_dt': config['control_timestep']}
    if task_id == 'grid-3x3':
        from copuzzle.games.lights_out.physics.env import PhysicalPuzzleEnv
        from copuzzle.games.lights_out.rules import get_rule
        env = PhysicalPuzzleEnv(get_rule(task_id), **common, action_scale=config['action_scale'],
            press_depth=config['press_depth'], release_depth=config['release_depth'],
            render_width=config['render_width'], render_height=config['render_height'],
            facility_profile=config['scene'].get('facility_profile', 'legacy_v1'))
    elif task_id == 'pancake-4':
        from copuzzle.games.pancake.physics.env import PancakeManipulationEnv
        env = PancakeManipulationEnv(**common, translation_scale=config['translation_scale'],
            rotation_scale=config['rotation_scale'], task_mode=config.get('task_mode', 'goal'),
            operation_prefix=config.get('operation_prefix'))
    else:
        from copuzzle.games.topspin.physics.env import TopSpinPhysicalEnv
        env = TopSpinPhysicalEnv(**common, action_scale=config['action_scale'],
                                facility_profile=config.get('facility_profile', 'legacy_v1'))
    try:
        if env.configuration_sha256 != episode.metadata['configuration_sha256']:
            raise ValueError('Reconstructed environment differs from native configuration')
        return _native_api(task_id)[2](env, episode)
    finally:
        env.close()


def _pancake_ready(episode):
    """Independent supported/released observation gate, without reading a plan."""
    obs, scene = episode.observations, episode.configuration['scene']
    positions = np.asarray(obs['token_positions'])
    xy = np.asarray(scene['stack_center_xy'])
    order = np.argsort(positions[:, :, 2], axis=1)[:, ::-1]
    target_z = (scene['support_top_m'] + scene['symmetric_boss_height_m'] / 2
                + scene['stack_pitch_m'] * np.arange(3, -1, -1))
    sorted_z = np.take_along_axis(positions[:, :, 2], order, axis=1)
    normal_z = 1 - 2 * np.sum(np.asarray(obs['token_orientations'])[:, :, 1:3] ** 2, axis=2)
    return (np.asarray(obs['board_valid'])[:, 0].astype(bool)
            & np.all(order + 1 == obs['board'], axis=1)
            & (np.max(np.linalg.norm(positions[:, :, :2] - xy, axis=2), axis=1) < .005)
            & (np.max(np.abs(sorted_z - target_z), axis=1) < .0025)
            & (np.min(np.abs(normal_z), axis=1) > .98)
            & (np.max(np.abs(obs['token_velocities']), axis=(1, 2)) < .015)
            & ~np.any(obs['tool_contacts'], axis=1)
            & ~np.any(obs['tool_contact_forces'] > 1e-6, axis=1)
            & (np.linalg.norm(obs['end_effector_position'][:, :2] - xy, axis=1)
               > scene['withdraw_distance_m']))


def measure_operations(task_id, episode):
    """Report observed primitive completions; unknown cost is never coerced to 0.

    Pancake measures differences between supported, stationary, released stack
    observations. This does not reconstruct rotations made while still holding
    the stack. Its controller phase and reference witness are never consulted.
    """
    _check_task_id(task_id)
    events = []
    if task_id == 'grid-3x3':
        operations = [int(value) for value in episode.event_buttons]
        semantics = 'native contact-gated press events'
    elif task_id == 'topspin-20-4':
        events = [deepcopy(event) for info in episode.infos for event in info['event_log']]
        operations = [event['operation'] for event in events]
        costs = [event.get('atomic_cost') for event in events]
        unknown = sum(cost is None for cost in costs)
        return {'measured_operations': operations, 'operation_events': events,
                'measured_atomic_cost': None if unknown else sum(costs),
                'known_atomic_cost': sum(cost for cost in costs if cost is not None),
                'unknown_atomic_cost_events': unknown,
                'operation_measurement': 'native released indexed events; unclassified cost remains unknown'}
    else:
        try:
            indices = np.flatnonzero(_pancake_ready(episode))
        except (KeyError, ValueError, TypeError, IndexError) as error:
            return {'measured_operations': None, 'operation_events': [], 'measured_atomic_cost': None,
                    'known_atomic_cost': 0, 'unknown_atomic_cost_events': None,
                    'operation_measurement': 'unavailable: missing or incompatible physical evidence',
                    'operation_measurement_error': str(error)}
        if not len(indices):
            return {'measured_operations': None, 'operation_events': [], 'measured_atomic_cost': None,
                    'known_atomic_cost': 0, 'unknown_atomic_cost_events': None,
                    'operation_measurement': 'unavailable: no supported released reference state was observed'}
        operations = []
        previous = None
        for index in indices:
            board = np.asarray(episode.observations['board'][index])
            if previous is not None and not np.array_equal(previous, board):
                matches = [k for k in (2, 3, 4)
                           if np.array_equal(np.r_[previous[:k][::-1], previous[k:]], board)]
                operation = matches[0] if len(matches) == 1 else None
                operations.append(operation)
                events.append({'control_step': int(index - 1), 'operation': operation,
                               'state_before': previous.tolist(), 'state_after': board.tolist()})
            previous = board.copy()
        semantics = 'changes between supported, stationary, released and withdrawn stack states'
    unknown = sum(value is None for value in operations)
    return {'measured_operations': operations, 'operation_events': events,
            'measured_atomic_cost': None if unknown else len(operations),
            'known_atomic_cost': len(operations) - unknown,
            'unknown_atomic_cost_events': unknown, 'operation_measurement': semantics}


class _ObservationPolicy:
    """Preserve native stop protocols while shielding live observation arrays."""
    def __init__(self, policy):
        if not callable(policy):
            raise TypeError('policy_factory must return a callable policy')
        self.policy = policy

    def reset(self):
        reset = getattr(self.policy, 'reset', None)
        if callable(reset):
            reset()

    def __call__(self, observation):
        return self.policy(deepcopy(observation))

    def __getattr__(self, name):
        return getattr(self.policy, name)


class _ZeroPolicy:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, observation):
        return np.zeros(self.shape, dtype=np.float32)


def _policy(task, env, policy_type, policy_factory):
    if policy_factory is not None:
        return _ObservationPolicy(policy_factory(task['task_id']))
    if policy_type == 'zero':
        return _ObservationPolicy(_ZeroPolicy(env.action_space.shape))
    # Only this explicitly privileged branch reads the witness.
    operations = deepcopy(task['reference_operations'])
    if task['task_id'] == 'grid-3x3':
        from copuzzle.games.lights_out.physics.scripted import ButtonSequencePolicy
        policy = ButtonSequencePolicy(operations, action_scale=env.action_scale)
    elif task['task_id'] == 'pancake-4':
        from copuzzle.games.pancake.physics.scripted import PancakeSequencePolicy
        policy = PancakeSequencePolicy(operations, translation_scale=env.translation_scale,
                                       rotation_scale=env.rotation_scale)
    else:
        from copuzzle.games.topspin.physics.scripted import SequencePolicy
        policy = SequencePolicy(operations, action_scale=env.action_scale)
    return _ObservationPolicy(policy)


def _callable_identity(value):
    target = value if inspect.isfunction(value) else type(value)
    identity = {'name': f'{target.__module__}.{target.__qualname__}', 'source_sha256': None}
    try:
        path = inspect.getsourcefile(target)
    except (TypeError, OSError):
        path = None
    if path and Path(path).is_file():
        identity.update(source_file=str(Path(path).absolute()), source_sha256=_sha(path))
    return identity


def _source_snapshot(task_ids):
    package = Path(__file__).resolve().parents[1]
    files = {Path(__file__), Path(__file__).with_name('suite.py')}
    files.add(package / 'games/lights_out/physics/control.py')
    families = {'grid-3x3': 'lights_out', 'pancake-4': 'pancake', 'topspin-20-4': 'topspin'}
    for task_id in task_ids:
        folder = package / 'games' / families[task_id]
        files.update((folder / 'physics').rglob('*.py'))
        if task_id == 'grid-3x3':
            files.update(folder / name for name in ('rules.py', 'structure.py', 'presets.py'))
        elif task_id == 'pancake-4':
            files.add(package / 'games/permutation/rules.py')
    assets = package / 'games/lights_out/physics/assets/ur5e'
    files.update(path for path in assets.rglob('*') if path.is_file())
    values = {str(path.relative_to(package)): _sha(path) for path in sorted(files) if path.is_file()}
    return {'scope': 'runner, suite, selected native physics/dependencies, shared arm controller and verified robot assets',
            'files': values, 'sha256': hashlib.sha256(_canonical(values)).hexdigest()}


def _summary(rows):
    count = len(rows)
    successes = sum(row['goal_success'] for row in rows)
    return {'attempts': count, 'successes': successes, 'failures': count - successes,
            'success_rate': successes / count if count else None,
            'timeouts': sum(row['truncated'] is True for row in rows),
            'collection_cutoffs': sum(row['collection_cutoff'] is True for row in rows),
            'errors': sum(row['error'] is not None for row in rows),
            'controller_failures': sum(row['outcome'] == 'policy_failed' for row in rows),
            'trajectories': sum(row['trajectory'] is not None for row in rows),
            'control_steps': sum(row['control_steps'] for row in rows),
            'control_seconds': sum(row['control_seconds'] for row in rows),
            'mean_control_steps_all_attempts': sum(row['control_steps'] for row in rows) / count if count else None,
            'mean_control_seconds_all_attempts': sum(row['control_seconds'] for row in rows) / count if count else None,
            'operation_cost_known_attempts': sum(row['measured_atomic_cost'] is not None for row in rows),
            'operation_cost_unknown_attempts': sum(row['measured_atomic_cost'] is None for row in rows),
            'measured_atomic_cost_total': (sum(row['measured_atomic_cost'] for row in rows)
                                           if all(row['measured_atomic_cost'] is not None for row in rows) else None),
            'known_completed_operations_lower_bound': sum(row['known_atomic_cost'] for row in rows),
            'known_measured_atomic_cost': sum(row['measured_atomic_cost'] for row in rows
                                             if row['measured_atomic_cost'] is not None)}


def _coverage_episode(task_id, episode):
    obs = episode.observations
    if task_id == 'pancake-4':
        mask = _pancake_ready(episode)
    elif 'board_valid' in obs:
        mask = obs['board_valid'][:, 0].astype(bool)
    else:
        mask = np.ones(len(obs['board']), dtype=bool)
    states = {tuple(map(int, board)) for board in obs['board'][mask]}
    goal = tuple(map(int, obs['goal'][0]))
    return {'states': states, 'goals': {goal},
            'reset_pairs': {(tuple(map(int, obs['board'][0])), goal)},
            'state_goal_pairs': {(state, goal) for state in states}}


def _actual_overlap(rows, observations):
    result = {}
    for task_id in sorted({row['task_id'] for row in rows}):
        sets = {split: {name: set() for name in ('states', 'goals', 'reset_pairs', 'state_goal_pairs')}
                for split in SPLITS}
        for row in rows:
            if row['task_id'] == task_id and row['id'] in observations:
                for name, values in observations[row['id']].items():
                    sets[row['split']][name].update(values)
        pairs = {}
        for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
            pairs[f'{a}:{b}'] = {name: {'intersection': len(sets[a][name] & sets[b][name]),
                'left_count': len(sets[a][name]), 'right_count': len(sets[b][name]),
                'left_fraction_shared': len(sets[a][name] & sets[b][name]) / len(sets[a][name]) if sets[a][name] else None,
                'right_fraction_shared': len(sets[a][name] & sets[b][name]) / len(sets[b][name]) if sets[b][name] else None}
                for name in sets[a]}
        result[task_id] = {'split_counts': {split: {name: len(values) for name, values in data.items()}
                                          for split, data in sets.items()}, 'pairwise': pairs}
    return {'definition': 'States are actually recorded valid boards; Pancake additionally requires supported, stationary, released and withdrawn. State-goal pairs retain each original episode goal.',
            'attempts_without_trajectory': sum(row['trajectory'] is None for row in rows),
            'not_a_state_disjointness_claim': True, 'by_task': result}


def _error(stage, error):
    return {'stage': stage, 'type': type(error).__name__, 'message': str(error)}


def _run_task(task, output, policy_type, policy_factory, max_steps, replay, suite_hash):
    task_id = task['task_id']
    row = {'id': task['id'], 'task_id': task_id, 'split': task['split'],
           'reset_seed': task['reset_seed'], 'state': deepcopy(task['state']), 'goal': deepcopy(task['goal']),
           'max_control_steps': task['max_control_steps'], 'policy_type': policy_type,
           'trajectory': None, 'configuration_sha256': None, 'control_steps': 0, 'control_seconds': 0.,
           'goal_success': False, 'native_goal_success': None, 'outcome': 'collection_error',
           'native_stop': None, 'terminated': None, 'truncated': None, 'collection_cutoff': None,
           'bootstrap_mask': None, 'measured_operations': None, 'measured_atomic_cost': None,
           'known_atomic_cost': 0, 'unknown_atomic_cost_events': None,
           'verified_replay': False, 'replay': None, 'error': None,
           'controller_error': None, 'failure_reason': None, 'files': {}}
    env = None
    coverage = None
    stage = 'environment'
    try:
        env = make_environment(task)
        row['configuration_sha256'] = env.configuration_sha256
        row['native_configuration'] = env.configuration()
        stage = 'policy_factory'
        controller = _policy(task, env, policy_type, policy_factory)
        row['policy_identity'] = _callable_identity(controller.policy)
        limit = min(task['max_control_steps'], max_steps) if max_steps is not None else task['max_control_steps']
        row['collector_max_steps'] = limit
        recorder = _native_api(task_id)[0]
        kwargs = {'seed': task['reset_seed'], 'options': {'state': deepcopy(task['state']), 'goal': deepcopy(task['goal'])},
                  'max_steps': limit, 'metadata': {'suite_sha256': suite_hash, 'suite_task_id': task['id'],
                  'split': task['split'], 'policy_type': policy_type,
                  'reference_witness_access': policy_type == 'reference'}}
        if task_id == 'pancake-4':
            kwargs['on_policy_error'] = 'record'
        stage = 'collection'
        episode = recorder(env, controller, **kwargs)
        row['control_steps'] = int(episode.num_steps)
        row['control_seconds'] = float(episode.num_steps * episode.configuration['control_timestep'])
        row['terminated'], row['truncated'] = bool(episode.terminated[-1]), bool(episode.truncated[-1])
        row['collection_cutoff'] = bool(episode.collection_cutoff[-1]) if hasattr(episode, 'collection_cutoff') else not (row['terminated'] or row['truncated'])
        row['bootstrap_mask'] = 0. if row['terminated'] else 1.
        row['native_goal_success'] = row['terminated']
        meta = episode.metadata
        row['native_stop'] = (meta['stop_reason'] if task_id == 'pancake-4' else meta['collector_stop']
                              if task_id == 'topspin-20-4' else 'terminated' if row['terminated']
                              else 'truncated' if row['truncated'] else 'max_steps')
        row['outcome'] = ('success' if row['terminated'] else 'timeout' if row['truncated']
                          else 'policy_failed' if row['native_stop'] in ('policy_failed', 'policy_error')
                          else 'collection_cutoff')
        row['goal_success'] = row['terminated']
        if task_id == 'pancake-4' and meta.get('policy_error'):
            row['controller_error'] = {'type': 'RuntimeError', 'message': meta['policy_error']}
            row['failure_reason'] = meta['policy_error']
            row['goal_success'] = False
        if row['native_stop'] == 'policy_failed':
            row['failure_reason'] = str(getattr(controller.policy, 'failure', None) or 'native policy finished with failure')
        row['native_configuration_schema'] = episode.configuration['schema_version']
        stage = 'save'
        folder = output / task_id / task['split']
        folder.mkdir(parents=True, exist_ok=True)
        path = episode.save(folder / f"{task['id']}.npz")
        row['trajectory'] = str(path.relative_to(output))
        stage = 'load'
        loaded = load_native(task_id, path)
        header = json.loads(path.with_suffix('.json').read_text())
        row['native_schema'] = header.get('schema_version')
        row.update(measure_operations(task_id, loaded))
        coverage = _coverage_episode(task_id, loaded)
        row['files'] = {str(p.relative_to(output)): _sha(p) for p in (path, path.with_suffix('.json'))}
        if replay:
            stage = 'replay'
            row['replay'] = replay_native(task_id, loaded)
            row['verified_replay'] = True
    except Exception as error:
        row['goal_success'] = False
        row['outcome'] = f'{stage}_error'
        row['error'] = _error(stage, error)
        if env is not None and row['control_steps'] == 0:
            # These are genuine live counters, not a reconstructed/fabricated
            # trajectory. The native recorder may have discarded its prefix.
            row['control_steps'] = int(getattr(env, '_steps', 0))
            row['control_seconds'] = float(getattr(getattr(env, 'data', None), 'time', 0.))
        row['trajectory_limitation'] = ('Native recorder returned no saveable trajectory; its attempted prefix is not fabricated'
                                        if row['trajectory'] is None else 'Saved native trajectory retained; see error stage')
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                row['goal_success'] = False
                row['outcome'] = 'close_error'
                row['close_error'] = _error('close', error)
                if row['error'] is None:
                    row['error'] = row['close_error']
    return row, coverage


def run_collection(suite, output, *, policy='reference', splits=None, task_ids=None,
                   max_tasks=None, max_steps=None, replay=True, policy_factory=None):
    """Attempt selected tasks, retain failures, and atomically publish a report.

    ``policy_factory(task_id)`` takes precedence over the default reference
    selection and is always labelled external. No task record or witness is
    passed to it. Native Lights Out recording does not stop on ``completed``;
    arbitrary policy exceptions may therefore produce an explicit error without
    a trajectory. No idle actions are inserted to manufacture a saved prefix.
    """
    from .suite import save_suite, split_overlap, suite_sha256, validate_suite
    suite = deepcopy(suite)
    validate_suite(suite)
    suite_hash = suite_sha256(suite)
    output = Path(output).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f'Refusing to overwrite collection directory: {output}')
    if not isinstance(policy, str) or not policy.strip():
        raise ValueError('policy must be a nonempty label')
    if policy_factory is not None and not callable(policy_factory):
        raise TypeError('policy_factory must be callable')
    if policy_factory is None and policy not in ('reference', 'zero'):
        raise ValueError('policy must be reference or zero unless a policy_factory is supplied')
    if type(replay) is not bool:
        raise ValueError('replay must be a boolean')
    policy_type = 'external' if policy_factory is not None else policy
    for value, name in ((max_tasks, 'max_tasks'), (max_steps, 'max_steps')):
        if value is not None:
            _positive(value, name)
    selection = {}
    for values, name, choices in ((splits, 'splits', SPLITS), (task_ids, 'task_ids', TASK_IDS)):
        values = choices if values is None else (values,) if isinstance(values, str) else tuple(values)
        if not values or len(set(values)) != len(values) or any(value not in choices for value in values):
            raise ValueError(f'{name} must contain distinct supported values')
        selection[name] = list(values)
    tasks = [task for task in suite['tasks'] if task['split'] in selection['splits'] and task['task_id'] in selection['task_ids']]
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    if not tasks:
        raise ValueError('No tasks match the collection selection')
    selection.update(max_tasks=max_tasks, max_steps=max_steps, replay=replay)
    policy_configuration = {'policy_type': policy_type, 'requested_policy': policy,
                            'reference_witness_access': policy_type == 'reference',
                            'external_factory': _callable_identity(policy_factory) if policy_factory is not None else None,
                            'external_parameter_freezing_enforced': False}
    fingerprint = hashlib.sha256(_canonical({'suite_sha256': suite_hash, 'selection': selection,
                                            'policy_configuration': policy_configuration})).hexdigest()
    before = _source_snapshot({task['task_id'] for task in tasks})
    output.mkdir(parents=True)
    save_suite(suite, output / 'suite.json')
    rows, observations = [], {}
    for task in tasks:
        row, coverage = _run_task(task, output, policy_type, policy_factory, max_steps, replay, suite_hash)
        rows.append(row)
        if coverage is not None:
            observations[task['id']] = coverage
        print(json.dumps({'id': row['id'], 'task_id': row['task_id'], 'split': row['split'],
                          'outcome': row['outcome'], 'control_steps': row['control_steps'],
                          'trajectory_saved': row['trajectory'] is not None, 'verified_replay': row['verified_replay']},
                         sort_keys=True, allow_nan=False), flush=True)
    after = _source_snapshot({task['task_id'] for task in tasks})
    manifest = {'schema_version': SCHEMA, 'status': 'complete' if before == after else 'incomplete_source_change',
                'suite_file': 'suite.json', 'suite_sha256': suite_hash,
                'policy_type': policy_type, 'policy_configuration': policy_configuration,
                'selection': selection, 'run_fingerprint': fingerprint,
                'source_snapshot': before, 'source_snapshot_after': after, 'source_stable': before == after,
                'selected_task_ids': [task['id'] for task in tasks],
                'attempted_task_ids': [row['id'] for row in rows],
                'episodes': rows, 'summary': _summary(rows),
                'by_task': {task_id: _summary([row for row in rows if row['task_id'] == task_id])
                            for task_id in sorted({row['task_id'] for row in rows})},
                'by_split': {split: _summary([row for row in rows if row['split'] == split]) for split in SPLITS},
                'by_task_split': {task_id: {split: _summary([row for row in rows if row['task_id'] == task_id and row['split'] == split])
                                          for split in SPLITS} for task_id in sorted({row['task_id'] for row in rows})},
                'planned_overlap': split_overlap(suite), 'observed_overlap': _actual_overlap(rows, observations),
                'bootstrap_semantics': 'row bootstrap_mask uses terminated_only: success=0, timeout/cutoff=1; the reader may explicitly choose finite_horizon',
                'native_trajectory_policy': 'original native schemas and original goals retained; no hindsight relabeling',
                'recorder_limitations': {'grid-3x3': 'completed does not stop its native recorder; policy exceptions may leave no saveable prefix',
                                        'pancake-4': 'native RuntimeError preservation enabled; other collection errors may leave no trajectory',
                                        'topspin-20-4': 'native completed/finished supported; arbitrary exceptions may leave no trajectory'},
                'files': {str(path.relative_to(output)): _sha(path) for path in sorted(output.rglob('*')) if path.is_file()}}
    manifest['data_snapshot'] = {'files_sha256': hashlib.sha256(_canonical(manifest['files'])).hexdigest(),
                                 'file_count': len(manifest['files']),
                                 'scope': 'saved suite and native trajectory files; final manifest excluded'}
    _atomic_json(output / 'manifest.json', manifest)
    return manifest


__all__ = ['make_environment', 'load_native', 'replay_native', 'measure_operations', 'run_collection']
