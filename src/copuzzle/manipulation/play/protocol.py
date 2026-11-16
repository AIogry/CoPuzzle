"""Portable, strictly checked continuing physical-play traces.

The archive contains one physical reset and T continuous actions. Behavior goals
are action-aligned annotations, not observations or episode terminations. Loading
uses NumPy and the public physical semantics only; no simulator is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

import numpy as np

from .semantics import TASK_IDS, frame, goal_matches, safe_boundary, validate_goal

SCHEMA = 'copuzzle.segmented_physical_play.v1'
_ARRAYS = ('actions', 'behavior_goals', 'segment_ids', 'native_rewards',
           'goal_reached', 'timeouts', 'collection_cutoff')
_COMMON = {'board': ('int32', ()), 'joint_positions': ('float64', (6,)),
           'joint_velocities': ('float64', (6,)),
           'end_effector_position': ('float64', (3,)),
           'steps_remaining': ('int32', (1,))}


def observation_spec(task_id):
    """Exact public fields for the three declared native facility profiles."""
    if task_id not in TASK_IDS:
        raise ValueError('Unsupported physical-play task')
    fields = dict(_COMMON)
    if task_id == 'grid-3x3':
        fields.update(board=('uint8', (9,)), end_effector_orientation=('float64', (4,)),
            button_positions=('float64', (9, 3)), button_rest_positions=('float64', (9, 3)),
            button_displacements=('float64', (9,)), button_velocities=('float64', (9,)),
            button_armed=('uint8', (9,)), previous_event_counts=('int32', (9,)),
            grid_shape=('int32', (2,)), grid_coordinates=('int32', (9, 2)))
    elif task_id == 'pancake-4':
        fields.update(board=('int32', (4,)), board_valid=('uint8', (1,)),
            token_positions=('float64', (4, 3)), token_orientations=('float64', (4, 4)),
            token_velocities=('float64', (4, 6)), tool_contacts=('uint8', (4,)),
            tool_contact_forces=('float64', (4,)), arm_joint_targets=('float64', (6,)),
            end_effector_orientation=('float64', (4,)), tool_target_position=('float64', (3,)),
            tool_target_orientation=('float64', (4,)), jaw_position=('float64', (1,)),
            jaw_velocity=('float64', (1,)), jaw_target=('float64', (1,)))
    else:
        fields.update(board=('int32', (20,)), board_valid=('uint8', (1,)),
            token_positions=('float64', (20, 3)), token_velocities=('float64', (20, 4)),
            turntable_angle=('float64', (1,)), turntable_velocity=('float64', (1,)),
            handle_position=('float64', (3,)), turntable_center=('float64', (3,)),
            slot_positions=('float64', (20, 3)), tool_token_contact_mask=('uint8', (20,)),
            tool_token_contact_normal_force=('float64', (20,)),
            tool_token_contact_tangent_speed=('float64', (20,)),
            tool_token_relative_velocity=('float64', (20, 3)),
            tool_linear_velocity=('float64', (3,)), socket_radius=('float64', (1,)),
            pin_radius=('float64', (1,)), socket_bottom_offset=('float64', (1,)),
            tool_clear=('uint8', (1,)), atomic_operation_count=('int32', (1,)))
    return fields


def json_value(value):
    """Convert numeric annotations without admitting objects or nonfinite JSON."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError('JSON keys must be strings')
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is float and not np.isfinite(value):
            raise ValueError('JSON values must be finite')
        return value
    raise ValueError('Unsupported JSON annotation value')


def canonical(value):
    return json.dumps(json_value(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _array(value, dtype, shape, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError(f'{name} must have dtype {dtype} and shape {shape}')
    if not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite')


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'Duplicate JSON key: {key}')
        value[key] = item
    return value


def _json_board(task_id, value, name):
    if not isinstance(value, list) or any(type(v) is not int for v in value):
        raise ValueError(f'{name} must be an integer JSON board')
    return validate_goal(task_id, value)


@dataclass
class PlayTrace:
    task_id: str
    observations: dict[str, np.ndarray]
    actions: np.ndarray
    behavior_goals: np.ndarray
    segment_ids: np.ndarray
    native_rewards: np.ndarray
    goal_reached: np.ndarray
    timeouts: np.ndarray
    collection_cutoff: np.ndarray
    infos: list
    segments: list
    configuration: dict
    metadata: dict

    @property
    def num_steps(self):
        return len(self.actions)

    @property
    def schema_version(self):
        return SCHEMA

    def arrays(self):
        return {**{f'observation__{key}': value for key, value in self.observations.items()},
                **{name: getattr(self, name) for name in _ARRAYS}}

    def validate(self):
        """Check identities, physical success evidence, segmentation and endings."""
        spec = observation_spec(self.task_id)
        if not isinstance(self.actions, np.ndarray) or self.actions.ndim != 2:
            raise ValueError('Actions must be a matrix')
        t = self.num_steps
        if t < 1:
            raise ValueError('A saved play trace requires at least one real transition')
        action_size = 7 if self.task_id == 'pancake-4' else 3
        board_dtype, board_shape = spec['board']
        for name, dtype, shape in [('actions', 'float32', (t, action_size)),
                ('behavior_goals', board_dtype, (t,) + board_shape),
                ('segment_ids', 'int32', (t,)), ('native_rewards', 'float32', (t,)),
                ('goal_reached', 'bool', (t,)), ('timeouts', 'bool', (t,)),
                ('collection_cutoff', 'bool', (t,))]:
            _array(getattr(self, name), dtype, shape, name)
        if np.any(np.abs(self.actions) > 1) or (action_size == 7 and np.any(self.actions[:, 6] < 0)):
            raise ValueError('Action outside the native continuous action bounds')
        if not isinstance(self.observations, dict) or set(self.observations) != set(spec):
            raise ValueError('Unexpected observation fields; goals and native goal monitors are excluded')
        for name, (dtype, shape) in spec.items():
            _array(self.observations[name], dtype, (t + 1,) + shape, name)
        for name in ('board',):
            for board in self.observations[name]:
                validate_goal(self.task_id, board)
        for goal in self.behavior_goals:
            validate_goal(self.task_id, goal)
        for name, value in self.observations.items():
            if value.dtype == np.uint8 and np.any(value > 1):
                raise ValueError(f'{name} must be binary')
            if 'orientation' in name and not np.allclose(np.linalg.norm(value, axis=-1), 1., atol=1e-6, rtol=0):
                raise ValueError(f'{name} must contain unit quaternions')
        self._validate_configuration()
        self._validate_observation_measurements()
        success = goal_matches(self.task_id, self.observations, self.behavior_goals, self.configuration)
        if not np.array_equal(self.goal_reached, success):
            raise ValueError('goal_reached differs from measured physical goal evidence')
        if not np.array_equal(self.native_rewards, np.where(success, 0., -1.).astype(np.float32)):
            raise ValueError('Native rewards differ from native goal achievement')
        ending = self.timeouts.astype(int) + self.collection_cutoff.astype(int)
        if np.any(ending[:-1]) or ending[-1] != 1:
            raise ValueError('Exactly one final timeout or collector cutoff is required')
        expected_timeout = self.observations['steps_remaining'][1:, 0] == 0
        if not np.array_equal(self.timeouts, expected_timeout):
            raise ValueError('Play timeout must match the global horizon, including simultaneous achievement')
        self._validate_segments()
        self._validate_infos()
        canonical({'configuration': self.configuration, 'metadata': self.metadata,
                   'segments': self.segments, 'infos': self.infos})

    def _validate_configuration(self):
        c, m, t = self.configuration, self.metadata, self.num_steps
        if not isinstance(c, dict) or not isinstance(m, dict):
            raise ValueError('Configuration and metadata must be dictionaries')
        expected = {'grid-3x3': 'copuzzle.physics_config.v1',
                    'pancake-4': 'copuzzle.pancake_manipulation_config.v1',
                    'topspin-20-4': 'copuzzle.topspin20_contact_config.v2'}[self.task_id]
        if c.get('schema_version') != expected:
            raise ValueError('Unsupported native configuration for physical play')
        if self.task_id == 'grid-3x3' and (c.get('task_name') != self.task_id
                or c.get('n_buttons') != 9 or c.get('scene', {}).get('facility_profile') != 'button_board_v2'):
            raise ValueError('Lights Out play requires the grid-3x3 button_board_v2 facility')
        if self.task_id == 'topspin-20-4' and c.get('facility_profile') != 'engaged_v2':
            raise ValueError('TopSpin play requires engaged_v2')
        required = {'seed', 'initial_state', 'initial_goal', 'global_horizon', 'collector_limit',
                    'requested_segments', 'stop_reason', 'sources', 'source_sha256',
                    'configuration_sha256', 'policy', 'reset_count'}
        if not required <= set(m):
            raise ValueError('Missing physical-play provenance')
        _integer(m['seed'], 'seed')
        if _integer(m['reset_count'], 'reset_count', 1) != 1:
            raise ValueError('A play trace must have exactly one physical reset')
        horizon = _integer(m['global_horizon'], 'global_horizon', 1)
        limit = _integer(m['collector_limit'], 'collector_limit', 1)
        _integer(m['requested_segments'], 'requested_segments', 1)
        if c.get('max_episode_steps') != horizon or type(c.get('max_episode_steps')) is not int:
            raise ValueError('Native and global horizons differ')
        if t > min(horizon, limit) or not np.array_equal(
                self.observations['steps_remaining'][:, 0], horizon - np.arange(t + 1)):
            raise ValueError('steps_remaining must decrease without resets across every segment')
        for field, value in [('initial_state', self.observations['board'][0]),
                             ('initial_goal', self.behavior_goals[0])]:
            if not np.array_equal(_json_board(self.task_id, m[field], field), value):
                raise ValueError(f'{field} differs from the saved trace')
        if m['configuration_sha256'] != hashlib.sha256(canonical(c)).hexdigest():
            raise ValueError('Configuration checksum mismatch')
        sources = m['sources']
        if not isinstance(sources, dict) or not sources or any(
                type(key) is not str or PurePosixPath(key).is_absolute() or '..' in PurePosixPath(key).parts
                or '\\' in key or not _digest(value) for key, value in sources.items()):
            raise ValueError('Sources require portable relative paths and SHA256 hashes')
        if m['source_sha256'] != hashlib.sha256(canonical(sources)).hexdigest():
            raise ValueError('Source manifest checksum mismatch')
        if not isinstance(m['policy'], str) or not m['policy']:
            raise ValueError('Missing collection policy identity')
        physics, control = c.get('physics_timestep'), c.get('control_timestep')
        if (type(physics) not in (int, float) or type(control) not in (int, float)
                or not np.isfinite([physics, control]).all() or physics <= 0 or control <= 0
                or abs(control / physics - round(control / physics)) > 1e-8 or control < physics):
            raise ValueError('Invalid native control/physics timestep ratio')
        if 'substeps' in c and (type(c['substeps']) is not int or c['substeps'] != round(control / physics)):
            raise ValueError('Native substeps disagree with the timesteps')

    def _validate_observation_measurements(self):
        o, task = self.observations, self.task_id
        if task == 'grid-3x3':
            if not np.all(o['grid_shape'] == [3, 3]) or not np.all(
                    o['grid_coordinates'] == np.array(list(np.ndindex(3, 3)), dtype=np.int32)):
                raise ValueError('Invalid public grid structure')
            if not np.all(o['button_rest_positions'] == o['button_rest_positions'][0]):
                raise ValueError('Button rest geometry changed inside play')
            if not np.array_equal(o['button_rest_positions'][0], self.configuration['button_rest_positions']):
                raise ValueError('Rest geometry differs from native configuration')
            if np.any(o['previous_event_counts'] < 0) or np.any(o['previous_event_counts'][0]):
                raise ValueError('Invalid measured button counts')
        elif task == 'pancake-4':
            if np.any(o['tool_contact_forces'] < 0) or np.any((o['jaw_target'] < 0) | (o['jaw_target'] > 1)):
                raise ValueError('Invalid force or jaw target')
        else:
            for name in ('tool_token_contact_normal_force', 'tool_token_contact_tangent_speed'):
                if np.any(o[name] < 0):
                    raise ValueError('Contact measurements must be nonnegative')
            if not np.array_equal(o['tool_token_contact_mask'], (o['tool_token_contact_normal_force'] > 1e-6).astype(np.uint8)):
                raise ValueError('Tool contact mask and measured forces disagree')
            for name in ('slot_positions', 'turntable_center', 'socket_radius', 'pin_radius', 'socket_bottom_offset'):
                if not np.all(o[name] == o[name][0]):
                    raise ValueError(f'Physical geometry changed inside play: {name}')
            if np.any(o['pin_radius'] <= 0) or np.any(o['socket_radius'] <= o['pin_radius']):
                raise ValueError('Invalid pin/socket dimensions')

    def _validate_segments(self):
        if not isinstance(self.segments, list) or not self.segments:
            raise ValueError('At least one nonempty segment must be retained')
        if len(self.segments) > self.metadata['requested_segments']:
            raise ValueError('More segments than requested')
        required = {'id', 'start_step', 'end_step', 'goal', 'requested_operation',
                    'goal_first_step', 'outcome', 'switch_safe', 'controller_completed'}
        outcomes = {'completed', 'controller_failure', 'controller_error', 'timeout',
                    'collection_cutoff', 'mismatch'}
        start = 0
        for index, segment in enumerate(self.segments):
            if not isinstance(segment, dict) or not required <= set(segment):
                raise ValueError('Missing segment fields')
            if _integer(segment['id'], 'segment id') != index or segment['start_step'] != start:
                raise ValueError('Segments must be numbered and contiguous')
            _integer(segment['start_step'], 'segment start')
            end = _integer(segment['end_step'], 'segment end', start + 1)
            if end > self.num_steps or not np.all(self.segment_ids[start:end] == index):
                raise ValueError('Segment action span/IDs disagree')
            goal = _json_board(self.task_id, segment['goal'], 'segment goal')
            if not np.all(self.behavior_goals[start:end] == goal):
                raise ValueError('A behavior goal must remain constant inside its segment')
            if np.array_equal(self.observations['board'][start], goal):
                raise ValueError('Every new goal must differ from its measured boundary board')
            op = segment['requested_operation']
            legal = ((type(op) is int and 0 <= op < 9) if self.task_id == 'grid-3x3' else
                     (type(op) is int and op in (2, 3, 4)) if self.task_id == 'pancake-4' else
                     (type(op) is str and op in ('reverse', 'shift_left', 'shift_right')))
            if not legal:
                raise ValueError('Invalid requested physical operation annotation')
            expected_goal = self.observations['board'][start].copy()
            if self.task_id == 'grid-3x3':
                coordinates = self.observations['grid_coordinates'][start]
                expected_goal[np.abs(coordinates - coordinates[op]).sum(axis=1) <= 1] ^= 1
            elif self.task_id == 'pancake-4':
                expected_goal[:op] = expected_goal[:op][::-1]
            elif op == 'reverse':
                expected_goal[:4] = expected_goal[:4][::-1]
            else:
                expected_goal = np.roll(expected_goal, -1 if op == 'shift_left' else 1)
            if not np.array_equal(goal, expected_goal):
                raise ValueError('Segment goal differs from the declared local operation at its boundary')
            found = np.flatnonzero(self.goal_reached[start:end])
            first = int(found[0]) + start if len(found) else None
            if segment['goal_first_step'] != first or (first is not None and type(segment['goal_first_step']) is not int):
                raise ValueError('First achievement step disagrees with measured evidence')
            safe = bool(safe_boundary(self.task_id, frame(self.observations, end), self.configuration))
            if type(segment['switch_safe']) is not bool or segment['switch_safe'] != safe:
                raise ValueError('switch_safe differs from measured physical boundary')
            outcome = segment['outcome']
            if outcome not in outcomes:
                raise ValueError('Unknown segment outcome')
            if type(segment['controller_completed']) is not bool:
                raise ValueError('controller_completed must be boolean')
            if outcome == 'completed' and (not safe or not self.goal_reached[end - 1]
                                           or not segment['controller_completed']):
                raise ValueError('Completed segments require controller completion, measured goal achievement and safe release')
            if index < len(self.segments) - 1 and outcome != 'completed':
                raise ValueError('A failed/partial segment must be the final retained segment')
            start = end
        if start != self.num_steps:
            raise ValueError('Segments must cover every recorded action')
        outcome = self.segments[-1]['outcome']
        stop = self.metadata['stop_reason']
        error = self.metadata.get('error')
        exception_stop = stop in ('controller_error', 'native_error')
        if exception_stop and (not isinstance(error, dict)
                or type(error.get('type')) is not str or not error['type']
                or type(error.get('message')) is not str
                or error.get('stage') not in ('policy_factory', 'controller', 'native_step')):
            raise ValueError('Exceptional collection stops must preserve their error diagnostic')
        if stop == 'native_error':
            failed = self.metadata.get('failed_action')
            if (error['stage'] != 'native_step' or not isinstance(failed, list)
                    or len(failed) != self.actions.shape[1]
                    or any(type(v) not in (int, float) for v in failed)
                    or not np.isfinite(failed).all() or np.any(np.abs(failed) > 1)
                    or (self.task_id == 'pancake-4' and failed[-1] < 0)):
                raise ValueError('Native error must retain its unrecorded attempted action')
        if self.timeouts[-1]:
            if outcome != 'timeout' or stop != 'timeout':
                raise ValueError('Global horizon exhaustion must be reported as timeout')
        elif outcome == 'timeout':
            raise ValueError('Timeout outcome without global horizon exhaustion')
        elif outcome == 'completed':
            if exception_stop:
                if len(self.segments) >= self.metadata['requested_segments']:
                    raise ValueError('No subsequent segment was requested for the declared error')
            elif stop == 'collection_cutoff':
                if self.num_steps != self.metadata['collector_limit']:
                    raise ValueError('Boundary cutoff differs from the collector limit')
            elif len(self.segments) != self.metadata['requested_segments'] or stop != 'requested_segments_complete':
                raise ValueError('Completed play must account for all requested segments')
        elif stop != outcome and not (stop == 'native_error' and outcome == 'collection_cutoff'):
            raise ValueError('Final segment outcome and collection stop reason disagree')
        if outcome == 'collection_cutoff' and stop != 'native_error' and self.num_steps != min(
                self.metadata['collector_limit'], self.metadata['global_horizon']):
            raise ValueError('Collector cutoff must occur at its declared step limit')

    def _validate_infos(self):
        if not isinstance(self.infos, list) or len(self.infos) != self.num_steps + 1:
            raise ValueError('Infos must retain every T+1 native report')
        c, o, count = self.configuration, self.observations, 0
        substeps = round(c['control_timestep'] / c['physics_timestep'])
        for index, info in enumerate(self.infos):
            if not isinstance(info, dict) or type(info.get('is_success')) is not bool:
                raise ValueError('Invalid native info report')
            if info['is_success'] != (bool(self.goal_reached[index - 1]) if index else False):
                raise ValueError('Native success must be an achievement flag, not a play ending')
            if _integer(info.get('steps'), 'native info step') != index:
                raise ValueError('Native step counter reset inside play')
            time = info.get('elapsed_time')
            if type(time) not in (int, float) or not np.isfinite(time) or not np.isclose(
                    time, index * c['control_timestep'], rtol=0, atol=1e-7):
                raise ValueError('Native physical time must remain globally continuous')
            if self.task_id == 'pancake-4':
                if info.get('no_token_equalities') is not True:
                    raise ValueError('Pancake play requires the native contact-only configuration')
                for name in ('ik_position_error', 'ik_rotation_error'):
                    if type(info.get(name)) not in (int, float) or not np.isfinite(info[name]) or info[name] < 0:
                        raise ValueError('Invalid IK diagnostic')
                continue
            events = info.get('event_log')
            if not isinstance(events, list) or (index == 0 and events):
                raise ValueError('Invalid measured event list')
            if self.task_id == 'grid-3x3':
                if info.get('rule_id') != c['rule_id']:
                    raise ValueError('Native rule identity changed inside play')
                counts = np.zeros(9, dtype=np.int32)
                previous = (-1, -1)
                for event in events:
                    if not isinstance(event, dict) or set(event) != {'substep', 'button'}:
                        raise ValueError('Invalid button event')
                    tick, button = event['substep'], event['button']
                    _integer(tick, 'event substep'); _integer(button, 'event button')
                    if tick >= substeps or button >= 9 or (tick, button) <= previous:
                        raise ValueError('Button events must be ordered within their physical interval')
                    previous = (tick, button)
                    counts[button] += 1
                if not np.array_equal(counts, o['previous_event_counts'][index]):
                    raise ValueError('Measured button events and counts disagree')
                if index:
                    board = o['board'][index - 1].copy()
                    coordinates = o['grid_coordinates'][index]
                    for button in np.flatnonzero(counts % 2):
                        board[np.abs(coordinates - coordinates[button]).sum(axis=1) <= 1] ^= 1
                    if not np.array_equal(board, o['board'][index]):
                        raise ValueError('Lamp state differs from measured button events')
                continue
            if len(events) > 1:
                raise ValueError('Only one measured TopSpin board commit is possible per interval')
            mask = o['tool_token_contact_mask'][index]
            contacts = info.get('tool_token_contact_ids')
            if (not isinstance(contacts, list) or any(type(v) is not int for v in contacts)
                    or contacts != (np.flatnonzero(mask) + 1).tolist()):
                raise ValueError('TopSpin contact identities disagree')
            if type(info.get('tool_handle_contact')) is not bool or type(info.get('tool_clear')) is not bool:
                raise ValueError('Invalid TopSpin contact/clear diagnostic')
            clear = bool(o['end_effector_position'][index, 2] > .15 and not mask.any() and not info['tool_handle_contact'])
            if info['tool_clear'] != clear or bool(o['tool_clear'][index, 0]) != clear:
                raise ValueError('TopSpin clearance disagrees with measured height/contact')
            if index:
                before, after = o['board'][index - 1:index + 1]
                if bool(events) != (not np.array_equal(before, after)):
                    raise ValueError('TopSpin measured events and committed boards disagree')
                if events:
                    event = events[0]
                    possibilities = {'reverse_four': np.r_[before[:4][::-1], before[4:]],
                                     'shift_left': np.roll(before, -1), 'shift_right': np.roll(before, 1)}
                    operation = next((key for key, value in possibilities.items() if np.array_equal(value, after)),
                                     'unclassified_measured_rearrangement')
                    cost = 1 if operation in possibilities else None
                    count += cost or 0
                    expected = {'operation': operation, 'state_before': before.tolist(), 'state_after': after.tolist(),
                                'atomic_cost': cost, 'operation_index': count if cost else None}
                    if (event != expected or not clear
                            or (cost is not None and (type(event['atomic_cost']) is not int
                                                     or type(event['operation_index']) is not int))):
                        raise ValueError('TopSpin measured operation/cost/withdrawal mismatch')
                    _json_board(self.task_id, event['state_before'], 'event initial board')
                    _json_board(self.task_id, event['state_after'], 'event final board')
            if (type(info.get('atomic_operation_count')) is not int or info['atomic_operation_count'] != count
                    or o['atomic_operation_count'][index, 0] != count):
                raise ValueError('TopSpin atomic counter reset or changed without an event')

    def save(self, path):
        """Publish a checked NPZ and commit-marker JSON without overwriting."""
        self.validate()
        path = Path(path).expanduser().absolute()
        if path.suffix != '.npz':
            raise ValueError('Play archive must use .npz')
        sidecar = path.with_suffix('.json')
        if os.path.lexists(path) or os.path.lexists(sidecar):
            raise FileExistsError('Refusing to overwrite a play artifact')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary, published = [], []
        try:
            arrays = self.arrays()
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                archive = Path(stream.name); temporary.append(archive)
                np.savez_compressed(stream, **arrays)
                stream.flush(); os.fsync(stream.fileno())
            header = {'schema_version': SCHEMA, 'task_id': self.task_id, 'num_steps': self.num_steps,
                'artifact': {'filename': path.name, 'sha256': _sha(archive)},
                'arrays': {key: {'dtype': value.dtype.name, 'shape': list(value.shape)} for key, value in arrays.items()},
                'configuration': self.configuration, 'metadata': self.metadata,
                'infos': self.infos, 'segments': self.segments}
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=path.parent, delete=False) as stream:
                description = Path(stream.name); temporary.append(description)
                json.dump(json_value(header), stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            for source, destination in ((archive, path), (description, sidecar)):
                identity = source.stat()
                os.link(source, destination)
                published.append((destination, identity.st_dev, identity.st_ino))
            return path
        except BaseException:
            for destination, device, inode in reversed(published):
                if destination.exists() and (destination.stat().st_dev, destination.stat().st_ino) == (device, inode):
                    destination.unlink()
            raise
        finally:
            for item in temporary:
                item.unlink(missing_ok=True)


def load_play_trace(path):
    """Load and validate all arrays and annotations using NumPy only."""
    path = Path(path).expanduser().absolute()
    if path.suffix != '.npz':
        raise ValueError('Play archive must use .npz')
    with path.with_suffix('.json').open(encoding='utf8') as stream:
        header = json.load(stream, object_pairs_hook=_unique,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    expected = {'schema_version', 'task_id', 'num_steps', 'artifact', 'arrays',
                'configuration', 'metadata', 'infos', 'segments'}
    if not isinstance(header, dict) or set(header) != expected or header['schema_version'] != SCHEMA:
        raise ValueError('Unsupported physical-play manifest')
    if header['artifact'] != {'filename': path.name, 'sha256': _sha(path)}:
        raise ValueError('Play NPZ filename/checksum mismatch')
    _integer(header['num_steps'], 'num_steps', 1)
    names = set(_ARRAYS) | {f'observation__{name}' for name in observation_spec(header['task_id'])}
    if not isinstance(header['arrays'], dict) or set(header['arrays']) != names:
        raise ValueError('Manifest array identity mismatch')
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(names) or set(archive.files) != names:
            raise ValueError('NPZ array identity mismatch')
        arrays = {name: archive[name].copy() for name in names}
    for name, array in arrays.items():
        descriptor = header['arrays'][name]
        if (descriptor != {'dtype': array.dtype.name, 'shape': list(array.shape)}
                or not isinstance(descriptor.get('shape'), list)
                or any(type(dim) is not int for dim in descriptor['shape'])):
            raise ValueError('NPZ array descriptor mismatch')
    trace = PlayTrace(task_id=header['task_id'],
        observations={name.removeprefix('observation__'): value for name, value in arrays.items()
                      if name.startswith('observation__')},
        **{name: arrays[name] for name in _ARRAYS}, configuration=header['configuration'],
        metadata=header['metadata'], infos=header['infos'], segments=header['segments'])
    if trace.num_steps != header['num_steps']:
        raise ValueError('Manifest transition count mismatch')
    trace.validate()
    return trace
