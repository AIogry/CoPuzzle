"""Offline protocol checks use synthetic measurements, never simulator replay."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from copuzzle.manipulation.play.protocol import (
    PlayTrace, canonical, load_play_trace, observation_spec,
)
from copuzzle.manipulation.play.semantics import goal_matches


def _slots():
    radius, half = .012 / np.sin(np.pi / 12), .05
    points = [[x, radius] for x in np.array([-3, -1, 1, 3]) * .012]
    points += [[half + radius*np.cos(a), radius*np.sin(a)] for a in np.deg2rad([75,45,15,-15,-45,-75])]
    points += [[x, -radius] for x in np.array([3,1,-1,-3]) * .012]
    points += [[-half + radius*np.cos(a), radius*np.sin(a)] for a in np.deg2rad([-105,-135,-165,-195,-225,-255])]
    return np.column_stack((np.array(points) + [.55, 0], np.full(20, .06)))


def _fixture(task='grid-3x3'):
    """Two local operation segments, with three stationary frames per target."""
    t, horizon = 6, 20
    obs = {key: np.zeros((t + 1,) + shape, dtype=dtype)
           for key, (dtype, shape) in observation_spec(task).items()}
    obs['steps_remaining'][:, 0] = horizon - np.arange(t + 1)
    obs['end_effector_position'][:, 2] = .5
    for key in obs:
        if 'orientation' in key:
            obs[key][..., 0] = 1
    initial = np.zeros(9, dtype=np.uint8) if task == 'grid-3x3' else np.arange(1, 5 if task == 'pancake-4' else 21, dtype=np.int32)
    first, second = initial.copy(), initial.copy()
    if task == 'grid-3x3':
        first[[0, 1, 3]] ^= 1
        second = first.copy(); second[[0, 1, 2, 4]] ^= 1
        operations = [0, 1]
    elif task == 'pancake-4':
        first[:2] = first[:2][::-1]
        second = first.copy(); second[:3] = second[:3][::-1]
        operations = [2, 3]
    else:
        first, second = np.roll(initial, 1), np.roll(initial, 2)
        operations = ['shift_right', 'shift_right']
    obs['board'][0] = initial
    obs['board'][1:4] = first
    obs['board'][4:] = second
    goals = np.stack([first] * 3 + [second] * 3)
    config = {'physics_timestep': .002, 'control_timestep': .05, 'max_episode_steps': horizon}
    if task == 'grid-3x3':
        config.update(schema_version='copuzzle.physics_config.v1', task_name=task, n_buttons=9,
                      rule_id='synthetic-grid-rule', scene={'facility_profile': 'button_board_v2'},
                      release_depth=.003, button_rest_positions=np.zeros((9, 3)).tolist())
        obs['grid_shape'][:] = [3, 3]
        obs['grid_coordinates'][:] = list(np.ndindex(3, 3))
        obs['button_armed'][:] = 1
        obs['previous_event_counts'][1, 0] = 1
        obs['previous_event_counts'][4, 1] = 1
    elif task == 'pancake-4':
        scene = {'stack_center_xy': [.5, 0], 'support_top_m': .02,
                 'symmetric_boss_height_m': .02, 'stack_pitch_m': .02, 'withdraw_distance_m': .1}
        config.update(schema_version='copuzzle.pancake_manipulation_config.v1', scene=scene)
        obs['board_valid'][:] = 1
        for i, board in enumerate(obs['board']):
            obs['token_positions'][i, :, :2] = [.5, 0]
            obs['token_positions'][i, board - 1, 2] = .03 + .02*np.arange(3, -1, -1)
    else:
        slots = _slots()
        config.update(schema_version='copuzzle.topspin20_contact_config.v2', facility_profile='engaged_v2',
                      scene={'slot_positions': slots.tolist()})
        obs['board_valid'][:] = obs['tool_clear'][:] = 1
        obs['slot_positions'][:] = slots
        obs['turntable_center'][:] = [.55, .012/np.sin(np.pi/12), .048]
        obs['socket_radius'][:] = .005
        obs['pin_radius'][:] = .0032
        obs['socket_bottom_offset'][:] = -.004
        obs['atomic_operation_count'][1:4] = 1
        obs['atomic_operation_count'][4:] = 2
        for i, board in enumerate(obs['board']):
            obs['token_positions'][i, board - 1] = slots
    reached = goal_matches(task, obs, goals, config)
    infos = []
    for index in range(t + 1):
        info = {'steps': index, 'elapsed_time': index * .05,
                'is_success': bool(reached[index - 1]) if index else False}
        if task == 'pancake-4':
            info.update(no_token_equalities=True, ik_position_error=0., ik_rotation_error=0.)
        elif task == 'grid-3x3':
            info.update(rule_id=config['rule_id'], event_log=(
                [{'substep': 2, 'button': 0 if index == 1 else 1}] if index in (1, 4) else []))
        else:
            event = []
            if index in (1, 4):
                event = [{'operation': 'shift_right', 'state_before': obs['board'][index - 1].tolist(),
                          'state_after': obs['board'][index].tolist(), 'atomic_cost': 1,
                          'operation_index': 1 if index == 1 else 2}]
            info.update(event_log=event, tool_handle_contact=False, tool_token_contact_ids=[],
                        max_token_speed=0., atomic_operation_count=int(obs['atomic_operation_count'][index, 0]), tool_clear=True)
        infos.append(info)
    sources = {'copuzzle/manipulation/play/protocol.py': 'a' * 64}
    metadata = {'seed': 3, 'initial_state': initial.tolist(), 'initial_goal': first.tolist(),
                'global_horizon': horizon, 'collector_limit': horizon, 'requested_segments': 2,
                'stop_reason': 'requested_segments_complete', 'sources': sources,
                'source_sha256': hashlib.sha256(canonical(sources)).hexdigest(),
                'configuration_sha256': hashlib.sha256(canonical(config)).hexdigest(), 'policy': 'synthetic-fixture'}
    metadata['reset_count'] = 1
    segments = [{'id': i, 'start_step': 3*i, 'end_step': 3*(i+1), 'goal': goal.tolist(),
                 'requested_operation': operations[i], 'goal_first_step': int(np.flatnonzero(reached[3*i:3*i+3])[0]) + 3*i,
                 'outcome': 'completed', 'switch_safe': True, 'controller_completed': True}
                for i, goal in enumerate((first, second))]
    return PlayTrace(task_id=task, observations=obs, actions=np.zeros((t, 7 if task == 'pancake-4' else 3), dtype=np.float32),
        behavior_goals=goals, segment_ids=np.repeat(np.arange(2, dtype=np.int32), 3),
        native_rewards=np.where(reached, 0., -1.).astype(np.float32), goal_reached=reached,
        timeouts=np.zeros(t, dtype=bool), collection_cutoff=np.array([False]*5 + [True]),
        infos=infos, segments=segments, configuration=config, metadata=metadata)


def _rehash(trace):
    trace.metadata['configuration_sha256'] = hashlib.sha256(canonical(trace.configuration)).hexdigest()


@pytest.mark.parametrize('task', ['grid-3x3', 'pancake-4', 'topspin-20-4'])
def test_roundtrip_multiple_achievements_do_not_end_play(task, tmp_path):
    trace = _fixture(task)
    path = trace.save(tmp_path / 'play.npz')
    restored = load_play_trace(path)
    assert restored.num_steps == 6
    assert restored.segments == trace.segments
    assert restored.goal_reached[:3].any() and restored.goal_reached[3:].any()
    assert not restored.timeouts.any() and not restored.collection_cutoff[:-1].any()
    for key, value in trace.arrays().items():
        np.testing.assert_array_equal(value, restored.arrays()[key])
    assert 'goal' not in restored.observations and 'success_streak' not in restored.observations


@pytest.mark.parametrize('field', ['actions', 'behavior_goals', 'segment_ids', 'native_rewards', 'goal_reached', 'timeouts', 'collection_cutoff'])
def test_wrong_dtype_rejected(field):
    trace = _fixture()
    setattr(trace, field, getattr(trace, field).astype(np.float64))
    with pytest.raises(ValueError, match='dtype'):
        trace.validate()


@pytest.mark.parametrize('mutation', [
    lambda t: t.actions.__setitem__((0, 0), np.nan),
    lambda t: t.actions.__setitem__((0, 0), 1.01),
    lambda t: t.observations['board'].__setitem__((0, 0), 2),
    lambda t: t.observations.__setitem__('goal', t.observations['board'].copy()),
    lambda t: t.observations['steps_remaining'].__setitem__((3, 0), 20),
    lambda t: t.infos[3].__setitem__('steps', 0),
    lambda t: t.infos[3].__setitem__('elapsed_time', 0.),
    lambda t: t.collection_cutoff.__setitem__(0, True),
    lambda t: t.goal_reached.__setitem__(0, False),
    lambda t: t.native_rewards.__setitem__(0, -1),
    lambda t: t.segment_ids.__setitem__(3, 0),
    lambda t: t.segments[1].__setitem__('start_step', 2),
    lambda t: t.segments[0].__setitem__('goal_first_step', 2),
    lambda t: t.segments[0].__setitem__('outcome', 'controller_failure'),
    lambda t: t.segments[0].__setitem__('switch_safe', False),
    lambda t: t.segments[1].__setitem__('end_step', 5),
    lambda t: t.metadata.__setitem__('seed', True),
    lambda t: t.metadata.__setitem__('reset_count', 2),
    lambda t: t.metadata.__setitem__('reset_count', True),
    lambda t: t.segments[0].__setitem__('controller_completed', False),
    lambda t: t.segments[0].__setitem__('requested_operation', 8),
    lambda t: t.metadata.__setitem__('configuration_sha256', '0'*64),
    lambda t: t.metadata.__setitem__('source_sha256', '0'*64),
    lambda t: t.metadata.__setitem__('untrusted', float('inf')),
    lambda t: t.infos[1]['event_log'][0].__setitem__('substep', 25),
    lambda t: t.infos[1]['event_log'][0].__setitem__('button', 2),
])
def test_inconsistent_trace_rejected(mutation):
    trace = _fixture()
    mutation(trace)
    with pytest.raises(ValueError):
        trace.validate()


def test_goal_achievement_cannot_substitute_for_safe_release():
    trace = _fixture()
    trace.observations['end_effector_position'][3, 2] = 0.
    trace.segments[0]['switch_safe'] = False
    with pytest.raises(ValueError, match='safe release'):
        trace.validate()


def test_pancake_stale_board_and_loaded_tool_cannot_claim_goal():
    for altered in ('floating', 'loaded'):
        trace = _fixture('pancake-4')
        if altered == 'floating':
            trace.observations['token_positions'][2, :, 2] += .1
        else:
            trace.observations['tool_contact_forces'][2, 0] = .1
        with pytest.raises(ValueError, match='physical goal evidence'):
            trace.validate()


def test_topspin_history_recomputed_and_raw_geometry_required():
    trace = _fixture('topspin-20-4')
    assert trace.goal_reached.tolist() == [False, False, True, False, False, True]
    trace.goal_reached[0] = True
    with pytest.raises(ValueError, match='physical goal evidence'):
        trace.validate()
    trace = _fixture('topspin-20-4')
    trace.observations['token_positions'][2, 0, 0] += .04
    with pytest.raises(ValueError, match='physical goal evidence'):
        trace.validate()


def test_topspin_unknown_event_cannot_disguise_known_atomic_operation():
    trace = _fixture('topspin-20-4')
    trace.infos[1]['event_log'][0]['operation'] = 'unclassified_measured_rearrangement'
    trace.infos[1]['event_log'][0]['atomic_cost'] = None
    with pytest.raises(ValueError, match='operation/cost'):
        trace.validate()


def test_horizon_timeout_takes_priority_even_when_goal_reached():
    trace = _fixture()
    trace.configuration['max_episode_steps'] = trace.metadata['global_horizon'] = 6
    trace.observations['steps_remaining'][:, 0] = 6 - np.arange(7)
    trace.timeouts[-1], trace.collection_cutoff[-1] = True, False
    trace.segments[-1]['outcome'] = trace.metadata['stop_reason'] = 'timeout'
    _rehash(trace)
    trace.validate()
    assert trace.goal_reached[-1] and trace.timeouts[-1]
    trace.timeouts[-1], trace.collection_cutoff[-1] = False, True
    with pytest.raises(ValueError, match='global horizon'):
        trace.validate()


def test_failure_and_native_error_prefix_are_retained():
    trace = _fixture()
    trace.segments[-1]['outcome'] = trace.metadata['stop_reason'] = 'controller_failure'
    trace.validate()
    trace.metadata.update(stop_reason='native_error', error={'stage': 'native_step', 'type': 'RuntimeError', 'message': 'test'},
                          failed_action=[0., 0., 0.])
    trace.segments[-1]['outcome'] = 'collection_cutoff'
    trace.validate()
    trace.metadata.pop('failed_action')
    with pytest.raises(ValueError, match='attempted action'):
        trace.validate()


def test_error_before_next_segment_does_not_require_empty_fake_segment():
    trace = _fixture()
    trace.metadata.update(requested_segments=3, stop_reason='controller_error',
                          error={'stage': 'policy_factory', 'type': 'RuntimeError', 'message': 'test'})
    trace.validate()


def test_no_overwrite_or_partial_publish(tmp_path, monkeypatch):
    trace = _fixture()
    path = trace.save(tmp_path / 'existing.npz')
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        trace.save(path)
    assert path.read_bytes() == before
    real_link, calls = os.link, []
    def fail_sidecar(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError('simulated publish failure')
        real_link(source, destination)
    monkeypatch.setattr(os, 'link', fail_sidecar)
    with pytest.raises(OSError):
        trace.save(tmp_path / 'partial.npz')
    assert not (tmp_path / 'partial.npz').exists()
    assert not (tmp_path / 'partial.json').exists()


def test_hash_descriptor_duplicate_json_and_array_identity(tmp_path):
    for mode in ('hash', 'descriptor', 'duplicate', 'extra_array', 'object'):
        path = _fixture().save(tmp_path / f'{mode}.npz')
        sidecar = path.with_suffix('.json')
        header = json.loads(sidecar.read_text())
        if mode == 'hash':
            path.write_bytes(path.read_bytes() + b'corrupt')
        elif mode == 'descriptor':
            header['arrays']['actions']['shape'] = [6, 7]
        elif mode == 'duplicate':
            sidecar.write_text(sidecar.read_text().replace('"schema_version":', '"task_id": "grid-3x3", "schema_version":'))
        else:
            with np.load(path, allow_pickle=False) as archive:
                values = {key: archive[key] for key in archive.files}
            if mode == 'extra_array':
                values['forged'] = np.array([0])
            else:
                values['actions'] = np.empty((6, 3), dtype=object)
            np.savez_compressed(path, **values)
            header['artifact']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        if mode != 'duplicate':
            sidecar.write_text(json.dumps(header))
        with pytest.raises(ValueError):
            load_play_trace(path)


def test_offline_load_never_imports_simulator_or_native_controller(tmp_path):
    path = _fixture('topspin-20-4').save(tmp_path / 'offline.npz')
    code = '''
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('mujoco') or fullname.endswith(('.physics.env', '.physics.scripted')):
            raise AssertionError('Offline loading imported ' + fullname)
sys.meta_path.insert(0, Block())
from copuzzle.manipulation.play.protocol import load_play_trace
assert load_play_trace(sys.argv[1]).num_steps == 6
'''
    subprocess.run([sys.executable, '-c', code, str(path)], check=True, env=os.environ.copy())
