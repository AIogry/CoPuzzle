"""Real robot-contact operations, composition, causality and portable traces."""
from copy import deepcopy
import json

import numpy as np
import pytest
from gymnasium.error import ResetNeeded

mujoco=pytest.importorskip('mujoco')

from copuzzle.games.topspin.physics.env import TopSpinPhysicalEnv
from copuzzle.games.topspin.physics.scene import ARC_RADIUS, HALF_STRAIGHT, TURN_CENTER, track_coordinates
from copuzzle.games.topspin.physics.scripted import SequencePolicy, TrackPushPolicy, TurntablePolicy
from copuzzle.games.topspin.physics.trace import load_trace, record_trace, replay_trace


@pytest.fixture(scope='module')
def single_traces():
    result={}
    for index,operation in enumerate(('reverse_four','shift_left','shift_right')):
        state=np.random.default_rng(310+index).permutation(np.arange(1,21,dtype=np.int32))
        goal=(np.r_[state[:4][::-1],state[4:]] if operation=='reverse_four'
              else np.roll(state,-1 if operation=='shift_left' else 1))
        policy=(TurntablePolicy() if operation=='reverse_four'
                else TrackPushPolicy(direction=-1 if operation=='shift_left' else 1))
        env=TopSpinPhysicalEnv(max_episode_steps=900)
        try:
            trace=record_trace(env,policy,seed=9,options={'state':state,'goal':goal})
            assert trace.terminated[-1],f'{operation} did not reach its measured goal'
            result[operation]=trace
        finally:env.close()
    return result


@pytest.mark.parametrize('operation',('reverse_four','shift_left','shift_right'))
def test_every_operation_requires_real_contact_and_reaches_arbitrary_goal(single_traces,operation):
    trace=single_traces[operation]
    events=[e for info in trace.infos for e in info['event_log']]
    assert [event['operation'] for event in events]==[operation]
    assert trace.terminated[-1] and not trace.truncated[-1]
    np.testing.assert_array_equal(trace.observations['board'][-1],trace.observations['goal'][-1])
    assert trace.observations['board_valid'][-1,0]
    assert np.any(trace.observations['board_valid']==0)
    assert np.max(np.abs(trace.observations['token_velocities'][-1]))<.015
    if operation=='reverse_four':
        assert any(info['tool_handle_contact'] for info in trace.infos)
        angle=trace.observations['turntable_angle'][:,0]
        assert np.any((angle>.5)&(angle<2.5))
        token_ids=trace.observations['board'][0,:4]-1
        # The selected four really move in the table plane, while all other
        # tokens remain in the nonrotating closed track.
        assert np.max(np.linalg.norm(trace.observations['token_positions'][:,token_ids,:2]
                    -trace.observations['token_positions'][0,token_ids,:2],axis=-1))>.06
        other=np.setdiff1d(np.arange(20),token_ids)
        assert np.max(np.linalg.norm(trace.observations['token_positions'][:,other,:2]
                    -trace.observations['token_positions'][0,other,:2],axis=-1))<.002
    else:
        assert any(info['tool_token_contact_ids'] for info in trace.infos)
        moved=np.linalg.norm(trace.observations['token_positions'][-1,:,:2]
                             -trace.observations['token_positions'][0,:,:2],axis=-1)
        # Free play redistributes: each puck moves forward, but not by one
        # identical nominal pitch. Check signed progress around the actual loop.
        first,_=track_coordinates(trace.observations['token_positions'][0])
        last,_=track_coordinates(trace.observations['token_positions'][-1])
        length=4*HALF_STRAIGHT+2*np.pi*ARC_RADIUS
        progress=((last-first+length/2)%length-length/2)*(1 if operation=='shift_right' else -1)
        assert np.all(progress>.002) and np.all(progress<.06)
        assert np.all(moved>.002), 'A real circulation must transport all twenty tokens'
        for positions in trace.observations['token_positions']:
            _,distance=track_coordinates(positions)
            assert distance.max()<.0035


def test_mixed_operations_and_inverse_turn_without_reset_or_pose_edits(monkeypatch):
    env=TopSpinPhysicalEnv(max_episode_steps=2400)
    state=np.arange(1,21,dtype=np.int32)
    expected=state.copy()
    for operation in ('reverse','shift_right','reverse','shift_left'):
        expected=np.r_[expected[:4][::-1],expected[4:]] if operation=='reverse' else np.roll(expected,1 if operation=='shift_right' else -1)
    holder={};calls=0
    original_reset,original_step=env.reset,mujoco.mj_step
    def audited_reset(*args,**kwargs):
        result=original_reset(*args,**kwargs)
        holder['qpos']=env.data.qpos.copy();holder['qvel']=env.data.qvel.copy()
        return result
    def audited_step(model,data,*args,**kwargs):
        nonlocal calls
        if data is env.data:
            np.testing.assert_array_equal(data.qpos,holder['qpos'])
            np.testing.assert_array_equal(data.qvel,holder['qvel'])
        original_step(model,data,*args,**kwargs)
        if data is env.data:
            holder['qpos']=data.qpos.copy();holder['qvel']=data.qvel.copy();calls+=1
    monkeypatch.setattr(env,'reset',audited_reset);monkeypatch.setattr(mujoco,'mj_step',audited_step)
    try:
        trace=record_trace(env,SequencePolicy(['reverse','shift_right','reverse','shift_left']),
                           options={'state':state,'goal':np.roll(state,7)})
        assert trace.metadata['collector_stop']=='policy_completed'
        assert [e['operation'] for info in trace.infos for e in info['event_log']]==[
            'reverse_four','shift_right','reverse_four','shift_left']
        np.testing.assert_array_equal(trace.observations['board'][-1],expected)
        assert abs(trace.observations['turntable_angle'][-1,0])<.005
        np.testing.assert_array_equal(env.data.qpos,holder['qpos'])
        np.testing.assert_array_equal(env.data.qvel,holder['qvel'])
        assert calls==trace.num_steps*env.substeps
    finally:env.close()


def test_saved_actions_replay_on_fresh_environment(single_traces,tmp_path):
    trace=single_traces['shift_right']
    path=trace.save(tmp_path/'shift.npz')
    restored=load_trace(path)
    env=TopSpinPhysicalEnv(max_episode_steps=900)
    try:
        report=replay_trace(env,restored)
        assert report['max_absolute_observation_error']==0
        assert report['measured_operations']==['shift_right']
        with pytest.raises(ResetNeeded):env.step(np.zeros(3,dtype=np.float32))
    finally:env.close()
    with pytest.raises(FileExistsError):trace.save(path)
    header=json.loads(path.with_suffix('.json').read_text())
    header['arrays']['actions']['dtype']='float64'
    path.with_suffix('.json').write_text(json.dumps(header))
    with pytest.raises(ValueError,match='metadata'):load_trace(path)


def test_trace_rejects_changed_actions_and_forged_events(single_traces):
    trace=deepcopy(single_traces['reverse_four'])
    trace.actions=trace.actions.astype(np.float64)
    with pytest.raises(ValueError,match='dtype'):trace.validate()
    trace=deepcopy(single_traces['reverse_four'])
    info=next(info for info in trace.infos if info['event_log'])
    info['event_log'][0]['operation']='shift_left'
    with pytest.raises(ValueError,match='operation'):trace.validate()


def test_no_contact_means_no_turntable_command_effect():
    env=TopSpinPhysicalEnv(max_episode_steps=280)
    try:
        observation,_=env.reset()
        initial=observation['board'].copy()
        env.model.geom_contype[env.scene.tool_geom_ids]=0
        env.model.geom_conaffinity[env.scene.tool_geom_ids]=0
        policy=TurntablePolicy()
        for _ in range(280):
            observation,_,term,trunc,info=env.step(policy(observation))
            assert not term and not info['event_log'] and not info['tool_handle_contact']
            assert env.observation_space.contains(observation)
            assert abs(observation['turntable_angle'][0])<1e-6
        np.testing.assert_array_equal(observation['board'],initial)
        assert trunc
    finally:env.close()


@pytest.mark.parametrize('options',[{'physics_dt':0},{'control_dt':float('nan')},{'action_scale':False},
                                   {'max_episode_steps':True}])
def test_invalid_configuration_rejected(options):
    with pytest.raises(ValueError):TopSpinPhysicalEnv(**options)


def test_boolean_state_rejected_before_reset_and_visible_crank_collides():
    env=TopSpinPhysicalEnv(max_episode_steps=2)
    try:
        with pytest.raises(ValueError):env.reset(options={'state':[True,*range(2,21)]})
        for name in ('crank_arm','crank_stem','topspin_housing','turntable_plate'):
            geom=env.model.geom(name).id
            assert env.model.geom_contype[geom] and env.model.geom_conaffinity[geom]
    finally:env.close()
