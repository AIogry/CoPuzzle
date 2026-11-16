"""Positive pin engagement, contact provenance, safe recovery and trace semantics."""
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip('mujoco')

from copuzzle.games.topspin.physics.env import TopSpinPhysicalEnv
from copuzzle.games.topspin.physics.scripted import AuditedPolicy, SequencePolicy, TrackPushPolicy
from copuzzle.games.topspin.physics.trace import engagement_report, load_trace, record_trace, replay_trace


@pytest.fixture(scope='module')
def samples():
    result={}
    cases=[('right',17,1,0.,False),('left_noisy',23,-1,.01,False),('recovered',37,1,0.,True)]
    for name,seed,direction,noise,disturb in cases:
        state=np.random.default_rng(seed).permutation(np.arange(1,21,dtype=np.int32))
        env=TopSpinPhysicalEnv(max_episode_steps=1600)
        driver=TrackPushPolicy(direction=direction)
        policy=AuditedPolicy(driver,noise_amplitude=noise,seed=seed+90,lift_disturbance=disturb)
        try:
            trace=record_trace(env,policy,seed=seed,options={'state':state,'goal':np.roll(state,direction)})
            trace.metadata['user']['controller_steps']=policy.controller_steps[:trace.num_steps]
            trace.metadata['user']['controller_failure']=policy.failure
            assert trace.terminated[-1] and not trace.truncated[-1],name
            result[name]=(trace,driver.audit)
        finally:env.close()
    return result


@pytest.mark.parametrize('name',['right','left_noisy','recovered'])
def test_single_selected_token_remains_engaged_until_verified_release(samples,name):
    trace,audit=samples[name]
    report=engagement_report(trace)
    assert report['wrong_token_contact_steps']==0
    assert report['atomic_operation_count']==1 and report['event_atomic_costs']==[1]
    assert report['all_events_follow_tool_withdrawal']
    assert len(report['transport_segments'])==1
    segment=report['transport_segments'][0]
    clearance=float(trace.observations['socket_radius'][0,0]-trace.observations['pin_radius'][0,0])
    assert segment['maximum_pin_socket_center_offset_m']<clearance+.0003
    assert segment['maximum_normal_force_N']>.1
    assert segment['loaded_steps_above_0_05N']>.75*segment['push_control_steps']
    assert segment['loaded_contact_tangent_speed_m_s']['p95']<.01
    assert trace.observations['tool_clear'][-1,0] and trace.observations['end_effector_position'][-1,2]>.15
    np.testing.assert_array_equal(trace.observations['board'][-1],trace.observations['goal'][-1])
    if name=='recovered':
        assert audit and audit[0]['reason']=='pin exited the socket during transport'
        assert all(row['selected_token_id']==segment['selected_token_id'] for row in audit)
        assert segment['reacquisitions']==1
    else:assert not audit


def test_exhausted_sequence_retries_withdraw_before_signalling_failure():
    env=TopSpinPhysicalEnv(max_episode_steps=700)
    policy=AuditedPolicy(SequencePolicy(['shift_right'],max_reacquisitions=0),lift_disturbance=True)
    state=np.arange(1,21,dtype=np.int32)
    try:
        trace=record_trace(env,policy,options={'state':state,'goal':np.roll(state,1)})
        assert trace.metadata['collector_stop']=='policy_failed'
        assert not trace.terminated[-1] and not trace.truncated[-1]
        assert policy.failure=='pin exited the socket during transport'
        assert trace.observations['end_effector_position'][-1,2]>.19
        assert trace.observations['tool_clear'][-1,0]
        assert trace.observations['atomic_operation_count'][-1,0]==0
        assert not trace.observations['tool_token_contact_mask'][-1].any()
        np.testing.assert_array_equal(trace.observations['board'][-1],state)
    finally:env.close()


def test_no_collision_cannot_produce_a_hidden_indexed_shift():
    env=TopSpinPhysicalEnv(max_episode_steps=650)
    driver=TrackPushPolicy(max_reacquisitions=0)
    try:
        observation,_=env.reset()
        original=observation['board'].copy()
        env.model.geom_contype[env.scene.tool_geom_ids]=0
        env.model.geom_conaffinity[env.scene.tool_geom_ids]=0
        for _ in range(650):
            observation,_,term,trunc,info=env.step(driver(observation))
            assert not term and not info['event_log']
            if driver.finished or trunc:break
        assert driver.failure and not driver.completed
        assert observation['end_effector_position'][2]>.19
        assert observation['atomic_operation_count'][0]==0
        np.testing.assert_array_equal(observation['board'],original)
    finally:env.close()


def test_new_trace_roundtrip_and_observable_contact_corruption(samples,tmp_path):
    trace,_=samples['right']
    path=trace.save(tmp_path/'engaged.npz');restored=load_trace(path)
    assert restored.configuration['schema_version']=='copuzzle.topspin20_contact_config.v2'
    env=TopSpinPhysicalEnv(max_episode_steps=1600)
    try:assert replay_trace(env,restored)['max_absolute_observation_error']==0
    finally:env.close()
    changed=deepcopy(trace)
    index,token=np.argwhere(changed.observations['tool_token_contact_normal_force']>.05)[0]
    changed.observations['tool_token_contact_mask'][index,token]=0
    with pytest.raises(ValueError,match='mask'):changed.validate()
    changed=deepcopy(trace)
    changed.observations['end_effector_position'][-1,2]=.10
    with pytest.raises(ValueError,match='clearance'):changed.validate()


def test_v2_rejects_forged_early_success_streak(samples):
    trace=deepcopy(samples['right'][0])
    candidate=(trace.observations['board_valid'][:,0].astype(bool)
               & np.all(trace.observations['board']==trace.observations['goal'],axis=1))
    first=int(np.flatnonzero(candidate)[0])
    assert first<trace.num_steps
    for key in trace.observations:trace.observations[key]=trace.observations[key][:first+1].copy()
    for key in ('actions','rewards','terminated','truncated'):setattr(trace,key,getattr(trace,key)[:first].copy())
    trace.infos=trace.infos[:first+1]
    trace.terminated[-1]=True;trace.rewards[-1]=0.;trace.infos[-1]['is_success']=True
    trace.observations['success_streak'][-1,0]=3
    trace.metadata['collector_stop']='terminated'
    with pytest.raises(ValueError,match='streak'):trace.validate()


def test_v2_cannot_hide_a_legal_atomic_operation_as_unclassified(samples):
    trace=deepcopy(samples['right'][0])
    for info in trace.infos:
        info['atomic_operation_count']=0
        for event in info['event_log']:
            event['operation']='unclassified_measured_rearrangement'
            event['atomic_cost']=None;event['operation_index']=None
    trace.observations['atomic_operation_count'][:]=0
    with pytest.raises(ValueError,match='classification'):trace.validate()


def test_legacy_profile_remains_explicit_and_configuration_separated():
    legacy=TopSpinPhysicalEnv(facility_profile='legacy_v1',max_episode_steps=10)
    try:trace=record_trace(legacy,lambda o:np.zeros(3,np.float32),max_steps=3)
    finally:legacy.close()
    assert 'tool_token_contact_mask' not in trace.observations
    current=TopSpinPhysicalEnv(max_episode_steps=10)
    try:
        with pytest.raises(ValueError,match='configuration'):replay_trace(current,trace)
    finally:current.close()
    legacy=TopSpinPhysicalEnv(facility_profile='legacy_v1',max_episode_steps=10)
    try:assert replay_trace(legacy,trace)['max_absolute_observation_error']==0
    finally:legacy.close()
