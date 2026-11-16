"""Small strict contact traces for the 20/4 tabletop prototype, not training data.

NPZ stores continuous actions and every observation, including in-flight states.
A checked JSON sidecar retains full measured info, configuration and provenance.
Loading uses NumPy only; replay requires an explicitly supplied live environment.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

SCHEMA='copuzzle.topspin20_contact_trace.v1'
ENGAGED_SCHEMA='copuzzle.topspin20_contact_trace.v2'
OBSERVATIONS={
    'board':('int32',(20,)), 'goal':('int32',(20,)), 'board_valid':('uint8',(1,)),
    'end_effector_position':('float64',(3,)), 'joint_positions':('float64',(6,)),
    'joint_velocities':('float64',(6,)), 'token_positions':('float64',(20,3)),
    'token_velocities':('float64',(20,4)), 'turntable_angle':('float64',(1,)),
    'turntable_velocity':('float64',(1,)), 'handle_position':('float64',(3,)),
    'turntable_center':('float64',(3,)), 'slot_positions':('float64',(20,3)),
    'steps_remaining':('int32',(1,)),
}


ENGAGED_OBSERVATIONS={**OBSERVATIONS,
    'tool_token_contact_mask':('uint8',(20,)),
    'tool_token_contact_normal_force':('float64',(20,)),
    'tool_token_contact_tangent_speed':('float64',(20,)),
    'tool_token_relative_velocity':('float64',(20,3)),
    'tool_linear_velocity':('float64',(3,)), 'socket_radius':('float64',(1,)),
    'pin_radius':('float64',(1,)), 'socket_bottom_offset':('float64',(1,)),
    'tool_clear':('uint8',(1,)), 'atomic_operation_count':('int32',(1,)),
    'success_streak':('int32',(1,)),
}


def observation_spec(configuration):
    if not isinstance(configuration,dict):raise ValueError('Unsupported environment configuration')
    version=configuration.get('schema_version')
    if version=='copuzzle.topspin20_contact_config.v1':return OBSERVATIONS
    if version=='copuzzle.topspin20_contact_config.v2':return ENGAGED_OBSERVATIONS
    raise ValueError('Unsupported environment configuration')


def trace_schema(configuration):
    return ENGAGED_SCHEMA if observation_spec(configuration) is ENGAGED_OBSERVATIONS else SCHEMA

# Exact historical source identity shipped with topspin-contact-v1. This is a
# narrow, explicit migration path: old configuration/model must still match,
# the caller must request legacy_v1, and every observation/action is replayed.
LEGACY_V1_SOURCE_HASHES={
  "games/lights_out/physics/control.py": "d09bf909fc409f1ee121bdcff4c3800e040b11943d488c97405f7f8f733189fd",
  "games/topspin/physics/__init__.py": "32c8703368d87ec536dd084a141da66d42c2aa77a1602c5270978d270d199511",
  "games/topspin/physics/env.py": "aab4de608f83ce69bc718172ba1a285d7cc1df05857e402abaa51f8b1ef6df49",
  "games/topspin/physics/scene.py": "3f460f8c211040fba241127e124edc1b7def0c5b2ce1595e8d377101dac4a232",
  "games/topspin/physics/scripted.py": "0c5cc85d6556e6c7a2bcad90c69ae06175c658331691c073ccae85b004a977cc",
  "games/topspin/physics/trace.py": "5d30e1922a51de9c04a765440fec0b55ecd681fb78dd53ac492bc0a57ec7b1a9"
}

def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json(v) for v in value]
    return value


def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def _hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    package=Path(__file__).resolve().parents[3]
    sources=list(Path(__file__).parent.glob('*.py'))+[package/'games/lights_out/physics/control.py']
    return {str(p.relative_to(package)):_hash(p) for p in sorted(sources)}


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate manifest key')
        result[key]=value
    return result


@dataclass
class ContactTrace:
    observations:dict
    actions:np.ndarray
    rewards:np.ndarray
    terminated:np.ndarray
    truncated:np.ndarray
    infos:list
    configuration:dict
    metadata:dict

    @property
    def num_steps(self):return len(self.actions)

    def validate(self):
        spec=observation_spec(self.configuration)
        engaged=spec is ENGAGED_OBSERVATIONS
        steps=self.num_steps
        if steps<1:raise ValueError('Trace needs at least one transition')
        for name,shape,dtype in [('actions',(steps,3),'float32'),('rewards',(steps,),'float32'),
                                  ('terminated',(steps,),'bool'),('truncated',(steps,),'bool')]:
            value=getattr(self,name)
            if not isinstance(value,np.ndarray) or value.shape!=shape or str(value.dtype)!=dtype:
                raise ValueError(f'Invalid {name} shape/dtype')
            if not np.isfinite(value).all():raise ValueError('Nonfinite trace value')
        if np.any(np.abs(self.actions)>1):raise ValueError('Actions outside [-1,1]')
        if np.any(self.terminated & self.truncated) or np.any((self.terminated|self.truncated)[:-1]):
            raise ValueError('Episode ending before final action or conflicting endings')
        if not np.array_equal(self.rewards,np.where(self.terminated,0.,-1.).astype(np.float32)):
            raise ValueError('Rewards disagree with environment endings')
        if not isinstance(self.observations,dict) or set(self.observations)!=set(spec):
            raise ValueError('Invalid observation fields')
        for name,(dtype,shape) in spec.items():
            value=self.observations[name]
            if not isinstance(value,np.ndarray) or value.shape!=(steps+1,)+shape or str(value.dtype)!=dtype:
                raise ValueError(f'Invalid observation {name} shape/dtype')
            if not np.isfinite(value).all():raise ValueError('Nonfinite observation')
        for name in ('board','goal'):
            if not np.all(np.sort(self.observations[name],axis=-1)==np.arange(1,21)):
                raise ValueError('Boards/goals must be permutations of 1..20')
        if np.any(self.observations['board_valid']>1):raise ValueError('board_valid must be binary')
        if engaged:
            for name in ('tool_token_contact_mask','tool_clear'):
                if np.any(self.observations[name]>1):raise ValueError('Contact/clear flags must be binary')
            for name in ('tool_token_contact_normal_force','tool_token_contact_tangent_speed'):
                if np.any(self.observations[name]<0):raise ValueError('Contact measurements must be nonnegative')
            if not np.array_equal(self.observations['tool_token_contact_mask'],
                                  (self.observations['tool_token_contact_normal_force']>1e-6).astype(np.uint8)):
                raise ValueError('Contact identity mask and positive forces disagree')
            if np.any((self.observations['success_streak']<0)|(self.observations['success_streak']>3)):
                raise ValueError('Invalid success streak')
            if np.any(self.terminated & (self.observations['tool_clear'][1:,0]==0)):
                raise ValueError('A coupled tool cannot terminate the task')
        if set(self.metadata)!={'reset_seed','reset_options','configuration_sha256','sources','collector_stop','user'}:
            raise ValueError('Invalid provenance fields')
        if hashlib.sha256(_canonical(self.configuration)).hexdigest()!=self.metadata['configuration_sha256']:
            raise ValueError('Configuration checksum mismatch')
        if type(self.metadata['reset_seed']) is not int or self.metadata['reset_seed']<0:
            raise ValueError('Invalid reset seed')
        if self.metadata['collector_stop'] not in ('policy_completed','policy_failed','max_steps','terminated','truncated'):
            raise ValueError('Invalid collector stop')
        expected_stop='terminated' if self.terminated[-1] else ('truncated' if self.truncated[-1] else None)
        if expected_stop is not None and self.metadata['collector_stop']!=expected_stop:
            raise ValueError('Collector and environment endings disagree')
        if not isinstance(self.metadata['sources'],dict) or not self.metadata['sources']:
            raise ValueError('Missing source hashes')
        if any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
               for v in self.metadata['sources'].values()):raise ValueError('Invalid source checksum')
        if not isinstance(self.infos,list) or len(self.infos)!=steps+1:raise ValueError('Info must retain T+1 records')
        fields={'is_success','steps','elapsed_time','event_log','tool_handle_contact','tool_token_contact_ids','max_token_speed'}
        if engaged:fields|={'atomic_operation_count','tool_clear'}
        count=0
        for index,info in enumerate(self.infos):
            if not isinstance(info,dict) or set(info)!=fields:raise ValueError('Invalid info fields')
            if type(info['is_success']) is not bool or type(info['tool_handle_contact']) is not bool:
                raise ValueError('Info flags must be booleans')
            if (not isinstance(info['tool_token_contact_ids'],list)
                    or any(type(v) is not int or not 1<=v<=20 for v in info['tool_token_contact_ids'])
                    or info['tool_token_contact_ids']!=sorted(set(info['tool_token_contact_ids']))):
                raise ValueError('Invalid tool/token contact IDs')
            if engaged:
                if info['tool_token_contact_ids']!=(np.flatnonzero(self.observations['tool_token_contact_mask'][index])+1).tolist():
                    raise ValueError('Contact identities differ from the observation')
                if type(info['tool_clear']) is not bool or info['tool_clear']!=bool(self.observations['tool_clear'][index,0]):
                    raise ValueError('Tool clearance info mismatch')
            if type(info['steps']) is not int or info['steps']!=index:raise ValueError('Info step mismatch')
            if not np.isclose(info['elapsed_time'],index*self.configuration['control_timestep'],rtol=0,atol=1e-8):
                raise ValueError('Elapsed time mismatch')
            if info['is_success'] != (bool(self.terminated[index-1]) if index else False):
                raise ValueError('Success info and termination disagree')
            if not isinstance(info['event_log'],list) or len(info['event_log'])>1:
                raise ValueError('Invalid measured event list')
            if index:
                before,after=self.observations['board'][index-1:index+1]
                changed=not np.array_equal(before,after)
                if bool(info['event_log'])!=changed:raise ValueError('Board changes and measured events disagree')
                if changed:
                    event=info['event_log'][0]
                    if set(event)!=({'operation','state_before','state_after','atomic_cost','operation_index'} if engaged else {'operation','state_before','state_after'}) or event['state_before']!=before.tolist() or event['state_after']!=after.tolist():
                        raise ValueError('Event state payload mismatch')
                    possibilities={'reverse_four':np.r_[before[:4][::-1],before[4:]],
                                   'shift_left':np.roll(before,-1),'shift_right':np.roll(before,1)}
                    if engaged:
                        classification=next((name for name,result in possibilities.items() if np.array_equal(after,result)),
                                            'unclassified_measured_rearrangement')
                        if event['operation']!=classification:raise ValueError('Measured operation classification is inconsistent')
                    if event['operation'] in possibilities and not np.array_equal(after,possibilities[event['operation']]):
                        raise ValueError('Measured event operation mismatch')
                    if event['operation'] not in {*possibilities,'unclassified_measured_rearrangement'}:
                        raise ValueError('Unknown measured operation')
                    if engaged:
                        cost=1 if event['operation'] in possibilities else None
                        count+=cost or 0
                        if event['atomic_cost']!=cost or event['operation_index']!=(count if cost else None):
                            raise ValueError('Atomic operation cost/index mismatch')
                        if not self.observations['tool_clear'][index,0]:
                            raise ValueError('Indexed completion must follow tool withdrawal')
            if engaged and (type(info['atomic_operation_count']) is not int or info['atomic_operation_count']!=count
                            or self.observations['atomic_operation_count'][index,0]!=count):
                raise ValueError('Atomic operation counter mismatch')

        if engaged:
            observed_clear=((self.observations['end_effector_position'][:,2]>.15)
                            & ~self.observations['tool_token_contact_mask'].any(axis=1)
                            & ~np.array([info['tool_handle_contact'] for info in self.infos],dtype=bool))
            if not np.array_equal(self.observations['tool_clear'][:,0].astype(bool),observed_clear):
                raise ValueError('Tool clearance disagrees with measured height/contact')
            valid=self.observations['board_valid'][:,0].astype(bool)
            if np.any(valid & ~observed_clear):raise ValueError('Coupled tool cannot have a completed indexed board')
            if np.any(valid & ((np.max(np.abs(self.observations['token_velocities']),axis=(1,2))>=.015)
                               | (np.abs(self.observations['turntable_velocity'][:,0])>=.05))):
                raise ValueError('Completed board must be physically settled')
            candidate=valid & observed_clear & np.all(self.observations['board']==self.observations['goal'],axis=1)
            streak=0
            for index,eligible in enumerate(candidate):
                streak=streak+1 if index and eligible else 0
                if self.observations['success_streak'][index,0]!=streak:
                    raise ValueError('Success streak must equal consecutive measured released-goal observations')
            success=(valid[1:] & observed_clear[1:]
                     & np.all(self.observations['board'][1:]==self.observations['goal'][1:],axis=1)
                     & (self.observations['success_streak'][1:,0]==3))
            if not np.array_equal(self.terminated,success):raise ValueError('Termination differs from released measured goal/stability')
        _canonical(_json({'configuration':self.configuration,'metadata':self.metadata,'infos':self.infos}))

    def save(self,path):
        self.validate()
        path=Path(path).expanduser().absolute()
        if path.suffix!='.npz':raise ValueError('Expected .npz path')
        sidecar=path.with_suffix('.json')
        if os.path.lexists(path) or os.path.lexists(sidecar):raise FileExistsError('Refusing to overwrite trace')
        path.parent.mkdir(parents=True,exist_ok=True)
        arrays={**{f'obs__{k}':v for k,v in self.observations.items()},
                **{k:getattr(self,k) for k in ('actions','rewards','terminated','truncated')}}
        temporary=[];published=False
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
                archive=Path(stream.name);temporary.append(archive);np.savez_compressed(stream,**arrays)
            header={'schema_version':trace_schema(self.configuration),'archive':{'filename':path.name,'sha256':_hash(archive)},
                    'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()},
                    'configuration':self.configuration,'metadata':self.metadata,'infos':self.infos}
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as stream:
                manifest=Path(stream.name);temporary.append(manifest);json.dump(header,stream,indent=2,sort_keys=True,allow_nan=False)
            os.link(archive,path);published=True;os.link(manifest,sidecar)
        except BaseException:
            if published and path.exists() and os.path.samefile(path,archive):path.unlink()
            raise
        finally:
            for item in temporary:item.unlink(missing_ok=True)
        return path


def load_trace(path):
    path=Path(path)
    header=json.loads(path.with_suffix('.json').read_text(),object_pairs_hook=_unique)
    if set(header)!={'schema_version','archive','arrays','configuration','metadata','infos'} or header['schema_version'] not in (SCHEMA,ENGAGED_SCHEMA):
        raise ValueError('Unsupported contact trace schema')
    spec=observation_spec(header['configuration'])
    if header['schema_version']!=trace_schema(header['configuration']):raise ValueError('Trace/configuration schema mismatch')
    if header['archive']!={'filename':path.name,'sha256':_hash(path)}:raise ValueError('Archive identity/checksum mismatch')
    with np.load(path,allow_pickle=False) as data:
        expected={f'obs__{k}' for k in spec}|{'actions','rewards','terminated','truncated'}
        if len(data.files)!=len(expected) or set(data.files)!=expected:raise ValueError('Archive fields mismatch')
        arrays={k:data[k] for k in data.files}
    if header['arrays']!={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()}:
        raise ValueError('Array metadata mismatch')
    trace=ContactTrace(observations={k:arrays.pop('obs__'+k) for k in spec},
                       **arrays,infos=header['infos'],configuration=header['configuration'],metadata=header['metadata'])
    trace.validate();return trace


def record_trace(env,policy,*,seed=0,options=None,max_steps=None,metadata=None):
    if type(seed) is not int or seed<0:raise ValueError('seed must be a nonnegative integer')
    limit=env.max_episode_steps if max_steps is None else max_steps
    if type(limit) is not int or limit<1:raise ValueError('max_steps must be positive')
    options={} if options is None else deepcopy(options)
    sources=source_hashes()
    observation,info=env.reset(seed=seed,options=options)
    if callable(getattr(policy,'reset',None)):policy.reset()
    history={k:[v.copy()] for k,v in observation.items()};infos=[_json(info)]
    actions=[];rewards=[];terminated=[];truncated=[];stop='max_steps'
    for _ in range(limit):
        if getattr(policy,'completed',False):stop='policy_completed';break
        if getattr(policy,'finished',False):stop='policy_failed';break
        action=np.asarray(policy(deepcopy(observation)),dtype=np.float32)
        if getattr(policy,'completed',False):stop='policy_completed';break
        if getattr(policy,'finished',False):stop='policy_failed';break
        observation,reward,term,trunc,info=env.step(action)
        actions.append(action.copy());rewards.append(reward);terminated.append(term);truncated.append(trunc);infos.append(_json(info))
        for key,value in observation.items():history[key].append(value.copy())
        if term or trunc:stop='terminated' if term else 'truncated';break
    if sources!=source_hashes():raise RuntimeError('Source changed during recording')
    trace=ContactTrace({k:np.stack(v) for k,v in history.items()},np.asarray(actions,dtype=np.float32),
                       np.asarray(rewards,np.float32),np.asarray(terminated,bool),np.asarray(truncated,bool),
                       infos,env.configuration(),{'reset_seed':seed,'reset_options':_json(options),
                        'configuration_sha256':env.configuration_sha256,'sources':sources,
                        'collector_stop':stop,'user':_json(metadata or {})})
    trace.validate();return trace


def replay_trace(env,trace,*,atol=1e-8):
    trace.validate()
    spec=observation_spec(trace.configuration)
    current_source=source_hashes()
    migrated=(getattr(env,'facility_profile',None)=='legacy_v1'
              and trace.metadata['sources']==LEGACY_V1_SOURCE_HASHES)
    if env.configuration_sha256!=trace.metadata['configuration_sha256'] or (current_source!=trace.metadata['sources'] and not migrated):
        raise ValueError('Replay requires matching environment configuration and source, or explicit verified legacy_v1 migration')
    if isinstance(atol,bool) or not np.isscalar(atol) or not np.isfinite(atol) or atol<0:
        raise ValueError('atol must be finite and nonnegative')
    observation,info=env.reset(seed=trace.metadata['reset_seed'],options=trace.metadata['reset_options'])
    maximum=0.
    def compare(observation,index,info):
        nonlocal maximum
        if set(observation)!=set(spec):raise ValueError('Replay observation fields differ')
        for key,value in observation.items():
            target=trace.observations[key][index]
            if value.shape!=target.shape or value.dtype!=target.dtype:raise ValueError('Replay shape/dtype mismatch')
            error=float(np.max(np.abs(value.astype(float)-target.astype(float))));maximum=max(maximum,error)
            if error>(atol if value.dtype.kind=='f' else 0):raise ValueError(f'Replay mismatch {index}:{key}')
        expected=trace.infos[index]
        for key in expected:
            if key in ('elapsed_time','max_token_speed'):
                if not np.isclose(info[key],expected[key],atol=atol,rtol=0):raise ValueError('Replay info mismatch')
            elif _json(info[key])!=expected[key]:raise ValueError('Replay info mismatch')
    compare(observation,0,info)
    for index,action in enumerate(trace.actions):
        observation,reward,term,trunc,info=env.step(action.copy());compare(observation,index+1,info)
        if reward!=trace.rewards[index] or term!=trace.terminated[index] or trunc!=trace.truncated[index]:
            raise ValueError('Replay reward/ending mismatch')
    return {'control_steps':trace.num_steps,'max_absolute_observation_error':maximum,
            'measured_operations':[e['operation'] for info in trace.infos for e in info['event_log']],
            'configuration_sha256':env.configuration_sha256,'atol':float(atol),
            'source_compatibility':'verified_legacy_v1_migration' if migrated else 'exact_source_match'}


def engagement_report(trace):
    """Audit declared controller selections against the recorded force/velocity.

    Controller phases and selected IDs are provenance, not simulator truth.
    Contact identity, force, offset, slip and indexed completion are independently
    measured observations. This report is specific to the engaged profile.
    """
    trace.validate()
    if observation_spec(trace.configuration) is not ENGAGED_OBSERVATIONS:
        raise ValueError('Engagement reports require v2 contact observations')
    log=trace.metadata['user'].get('controller_steps')
    if not isinstance(log,list) or len(log)!=trace.num_steps:
        raise ValueError('controller_steps must align with every recorded action')
    groups={}
    for step,row in enumerate(log):
        if not isinstance(row,dict) or set(row)!={'phase','selected_token_id','operation_index','reacquisitions'}:
            raise ValueError('Invalid controller audit row')
        token=row['selected_token_id']
        if token is not None and (type(token) is not int or not 1<=token<=20):
            raise ValueError('Invalid selected token ID')
        if row['phase']=='push' and token is not None:
            key=(int(row['operation_index']),token)
            groups.setdefault(key,[]).append(step)
    observations=trace.observations
    reports=[]
    for (operation_index,token),steps in sorted(groups.items()):
        token_index=token-1
        indices=np.asarray(steps)+1
        masks=observations['tool_token_contact_mask'][indices]
        other=masks.copy();other[:,token_index]=0
        force=observations['tool_token_contact_normal_force'][indices,token_index]
        loaded=force>.05
        offsets=observations['end_effector_position'][indices,:2]-observations['token_positions'][indices,token_index,:2]
        radial=np.linalg.norm(offsets,axis=1)
        speeds=observations['tool_token_contact_tangent_speed'][indices,token_index]
        relative=observations['tool_token_relative_velocity'][indices,token_index,:2]
        loaded_offsets=offsets[loaded]
        loaded_speeds=speeds[loaded]
        run=maximum_run=0
        acquired=False
        for present in loaded:
            acquired=acquired or bool(present)
            run=0 if present else run+1 if acquired else 0
            maximum_run=max(maximum_run,run)
        reports.append({
            'operation_index':operation_index,'selected_token_id':token,'push_control_steps':len(steps),
            'wrong_token_contact_steps':int(np.any(other,axis=1).sum()),
            'selected_contact_steps':int(masks[:,token_index].sum()),
            'loaded_steps_above_0_05N':int(loaded.sum()),
            'maximum_missing_load_run_after_acquisition':maximum_run,
            'maximum_pin_socket_center_offset_m':float(radial.max()),
            'loaded_offset_xy_peak_to_peak_m':np.ptp(loaded_offsets,axis=0).tolist() if len(loaded_offsets) else None,
            'loaded_contact_tangent_speed_m_s':{
                'median':float(np.median(loaded_speeds)) if len(loaded_speeds) else None,
                'p95':float(np.quantile(loaded_speeds,.95)) if len(loaded_speeds) else None,
                'max':float(loaded_speeds.max()) if len(loaded_speeds) else None},
            'maximum_normal_force_N':float(force.max()),
            'maximum_relative_xy_speed_m_s':float(np.linalg.norm(relative,axis=1).max()),
            'reacquisitions':max(log[step]['reacquisitions'] for step in steps),
        })
    events=[event for info in trace.infos for event in info['event_log']]
    return {'schema_version':'copuzzle.topspin20_engagement_audit.v1',
            'configuration_sha256':trace.metadata['configuration_sha256'],
            'atomic_operation_count':int(observations['atomic_operation_count'][-1,0]),
            'measured_operations':[event['operation'] for event in events],
            'event_atomic_costs':[event['atomic_cost'] for event in events],
            'all_events_follow_tool_withdrawal':all(info['tool_clear'] for info in trace.infos if info['event_log']),
            'wrong_token_contact_steps':sum(row['wrong_token_contact_steps'] for row in reports),
            'transport_segments':reports}
