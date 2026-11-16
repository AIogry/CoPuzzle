"""Validate three direct-contact TopSpin operations and a mixed sequence.

MUJOCO_GL=egl python examples/topspin_physical.py --output artifacts/topspin-contact-v1

Produces strict continuous-action traces, independent action replays, native
renders and a four-times-speed GIF. No exact solver or permutation actuator is
used. Reference permutations are computed only for preparing/verifying goals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from copuzzle.games.topspin.physics.env import TopSpinPhysicalEnv
from copuzzle.games.topspin.physics.scripted import AuditedPolicy, SequencePolicy, TrackPushPolicy, TurntablePolicy
from copuzzle.games.topspin.physics.trace import engagement_report, load_trace, record_trace, replay_trace, source_hashes


def _hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)


def _reference(state,operation):
    if operation=='reverse':return np.r_[state[:4][::-1],state[4:]]
    return np.roll(state,1 if operation=='shift_right' else -1)


class NativeCapture:
    def __init__(self,env,driver,folder,state):
        self.env,self.driver,self.folder=env,driver,folder
        self.after_shift=_reference(_reference(state,'reverse'),'shift_right')
        self.renderer=mujoco.Renderer(env.model,height=384,width=512)
        self.reset()

    def reset(self):
        self.driver.reset();self.steps=0;self.frames=[];self.snapshots={}

    @property
    def completed(self):return self.driver.completed

    @property
    def finished(self):return getattr(self.driver,'finished',self.completed)

    @property
    def failure(self):return getattr(self.driver,'failure',None)

    @property
    def controller_steps(self):return self.driver.controller_steps

    def _snapshot(self,name,cameras,observation):
        import imageio.v2 as imageio
        self.snapshots[name]={'control_step':self.steps,'turntable_angle':float(observation['turntable_angle'][0]),
                              'board':observation['board'].tolist()}
        original=self.env.camera
        for camera in cameras:
            self.env.camera=camera
            imageio.imwrite(self.folder/f'{name}-{camera}.png',self.env.render())
        self.env.camera=original

    def __call__(self,observation):
        if self.steps==0:self._snapshot('initial',('puzzle','top','overview'),observation)
        angle=float(observation['turntable_angle'][0])
        if 'rotor-90' not in self.snapshots and angle>=np.pi/2:
            self._snapshot('rotor-90',('puzzle','top'),observation)
        if 'rotor-180' not in self.snapshots and angle>np.pi-.005 and observation['board_valid'][0]:
            self._snapshot('rotor-180',('puzzle','top'),observation)
        if ('circulated' not in self.snapshots and observation['board_valid'][0]
                and np.array_equal(observation['board'],self.after_shift)):
            self._snapshot('circulated',('puzzle','top'),observation)
        if self.steps%8==0:
            self.renderer.update_scene(self.env.data,camera='puzzle');self.frames.append(self.renderer.render().copy())
        self.steps+=1
        return self.driver(observation)

    def save(self):
        from PIL import Image
        if set(self.snapshots)!={'initial','rotor-90','rotor-180','circulated'}:
            raise RuntimeError('Did not capture every required physical stage')
        frames=[Image.fromarray(frame) for frame in self.frames]
        frames[0].save(self.folder/'mixed-operations.gif',save_all=True,append_images=frames[1:],
                       duration=100,loop=0,optimize=False)
        report={'native_render':True,'camera':'puzzle','image_dimensions':[512,384],
                'frame_stride_control_steps':8,'control_timestep':self.env.control_timestep,
                'frame_duration_ms':100,'playback_speed':4,'snapshots':self.snapshots}
        _write(self.folder/'visualization.json',report)
        self.renderer.close()
        return report


CASE_DEFINITIONS={
    'mixed':(['reverse','shift_right','reverse','shift_left'],0.,False,False),
    'reverse':(['reverse'],0.,False,False),
    'shift_left':(['shift_left'],0.,False,False),
    'shift_right':(['shift_right'],0.,False,False),
    'shift3':(['shift_right']*3,0.,False,False),
    'noisy_right':(['shift_right'],.01,False,False),
    'reacquire_right':(['shift_right'],0.,True,False),
    'abort_right':(['shift_right'],0.,True,True),
}


def generate(output,seed=17,*,cases=None,facility_profile='engaged_v2',render=False):
    output=Path(output).expanduser().absolute()
    if output.exists():raise FileExistsError('Refusing to overwrite prototype artifacts')
    if type(seed) is not int or seed<0:raise ValueError('seed must be a nonnegative integer')
    if facility_profile not in ('legacy_v1','engaged_v2'):raise ValueError('Unsupported facility profile')
    cases=tuple(cases or (CASE_DEFINITIONS if facility_profile=='engaged_v2' else ('mixed','reverse','shift_left','shift_right')))
    if len(set(cases))!=len(cases) or any(name not in CASE_DEFINITIONS for name in cases):
        raise ValueError('Cases must be distinct supported names')
    if facility_profile=='legacy_v1' and any(name not in ('mixed','reverse','shift_left','shift_right') for name in cases):
        raise ValueError('Recovery/noise/macro cases require engaged_v2')
    source=source_hashes();script_sha=_hash(__file__)
    output.mkdir(parents=True)
    state=np.random.default_rng(seed).permutation(np.arange(1,21,dtype=np.int32))
    results={};visualization=None
    for name in cases:
        operations,noise,disturb,abort=CASE_DEFINITIONS[name]
        expected=state.copy();visited={tuple(state)}
        for operation in operations:
            expected=_reference(expected,operation);visited.add(tuple(expected))
        goal=expected if name!='mixed' else next(np.roll(state,k) for k in range(1,20) if tuple(np.roll(state,k)) not in visited)
        limit=3200 if name in ('mixed','shift3') else 1600
        env=TopSpinPhysicalEnv(max_episode_steps=limit,facility_profile=facility_profile,
                               render_mode='rgb_array' if render else None)
        driver=SequencePolicy(operations,max_reacquisitions=0 if abort else 3)
        audited=AuditedPolicy(driver,noise_amplitude=noise,seed=seed+200,lift_disturbance=disturb)
        capture=None
        try:
            policy=audited
            if render and name=='mixed':
                capture=NativeCapture(env,audited,output,state);policy=capture
            trace=record_trace(env,policy,seed=seed,options={'state':state,'goal':goal},metadata={
                'purpose':'direct-contact engagement validation; not a learned-policy benchmark',
                'intended_operations':operations,'reference_final_board':(state if abort else expected).tolist(),
                'goal_is_off_path_collection_target':name=='mixed','collector_script_sha256':script_sha,
                'facility_profile':facility_profile,'action_noise_amplitude':noise,'action_noise_seed':seed+200,
                'injected_lift_disturbance':disturb,'expected_outcome':'safe_abort' if abort else 'complete'})
            active=driver.driver
            trace.metadata['user'].update(controller_steps=audited.controller_steps[:trace.num_steps],
                controller_failure=audited.failure,disturbance_applied=audited.disturbance_applied,
                recovery_audit=list(getattr(active,'audit',[])))
            path=trace.save(output/f'{name}.npz')
            print(json.dumps({'case':name,'trace_saved':str(path),'control_steps':trace.num_steps}),flush=True)
            loaded=load_trace(path)
            if capture is not None:visualization=capture.save();capture=None
            elif render:
                import imageio.v2 as imageio
                imageio.imwrite(output/f'{name}-final.png',env.render())
        finally:
            if capture is not None:capture.renderer.close()
            env.close()
        fresh=TopSpinPhysicalEnv(max_episode_steps=limit,facility_profile=facility_profile)
        try:replay=replay_trace(fresh,loaded)
        finally:fresh.close()
        actual=replay['measured_operations']
        intended=[] if abort else ['reverse_four' if operation=='reverse' else operation for operation in operations]
        audit=engagement_report(loaded) if facility_profile=='engaged_v2' else None
        result={**replay,'trace':path.name,'requested_operations':operations,
                'expected_measured_operations':intended,'ordered_operations_correct':actual==intended,
                'reference_final_board':(state if abort else expected).tolist(),
                'actual_final_board':loaded.observations['board'][-1].tolist(),
                'expected_outcome':'safe_abort' if abort else 'complete',
                'terminal_goal_success':bool(loaded.terminated[-1]),'collector_stop':loaded.metadata['collector_stop'],
                'controller_failure':audited.failure,'disturbance_applied':audited.disturbance_applied,
                'simulated_seconds':loaded.infos[-1]['elapsed_time'],
                'handle_contact_control_steps':sum(info['tool_handle_contact'] for info in loaded.infos),
                'token_contact_control_steps':sum(bool(info['tool_token_contact_ids']) for info in loaded.infos),
                'engagement':audit}
        _write(output/f'{name}-replay.json',result)
        if actual!=intended or not np.array_equal(loaded.observations['board'][-1],state if abort else expected):
            raise RuntimeError(f'{name} did not reach its expected measured outcome')
        if audit is not None and audit['wrong_token_contact_steps']:
            raise RuntimeError(f'{name} contacted a token other than its declared transport target')
        if abort:
            if (loaded.metadata['collector_stop']!='policy_failed' or not loaded.observations['tool_clear'][-1,0]
                    or loaded.observations['end_effector_position'][-1,2]<.19):
                raise RuntimeError('Failure did not finish the safe physical withdrawal')
        elif name!='mixed' and not loaded.terminated[-1]:raise RuntimeError(f'{name} did not terminate at its goal')
        if name=='mixed' and loaded.metadata['collector_stop']!='policy_completed':raise RuntimeError('Mixed sequence did not finish')
        results[name]=result
        print(json.dumps({'case':name,'verified':True,'control_steps':trace.num_steps,'operations':actual,
                          'replay_error':replay['max_absolute_observation_error']}),flush=True)
    historical=Path(__file__).resolve().parents[1]/'artifacts/topspin-contact-v1/shift_right.npz'
    legacy_report=None
    if facility_profile=='engaged_v2' and historical.exists():
        old=load_trace(historical)
        legacy=TopSpinPhysicalEnv(facility_profile='legacy_v1',max_episode_steps=old.configuration['max_episode_steps'],
                                 physics_dt=old.configuration['physics_timestep'],control_dt=old.configuration['control_timestep'],
                                 action_scale=old.configuration['action_scale'])
        try:legacy_report=replay_trace(legacy,old)
        finally:legacy.close()
        legacy_report.update(archive_sha256=_hash(historical),original_archive='artifacts/topspin-contact-v1/shift_right.npz')
        _write(output/'legacy-v1-replay.json',legacy_report)
    if source!=source_hashes() or script_sha!=_hash(__file__):raise RuntimeError('Source changed during artifact generation')
    manifest={'schema_version':'copuzzle.topspin20_engaged_validation.v2','status':'complete',
              'task':'topspin-20-4','facility_profile':facility_profile,
              'purpose':'positive-engagement contact validation; no global exact solver or frozen benchmark',
              'seed':seed,'initial_state':state.tolist(),'sources':source,'collector_script_sha256':script_sha,
              'versions':{'python':platform.python_version(),'mujoco':mujoco.__version__,'numpy':np.__version__},
              'cases':results,'visualization':visualization,'legacy_v1_replay':legacy_report,
              'limitations':['robot-friendly blind sockets and physical pin are an explicit facility adaptation',
                             'engineered bearing friction and token retaining-lip constraints',
                             'small declared test set, not a calibrated real-robot robustness benchmark',
                             '20-token geometry is distinct from abstract TopSpin-8-4'],
              'files':{str(p.relative_to(output)):_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    _write(output/'manifest.json',manifest)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--cases',nargs='+',choices=tuple(CASE_DEFINITIONS))
    parser.add_argument('--profile',choices=('legacy_v1','engaged_v2'),default='engaged_v2')
    parser.add_argument('--render',action='store_true',help='Also save native PNG/GIF previews; MP4 can be exported separately')
    args=parser.parse_args(argv)
    manifest=generate(args.output,args.seed,cases=args.cases,facility_profile=args.profile,render=args.render)
    print(json.dumps({'status':manifest['status'],'manifest':str(args.output/'manifest.json')},indent=2))


if __name__=='__main__':main()
