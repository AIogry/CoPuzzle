"""Contact-actuated permutation fixtures with measured token-state decoding."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import ResetNeeded
import mujoco
import numpy as np

from ..oracle import ExactOracle
from ..rules import get_rule
from .scene import build_scene, latch_current, STROKE, RACK_TRAVEL
from copuzzle.games.lights_out.physics.control import DampedIK


class PhysicalPermutationEnv(gym.Env):
    """UR5e probe drives spring racks linked to captured rigid token carriers.

    Phase 0=ready, 1=carrying, 2=empty return, 3=mechanism fault. Latch transfers
    preserve qpos/qvel. Rule.transition is used only as an integrity cross-check;
    the next board is decoded from actual token positions. Ideal switchable weld
    clamps are an explicit mechanism abstraction, not learned grasping.
    """
    metadata = {'render_modes':['rgb_array'], 'render_fps':20}

    def __init__(self, task_id='topspin-8-4', *, max_episode_steps=3000,
                 render_mode=None, physics_dt=.002, control_dt=.05, action_scale=.012):
        super().__init__()
        self.rule = get_rule(task_id)
        if type(max_episode_steps) is not int or max_episode_steps < 1:
            raise ValueError('max_episode_steps must be positive')
        for name, value in [('physics_dt',physics_dt),('control_dt',control_dt),('action_scale',action_scale)]:
            if isinstance(value,bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be finite and positive')
        ratio = control_dt / physics_dt
        if round(ratio) < 1 or not np.isclose(ratio, round(ratio), atol=1e-10, rtol=0):
            raise ValueError('control_dt must be an integer multiple of physics_dt')
        if render_mode not in (None,'rgb_array'):
            raise ValueError('Only rgb_array rendering is supported')
        self.max_episode_steps, self.render_mode = max_episode_steps, render_mode
        self.physics_timestep, self.control_timestep = float(physics_dt), float(control_dt)
        self.substeps, self.action_scale = round(ratio), float(action_scale)
        self.model, self.scene, self.fixture = build_scene(self.rule, physics_dt=physics_dt)
        self.data = mujoco.MjData(self.model)
        self.controller = DampedIK(self.model, self.scene)
        self.data.qpos[self.scene.arm_qpos_ids] = self.scene.home_qpos
        mujoco.mj_forward(self.model,self.data)
        self._home = self.controller.solve(self.data,self.scene.home_tool_position).copy()
        self._default_equalities = self.model.eq_data.copy()
        self._oracle = ExactOracle(self.rule)
        self._closed, self._renderer, self._needs_reset = False,None,True
        self._steps, self._phase, self._active = 0,0,-1
        self._board = self._goal = None
        self._counts = np.zeros(3,dtype=np.int32)
        self._event_log, self._attempt_log = [],[]
        self._selected = np.array([],dtype=np.int32)
        self._rack_start, self._settled, self._started_step = 0.,0,0
        self._failure = None
        self.action_space = spaces.Box(-1.,1.,shape=(3,),dtype=np.float32)
        N=self.rule.n_tokens
        vec=lambda shape,dtype=np.float64,low=-np.inf,high=np.inf: spaces.Box(low,high,shape=shape,dtype=dtype)
        self.observation_space=spaces.Dict({
            'board':vec((N,),np.int32,1,N), 'goal':vec((N,),np.int32,1,N),
            'board_valid':vec((1,),np.uint8,0,1), 'mechanism_phase':vec((1,),np.int32,0,3),
            'active_operation':vec((1,),np.int32,-1,2), 'rack_start':vec((1,)),
            'joint_positions':vec((6,)), 'joint_velocities':vec((6,)),
            'end_effector_position':vec((3,)), 'end_effector_orientation':vec((4,),low=-1,high=1),
            'token_positions':vec((N,3)), 'token_orientations':vec((N,4),low=-1,high=1),
            'token_velocities':vec((N,6)), 'slot_positions':vec((N,3)),
            'rack_positions':vec((3,3)), 'rack_rest_positions':vec((3,3)),
            'rack_displacements':vec((3,)), 'rack_velocities':vec((3,)),
            'carrier_angles':vec((3,)), 'carrier_velocities':vec((3,)),
            'previous_event_counts':vec((3,),np.int32,0,self.substeps),
            'steps_remaining':vec((1,),np.int32,0,max_episode_steps),
        })
        self._configuration={
            'schema_version':'copuzzle.permutation_physics_config.v1', 'task_name':task_id,
            'rule_spec':self.rule.spec(), 'model_sha256':self.scene.model_sha256,
            'scene':deepcopy(self.scene.metadata), 'physics_timestep':self.physics_timestep,
            'control_timestep':self.control_timestep,'substeps':self.substeps,
            'action_scale':self.action_scale,'max_episode_steps':max_episode_steps,
            'action_semantics':'UR5e measured tool XYZ increments; contact drives geared racks',
            'success_semantics':'decoded target permutation with all mechanisms docked and stationary',
            'event_semantics':'ordered measured token permutations after physical carrier motion',
            'controller':'existing fixed-downward damped IK with joint-position actuators',
            'mujoco_version':mujoco.__version__,
        }
        self.configuration_sha256=hashlib.sha256(json.dumps(self._configuration,sort_keys=True,
                                                             separators=(',',':')).encode()).hexdigest()

    def configuration(self):
        return deepcopy(self._configuration)

    def _decode(self):
        positions=self.data.xpos[self.fixture['token_bodies']]
        distances=np.linalg.norm(positions[:,None,:]-self.fixture['slots'][None,:,:],axis=-1)
        slots=distances.argmin(axis=1)
        valid=len(set(slots.tolist()))==self.rule.n_tokens and np.max(distances[np.arange(len(slots)),slots])<.0025
        board=self._board.copy() if self._board is not None else np.arange(1,self.rule.n_tokens+1,dtype=np.int32)
        if valid:
            board[slots]=np.arange(1,self.rule.n_tokens+1,dtype=np.int32)
        return board,bool(valid)

    def _is_success(self):
        board,valid=self._decode()
        return self._phase==0 and valid and self._fixture_stationary() and np.array_equal(board,self._goal)

    def _fixture_stationary(self, *, ignore_racks=False):
        token_velocities=self._token_velocities()
        return (np.max(np.abs(token_velocities))<.01
                and np.max(np.abs(self.data.qvel[self.fixture['hinge_dofs']]))<.1
                and (ignore_racks or np.max(np.abs(self.data.qvel[self.scene.button_dof_ids]))<.01))

    def _token_velocities(self):
        velocities=[]
        for body in self.fixture['token_bodies']:
            velocity=np.empty(6)
            mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,int(body),velocity,0)
            velocities.append(velocity)
        return np.asarray(velocities)

    def _observation(self):
        f,s,d=self.fixture,self.scene,self.data
        _,valid=self._decode()
        quat=np.empty(4)
        mujoco.mju_mat2Quat(quat,d.site_xmat[s.tool_site_id])
        if quat[0]<0: quat*=-1
        return {
            'board':self._board.copy(),'goal':self._goal.copy(),
            'board_valid':np.array([valid and self._phase in (0,2)],dtype=np.uint8),
            'mechanism_phase':np.array([self._phase],dtype=np.int32),
            'active_operation':np.array([self._active],dtype=np.int32),
            'rack_start':np.array([self._rack_start],dtype=np.float64),
            'joint_positions':d.qpos[s.arm_qpos_ids].copy(),'joint_velocities':d.qvel[s.arm_dof_ids].copy(),
            'end_effector_position':d.site_xpos[s.tool_site_id].copy(),'end_effector_orientation':quat,
            'token_positions':d.xpos[f['token_bodies']].copy(),'token_orientations':d.xquat[f['token_bodies']].copy(),
            'token_velocities':self._token_velocities(),'slot_positions':f['slots'].copy(),
            'rack_positions':d.site_xpos[s.button_site_ids].copy(),'rack_rest_positions':s.button_rest_positions.copy(),
            'rack_displacements':d.qpos[s.button_qpos_ids].copy(),'rack_velocities':d.qvel[s.button_dof_ids].copy(),
            'carrier_angles':d.qpos[f['hinge_qpos']].copy(),'carrier_velocities':d.qvel[f['hinge_dofs']].copy(),
            'previous_event_counts':self._counts.copy(),
            'steps_remaining':np.array([self.max_episode_steps-self._steps],dtype=np.int32),
        }

    def _info(self):
        return {'is_success':bool(self._is_success()),'steps':self._steps,'elapsed_time':float(self.data.time),
                'rule_id':self.rule.rule_id,'event_log':deepcopy(self._event_log),
                'attempt_log':deepcopy(self._attempt_log),'failure_reason':self._failure}

    def reset(self,*,seed=None,options=None):
        if self._closed: raise RuntimeError('Environment is closed')
        super().reset(seed=seed)
        if options is not None and not isinstance(options,dict): raise ValueError('options must be a dict')
        options={} if options is None else options
        if set(options)-{'state','goal'}: raise ValueError('Unknown reset options')
        state=self.rule.validate_state(options['state']) if 'state' in options else self.np_random.permutation(
            np.arange(1,self.rule.n_tokens+1,dtype=np.int32))
        if 'goal' in options:
            goal=self.rule.validate_state(options['goal'])
            if self._oracle.distance(state,goal) is None: raise ValueError('Unreachable goal')
        else:
            goal=state[self._oracle.reachable_states[self.np_random.integers(1,len(self._oracle.reachable_states))]-1]
        mujoco.mj_resetData(self.model,self.data)
        self.model.eq_data[:]=self._default_equalities
        self.model.jnt_range[self.scene.button_joint_ids,1]=RACK_TRAVEL
        self.data.qpos[self.scene.arm_qpos_ids]=self._home
        self.data.ctrl[self.scene.arm_actuator_ids]=self._home
        for slot,token in enumerate(state):
            address=self.fixture['token_qpos'][token-1]
            self.data.qpos[address:address+3]=self.fixture['slots'][slot]
            self.data.qpos[address+3:address+7]=[1,0,0,0]
        mujoco.mj_forward(self.model,self.data)
        for eq in self.fixture['dock']: latch_current(self.model,self.data,eq)
        self._board,self._goal=state.copy(),goal.copy()
        self._steps,self._phase,self._active=0,0,-1
        self._rack_start,self._settled=0.,0
        self._counts[:]=0
        self._event_log,self._attempt_log=[],[]
        self._selected=np.array([],dtype=np.int32)
        self._failure=None
        self._return_locked=False
        self._started_time=0.
        self._needs_reset=np.array_equal(state,goal)
        return self._observation(),self._info()

    def _contact_racks(self):
        tool=set(map(int,self.scene.tool_geom_ids))
        racks={int(geom):i for i,geom in enumerate(self.scene.button_geom_ids)}
        found=set()
        for contact in self.data.contact[:self.data.ncon]:
            a,b=int(contact.geom1),int(contact.geom2)
            if a in tool and b in racks: found.add(racks[b])
            if b in tool and a in racks: found.add(racks[a])
        return found

    def _start(self,action,substep):
        f,d,m=self.fixture,self.data,self.model
        self._active,self._phase,self._settled=action,1,0
        self._started_step=self._steps
        self._started_substep=substep
        self._started_time=float(d.time)
        self._pre_board=self._board.copy()
        indices=range(self.rule.n_tokens) if self.rule.family=='topspin' and action<2 else range(
            4 if self.rule.family=='topspin' else action+2)
        self._selected=np.array([self._board[i]-1 for i in indices],dtype=np.int32)
        for token in self._selected:
            latch_current(m,d,f['carry'][action,token])
            d.eq_active[f['dock'][token]]=False
        self._rack_start=float(d.qpos[self.scene.button_qpos_ids[action]])
        gear=f['gear'][action]
        ratio=f['angles'][action]/STROKE
        m.eq_data[gear,0]=float(d.qpos[f['hinge_qpos'][action]])-ratio*self._rack_start
        d.eq_active[gear]=True
        d.eq_active[f['carrier_lock'][action]]=False
        for other in range(3):
            d.eq_active[f['rack_lock'][other]]=other!=action
        m.jnt_range[self.scene.button_joint_ids[action],1]=self._rack_start+STROKE

    def _transfer_to_docks(self,board):
        for token in self._selected:
            eq=self.fixture['dock'][token]
            latch_current(self.model,self.data,eq)
            # Ideal slot docking pulls only the final small residual into the
            # indexed seat through constraint forces; no generalized pose edit.
            slot=int(np.flatnonzero(board==token+1)[0])
            self.model.eq_data[eq,3:6]=self.fixture['slots'][slot]
            self.data.eq_active[self.fixture['carry'][self._active,token]]=False
        self._board=board.copy()
        self._phase,self._settled=2,0
        self._return_locked=False

    def _mechanism_step(self,substep):
        f,d=self.fixture,self.data
        if self._phase==0:
            contacts=self._contact_racks()
            candidates=[a for a in contacts if d.qpos[self.scene.button_qpos_ids[a]]>.0004]
            if len(candidates)==1 and self._fixture_stationary(ignore_racks=True): self._start(candidates[0],substep)
        elif self._phase==1:
            a=self._active
            angle=d.qpos[f['hinge_qpos'][a]]
            velocity=d.qvel[f['hinge_dofs'][a]]
            board,valid=self._decode()
            complete=abs(angle-f['angles'][a])<.02 and abs(velocity)<.5 and valid
            self._settled=self._settled+1 if complete else 0
            if self._settled>=5:
                expected=self.rule.transition(self._pre_board,a)
                if not np.array_equal(board,expected):
                    self._phase,self._failure=3,'physical permutation differs from allowed mechanism'
                    return
                self._transfer_to_docks(board)
                self._counts[a]+=1
                self._event_log.append({'substep':substep,'operation':a,'started_step':self._started_step,
                                        'started_substep':self._started_substep,
                                        'duration_seconds':float(self.data.time-self._started_time)})
            elif (self._steps-self._started_step>2 and abs(angle)<.015 and abs(velocity)<.1
                  and d.qpos[self.scene.button_qpos_ids[a]]<self._rack_start+.0002
                  and a not in self._contact_racks() and valid):
                if np.array_equal(board,self._pre_board):
                    self._transfer_to_docks(board)
                    self._attempt_log.append({'substep':substep,'operation':a,'result':'aborted_and_returned'})
        elif self._phase==2:
            a=self._active
            ready=(abs(d.qpos[f['hinge_qpos'][a]])<.02 and abs(d.qvel[f['hinge_dofs'][a]])<.1
                   and abs(d.qpos[self.scene.button_qpos_ids[a]])<.001)
            if self._return_locked:
                ready=ready and self._fixture_stationary()
            self._settled=self._settled+1 if ready else 0
            if self._settled>=5:
                if not self._return_locked:
                    d.eq_active[f['gear'][a]]=False
                    d.eq_active[f['carrier_lock'][a]]=True
                    self._return_locked=True
                    self._settled=0
                else:
                    d.eq_active[f['rack_lock']]=False
                    self.model.jnt_range[self.scene.button_joint_ids,1]=RACK_TRAVEL
                    self._phase,self._active,self._rack_start=0,-1,0.

    def step(self,action):
        if self._closed: raise RuntimeError('Environment is closed')
        if self._needs_reset: raise ResetNeeded('Reset before stepping or after episode completion')
        action=np.asarray(action)
        if action.shape!=(3,) or action.dtype.kind not in 'fiu' or not np.all(np.isfinite(action)) or np.any(np.abs(action)>1):
            raise ValueError('action must be a finite vector in [-1,1]^3')
        target=self.data.site_xpos[self.scene.tool_site_id]+self.action_scale*action
        joints=self.controller.solve(self.data,target)
        self.data.ctrl[self.scene.arm_actuator_ids]=joints
        self._counts[:]=0
        self._event_log,self._attempt_log=[],[]
        for substep in range(self.substeps):
            mujoco.mj_step(self.model,self.data)
            mujoco.mj_forward(self.model,self.data)
            self._mechanism_step(substep)
        self._steps+=1
        success=bool(self._is_success())
        timeout=self._steps>=self.max_episode_steps and not success
        self._needs_reset=success or timeout
        return self._observation(),0. if success else -1.,success,timeout,self._info()

    def render(self):
        if self.render_mode is None: return None
        if self._renderer is None: self._renderer=mujoco.Renderer(self.model,height=480,width=640)
        # Sites locate joints, slots and contacts for diagnostics. Their
        # translucent spheres are not physical geometry and obscure token faces.
        options=mujoco.MjvOption()
        options.sitegroup[:]=0
        self._renderer.update_scene(self.data,camera='overview',scene_option=options)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None: self._renderer.close()
        self._renderer=None
        self._closed=True
