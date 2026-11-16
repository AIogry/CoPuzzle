"""Continuous UR5e contact control of a passive planar 20/4 TopSpin board."""
from copy import deepcopy
import hashlib
import json

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import ResetNeeded
import mujoco
import numpy as np

from copuzzle.games.lights_out.physics.control import DampedIK
from .scene import build_scene, TURN_CENTER, TOKEN_Z, TOKEN_HALF_HEIGHT, SOCKET_RADIUS, SOCKET_BOTTOM, PIN_RADIUS, track_coordinates


class TopSpinPhysicalEnv(gym.Env):
    metadata = {'render_modes':['rgb_array'], 'render_fps':20}

    def __init__(self, *, max_episode_steps=1800, physics_dt=.002, control_dt=.05,
                 action_scale=.012, render_mode=None, camera='puzzle', facility_profile='engaged_v2'):
        if type(max_episode_steps) is not int or max_episode_steps < 1:
            raise ValueError('max_episode_steps must be positive')
        if render_mode not in (None, 'rgb_array') or camera not in ('overview','puzzle','top'):
            raise ValueError('Unsupported render mode or camera')
        for name,value in (('physics_dt',physics_dt),('control_dt',control_dt)):
            value=np.asarray(value)
            if value.ndim!=0 or value.dtype.kind not in 'iuf' or not np.isfinite(value) or value<=0:
                raise ValueError(name+' must be finite and positive')
        ratio=control_dt/physics_dt
        if not np.isfinite(ratio) or ratio<1 or not np.isclose(round(ratio),ratio):
            raise ValueError('control_dt must be an integer multiple of physics_dt')
        if isinstance(action_scale,bool) or not np.isfinite(action_scale) or action_scale<=0:
            raise ValueError('action_scale must be positive')
        self.facility_profile=facility_profile
        self.model,self.scene=build_scene(physics_dt=physics_dt,facility_profile=facility_profile)
        self.data=mujoco.MjData(self.model)
        self.controller=DampedIK(self.model,self.scene)
        self.data.qpos[self.scene.arm_qpos_ids]=self.scene.home_qpos
        mujoco.mj_forward(self.model,self.data)
        self._home=self.controller.solve(self.data,self.scene.home_tool_position).copy()
        self.max_episode_steps,self.substeps=max_episode_steps,round(ratio)
        self.physics_timestep,self.control_timestep=float(physics_dt),float(control_dt)
        self.action_scale=float(action_scale)
        self.render_mode,self.camera=render_mode,camera
        self._renderer=None
        self._closed,self._needs_reset=False,True
        self._steps=0
        self._success_streak=0
        self._operation_count=0
        self._board=self._goal=None
        self._events=[]
        self.action_space=spaces.Box(-1.,1.,shape=(3,),dtype=np.float32)
        box=lambda shape,dtype=np.float64,low=-np.inf,high=np.inf:spaces.Box(low,high,shape=shape,dtype=dtype)
        self.observation_space=spaces.Dict({
            'board':box((20,),np.int32,1,20),'goal':box((20,),np.int32,1,20),
            'board_valid':box((1,),np.uint8,0,1),'end_effector_position':box((3,)),
            'joint_positions':box((6,)),'joint_velocities':box((6,)),
            'token_positions':box((20,3)),'token_velocities':box((20,4)),
            'turntable_angle':box((1,)),'turntable_velocity':box((1,)),
            'handle_position':box((3,)),'turntable_center':box((3,)),
            'slot_positions':box((20,3)), 'steps_remaining':box((1,),np.int32,0,max_episode_steps),
        })
        if facility_profile=='engaged_v2':
            self.observation_space.spaces.update({
                'tool_token_contact_mask':box((20,),np.uint8,0,1),
                'tool_token_contact_normal_force':box((20,),low=0),
                'tool_token_contact_tangent_speed':box((20,),low=0),
                'tool_token_relative_velocity':box((20,3)),
                'tool_linear_velocity':box((3,)),
                'socket_radius':box((1,),low=0), 'pin_radius':box((1,),low=0),
                'socket_bottom_offset':box((1,)), 'tool_clear':box((1,),np.uint8,0,1),
                'atomic_operation_count':box((1,),np.int32,0,max_episode_steps),
                'success_streak':box((1,),np.int32,0,3),
            })
        self._configuration={'schema_version':'copuzzle.topspin20_contact_config.v1',
            'scene':deepcopy(self.scene.metadata),'model_sha256':self.scene.model_sha256,
            'physics_timestep':self.physics_timestep,'control_timestep':self.control_timestep,
            'action_scale':self.action_scale,'max_episode_steps':max_episode_steps,
            'mujoco_version':mujoco.__version__,'state_semantics':'clockwise order starting at the leftmost token fully inside the four-token straight window',
            'slot_positions_semantics':'nominal reset layout; tokens may redistribute spacing along the physical track',
            'token_velocity_semantics':'world x/y/z slide rates and yaw rate; m/s,m/s,m/s,rad/s',
            'success_semantics':'measured goal slots and settled passive mechanism',
            'prototype_scope':'20/4 physical track circulation and planar four-token reversal; no exact global solver'}
        if facility_profile=='engaged_v2':
            self._configuration.update(schema_version='copuzzle.topspin20_contact_config.v2',
                facility_profile=facility_profile,
                success_semantics='exact measured goal after pin withdrawal, tool clearance and three settled control intervals',
                event_semantics='one measured indexed shift/reversal after release; unclassified rearrangements have unknown atomic cost',
                contact_observation_semantics='positive-load contact identity, summed normal force, maximum contact-plane slip speed, and site-relative world velocity')
        self.configuration_sha256=hashlib.sha256(json.dumps(self._configuration,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def configuration(self): return deepcopy(self._configuration)

    def _validate_state(self,value):
        def has_boolean(item):
            return any(has_boolean(v) for v in item) if isinstance(item,(list,tuple)) else isinstance(item,(bool,np.bool_))
        if has_boolean(value):raise ValueError('state values cannot be booleans')
        array=np.asarray(value)
        if array.shape!=(20,) or array.dtype.kind not in 'iu' or not np.array_equal(np.sort(array),np.arange(1,21)):
            raise ValueError('state and goal must be integer permutations of 1..20')
        return array.astype(np.int32,copy=True)

    def _decode(self):
        positions=self.data.xpos[self.scene.token_bodies]
        coordinates,distances=track_coordinates(positions)
        angle=float(self.data.qpos[self.scene.turntable_qpos])
        window=np.flatnonzero((np.abs(positions[:,0]-TURN_CENTER[0])<.039)
                              & (np.abs(positions[:,1]-TURN_CENTER[1])<.002))
        valid=(len(window)==4 and np.max(distances)<.0035
               and np.max(np.abs(positions[:,2]-TOKEN_Z))<.0015
               and min(abs(angle),abs(angle-np.pi))<.005)
        board=self._board.copy() if self._board is not None else np.arange(1,21,dtype=np.int32)
        if valid:
            first=int(window[np.argmin(positions[window,0])])
            ordered=np.argsort(coordinates)
            start=int(np.flatnonzero(ordered==first)[0])
            board=(np.roll(ordered,-start)+1).astype(np.int32)
        return board,bool(valid)

    def _settled(self):
        return (np.max(np.abs(self.data.qvel[self.scene.token_dofs]))<.015
                and abs(self.data.qvel[self.scene.turntable_dof])<.05)

    def _observation(self):
        _,valid=self._decode()
        valid=valid and self._settled() and (self.facility_profile=='legacy_v1' or self._tool_clear())
        s,d=self.scene,self.data
        observation={'board':self._board.copy(),'goal':self._goal.copy(),'board_valid':np.array([valid],np.uint8),
                'end_effector_position':d.site_xpos[s.tool_site_id].copy(),
                'joint_positions':d.qpos[s.arm_qpos_ids].copy(),'joint_velocities':d.qvel[s.arm_dof_ids].copy(),
                'token_positions':d.xpos[s.token_bodies].copy(),'token_velocities':d.qvel[s.token_dofs].copy(),
                'turntable_angle':np.array([d.qpos[s.turntable_qpos]]),
                'turntable_velocity':np.array([d.qvel[s.turntable_dof]]),
                'handle_position':d.site_xpos[s.handle_site].copy(),'turntable_center':TURN_CENTER.copy(),
                'slot_positions':s.slots.copy(),'steps_remaining':np.array([self.max_episode_steps-self._steps],np.int32)}
        if self.facility_profile=='engaged_v2':
            mask,forces,slip,relative,velocity=self._contact_measurements()
            observation.update(tool_token_contact_mask=mask,tool_token_contact_normal_force=forces,
                               tool_token_contact_tangent_speed=slip,tool_token_relative_velocity=relative,
                               tool_linear_velocity=velocity,socket_radius=np.array([SOCKET_RADIUS]),
                               pin_radius=np.array([PIN_RADIUS]),socket_bottom_offset=np.array([SOCKET_BOTTOM]),
                               tool_clear=np.array([self._tool_clear()],np.uint8),
                               atomic_operation_count=np.array([self._operation_count],np.int32),
                               success_streak=np.array([self._success_streak],np.int32))
        return observation

    def _contact_measurements(self):
        d,s=self.data,self.scene
        velocity=np.empty(6)
        mujoco.mj_objectVelocity(self.model,d,mujoco.mjtObj.mjOBJ_SITE,s.tool_site_id,velocity,0)
        tip=d.site_xpos[s.tool_site_id]
        centers=d.xpos[s.token_bodies]
        token_v=d.qvel[s.token_dofs]
        angular=np.column_stack((np.zeros((20,2)),token_v[:,3]))
        relative=velocity[3:]-token_v[:,:3]-np.cross(angular,tip-centers)
        mask=np.zeros(20,np.uint8);forces=np.zeros(20);slip=np.zeros(20)
        tools=set(s.tool_geom_ids.tolist())
        for index,contact in enumerate(d.contact[:d.ncon]):
            a,b=int(contact.geom1),int(contact.geom2)
            token=(s.geom_token_ids.get(b) if a in tools else s.geom_token_ids.get(a) if b in tools else None)
            if token is None:continue
            wrench=np.zeros(6);mujoco.mj_contactForce(self.model,d,index,wrench)
            if wrench[0]<=1e-6:continue
            token-=1;mask[token]=1;forces[token]+=wrench[0]
            tool_velocity=velocity[3:]+np.cross(velocity[:3],contact.pos-tip)
            token_velocity=token_v[token,:3]+np.cross(angular[token],contact.pos-centers[token])
            delta=tool_velocity-token_velocity;normal=contact.frame[:3]
            slip[token]=max(slip[token],float(np.linalg.norm(delta-np.dot(delta,normal)*normal)))
        return mask,forces,slip,relative,velocity[3:].copy()

    def _tool_clear(self):
        return (self.data.site_xpos[self.scene.tool_site_id,2]>.15
                and not self.has_handle_contact() and not self.token_contacts())


    def _info(self):
        board,valid=self._decode()
        eligible=valid and self._settled() and (self.facility_profile=='legacy_v1' or self._tool_clear())
        info={'is_success':bool(eligible and np.array_equal(board,self._goal)),
                'steps':self._steps,'elapsed_time':float(self.data.time),'event_log':deepcopy(self._events),
                'tool_handle_contact':self.has_handle_contact(),'tool_token_contact_ids':self.token_contacts(),'max_token_speed':float(np.max(np.abs(self.data.qvel[self.scene.token_dofs])))}
        if self.facility_profile=='engaged_v2':
            info.update(atomic_operation_count=self._operation_count,tool_clear=bool(self._tool_clear()))
        return info

    def has_handle_contact(self):
        tools=set(self.scene.tool_geom_ids.tolist())
        handles=set(self.scene.handle_geom_ids.tolist())
        for index,contact in enumerate(self.data.contact[:self.data.ncon]):
            a,b=int(contact.geom1),int(contact.geom2)
            if (a in tools and b in handles) or (b in tools and a in handles):
                if self.facility_profile=='legacy_v1':return True
                force=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,index,force)
                if force[0]>1e-6:return True
        return False

    def token_contacts(self):
        if self.facility_profile=='engaged_v2':
            return (np.flatnonzero(self._contact_measurements()[0])+1).tolist()
        tools=set(self.scene.tool_geom_ids.tolist())
        tokens={int(geom):index+1 for index,geom in enumerate(self.scene.token_geoms)}
        touched=set()
        for contact in self.data.contact[:self.data.ncon]:
            a,b=int(contact.geom1),int(contact.geom2)
            if a in tools and b in tokens:touched.add(tokens[b])
            if b in tools and a in tokens:touched.add(tokens[a])
        return sorted(touched)

    def reset(self,*,seed=None,options=None):
        if self._closed: raise RuntimeError('Environment is closed')
        super().reset(seed=seed)
        options={} if options is None else options
        if not isinstance(options,dict) or set(options)-{'state','goal'}: raise ValueError('Unknown reset options')
        state=self._validate_state(options['state']) if 'state' in options else np.arange(1,21,dtype=np.int32)
        goal=self._validate_state(options['goal']) if 'goal' in options else np.concatenate((state[:4][::-1],state[4:]))
        if np.array_equal(state,goal): raise ValueError('Reset goal must differ from state')
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[self.scene.arm_qpos_ids]=self._home
        self.data.ctrl[self.scene.arm_actuator_ids]=self._home
        for index,token in enumerate(state):
            self.data.qpos[self.scene.token_qpos[token-1,:2]]=self.scene.slots[index,:2]-self.scene.slots[token-1,:2]
        mujoco.mj_forward(self.model,self.data)
        self._board,self._goal=state.copy(),goal.copy()
        self._steps,self._events,self._needs_reset=0,[],False
        self._success_streak=0
        self._operation_count=0
        return self._observation(),self._info()

    def step(self,action):
        if self._closed: raise RuntimeError('Environment is closed')
        if self._needs_reset: raise ResetNeeded('Reset before stepping')
        action=np.asarray(action)
        if action.shape!=(3,) or action.dtype.kind not in 'iuf' or not np.isfinite(action).all() or np.any(np.abs(action)>1):
            raise ValueError('Expected finite action in [-1,1]^3')
        target=self.data.site_xpos[self.scene.tool_site_id]+self.action_scale*action
        self.data.ctrl[self.scene.arm_actuator_ids]=self.controller.solve(self.data,target)
        self._events=[]
        for _ in range(self.substeps):
            mujoco.mj_step(self.model,self.data)
        mujoco.mj_forward(self.model,self.data)
        board,valid=self._decode()
        if (valid and self._settled() and (self.facility_profile=='legacy_v1' or self._tool_clear())
                and not np.array_equal(board,self._board)):
            before=self._board.copy()
            if np.array_equal(board,np.concatenate((before[:4][::-1],before[4:]))): operation='reverse_four'
            elif np.array_equal(board,np.roll(before,-1)): operation='shift_left'
            elif np.array_equal(board,np.roll(before,1)): operation='shift_right'
            else: operation='unclassified_measured_rearrangement'
            event={'operation':operation,'state_before':before.tolist(),'state_after':board.tolist()}
            if self.facility_profile=='engaged_v2':
                cost=1 if operation!='unclassified_measured_rearrangement' else None
                if cost is not None:self._operation_count+=cost
                event.update(atomic_cost=cost,operation_index=self._operation_count if cost else None)
            self._events.append(event)
            self._board=board
        self._steps+=1
        info=self._info()
        self._success_streak=self._success_streak+1 if info['is_success'] else 0
        terminated=self._success_streak>=3
        info['is_success']=bool(terminated)
        truncated=self._steps>=self.max_episode_steps and not terminated
        self._needs_reset=terminated or truncated
        return self._observation(),0. if terminated else -1.,terminated,truncated,info

    def render(self):
        if self.render_mode is None:return None
        if self._renderer is None:self._renderer=mujoco.Renderer(self.model,height=768,width=1024)
        self._renderer.update_scene(self.data,camera=self.camera)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:self._renderer.close()
        self._renderer=None;self._closed=True
