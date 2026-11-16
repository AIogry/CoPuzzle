"""Public-observation contact executor for the passive turntable handle."""
import numpy as np


class TurntablePolicy:
    """Approach the raised handle and push tangentially through 180 degrees."""
    def __init__(self, *, action_scale=.012, direction=1):
        if direction not in (-1,1): raise ValueError('direction must be -1 or 1')
        self.action_scale,self.direction=action_scale,direction
        self.reset()

    def reset(self):
        self.phase='lift';self._xy=None;self.steps=0
        self._release_xy=None
        self.expected=None;self.initial_count=None;self.failure=None

    @property
    def completed(self): return self.phase=='complete'

    def __call__(self,observation):
        self.steps+=1
        if self.completed:return np.zeros(3,dtype=np.float32)
        tip=observation['end_effector_position']
        angle=float(observation['turntable_angle'][0])
        tangent=np.array([-np.cos(angle),-np.sin(angle)])*self.direction
        target=observation['handle_position'].copy()
        offset=(float(np.clip(.004+.025*abs(observation['turntable_velocity'][0]),.004,.018))
                if self.phase=='push' else .018)
        target[:2]-=offset*tangent
        target[2]=.080
        if self._xy is None:self._xy=tip[:2].copy()
        if 'atomic_operation_count' in observation and self.expected is None:
            self.expected=np.r_[observation['board'][:4][::-1],observation['board'][4:]]
            self.initial_count=int(observation['atomic_operation_count'][0])
        target_angle=np.pi if self.direction==1 else 0.
        if (self.phase=='push' and abs(angle-target_angle)<.0025
                and abs(observation['turntable_velocity'][0])<.05
                and np.max(np.abs(observation['token_velocities']))<.015):
            self.phase='release';self._release_xy=tip[:2].copy()
        if self.phase=='lift':
            target=np.array([*self._xy,.20])
            if abs(tip[2]-.20)<.003:self.phase='travel'
        elif self.phase=='travel':
            target[2]=.20
            if np.linalg.norm(tip-target)<.003:self.phase='lower'
        elif self.phase=='release':
            target=np.array([*self._release_xy,.20])
            if abs(tip[2]-.20)<.003:self.phase='settle' if self.expected is not None else 'complete'
        elif self.phase=='settle':
            target=np.array([*self._release_xy,.20])
            if (observation['board_valid'][0] and np.array_equal(observation['board'],self.expected)
                    and int(observation['atomic_operation_count'][0])==self.initial_count+1):self.phase='complete'
        elif self.phase=='lower':
            if abs(tip[2]-.080)<.003:self.phase='push'
        return np.clip((target-tip)/self.action_scale,-1,1).astype(np.float32)


class LegacyTrackPushPolicy:
    """Press a lower straight token and friction-drag the filled train.

    This experimental controller uses a measured token entering the four-token
    window as its stop condition; it never assigns a board or token position.
    """
    def __init__(self, *, action_scale=.012, direction=1):
        if direction not in (-1,1):raise ValueError('direction must be -1 or 1')
        self.action_scale,self.direction=action_scale,direction
        self.reset()

    def reset(self):
        self.phase='lift';self._xy=None;self.token=None;self.anchor=None
        self._release_xy=None;self.steps=0;self._drag_x=None;self._press_steps=0;self._goal_ids=None

    @property
    def completed(self):return self.phase=='complete'

    def __call__(self,observation):
        self.steps+=1
        if self.completed:return np.zeros(3,dtype=np.float32)
        tip=observation['end_effector_position']
        if self.token is None:
            self.token=int(observation['board'][11 if self.direction==1 else 12])-1
            self.anchor=int(observation['board'][-1 if self.direction==1 else 1])-1
            self._goal_ids=np.roll(observation['board'],self.direction)[:4]-1
            self._xy=tip[:2].copy()
        token=observation['token_positions'][self.token]
        anchor=observation['token_positions'][self.anchor]
        target=np.array([token[0],token[1],token[2]+.0045])
        target_anchor=observation['turntable_center'][0]-.035
        if self.phase=='lift':
            target=np.array([*self._xy,.20])
            if abs(tip[2]-.20)<.003:self.phase='travel'
        elif self.phase=='travel':
            target[2]=.20
            if np.linalg.norm(tip-target)<.003:self.phase='lower'
        elif self.phase=='lower':
            if abs(tip[2]-target[2])<.0025:
                self.phase='push';self._drag_x=tip[0]
        elif self.phase=='push':
            self._press_steps+=1
            if self._press_steps>10:self._drag_x-=self.direction*.0005
            target[0]=self._drag_x
            window=observation['token_positions'][self._goal_ids]
            centered=window[:,:2]-observation['turntable_center'][:2]
            if (np.all(np.abs(centered[:,0])<.0368) and np.all(np.abs(centered[:,1])<.0018)
                    and np.all(np.diff(window[:,0])>0)):
                self.phase='release';self._release_xy=tip[:2].copy()
        elif self.phase=='release':
            target=np.array([*self._release_xy,.20])
            if abs(tip[2]-.20)<.003:self.phase='complete'
        return np.clip((target-tip)/self.action_scale,-1,1).astype(np.float32)


class TrackPushPolicy:
    """Insert a physical pin, transport one selected token, and verify one index.

    Positive engagement is only collision against socket walls. The controller
    monitors contact identity, radial pin clearance and measured relative speed;
    it withdraws and reacquires the same token on loss instead of silently
    pushing a neighbor. A macro is composed from separately verified steps.
    """
    def __init__(self, *, action_scale=.012, direction=1, max_reacquisitions=3):
        if direction not in (-1,1):raise ValueError('direction must be -1 or 1')
        self.action_scale,self.direction=action_scale,direction
        if type(max_reacquisitions) is not int or max_reacquisitions<0:
            raise ValueError('max_reacquisitions must be a nonnegative integer')
        self.max_reacquisitions=max_reacquisitions
        self.reset()

    def reset(self):
        self.phase='lift';self.steps=0;self.phase_steps=0;self.token=None
        self._xy=None;self._release_xy=None;self.expected=None;self.initial=None
        self.initial_count=None;self.reacquisitions=0;self.failure=None;self.audit=[]
        self._missing_contact=0;self._legacy=None

    @property
    def selected_token_id(self):return None if self.token is None else self.token+1

    @property
    def completed(self):return self._legacy.completed if self._legacy is not None else self.phase=='complete'

    @property
    def finished(self):return self.completed or self.phase=='failed'

    def _phase(self,name):self.phase=name;self.phase_steps=0

    def _recover(self,reason,observation):
        self.reacquisitions+=1
        self.audit.append({'step':self.steps,'selected_token_id':self.token+1,'reason':reason,
                           'contact_ids':(np.flatnonzero(observation['tool_token_contact_mask'])+1).tolist()})
        self._xy=observation['end_effector_position'][:2].copy()
        if self.reacquisitions>self.max_reacquisitions:
            self.failure=reason;self._phase('abort_lift')
        else:self._phase('reacquire_lift')
        self._missing_contact=0

    def __call__(self,observation):
        if 'tool_token_contact_mask' not in observation:
            if self._legacy is None:self._legacy=LegacyTrackPushPolicy(action_scale=self.action_scale,direction=self.direction)
            return self._legacy(observation)
        self.steps+=1;self.phase_steps+=1
        tip=observation['end_effector_position']
        if self.finished:return np.zeros(3,np.float32)
        if self.token is None:
            self.token=int(observation['board'][11 if self.direction==1 else 12])-1
            self.initial=observation['board'].copy();self.expected=np.roll(self.initial,self.direction)
            self.initial_count=int(observation['atomic_operation_count'][0]);self._xy=tip[:2].copy()
        token=observation['token_positions'][self.token]
        offset=tip[:2]-token[:2]
        mask=observation['tool_token_contact_mask']
        clearance=float(observation['socket_radius'][0]-observation['pin_radius'][0])
        target=np.array([token[0],token[1],token[2]-.002])
        if self.phase=='push':
            wrong=np.flatnonzero(mask)
            wrong=wrong[wrong!=self.token]
            if len(wrong):self._recover('contact switched away from selected token',observation)
            elif np.linalg.norm(offset)>clearance+.0008:self._recover('pin exceeded socket clearance',observation)
            elif tip[2]>token[2]+.006:self._recover('pin exited the socket during transport',observation)
            elif np.linalg.norm(observation['tool_token_relative_velocity'][self.token,:2])>.03:
                self._recover('excess relative tangential velocity',observation)
            else:
                self._missing_contact=0 if mask[self.token] else self._missing_contact+1
                if self._missing_contact>12:self._recover('selected socket contact lost',observation)
        if self.phase in ('lift','reacquire_lift','abort_lift'):
            target=np.array([*self._xy,.20])
            if abs(tip[2]-.20)<.003:
                self._phase('failed' if self.phase=='abort_lift' else 'travel')
        elif self.phase=='travel':
            target[2]=.12
            if np.linalg.norm(tip-target)<.001:self._phase('align')
        elif self.phase=='align':
            target[2]=token[2]+.012
            if np.linalg.norm(tip-target)<.0006:self._phase('insert')
        elif self.phase=='insert':
            if np.linalg.norm(offset)<clearance-.0004 and abs(tip[2]-target[2])<.0008:
                self._phase('push');self._missing_contact=0
            elif self.phase_steps>100:self._recover('socket insertion did not reach depth',observation)
        elif self.phase=='push':
            # The selected token remains in the lower straight channel during
            # one indexed shift. Wall-normal force, not top-face friction,
            # transfers the load to the closed train.
            target[0]-=self.direction*(clearance+.0010)
            window=observation['token_positions'][self.expected[:4]-1]
            centered=window[:,:2]-observation['turntable_center'][:2]
            if (np.all(np.abs(centered[:,0])<.0368) and np.all(np.abs(centered[:,1])<.0018)
                    and np.all(np.diff(window[:,0])>0)):
                self._release_xy=tip[:2].copy();self._phase('retract')
            elif self.phase_steps>500:self._recover('transport did not reach the next index',observation)
        elif self.phase=='retract':
            target=np.array([*self._release_xy,.20])
            if abs(tip[2]-.20)<.003:self._phase('settle')
        elif self.phase=='settle':
            target=np.array([*self._release_xy,.20])
            if (observation['board_valid'][0] and observation['tool_clear'][0]
                    and np.array_equal(observation['board'],self.expected)
                    and int(observation['atomic_operation_count'][0])==self.initial_count+1):
                self._phase('complete')
            elif self.phase_steps>30:
                if observation['board_valid'][0] and not np.array_equal(observation['board'],self.initial):
                    self.failure='released result was not exactly the requested one-step shift'
                    self._phase('failed')
                else:self._recover('released arrangement did not settle at the next index',observation)
        return np.clip((target-tip)/self.action_scale,-1,1).astype(np.float32)


class SequencePolicy:
    """Execute a caller-specified sequence using only the measured observations."""
    def __init__(self,operations,*,action_scale=.012,max_reacquisitions=3):
        if not operations or any(value not in ('reverse','shift_left','shift_right') for value in operations):
            raise ValueError('Expected nonempty reverse/shift_left/shift_right sequence')
        self.operations=tuple(operations);self.action_scale=action_scale;self.max_reacquisitions=max_reacquisitions;self.reset()

    def reset(self):self.index=0;self.driver=None;self.failure=None

    @property
    def completed(self):return self.index==len(self.operations)

    @property
    def finished(self):return self.completed or self.failure is not None

    def __call__(self,observation):
        if self.finished:return np.zeros(3,dtype=np.float32)
        if (self.driver is not None and getattr(self.driver,'failure',None) is not None
                and getattr(self.driver,'finished',False)):
            self.failure=self.driver.failure
            return np.zeros(3,dtype=np.float32)
        if self.driver is not None and self.driver.completed:
            self.index+=1;self.driver=None
            if self.completed:return np.zeros(3,dtype=np.float32)
        if self.driver is None:
            operation=self.operations[self.index]
            if operation=='reverse':
                self.driver=TurntablePolicy(action_scale=self.action_scale,
                    direction=1 if observation['turntable_angle'][0]<np.pi/2 else -1)
            else:
                self.driver=TrackPushPolicy(action_scale=self.action_scale,
                    direction=1 if operation=='shift_right' else -1,max_reacquisitions=self.max_reacquisitions)
        return self.driver(observation)


class AuditedPolicy:
    """Record controller intent separately and optionally apply bounded noise.

    ``lift_disturbance`` injects a real upward robot action after twenty loaded
    transport observations, then returns control to the same controller. It is
    a reproducible recovery test, not a simulator-state intervention.
    """
    def __init__(self,driver,*,noise_amplitude=0.,seed=0,lift_disturbance=False):
        if not np.isfinite(noise_amplitude) or not 0<=noise_amplitude<=.05:
            raise ValueError('noise_amplitude must lie in [0,.05]')
        self.driver=driver;self.noise_amplitude=float(noise_amplitude)
        self.seed=seed;self.lift_disturbance=bool(lift_disturbance);self.reset()

    def reset(self):
        self.driver.reset();self.rng=np.random.default_rng(self.seed);self.controller_steps=[]
        self._loaded=0;self._disturbance_remaining=0;self.disturbance_applied=False

    @property
    def completed(self):return self.driver.completed

    @property
    def finished(self):return getattr(self.driver,'finished',self.completed)

    @property
    def failure(self):return getattr(self.driver,'failure',None)

    def _active(self):
        return self.driver.driver if isinstance(self.driver,SequencePolicy) else self.driver

    def __call__(self,observation):
        active=self._active()
        selected=getattr(active,'selected_token_id',None) if active is not None else None
        if (active is not None and active.phase=='push' and selected is not None
                and observation['tool_token_contact_normal_force'][selected-1]>.05):
            self._loaded+=1
        if self.lift_disturbance and not self.disturbance_applied and self._loaded>=20:
            self._disturbance_remaining=14;self.disturbance_applied=True
        disturbed=self._disturbance_remaining>0
        if disturbed:
            action=np.array([0,0,1],np.float32);self._disturbance_remaining-=1
        else:action=self.driver(observation)
        active=self._active()
        selected=getattr(active,'selected_token_id',None) if active is not None else None
        self.controller_steps.append({
            'phase':'external_lift_disturbance' if disturbed else getattr(active,'phase','complete'),
            'selected_token_id':selected,
            'operation_index':int(getattr(self.driver,'index',0)),
            'reacquisitions':int(getattr(active,'reacquisitions',0))})
        if self.noise_amplitude:action=action+self.rng.uniform(-self.noise_amplitude,self.noise_amplitude,size=3)
        return np.clip(action,-1,1).astype(np.float32)
