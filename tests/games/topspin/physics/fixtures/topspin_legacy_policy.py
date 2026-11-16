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
            if abs(tip[2]-.20)<.003:self.phase='complete'
        elif self.phase=='lower':
            if abs(tip[2]-.080)<.003:self.phase='push'
        return np.clip((target-tip)/self.action_scale,-1,1).astype(np.float32)


class TrackPushPolicy:
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


class SequencePolicy:
    """Execute a caller-specified sequence using only the measured observations."""
    def __init__(self,operations,*,action_scale=.012):
        if not operations or any(value not in ('reverse','shift_left','shift_right') for value in operations):
            raise ValueError('Expected nonempty reverse/shift_left/shift_right sequence')
        self.operations=tuple(operations);self.action_scale=action_scale;self.reset()

    def reset(self):self.index=0;self.driver=None

    @property
    def completed(self):return self.index==len(self.operations)

    def __call__(self,observation):
        if self.completed:return np.zeros(3,dtype=np.float32)
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
                    direction=1 if operation=='shift_right' else -1)
        return self.driver(observation)
