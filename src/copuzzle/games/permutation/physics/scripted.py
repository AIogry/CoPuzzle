"""Observation-only operation executor for the rack-and-carrier fixture."""
import numpy as np


class FixtureSequencePolicy:
    def __init__(self,operations,*,action_scale=.012,clearance=.09,phase_timeout=400):
        operations=tuple(operations)
        if any(isinstance(a,(bool,np.bool_)) or not isinstance(a,(int,np.integer)) or not 0<=a<3 for a in operations):
            raise ValueError('operations must be IDs0,1,2')
        for name,value in (('action_scale',action_scale),('clearance',clearance)):
            if isinstance(value,bool) or not np.isscalar(value) or not np.isfinite(value) or value<=0:
                raise ValueError(name+' must be finite and positive')
        if type(phase_timeout) is not int or phase_timeout<1:
            raise ValueError('phase_timeout must be positive')
        self.operations=tuple(map(int,operations))
        self.action_scale,self.clearance,self.phase_timeout=action_scale,clearance,phase_timeout
        self.reset()

    def reset(self):
        self.index,self.phase,self.phase_steps=0,'lift',0
        self.confirmed=[]
        self._xy=None

    @property
    def completed(self): return self.index==len(self.operations)

    def _phase(self,value):
        self.phase,self.phase_steps=value,0

    def __call__(self,observation):
        tip=observation['end_effector_position']
        if self.completed: return np.zeros(3,dtype=np.float32)
        op=self.operations[self.index]
        rest=observation['rack_rest_positions'][op]
        hover=rest+[0,0,self.clearance]
        if self._xy is None: self._xy=tip[:2].copy()
        self.phase_steps+=1
        if self.phase_steps>self.phase_timeout:
            raise RuntimeError(f'Operation {op} timed out in {self.phase}; phase={observation["mechanism_phase"].tolist()}')
        if self.phase=='lift':
            target=np.array([*self._xy,hover[2]])
            if abs(tip[2]-hover[2])<.003: self._phase('travel')
        elif self.phase=='travel':
            target=hover
            if np.linalg.norm(tip-hover)<.003: self._phase('press')
        elif self.phase=='press':
            target=rest-[0,0,.052]
            if observation['previous_event_counts'][op]>0:
                self.confirmed.append(op)
                self._phase('release')
                target=hover
        else:
            target=hover
            if np.linalg.norm(tip-hover)<.003 and observation['mechanism_phase'][0]==0:
                self.index+=1
                self._xy=tip[:2].copy()
                self._phase('lift')
        return np.clip((target-tip)/self.action_scale,-1,1).astype(np.float32)
