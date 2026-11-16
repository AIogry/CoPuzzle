"""Observable-state scripted contact manipulation reference, not a learner."""
import mujoco
import numpy as np

from .scene import N, STACK_XY, SUPPORT_TOP, PITCH, JAW_OPEN_GAP, JAW_TRAVEL, WITHDRAW_DISTANCE


class PancakeFlipPolicy:
    """Insert forks, clamp a prefix, lift, turn, replace, release, and withdraw."""
    def __init__(self, prefix=2, *, translation_scale=.004, rotation_scale=.04):
        if type(prefix) is not int or not 1 <= prefix <= N:
            raise ValueError('physical prefix must be 1..4; prefix 1 is an execution diagnostic')
        for value in (translation_scale, rotation_scale):
            if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError('Control scales must be positive and finite')
        self.prefix = prefix
        self.translation_scale, self.rotation_scale = translation_scale, rotation_scale
        self.reset()

    def reset(self):
        self.index, self.steps, self.completed = 0, 0, False
        self.phase = 'approach'
        seam = SUPPORT_TOP + (N - self.prefix) * PITCH
        down = np.array([0., 1., 0., 0.])
        # Quaternion sign chooses the +X half-turn path from the downward pose.
        # The equivalent +identity endpoint takes -X and folds into the forearm.
        flip = np.array([-1., 0., 0., 0.])
        nominal = JAW_OPEN_GAP - (self.prefix * PITCH - .002)
        grip = (nominal + .004) / JAW_TRAVEL
        release = (nominal - .0007) / JAW_TRAVEL
        self.waypoints = [
            ('approach', np.array([STACK_XY[0] - .10, 0, seam]), down, 0., 20),
            ('insert', np.array([*STACK_XY, seam]), down, 0., 30),
            ('clamp', np.array([*STACK_XY, seam]), down, grip, 60),
            ('lift', np.array([*STACK_XY, .46]), down, grip, 40),
            ('turn_clearance', np.array([.50, 0, .46]), down, grip, 25),
            ('flip', np.array([.50, 0, .46]), flip, grip, 60),
            ('place', np.array([*STACK_XY, SUPPORT_TOP + N * PITCH]), flip, grip, 40),
            ('release', np.array([*STACK_XY, SUPPORT_TOP + N * PITCH]), flip, release, 40),
            ('withdraw', np.array([STACK_XY[0] - WITHDRAW_DISTANCE - .002, 0,
                                   SUPPORT_TOP + N * PITCH]), flip, release, 50),
        ]
        self._dwell = 0

    def __call__(self, observation):
        if self.completed:
            return np.r_[np.zeros(6), observation['jaw_target'][0]].astype(np.float32)
        if self.steps == 0:
            down = np.array([0., 1., 0., 0.])
            if abs(np.dot(observation['end_effector_orientation'], down)) < .99:
                if np.any(observation['tool_contacts']) or not observation['board_valid'][0]:
                    raise RuntimeError('Cannot reorient a clamp that has not released a supported stack')
                inverted = np.array([-1., 0., 0., 0.])
                if abs(np.dot(observation['end_effector_orientation'], inverted)) < .99:
                    raise RuntimeError('Scripted recovery requires the known inverted post-flip pose')
                position = observation['end_effector_position']
                released_grip = float(observation['jaw_target'][0])
                recovery = [
                    ('empty_lift', np.array([position[0], position[1], .46]), inverted, released_grip, 30),
                    ('empty_clearance', np.array([.50, 0., .46]), inverted, released_grip, 25),
                    ('empty_turn_down', np.array([.50, 0., .46]), down, released_grip, 40),
                    ('empty_open', np.array([.50, 0., .46]), down, 0., 30),
                ]
                self.waypoints = recovery + self.waypoints
        phase, target, quaternion, grip, dwell = self.waypoints[self.index]
        self.phase = phase
        self.steps += 1
        inverse, error = np.empty(4), np.empty(4)
        mujoco.mju_negQuat(inverse, observation['tool_target_orientation'])
        mujoco.mju_mulQuat(error, quaternion, inverse)
        rotation = np.empty(3)
        mujoco.mju_quat2Vel(rotation, error, 1.)
        action = np.r_[np.clip((target - observation['tool_target_position']) / self.translation_scale, -.4, .4),
                       np.clip(rotation / self.rotation_scale, -.35, .35), grip].astype(np.float32)
        inverse_actual = np.empty(4)
        actual_error = np.empty(4)
        actual_rotation = np.empty(3)
        mujoco.mju_negQuat(inverse_actual, observation['end_effector_orientation'])
        mujoco.mju_mulQuat(actual_error, quaternion, inverse_actual)
        mujoco.mju_quat2Vel(actual_rotation, actual_error, 1.)
        near = (np.linalg.norm(observation['end_effector_position'] - target) < .002
                and np.linalg.norm(actual_rotation) < .015)
        self._dwell = self._dwell + 1 if near else 0
        if self._dwell >= dwell:
            self.index += 1
            self._dwell = 0
            if self.index >= len(self.waypoints):
                self.completed = True
        if self.steps > 2300:
            raise RuntimeError(f'Contact flip timed out in {phase}')
        return action


class PancakeSequencePolicy:
    """Chain real prefix flips, returning the empty clamp safely between them."""
    def __init__(self, prefixes, *, translation_scale=.004, rotation_scale=.04):
        self.prefixes = tuple(prefixes)
        if not self.prefixes or any(type(k) is not int or not 1 <= k <= N for k in self.prefixes):
            raise ValueError('prefixes must be a nonempty sequence of integers 1..4')
        self.translation_scale, self.rotation_scale = translation_scale, rotation_scale
        self.reset()

    def reset(self):
        self.index, self.completed, self.phase = 0, False, 'approach'
        self._policy = PancakeFlipPolicy(self.prefixes[0], translation_scale=self.translation_scale,
                                         rotation_scale=self.rotation_scale)

    def __call__(self, observation):
        if self._policy.completed:
            self.index += 1
            if self.index == len(self.prefixes):
                self.completed = True
                return np.r_[np.zeros(6), observation['jaw_target'][0]].astype(np.float32)
            self._policy = PancakeFlipPolicy(self.prefixes[self.index], translation_scale=self.translation_scale,
                                             rotation_scale=self.rotation_scale)
        action = self._policy(observation)
        self.phase = f'flip_{self.index + 1}/{self._policy.phase}'
        return action
