"""Six-dimensional Cartesian IK; only actuator targets move the live robot."""
import mujoco
import numpy as np

from copuzzle.games.lights_out.physics.control import DampedIK


class PoseIK(DampedIK):
    def __init__(self, model, scene):
        super().__init__(model, scene, damping=.025, max_iterations=120,
                         position_tolerance=3e-5, rotation_tolerance=3e-4)
        self.target_quaternion = scene.tool_down_quat.copy()
        self._command_seed = mujoco.MjData(model)
        self._last_command = None

    def reset(self, joint_seed=None):
        self._last_command = None if joint_seed is None else np.asarray(joint_seed, dtype=np.float64).copy()

    def _pose_error(self, target):
        mujoco.mj_kinematics(self.model, self._scratch)
        mujoco.mj_comPos(self.model, self._scratch)
        site = self.scene.tool_site_id
        error = target - self._scratch.site_xpos[site]
        mujoco.mju_mat2Quat(self._current_quat, self._scratch.site_xmat[site])
        mujoco.mju_negQuat(self._inverse_quat, self._current_quat)
        mujoco.mju_mulQuat(self._error_quat, self.target_quaternion, self._inverse_quat)
        mujoco.mju_quat2Vel(self._rotation_error, self._error_quat, 1.)
        return np.concatenate([error, self._rotation_error])

    def solve(self, data, target_xyz, quaternion=None):
        if quaternion is not None:
            quaternion = np.asarray(quaternion, dtype=np.float64)
            if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-10:
                raise ValueError('quaternion must be a finite nonzero four-vector')
            self.target_quaternion = quaternion / np.linalg.norm(quaternion)
        # Use the continuous previous target branch rather than chasing a
        # delayed measured wrist through an IK singularity. Only scratch
        # kinematics are changed; live qpos remains untouched.
        self._command_seed.qpos[:] = data.qpos
        if self._last_command is not None:
            self._command_seed.qpos[self.scene.arm_qpos_ids] = self._last_command
        result = super().solve(self._command_seed, target_xyz)
        self._last_command = result.copy()
        return result
