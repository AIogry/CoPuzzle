"""Cartesian targets for a position-controlled UR5e, using MuJoCo Jacobians.

Only scratch forward kinematics are changed during IK. The live robot moves
through actuator forces and mj_step; solve never teleports simulation qpos.
"""
from __future__ import annotations

import mujoco
import numpy as np


class DampedIK:
    """Six-dimensional damped least-squares IK with a fixed down orientation.

    This is a kinematic controller, not a collision-free path planner. Targets
    are clipped to the declared workspace. The returned arm positions respect
    joint/actuator ranges; difficult targets can retain a nonzero pose residual.
    """

    def __init__(self, model, scene, *, damping=0.01, max_iterations=80,
                 max_joint_step=0.20, position_tolerance=2e-5, rotation_tolerance=2e-4):
        for name, value in (("damping", damping), ("max_joint_step", max_joint_step),
                            ("position_tolerance", position_tolerance),
                            ("rotation_tolerance", rotation_tolerance)):
            numeric = np.asarray(value)
            if (numeric.ndim != 0 or numeric.dtype.kind not in "iuf"
                    or not np.isfinite(numeric) or float(numeric) <= 0):
                raise ValueError(f"{name} must be positive and finite")
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.model = model
        self.scene = scene
        self.damping = float(damping)
        self.max_iterations = max_iterations
        self.max_joint_step = float(max_joint_step)
        self.position_tolerance = float(position_tolerance)
        self.rotation_tolerance = float(rotation_tolerance)
        self._scratch = mujoco.MjData(model)
        self._jac_position = np.empty((3, model.nv))
        self._jac_rotation = np.empty((3, model.nv))
        self._current_quat = np.empty(4)
        self._inverse_quat = np.empty(4)
        self._error_quat = np.empty(4)
        self._rotation_error = np.empty(3)
        joints = scene.arm_joint_ids
        limits = model.jnt_range[joints].copy()
        limits[~model.jnt_limited[joints].astype(bool)] = [-np.inf, np.inf]
        controls = model.actuator_ctrlrange[scene.arm_actuator_ids].copy()
        controls[~model.actuator_ctrllimited[scene.arm_actuator_ids].astype(bool)] = [-np.inf, np.inf]
        self._lower = np.maximum(limits[:, 0], controls[:, 0])
        self._upper = np.minimum(limits[:, 1], controls[:, 1])
        self.last_position_error = float("inf")
        self.last_rotation_error = float("inf")
        self.last_iterations = 0

    def _pose_error(self, target):
        mujoco.mj_kinematics(self.model, self._scratch)
        mujoco.mj_comPos(self.model, self._scratch)
        site = self.scene.tool_site_id
        error = target - self._scratch.site_xpos[site]
        mujoco.mju_mat2Quat(self._current_quat, self._scratch.site_xmat[site])
        mujoco.mju_negQuat(self._inverse_quat, self._current_quat)
        mujoco.mju_mulQuat(self._error_quat, self.scene.tool_down_quat, self._inverse_quat)
        mujoco.mju_quat2Vel(self._rotation_error, self._error_quat, 1.0)
        return np.concatenate((error, self._rotation_error))

    def solve(self, data, target_xyz):
        target = np.asarray(target_xyz)
        if (target.shape != (3,) or target.dtype.kind not in "iuf"
                or not np.isfinite(target).all()):
            raise ValueError("target_xyz must contain three finite real numbers")
        target = np.clip(target.astype(np.float64), *self.scene.workspace_bounds)
        if not np.isfinite(data.qpos).all():
            raise ValueError("Simulation qpos must be finite")
        self._scratch.qpos[:] = data.qpos
        arm = self.scene.arm_qpos_ids
        self._scratch.qpos[arm] = np.clip(self._scratch.qpos[arm], self._lower, self._upper)
        for iteration in range(self.max_iterations):
            error = self._pose_error(target)
            if (np.linalg.norm(error[:3]) <= self.position_tolerance
                    and np.linalg.norm(error[3:]) <= self.rotation_tolerance):
                break
            mujoco.mj_jacSite(self.model, self._scratch, self._jac_position,
                             self._jac_rotation, self.scene.tool_site_id)
            jacobian = np.vstack((self._jac_position[:, self.scene.arm_dof_ids],
                                  self._jac_rotation[:, self.scene.arm_dof_ids]))
            update = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + self.damping ** 2 * np.eye(6), error)
            scale = max(1.0, float(np.max(np.abs(update))) / self.max_joint_step)
            self._scratch.qpos[arm] = np.clip(
                self._scratch.qpos[arm] + update / scale, self._lower, self._upper)
        residual = self._pose_error(target)
        self.last_position_error = float(np.linalg.norm(residual[:3]))
        self.last_rotation_error = float(np.linalg.norm(residual[3:]))
        self.last_iterations = iteration + 1
        return self._scratch.qpos[arm].copy()
