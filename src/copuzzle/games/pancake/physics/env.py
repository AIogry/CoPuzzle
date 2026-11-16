"""Continuous UR5e contact manipulation of a real supported rigid-disc stack."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import ResetNeeded
import mujoco
import numpy as np

from copuzzle.games.permutation.rules import get_rule
from .control import PoseIK
from .scene import build_scene, N, STACK_XY, SUPPORT_TOP, BOSS_HALF, PITCH, JAW_TRAVEL, WITHDRAW_DISTANCE


class PancakeManipulationEnv(gym.Env):
    """Action = Cartesian XYZ/rotation-vector increments plus absolute jaw close.

    The seventh component is in [0,1], corresponding to 0..JAW_TRAVEL metres.
    The first six are normalized [-1,1] increments. Motion is produced by arm
    and gripper actuators and contacts; token qpos is assigned only at reset.
    """
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 50}

    def __init__(self, *, max_episode_steps=2500, physics_dt=.001, control_dt=.02,
                 translation_scale=.004, rotation_scale=.04, render_mode=None,
                 task_mode='goal', operation_prefix=None):
        if task_mode not in ('goal', 'operation'):
            raise ValueError("task_mode must be 'goal' or 'operation'")
        if task_mode == 'operation':
            if type(operation_prefix) is not int or not 1 <= operation_prefix <= N:
                raise ValueError('operation mode requires an operation_prefix in 1..4')
        elif operation_prefix is not None:
            raise ValueError('operation_prefix is only valid in operation mode')
        self.task_mode, self.operation_prefix = task_mode, operation_prefix
        if type(max_episode_steps) is not int or max_episode_steps < 1:
            raise ValueError('max_episode_steps must be a positive integer')
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('Only rgb_array rendering is supported')
        for value in (physics_dt, control_dt, translation_scale, rotation_scale):
            if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError('Intervals and control scales must be positive finite numbers')
        substeps = round(control_dt / physics_dt)
        if substeps < 1 or not np.isclose(substeps * physics_dt, control_dt):
            raise ValueError('control_dt must be an integer multiple of physics_dt')
        self.max_episode_steps, self.render_mode = max_episode_steps, render_mode
        self.physics_dt, self.control_dt, self.substeps = physics_dt, control_dt, substeps
        self.translation_scale, self.rotation_scale = translation_scale, rotation_scale
        self.rule = get_rule('pancake-4')
        self.model, self.scene = build_scene(physics_dt=physics_dt)
        self.data = mujoco.MjData(self.model)
        self.controller = PoseIK(self.model, self.scene)
        self.data.qpos[self.scene.arm_qpos_ids] = self.scene.home_qpos
        mujoco.mj_forward(self.model, self.data)
        self._home = self.controller.solve(self.data, self.scene.home_tool_position,
                                           self.scene.tool_down_quat).copy()
        self._home_ik_errors = (self.controller.last_position_error, self.controller.last_rotation_error)
        self._needs_reset, self._closed, self._renderer = True, False, None
        self._steps = 0
        self._board = np.arange(1, N + 1, dtype=np.int32)
        self._goal = self._board[::-1].copy()
        self._target_position = self.scene.home_tool_position.copy()
        self._target_quaternion = self.scene.tool_down_quat.copy()
        self._grip = 0.
        self.action_space = spaces.Box(np.array([-1.] * 6 + [0.], dtype=np.float32),
                                       np.ones(7, dtype=np.float32), dtype=np.float32)
        vector = lambda shape, dtype=np.float64, low=-np.inf, high=np.inf: spaces.Box(
            low, high, shape=shape, dtype=dtype)
        self.observation_space = spaces.Dict({
            'board': vector((N,), np.int32, 1, N), 'goal': vector((N,), np.int32, 1, N),
            'board_valid': vector((1,), np.uint8, 0, 1),
            'token_positions': vector((N, 3)), 'token_orientations': vector((N, 4), low=-1, high=1),
            'token_velocities': vector((N, 6)), 'tool_contacts': vector((N,), np.uint8, 0, 1),
            'tool_contact_forces': vector((N,), low=0),
            'joint_positions': vector((6,)), 'joint_velocities': vector((6,)),
            'arm_joint_targets': vector((6,)),
            'end_effector_position': vector((3,)), 'end_effector_orientation': vector((4,), low=-1, high=1),
            'tool_target_position': vector((3,)), 'tool_target_orientation': vector((4,), low=-1, high=1),
            'jaw_position': vector((1,)), 'jaw_velocity': vector((1,)), 'jaw_target': vector((1,), low=0, high=1),
            'steps_remaining': vector((1,), np.int32, 0, max_episode_steps),
        })
        if self.task_mode == 'operation':
            self.observation_space = spaces.Dict({**self.observation_space.spaces,
                'operation_prefix': vector((1,), np.int32, 1, N),
                'operation_stage': vector((1,), np.int32, 0, 3),
                'operation_selected': vector((N,), np.uint8, 0, 1),
                'operation_initial_board': vector((N,), np.int32, 1, N),
                'operation_initial_positions': vector((N, 3)),
                'operation_initial_orientations': vector((N, 4), low=-1, high=1),
                'operation_peak_lift': vector((N,), low=0),
                'operation_peak_displacement': vector((N,), low=0),
                'operation_minimum_normal_dot': vector((N,), low=-1, high=1),
                'operation_violation': vector((1,), np.uint8, 0, 1),
            })
        self._configuration = {'schema_version': 'copuzzle.pancake_manipulation_config.v1',
            'task_name': 'pancake-4-contact-clamp', 'rule_spec': self.rule.spec(),
            'model_sha256': self.scene.model_sha256, 'scene': deepcopy(self.scene.metadata),
            'physics_timestep': physics_dt, 'control_timestep': control_dt, 'substeps': substeps,
            'translation_scale': translation_scale, 'rotation_scale': rotation_scale,
            'max_episode_steps': max_episode_steps, 'mujoco_version': mujoco.__version__,
            'action_semantics': 'XYZ/world rotation increments and absolute normalized jaw closing',
            'controller_state_observations': ['tool_target_position', 'tool_target_orientation',
                                              'arm_joint_targets', 'jaw_target'],
            'success_semantics': 'supported stable target stack after the clamp releases all tokens and withdraws'}
        if self.task_mode == 'operation':
            self._configuration.update({
                'schema_version': 'copuzzle.pancake_operation_config.v1',
                'task_name': f'pancake-4-contact-operation-prefix-{operation_prefix}',
                'task_mode': 'operation', 'operation_prefix': operation_prefix,
                'operation_protocol': {'lift_height_m': .06, 'contact_force_min_n': .05,
                    'inversion_cosine_max': -.90, 'unselected_motion_limit_m': .01,
                    'unselected_normal_cosine_min': .98},
                'success_semantics': 'measured selected-prefix grasp, lift, inversion, supported restack, release and withdrawal',
                'permutation_semantics': 'unsigned prefix reversal; prefix 1 changes orientation but not permutation',
            })
        self.configuration_sha256 = hashlib.sha256(json.dumps(self._configuration, sort_keys=True,
                                                        separators=(',', ':')).encode()).hexdigest()

    def configuration(self):
        return deepcopy(self._configuration)

    def _velocities(self):
        values = []
        for body in self.scene.token_body_ids:
            value = np.empty(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, int(body), value, 0)
            values.append(value)
        return np.array(values)

    def _decode(self):
        p = self.data.xpos[self.scene.token_body_ids]
        order = np.argsort(p[:, 2])[::-1]
        expected = SUPPORT_TOP + BOSS_HALF + PITCH * np.arange(N - 1, -1, -1)
        normals = self.data.xmat[self.scene.token_body_ids].reshape(N, 3, 3)[:, 2, 2]
        valid = (np.max(np.linalg.norm(p[:, :2] - STACK_XY, axis=-1)) < .005
                 and np.max(np.abs(p[order, 2] - expected)) < .0025
                 and np.min(np.abs(normals)) > .98)
        return (order + 1).astype(np.int32), bool(valid)

    def _contacts(self):
        tool = set(self.scene.tool_geom_ids.tolist())
        tokens = {int(geom): index for index, row in enumerate(self.scene.token_geom_ids) for geom in row}
        touching = np.zeros(N, dtype=np.uint8)
        forces = np.zeros(N, dtype=np.float64)
        force = np.empty(6)
        for index, contact in enumerate(self.data.contact[:self.data.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            token = tokens.get(b) if a in tool else tokens.get(a) if b in tool else None
            if token is not None:
                touching[token] = 1
                mujoco.mj_contactForce(self.model, self.data, index, force)
                forces[token] += abs(force[0])
        return touching, forces

    def _observation(self):
        board, valid = self._decode()
        if valid:
            self._board = board.copy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.scene.tool_site_id])
        contacts, forces = self._contacts()
        observation = {'board': self._board.copy(), 'goal': self._goal.copy(),
            'board_valid': np.array([valid], dtype=np.uint8),
            'token_positions': self.data.xpos[self.scene.token_body_ids].copy(),
            'token_orientations': self.data.xquat[self.scene.token_body_ids].copy(),
            'token_velocities': self._velocities(), 'tool_contacts': contacts, 'tool_contact_forces': forces,
            'joint_positions': self.data.qpos[self.scene.arm_qpos_ids].copy(),
            'joint_velocities': self.data.qvel[self.scene.arm_dof_ids].copy(),
            'arm_joint_targets': self.data.ctrl[self.scene.arm_actuator_ids].copy(),
            'end_effector_position': self.data.site_xpos[self.scene.tool_site_id].copy(),
            'end_effector_orientation': quat, 'tool_target_position': self._target_position.copy(),
            'tool_target_orientation': self._target_quaternion.copy(),
            'jaw_position': np.array([self.data.qpos[self.scene.jaw_qpos_id]]),
            'jaw_velocity': np.array([self.data.qvel[self.scene.jaw_dof_id]]),
            'jaw_target': np.array([self._grip]),
            'steps_remaining': np.array([self.max_episode_steps - self._steps], dtype=np.int32)}
        if self.task_mode == 'operation':
            observation.update(self._operation_fields())
        return observation

    def _goal_ready(self, obs):
        return bool(obs['board_valid'][0] and np.array_equal(obs['board'], self._goal)
                    and np.max(np.abs(obs['token_velocities'])) < .015 and not np.any(obs['tool_contacts'])
                    and np.linalg.norm(obs['end_effector_position'][:2] - STACK_XY) > WITHDRAW_DISTANCE)

    def _success(self, obs):
        return self._goal_ready(obs) and (self.task_mode == 'goal' or self._operation_stage == 3)

    @staticmethod
    def _normal_vectors(quaternions):
        w, x, y, z = np.asarray(quaternions).T
        return np.column_stack((2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x*x + y*y)))

    def _reset_operation(self, state):
        self._operation_stage = 0
        self._operation_initial_board = state.copy()
        self._operation_selected = np.isin(np.arange(1, N + 1), state[:self.operation_prefix])
        self._operation_initial_positions = self.data.xpos[self.scene.token_body_ids].copy()
        self._operation_initial_orientations = self.data.xquat[self.scene.token_body_ids].copy()
        self._operation_peak_lift = np.zeros(N)
        self._operation_peak_displacement = np.zeros(N)
        self._operation_minimum_normal_dot = np.ones(N)
        self._operation_violation = False

    def _operation_fields(self):
        return {
            'operation_prefix': np.array([self.operation_prefix], dtype=np.int32),
            'operation_stage': np.array([self._operation_stage], dtype=np.int32),
            'operation_selected': self._operation_selected.astype(np.uint8),
            'operation_initial_board': self._operation_initial_board.copy(),
            'operation_initial_positions': self._operation_initial_positions.copy(),
            'operation_initial_orientations': self._operation_initial_orientations.copy(),
            'operation_peak_lift': self._operation_peak_lift.copy(),
            'operation_peak_displacement': self._operation_peak_displacement.copy(),
            'operation_minimum_normal_dot': self._operation_minimum_normal_dot.copy(),
            'operation_violation': np.array([self._operation_violation], dtype=np.uint8),
        }

    def _update_operation(self, obs):
        """Measured execution evidence only; collector phases are never consulted."""
        protocol = self._configuration['operation_protocol']
        selected = self._operation_selected
        delta = obs['token_positions'] - self._operation_initial_positions
        lift = delta[:, 2]
        displacement = np.linalg.norm(delta, axis=1)
        normal_dot = np.clip(np.sum(self._normal_vectors(obs['token_orientations']) *
            self._normal_vectors(self._operation_initial_orientations), axis=1), -1, 1)
        self._operation_peak_lift = np.maximum(self._operation_peak_lift, lift)
        self._operation_peak_displacement = np.maximum(self._operation_peak_displacement, displacement)
        self._operation_minimum_normal_dot = np.minimum(self._operation_minimum_normal_dot, normal_dot)
        self._operation_violation |= bool(
            np.any(displacement[~selected] > protocol['unselected_motion_limit_m'])
            or np.any(normal_dot[~selected] < protocol['unselected_normal_cosine_min']))
        lifted = np.all(lift[selected] >= protocol['lift_height_m'])
        grasped = (np.any(obs['tool_contacts'][selected])
                   and np.sum(obs['tool_contact_forces'][selected]) > protocol['contact_force_min_n'])
        inverted = np.all(normal_dot[selected] <= protocol['inversion_cosine_max'])
        if self._operation_stage == 0 and lifted and grasped:
            self._operation_stage = 1
        if self._operation_stage >= 1 and lifted and inverted:
            self._operation_stage = 2
        if (self._operation_stage >= 2 and inverted and not self._operation_violation
                and self._goal_ready(obs)):
            self._operation_stage = 3
        obs.update(self._operation_fields())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._closed:
            raise RuntimeError('Environment is closed')
        options = {} if options is None else options
        if not isinstance(options, dict) or set(options) - {'state', 'goal'}:
            raise ValueError('options may contain only state and goal')
        state = self.rule.validate_state(options.get('state', [1, 2, 3, 4]))
        if self.task_mode == 'operation':
            expected = state.copy()
            expected[:self.operation_prefix] = expected[:self.operation_prefix][::-1]
            self._goal = self.rule.validate_state(options.get('goal', expected))
            if not np.array_equal(self._goal, expected):
                raise ValueError('Operation diagnostic goal must equal the requested unsigned prefix reversal')
        else:
            self._goal = self.rule.validate_state(options.get('goal', [2, 1, 3, 4]))
        mujoco.mj_resetData(self.model, self.data)
        self.controller.reset(self._home)
        self.controller.last_position_error, self.controller.last_rotation_error = self._home_ik_errors
        self.data.qpos[self.scene.arm_qpos_ids] = self._home
        self.data.ctrl[self.scene.arm_actuator_ids] = self._home
        for top_index, token in enumerate(state):
            address = self.scene.token_qpos_ids[token - 1]
            self.data.qpos[address:address + 3] = [*STACK_XY,
                SUPPORT_TOP + BOSS_HALF + (N - top_index - 1) * PITCH]
            self.data.qpos[address + 3:address + 7] = [1, 0, 0, 0]
        self._target_position, self._target_quaternion = self.scene.home_tool_position.copy(), self.scene.tool_down_quat.copy()
        self._grip, self._steps = 0., 0
        self._board = state.copy()
        mujoco.mj_forward(self.model, self.data)
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.
        mujoco.mj_forward(self.model, self.data)
        if self.task_mode == 'operation':
            self._reset_operation(state)
        obs = self._observation()
        self._needs_reset = self._success(obs)
        return obs, self._info(obs)

    def _info(self, obs):
        info = {'is_success': self._success(obs), 'steps': self._steps,
                'elapsed_time': float(self.data.time), 'no_token_equalities': self.model.neq == 0,
                'ik_position_error': self.controller.last_position_error,
                'ik_rotation_error': self.controller.last_rotation_error}
        if self.task_mode == 'operation':
            info.update({'task_mode': 'operation', 'operation_prefix': self.operation_prefix,
                         'execution_success': self._success(obs), 'operation_stage': self._operation_stage,
                         'permutation_changed': not np.array_equal(obs['board'], self._operation_initial_board)})
        return info

    def step(self, action):
        if self._closed:
            raise RuntimeError('Environment is closed')
        if self._needs_reset:
            raise ResetNeeded('Reset before stepping or after episode completion')
        action = np.asarray(action)
        if (action.shape != (7,) or action.dtype.kind not in 'fiu' or not np.isfinite(action).all()
                or np.any(np.abs(action[:6]) > 1) or not 0 <= action[6] <= 1):
            raise ValueError('Expected normalized 6D pose increments and a jaw target in [0,1]')
        self._target_position = np.clip(self._target_position + self.translation_scale * action[:3],
                                       *self.scene.workspace_bounds)
        rotation = self.rotation_scale * action[3:6]
        angle = np.linalg.norm(rotation)
        if angle:
            increment = np.r_[np.cos(angle / 2), np.sin(angle / 2) * rotation / angle]
            target = np.empty(4)
            mujoco.mju_mulQuat(target, increment, self._target_quaternion)
            self._target_quaternion = target / np.linalg.norm(target)
        self._grip = float(action[6])
        joints = self.controller.solve(self.data, self._target_position, self._target_quaternion)
        self.data.ctrl[self.scene.arm_actuator_ids] = joints
        self.data.ctrl[self.scene.jaw_actuator_id] = self._grip * JAW_TRAVEL
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._steps += 1
        obs = self._observation()
        if self.task_mode == 'operation':
            self._update_operation(obs)
        success = self._success(obs)
        timeout = self._steps >= self.max_episode_steps and not success
        self._needs_reset = success or timeout
        return obs, 0. if success else -1., success, timeout, self._info(obs)

    def render(self, camera='apparatus'):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=720, width=960)
        options = mujoco.MjvOption()
        options.sitegroup[:] = 0
        self._renderer.update_scene(self.data, camera=camera, scene_option=options)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
        self._renderer, self._closed = None, True
