"""UR5e contact-controlled Lights Out with public logical and physical state.

Continuous commands move a rigid tool. Only contact-gated button displacement
changes the board; intended button IDs are never an environment action.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import operator
from pathlib import Path
import inspect

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import ResetNeeded
import numpy as np

try:
    import mujoco
except ImportError as error:
    raise ImportError('PhysicalPuzzleEnv requires the optional physics dependency: install copuzzle[physics]') from error

from ..oracle import is_reachable
from ..rules import PuzzleRule
from ..structure import export_structure
from .control import DampedIK
from .events import ButtonEventDetector
from .scene import build_scene


PHYSICS_OBSERVATION_PROFILE = 'physical_state_structure_v1'


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a positive integer')
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f'{name} must be a positive integer') from error
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _positive_float(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f'{name} must be a positive finite number')
    return float(value)


def _binary_space(shape):
    space = spaces.MultiBinary(shape)
    space.dtype = np.dtype(np.uint8)
    return space


def _vector_space(shape, *, low=-np.inf, high=np.inf, dtype=np.float64):
    return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


class PhysicalPuzzleEnv(gym.Env):
    """Operate a fixed Lights Out rule by continuous arm control.

    Actions are normalized XYZ increments relative to the measured tool pose.
    Tool orientation stays fixed downward. Goal success depends only on bits,
    evaluated after the complete control interval. All rewards are post-state:
    zero for success, otherwise minus one. Horizon counts control intervals.
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(self, rule: PuzzleRule, max_episode_steps: int = 1500,
                 render_mode: str | None = None, *, physics_dt: float = 0.002,
                 control_dt: float = 0.05, action_scale: float = 0.02,
                 press_depth: float = 0.008, release_depth: float = 0.003,
                 render_width: int = 640, render_height: int = 480):
        super().__init__()
        if not isinstance(rule, PuzzleRule):
            raise TypeError('rule must be a PuzzleRule')
        self.max_episode_steps = _positive_integer(max_episode_steps, 'max_episode_steps')
        self.physics_timestep = _positive_float(physics_dt, 'physics_dt')
        self.control_timestep = _positive_float(control_dt, 'control_dt')
        self.action_scale = _positive_float(action_scale, 'action_scale')
        ratio = self.control_timestep / self.physics_timestep
        self.substeps = int(round(ratio))
        if self.substeps < 1 or not np.isclose(ratio, self.substeps, rtol=0, atol=1e-10):
            raise ValueError('control_dt must be a positive integer multiple of physics_dt')
        if render_mode not in (None, 'rgb_array'):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.render_mode = render_mode
        self.render_width = _positive_integer(render_width, 'render_width')
        self.render_height = _positive_integer(render_height, 'render_height')
        self._events = ButtonEventDetector(rule.n_buttons, press_depth=press_depth,
                                           release_depth=release_depth)
        self.rule = rule
        self.public_structure = export_structure(rule)
        self.model, self.scene = build_scene(rule, physics_dt=self.physics_timestep)
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.render_width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.render_height)
        self.data = mujoco.MjData(self.model)
        self._controller = DampedIK(self.model, self.scene)
        if np.any(self.model.jnt_range[self.scene.button_joint_ids, 1] < self._events.press_depth):
            raise ValueError('press_depth exceeds the physical button travel')
        self.data.qpos[self.scene.arm_qpos_ids] = self.scene.home_qpos
        mujoco.mj_forward(self.model, self.data)
        self._home_joint_positions = self._controller.solve(self.data, self.scene.home_tool_position).copy()
        self.model_sha256 = self.scene.model_sha256
        self.observation_profile = PHYSICS_OBSERVATION_PROFILE
        self.metadata = dict(self.metadata, render_fps=round(1 / self.control_timestep))
        self._renderer = None
        self._closed = False
        self._board = None
        self._goal = None
        self._steps = 0
        self._needs_reset = True
        self._previous_event_counts = np.zeros(rule.n_buttons, dtype=np.int32)
        self._event_log = []
        self._tool_geom_ids = frozenset(map(int, self.scene.tool_geom_ids))
        self._button_for_geom = {int(geom): index for index, geom in enumerate(self.scene.button_geom_ids)}
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        count = rule.n_buttons
        fields = {
            'board': _binary_space(count), 'goal': _binary_space(count),
            'joint_positions': _vector_space((6,)),
            'joint_velocities': _vector_space((6,)),
            'end_effector_position': _vector_space((3,)),
            'end_effector_orientation': _vector_space((4,), low=-1.0, high=1.0),
            'button_positions': _vector_space((count, 3)),
            'button_rest_positions': _vector_space((count, 3)),
            'button_displacements': _vector_space((count,)),
            'button_velocities': _vector_space((count,)),
            'button_armed': _binary_space(count),
            'steps_remaining': _vector_space((1,), low=0, high=self.max_episode_steps, dtype=np.int32),
            'previous_event_counts': _vector_space((count,), low=0,
                                                  high=self.substeps, dtype=np.int32),
        }
        for name, array in self.public_structure.observation_fields().items():
            fields[name] = (_binary_space(array.shape) if name == 'adjacency' else
                            _vector_space(array.shape, low=0, high=np.iinfo(np.int32).max, dtype=np.int32))
        self.observation_space = spaces.Dict(fields)
        self._configuration = {
            'schema_version': 'copuzzle.physics_config.v1',
            'observation_profile': self.observation_profile,
            'task_name': rule.name, 'rule_id': rule.rule_id,
            'family': self.public_structure.family, 'n_buttons': count,
            'structure_sha256': self.public_structure.structure_sha256,
            'model_sha256': self.model_sha256,
            'scene': deepcopy(self.scene.metadata),
            'controller': {
                'algorithm': 'damped_least_squares_fixed_downward_orientation',
                'version': 'copuzzle.damped_ik.v1',
                'source_sha256': hashlib.sha256(Path(inspect.getfile(DampedIK)).read_bytes()).hexdigest(),
                'damping': self._controller.damping,
                'max_iterations': self._controller.max_iterations,
                'max_joint_step': self._controller.max_joint_step,
                'position_tolerance': self._controller.position_tolerance,
                'rotation_tolerance': self._controller.rotation_tolerance,
                'tool_down_quaternion_wxyz': np.asarray(self.scene.tool_down_quat).tolist(),
            },
            'physics_timestep': self.physics_timestep,
            'control_timestep': self.control_timestep, 'substeps': self.substeps,
            'action_scale': self.action_scale, 'action_semantics': 'measured_tool_xyz_plus_scaled_delta_fixed_downward_orientation',
            'workspace_bounds': np.asarray(self.scene.workspace_bounds).tolist(),
            'press_depth': self._events.press_depth, 'release_depth': self._events.release_depth,
            'event_semantics': 'contact_gated_hysteresis_at_every_physics_substep',
            'event_substep_indexing': 'zero_based_within_control_step',
            'max_episode_steps': self.max_episode_steps,
            'success_timing': 'end_of_control_interval',
            'reward': {'success': 0.0, 'otherwise': -1.0},
            'initial_joint_positions': self._home_joint_positions.tolist(),
            'initial_tool_position': np.asarray(self.scene.home_tool_position).tolist(),
            'finite_horizon_observation': 'steps_remaining',
            'button_rest_positions': np.asarray(self.scene.button_rest_positions).tolist(),
            'goal_representation': 'binary_board_only',
            'render_width': self.render_width, 'render_height': self.render_height,
            'mujoco_version': mujoco.__version__,
        }
        canonical = json.dumps(self._configuration, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        self.configuration_sha256 = hashlib.sha256(canonical).hexdigest()

    def configuration(self) -> dict:
        """Return plain preparation/replay metadata, never a solver object."""
        return deepcopy(self._configuration)

    @property
    def button_rest_positions(self) -> np.ndarray:
        return np.array(self.scene.button_rest_positions, dtype=np.float64, copy=True)

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError('This physical environment is closed')

    def _observation(self):
        if self._board is None:
            raise ResetNeeded('Call reset before requesting an observation')
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.scene.tool_site_id])
        # q and -q represent the same pose; choose a stable public convention.
        if quat[0] < 0:
            quat *= -1
        observation = {
            'board': self._board.copy(), 'goal': self._goal.copy(),
            'joint_positions': self.data.qpos[self.scene.arm_qpos_ids].copy(),
            'joint_velocities': self.data.qvel[self.scene.arm_dof_ids].copy(),
            'end_effector_position': self.data.site_xpos[self.scene.tool_site_id].copy(),
            'end_effector_orientation': quat,
            'button_positions': self.data.site_xpos[self.scene.button_site_ids].copy(),
            'button_rest_positions': self.button_rest_positions,
            'button_displacements': self.data.qpos[self.scene.button_qpos_ids].copy(),
            'button_velocities': self.data.qvel[self.scene.button_dof_ids].copy(),
            'button_armed': self._events.armed,
            'previous_event_counts': self._previous_event_counts.copy(),
            'steps_remaining': np.array([self.max_episode_steps - self._steps], dtype=np.int32),
            **self.public_structure.observation_fields(),
        }
        return observation

    def _info(self):
        return {'is_success': bool(np.array_equal(self._board, self._goal)),
                'steps': self._steps, 'elapsed_time': float(self.data.time),
                'rule_id': self.rule.rule_id, 'event_log': deepcopy(self._event_log)}

    def _update_colors(self):
        for index, geom in enumerate(self.scene.button_geom_ids):
            self.model.geom_rgba[geom] = ((0.12, 0.75, 0.25, 1.0) if self._board[index]
                                          else (0.80, 0.16, 0.12, 1.0))

    def reset(self, *, seed=None, options=None):
        self._ensure_open()
        super().reset(seed=seed)
        if options is not None and not isinstance(options, dict):
            raise TypeError('options must be a dictionary or None')
        options = {} if options is None else options
        unknown = set(options) - {'state', 'goal'}
        if unknown:
            raise ValueError(f'Unknown reset option(s): {sorted(unknown)}')
        state = (self.rule.validate_state(options['state']) if 'state' in options else
                 self.np_random.integers(0, 2, self.rule.n_buttons, dtype=np.uint8))
        if 'goal' in options:
            goal = self.rule.validate_state(options['goal'], name='goal')
            if not is_reachable(self.rule, state, goal):
                raise ValueError('goal is unreachable from state under this rule')
        else:
            while True:
                event = self.np_random.integers(0, 2, self.rule.n_buttons, dtype=np.uint8)
                goal = self.rule.apply_event(state, event)
                if not np.array_equal(state, goal):
                    break
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.scene.arm_qpos_ids] = self._home_joint_positions
        self.data.ctrl[self.scene.arm_actuator_ids] = self._home_joint_positions
        mujoco.mj_forward(self.model, self.data)
        self._board, self._goal = state.copy(), goal.copy()
        self._events.reset(self.data.qpos[self.scene.button_qpos_ids])
        self._previous_event_counts.fill(0)
        self._event_log = []
        self._steps = 0
        self._needs_reset = bool(np.array_equal(state, goal))
        self._update_colors()
        return self._observation(), self._info()

    def _contacts(self):
        touching = np.zeros(self.rule.n_buttons, dtype=bool)
        for contact in self.data.contact[:self.data.ncon]:
            if contact.dist > 0:
                continue
            first, second = int(contact.geom1), int(contact.geom2)
            if first in self._tool_geom_ids and second in self._button_for_geom:
                touching[self._button_for_geom[second]] = True
            elif second in self._tool_geom_ids and first in self._button_for_geom:
                touching[self._button_for_geom[first]] = True
        return touching

    def step(self, action):
        self._ensure_open()
        if self._needs_reset:
            raise ResetNeeded('Call reset before stepping or after an episode ends')
        action = np.asarray(action)
        if (action.shape != (3,) or action.dtype.kind not in 'iuf'
                or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1)):
            raise ValueError('action must be a finite XYZ vector in [-1, 1]')
        target = (self.data.site_xpos[self.scene.tool_site_id]
                  + self.action_scale * action.astype(np.float64))
        target = np.clip(target, self.scene.workspace_bounds[0], self.scene.workspace_bounds[1])
        joint_target = self._controller.solve(self.data, target)
        self.data.ctrl[self.scene.arm_actuator_ids] = joint_target
        self._previous_event_counts.fill(0)
        self._event_log = []
        for substep in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            # These contacts generated the impulses for this integration substep.
            fired = self._events.update(self.data.qpos[self.scene.button_qpos_ids], self._contacts())
            for button in fired:
                button = int(button)
                self._board = self.rule.transition(self._board, button)
                self._previous_event_counts[button] += 1
                self._event_log.append({'substep': substep, 'button': button})
        # Integration updates qpos; refresh derived poses for the returned state
        # and for the next command's measured-pose reference.
        mujoco.mj_forward(self.model, self.data)
        if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
            raise RuntimeError('Physics produced nonfinite positions or velocities')
        self._steps += 1
        self._update_colors()
        success = bool(np.array_equal(self._board, self._goal))
        terminated = success
        truncated = self._steps >= self.max_episode_steps and not success
        self._needs_reset = terminated or truncated
        return self._observation(), 0.0 if success else -1.0, terminated, truncated, self._info()

    def render(self):
        self._ensure_open()
        if self.render_mode is None:
            return None
        if self._board is None:
            raise ResetNeeded('Call reset before rendering')
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        self._renderer.update_scene(self.data, camera=getattr(self.scene, 'camera_name', 'overview'))
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._closed = True
