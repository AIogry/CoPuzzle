"""Observation-only reference executor for testing physical button sequences.

The caller supplies an operation sequence. This is a diagnostic controller,
not a learned policy, planner, or offline RL result. Every board change still
requires physical contact in the environment.
"""
from __future__ import annotations

import operator
import numpy as np


class ButtonSequencePolicy:
    """Lift, travel, press, and release using only public physical observations."""

    def __init__(self, buttons, *, action_scale=0.02, clearance=0.07,
                 compression=0.011, tolerance=0.003, phase_timeout=200):
        sequence = []
        for button in buttons:
            if isinstance(button, (bool, np.bool_)):
                raise ValueError("buttons must contain nonnegative integer IDs")
            try:
                button = operator.index(button)
            except TypeError as error:
                raise ValueError("buttons must contain nonnegative integer IDs") from error
            if button < 0:
                raise ValueError("buttons must contain nonnegative integer IDs")
            sequence.append(button)
        for name, value in (("action_scale", action_scale), ("clearance", clearance),
                            ("compression", compression), ("tolerance", tolerance)):
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if type(phase_timeout) is not int or phase_timeout < 1:
            raise ValueError("phase_timeout must be a positive integer")
        self.buttons = tuple(sequence)
        self.action_scale = float(action_scale)
        self.clearance = float(clearance)
        self.compression = float(compression)
        self.tolerance = float(tolerance)
        self.phase_timeout = phase_timeout
        self.reset()

    def reset(self):
        self.index = 0
        self.phase = "lift"
        self.phase_steps = 0
        self.confirmed_buttons = []
        self._lift_xy = None

    @property
    def completed(self):
        return self.index == len(self.buttons)

    def _set_phase(self, phase):
        self.phase, self.phase_steps = phase, 0

    def __call__(self, observation):
        tip = np.asarray(observation["end_effector_position"], dtype=float)
        rest = np.asarray(observation["button_rest_positions"], dtype=float)
        if self.completed:
            return np.zeros(3, dtype=np.float32)
        button = self.buttons[self.index]
        if button >= len(rest):
            raise ValueError(f"button {button} is outside this scene")
        if self._lift_xy is None:
            self._lift_xy = tip[:2].copy()
        hover = rest[button].copy()
        hover[2] += self.clearance
        self.phase_steps += 1
        if self.phase_steps > self.phase_timeout:
            raise RuntimeError(f"Button {button}: {self.phase} did not finish within {self.phase_timeout} control steps")
        if self.phase == "lift":
            target = np.array([*self._lift_xy, hover[2]])
            if abs(tip[2] - hover[2]) < self.tolerance:
                self._set_phase("travel")
        elif self.phase == "travel":
            target = hover
            if np.linalg.norm(tip - hover) < self.tolerance:
                self._set_phase("press")
        elif self.phase == "press":
            target = rest[button].copy()
            target[2] -= self.compression
            if observation["previous_event_counts"][button] > 0:
                self.confirmed_buttons.append(button)
                self._set_phase("release")
                target = hover
        else:
            target = hover
            if (np.linalg.norm(tip - hover) < self.tolerance
                    and observation["button_armed"][button]):
                self.index += 1
                self._lift_xy = tip[:2].copy()
                self._set_phase("lift")
        return np.clip((target - tip) / self.action_scale, -1, 1).astype(np.float32)
