"""Contact-gated button switches with separate press and release thresholds."""
from __future__ import annotations

import operator

import numpy as np


class ButtonEventDetector:
    """Detect completed presses at each physics substep.

    A switch fires while armed when its compression reaches ``press_depth``
    and the tool contacts its cap. It must return to ``release_depth`` before
    firing again. Thresholds are in metres; compression is positive downward.
    """

    def __init__(self, n_buttons: int, *, press_depth: float = 0.008,
                 release_depth: float = 0.003):
        if isinstance(n_buttons, (bool, np.bool_)):
            raise ValueError('n_buttons must be a positive integer')
        try:
            n_buttons = operator.index(n_buttons)
        except TypeError as error:
            raise ValueError('n_buttons must be a positive integer') from error
        if n_buttons <= 0:
            raise ValueError('n_buttons must be a positive integer')
        for name, value in (('press_depth', press_depth), ('release_depth', release_depth)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
                raise ValueError(f'{name} must be a finite number')
        if not 0 <= release_depth < press_depth:
            raise ValueError('Require 0 <= release_depth < press_depth')
        self.n_buttons = n_buttons
        self.press_depth = float(press_depth)
        self.release_depth = float(release_depth)
        self._armed = np.ones(n_buttons, dtype=bool)

    @property
    def armed(self) -> np.ndarray:
        """Return a copy of the switch state needed by state observations."""
        return self._armed.astype(np.uint8, copy=True)

    def _depths(self, value) -> np.ndarray:
        depths = np.asarray(value)
        if (depths.shape != (self.n_buttons,) or depths.dtype.kind not in 'iuf'
                or not np.all(np.isfinite(depths))):
            raise ValueError('displacements must be a finite vector of length n_buttons')
        return depths

    def reset(self, displacements=None) -> None:
        """Start without firing; switches already compressed remain disarmed."""
        if displacements is None:
            self._armed.fill(True)
        else:
            self._armed = self._depths(displacements) <= self.release_depth

    def update(self, displacements, contacts) -> np.ndarray:
        """Consume one substep and return fired button IDs in index order."""
        depths = self._depths(displacements)
        touching = np.asarray(contacts)
        if (touching.shape != (self.n_buttons,) or touching.dtype.kind not in 'biuf'
                or not np.all((touching == 0) | (touching == 1))):
            raise ValueError('contacts must be a binary vector of length n_buttons')
        self._armed[depths <= self.release_depth] = True
        fired = self._armed & (depths >= self.press_depth) & touching.astype(bool)
        self._armed[fired] = False
        return np.flatnonzero(fired)
