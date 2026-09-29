"""Binary Lights Out rules, independent of a simulator or learning method.

Columns of ``toggle_matrix`` are operation effects: entry ``[j, i]`` says
whether pressing button i flips light j. Graphs are simple and undirected;
every operation also flips its own light.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import operator
import re

import numpy as np


def _binary_array(value: object, *, name: str) -> np.ndarray:
    """Validate before casting, so fractional and overflowing values fail."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular binary array") from error
    if array.dtype.kind not in "biuf" or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary values 0 or 1")
    return array.astype(np.uint8, copy=True)


def _immutable(array: np.ndarray) -> np.ndarray:
    # A bytes-backed array cannot be made writeable again with setflags().
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


@dataclass(frozen=True, eq=False)
class PuzzleRule:
    """One fixed rule, with optional display coordinates.

    ``rule_id`` identifies the ordered logical graph. Names and coordinates
    do not affect it. Node relabeling generally changes the identifier.
    Arrays are copied and made immutable during construction.
    """

    name: str
    adjacency: np.ndarray
    coordinates: np.ndarray | None = None
    toggle_matrix: np.ndarray = field(init=False, repr=False)
    rule_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        adjacency = _binary_array(self.adjacency, name="adjacency")
        if (
            adjacency.ndim != 2
            or adjacency.shape[0] == 0
            or adjacency.shape[0] != adjacency.shape[1]
        ):
            raise ValueError("adjacency must be a nonempty square matrix")
        if np.any(np.diag(adjacency)):
            raise ValueError("adjacency must have no self-loops (zero diagonal)")
        if not np.array_equal(adjacency, adjacency.T):
            raise ValueError("adjacency must be symmetric for an undirected graph")
        n_buttons = adjacency.shape[0]
        toggle_matrix = adjacency ^ np.eye(n_buttons, dtype=np.uint8)
        digest = hashlib.sha256()
        digest.update(b"copuzzle/undirected-self-toggle/v1\0")
        digest.update(str(n_buttons).encode("ascii") + b"\0")
        digest.update(adjacency.tobytes(order="C"))
        object.__setattr__(self, "adjacency", _immutable(adjacency))
        object.__setattr__(self, "toggle_matrix", _immutable(toggle_matrix))
        object.__setattr__(self, "rule_id", digest.hexdigest())

        if self.coordinates is not None:
            try:
                coordinates = np.asarray(self.coordinates)
            except (TypeError, ValueError) as error:
                raise ValueError("coordinates must be a finite numeric array") from error
            if (
                coordinates.ndim != 2
                or coordinates.shape[0] != n_buttons
                or coordinates.shape[1] not in (2, 3)
                or coordinates.dtype.kind not in "iuf"
                or not np.all(np.isfinite(coordinates))
            ):
                raise ValueError("coordinates must have shape (n_buttons, 2 or 3) and finite values")
            object.__setattr__(self, "coordinates", _immutable(coordinates.astype(np.float64)))

    @property
    def n_buttons(self) -> int:
        return self.adjacency.shape[0]

    def validate_state(self, value: object, name: str = "state") -> np.ndarray:
        """Return an independent uint8 vector after strict binary validation."""
        state = _binary_array(value, name=name)
        if state.shape != (self.n_buttons,):
            raise ValueError(f"{name} must have shape ({self.n_buttons},), got {state.shape}")
        return state

    def transition(self, state: object, action: int) -> np.ndarray:
        """Apply one button press without mutating the input state."""
        board = self.validate_state(state)
        if isinstance(action, (bool, np.bool_)):
            raise ValueError("action must be an integer button index")
        try:
            action = operator.index(action)
        except TypeError as error:
            raise ValueError("action must be an integer button index") from error
        if not 0 <= action < self.n_buttons:
            raise ValueError(f"action must be in [0, {self.n_buttons}), got {action}")
        return board ^ self.toggle_matrix[:, action]

    def effect(self, event: object) -> np.ndarray:
        """Return the XOR effect of a binary press-parity vector."""
        parity = self.validate_state(event, name="event")
        columns = self.toggle_matrix[:, parity.astype(bool)]
        return np.bitwise_xor.reduce(columns, axis=1, initial=np.uint8(0))

    def apply_event(self, state: object, event: object) -> np.ndarray:
        """Apply press parity; this vector is distinct from the resulting effect."""
        return self.validate_state(state) ^ self.effect(event)


def grid_rule(rows: int, cols: int) -> PuzzleRule:
    """Create a rectangular grid with row-major button indices."""
    rows = _positive_integer(rows, name="rows")
    cols = _positive_integer(cols, name="cols")
    n_buttons = rows * cols
    adjacency = np.zeros((n_buttons, n_buttons), dtype=np.uint8)
    coordinates = np.empty((n_buttons, 2), dtype=np.float64)
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            coordinates[index] = (col, row)
            for next_row, next_col in ((row + 1, col), (row, col + 1)):
                if next_row < rows and next_col < cols:
                    neighbor = next_row * cols + next_col
                    adjacency[index, neighbor] = adjacency[neighbor, index] = 1
    return PuzzleRule(f"grid-{rows}x{cols}", adjacency, coordinates)


def graph_rule(
    adjacency: object,
    name: str = "custom",
    coordinates: object | None = None,
) -> PuzzleRule:
    """Create a fixed simple undirected graph rule."""
    return PuzzleRule(name=name, adjacency=adjacency, coordinates=coordinates)


def get_rule(task_id: str) -> PuzzleRule:
    """Resolve grid-RxC, graph-path9, or graph-cycle9 task identifiers."""
    if not isinstance(task_id, str):
        raise ValueError("task_id must be a string")
    match = re.fullmatch(r"grid-([1-9][0-9]*)x([1-9][0-9]*)", task_id)
    if match:
        return grid_rule(int(match.group(1)), int(match.group(2)))
    if task_id in {"graph-path9", "graph-cycle9"}:
        n_buttons = 9
        adjacency = np.zeros((n_buttons, n_buttons), dtype=np.uint8)
        for index in range(n_buttons - 1):
            adjacency[index, index + 1] = adjacency[index + 1, index] = 1
        if task_id == "graph-cycle9":
            adjacency[0, -1] = adjacency[-1, 0] = 1
            angles = 2 * np.pi * np.arange(n_buttons) / n_buttons
            coordinates = np.column_stack((np.cos(angles), np.sin(angles)))
        else:
            coordinates = np.column_stack((np.arange(n_buttons), np.zeros(n_buttons)))
        return graph_rule(adjacency, name=task_id, coordinates=coordinates)
    raise ValueError(f"unknown task_id {task_id!r}; expected grid-RxC, graph-path9, or graph-cycle9")
