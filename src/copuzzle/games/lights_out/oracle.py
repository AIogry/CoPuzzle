"""Privileged GF(2) diagnostics, not a learned baseline.

The solver reads the true rule. Minimum press count is found by enumerating
the affine solution space, with a configurable nullity limit.
"""

from __future__ import annotations

import operator

import numpy as np

from .rules import PuzzleRule, _binary_array


def _matrix(value: object) -> np.ndarray:
    matrix = _binary_array(value, name="matrix")
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    return matrix


def _eliminate(matrix: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Reduced row echelon form using XOR row operations."""
    matrix = matrix.copy()
    rhs = rhs.copy()
    pivots: list[int] = []
    row = 0
    for column in range(matrix.shape[1]):
        candidates = np.flatnonzero(matrix[row:, column])
        if not candidates.size:
            continue
        pivot_row = row + int(candidates[0])
        if pivot_row != row:
            matrix[[row, pivot_row]] = matrix[[pivot_row, row]]
            rhs[[row, pivot_row]] = rhs[[pivot_row, row]]
        targets = np.flatnonzero(matrix[:, column])
        targets = targets[targets != row]
        matrix[targets] ^= matrix[row]
        rhs[targets] ^= rhs[row]
        pivots.append(column)
        row += 1
        if row == matrix.shape[0]:
            break
    return matrix, rhs, pivots


def rank(matrix: object) -> int:
    """Return binary matrix rank over GF(2), including rectangular matrices."""
    matrix = _matrix(matrix)
    _, _, pivots = _eliminate(matrix, np.zeros(matrix.shape[0], dtype=np.uint8))
    return len(pivots)


def _system(rule: PuzzleRule, start: object, goal: object) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if not isinstance(rule, PuzzleRule):
        raise TypeError("rule must be a PuzzleRule")
    start = rule.validate_state(start, name="start")
    goal = rule.validate_state(goal, name="goal")
    return _eliminate(rule.toggle_matrix, start ^ goal)


def is_reachable(rule: PuzzleRule, start: object, goal: object) -> bool:
    """Check consistency by elimination, without enumerating solutions."""
    _, rhs, pivots = _system(rule, start, goal)
    return not bool(np.any(rhs[len(pivots):]))


def _bits(vector: np.ndarray) -> int:
    return sum(1 << int(index) for index in np.flatnonzero(vector))


def solve(
    rule: PuzzleRule,
    start: object,
    goal: object,
    max_nullity: int = 20,
) -> list[int] | None:
    """Return a shortest press list, or None when the target is unreachable.

    Inputs are never mutated. Presses are returned in ascending button order;
    their order has no effect for these fixed XOR rules. When nullity exceeds
    ``max_nullity``, a reachable system raises ValueError rather than returning
    a potentially nonminimal answer. Use ``is_reachable`` for a cheap check.
    """
    if isinstance(max_nullity, (bool, np.bool_)):
        raise ValueError("max_nullity must be a nonnegative integer")
    try:
        max_nullity = operator.index(max_nullity)
    except TypeError as error:
        raise ValueError("max_nullity must be a nonnegative integer") from error
    if max_nullity < 0:
        raise ValueError("max_nullity must be a nonnegative integer")

    matrix, rhs, pivots = _system(rule, start, goal)
    if np.any(rhs[len(pivots):]):
        return None
    pivot_set = set(pivots)
    free_columns = [column for column in range(rule.n_buttons) if column not in pivot_set]
    nullity = len(free_columns)
    if nullity > max_nullity:
        raise ValueError(
            f"nullity {nullity} exceeds max_nullity={max_nullity}; minimum-press solving "
            f"would enumerate 2**{nullity} solutions. Use is_reachable for reachability only."
        )

    particular = np.zeros(rule.n_buttons, dtype=np.uint8)
    particular[pivots] = rhs[:len(pivots)]
    current = _bits(particular)
    best = current
    best_weight = best.bit_count()
    basis: list[int] = []
    for column in free_columns:
        vector = np.zeros(rule.n_buttons, dtype=np.uint8)
        vector[column] = 1
        vector[pivots] = matrix[:len(pivots), column]
        basis.append(_bits(vector))

    # Gray code changes exactly one nullspace basis vector each iteration.
    for index in range(1, 1 << nullity):
        if best_weight == 0:
            break
        changed_bit = (index & -index).bit_length() - 1
        current ^= basis[changed_bit]
        weight = current.bit_count()
        if weight < best_weight or (weight == best_weight and current < best):
            best, best_weight = current, weight
    return [index for index in range(rule.n_buttons) if best & (1 << index)]
