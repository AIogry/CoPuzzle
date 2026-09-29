"""Compatibility exports for Lights Out; use copuzzle.games.lights_out.rules."""

from .games.lights_out.rules import (
    PuzzleRule,
    grid_rule,
    graph_rule,
    get_rule,
)

__all__ = [
    "PuzzleRule",
    "grid_rule",
    "graph_rule",
    "get_rule",
]
