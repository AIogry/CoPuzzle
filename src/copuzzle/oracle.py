"""Compatibility exports for Lights Out; use copuzzle.games.lights_out.oracle."""

from .games.lights_out.oracle import (
    rank,
    is_reachable,
    solve,
)

__all__ = [
    "rank",
    "is_reachable",
    "solve",
]
