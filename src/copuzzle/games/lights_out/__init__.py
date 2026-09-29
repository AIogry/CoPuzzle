"""Lights Out environments for fixed grid and graph rules.

This family contains the current binary button games. Its rules, trajectory
schemas, and evaluators are specific to Lights Out.
"""

from .env import PuzzleEnv
from .rules import PuzzleRule, get_rule, graph_rule, grid_rule


def make_env(task_id="grid-3x3", **kwargs):
    """Create a fixed-rule Lights Out environment; graph disclosure is opt-in."""
    return PuzzleEnv(get_rule(task_id), **kwargs)


__all__ = ["PuzzleEnv", "PuzzleRule", "get_rule", "grid_rule", "graph_rule", "make_env"]
