"""Lights Out environments for fixed grid and graph rules.

This family contains the current binary button games. Its rules, trajectory
schemas, and evaluators are specific to Lights Out.
"""

from .env import PuzzleEnv
from .rules import PuzzleRule, get_rule, graph_rule, grid_rule
from .structure import PublicStructure, export_structure, load_structure, save_structure


def make_env(task_id="grid-3x3", **kwargs):
    """Create a fixed rule; opt into public structure with structure_v1."""
    return PuzzleEnv(get_rule(task_id), **kwargs)


__all__ = ["PuzzleEnv", "PuzzleRule", "get_rule", "grid_rule", "graph_rule", "make_env",
           "PublicStructure", "export_structure", "load_structure", "save_structure"]
