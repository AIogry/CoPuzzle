"""Compatibility exports for Lights Out; use copuzzle.games.lights_out.data."""

from .games.lights_out.data import (
    SCHEMA_VERSION,
    GENERATOR_VERSION,
    TrajectoryDataset,
    GoalSampler,
    generate_dataset,
    load_dataset,
    relabel_transitions,
    sample_goal_transitions,
)

__all__ = [
    "SCHEMA_VERSION",
    "GENERATOR_VERSION",
    "TrajectoryDataset",
    "GoalSampler",
    "generate_dataset",
    "load_dataset",
    "relabel_transitions",
    "sample_goal_transitions",
]
