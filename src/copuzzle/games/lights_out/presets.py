"""Named development tasks and explicit evaluation-depth suggestions.

These presets are a catalog, not a frozen cross-size or cross-graph protocol.
Explicit depths remain available for every parameterized rule. Existing CLI
calls retain their original depths unless the caller selects these presets.
Full-rank tasks omit depth N: that boundary always presses every button and
is a weak sole measure of goal-dependent decision making.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskPreset:
    task_id: str
    gym_id: str
    evaluation_depths: tuple[int, ...]


TASK_PRESETS = (
    TaskPreset("grid-3x3", "CoPuzzle/Grid3x3-v0", (1, 2, 3, 4, 6, 8)),
    TaskPreset("grid-4x4", "CoPuzzle/Grid4x4-v0", (1, 2, 3, 4, 5, 6, 7)),
    TaskPreset("grid-4x5", "CoPuzzle/Grid4x5-v0", (1, 2, 3, 4, 8, 12, 16, 18)),
    TaskPreset("grid-4x6", "CoPuzzle/Grid4x6-v0", (1, 2, 3, 4, 8, 12, 16, 20, 22)),
    TaskPreset("graph-path9", "CoPuzzle/GraphPath9-v0", (1, 2, 3, 4, 6, 8)),
    TaskPreset("graph-path16", "CoPuzzle/GraphPath16-v0", (1, 2, 3, 4, 8, 12, 14)),
    TaskPreset("graph-path20", "CoPuzzle/GraphPath20-v0", (1, 2, 3, 4, 8, 10, 12, 13)),
    TaskPreset("graph-path24", "CoPuzzle/GraphPath24-v0", (1, 2, 3, 4, 8, 12, 16, 20, 22)),
    TaskPreset("graph-cycle9", "CoPuzzle/GraphCycle9-v0", (1, 2, 3, 4)),
    TaskPreset("graph-cycle16", "CoPuzzle/GraphCycle16-v0", (1, 2, 3, 4, 8, 12, 14)),
    TaskPreset("graph-cycle20", "CoPuzzle/GraphCycle20-v0", (1, 2, 3, 4, 8, 12, 16, 18)),
    TaskPreset("graph-cycle24", "CoPuzzle/GraphCycle24-v0", (1, 2, 3, 4, 6, 8, 10, 12)),
)


def get_task_preset(task_id: str) -> TaskPreset:
    """Look up suggested settings; other valid rules need explicit settings."""
    for preset in TASK_PRESETS:
        if preset.task_id == task_id:
            return preset
    raise ValueError(f"No evaluation preset for {task_id!r}; specify explicit depths")


__all__ = ["TaskPreset", "TASK_PRESETS", "get_task_preset"]
