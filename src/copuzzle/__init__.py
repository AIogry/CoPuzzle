"""CoPuzzle: independent abstract puzzle environments and offline data."""
from gymnasium.envs.registration import register, registry

from copuzzle.games.lights_out import PuzzleEnv, PuzzleRule, get_rule, graph_rule, grid_rule, make_env

__version__ = "0.1.0.dev1"


for _gym_id, _task in {
    "CoPuzzle/Grid3x3-v0": "grid-3x3",
    "CoPuzzle/Grid4x4-v0": "grid-4x4",
    "CoPuzzle/GraphPath9-v0": "graph-path9",
    "CoPuzzle/GraphCycle9-v0": "graph-cycle9",
}.items():
    if _gym_id not in registry:
        register(id=_gym_id, entry_point="copuzzle:make_env", kwargs={"task_id": _task})

__all__ = ["PuzzleEnv", "PuzzleRule", "get_rule", "graph_rule", "grid_rule", "make_env"]
