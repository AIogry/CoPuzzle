"""CoPuzzle: independent goal-conditioned puzzles and offline data."""
from gymnasium.envs.registration import register, registry

from copuzzle.games.lights_out import PuzzleEnv, PuzzleRule, get_rule, graph_rule, grid_rule, make_env
from copuzzle.games.lights_out.presets import TASK_PRESETS
from copuzzle.games.lights_out.structure import PublicStructure, export_structure, load_structure, save_structure

__version__ = "0.1.0.dev3"


def make_physics_env(task_id="grid-3x3", **kwargs):
    """Load optional MuJoCo support only when a physical environment is created."""
    from copuzzle.games.lights_out.physics import make_physics_env as factory
    return factory(task_id, **kwargs)


for _preset in TASK_PRESETS:
    if _preset.gym_id not in registry:
        register(id=_preset.gym_id, entry_point="copuzzle:make_env",
                 kwargs={"task_id": _preset.task_id})
    _physics_id = _preset.gym_id.replace("CoPuzzle/", "CoPuzzle/Physical")
    if _physics_id not in registry:
        register(id=_physics_id, entry_point="copuzzle:make_physics_env",
                 kwargs={"task_id": _preset.task_id})

__all__ = ["PuzzleEnv", "PuzzleRule", "get_rule", "graph_rule", "grid_rule", "make_env",
           "PublicStructure", "export_structure", "load_structure", "save_structure", "make_physics_env"]
