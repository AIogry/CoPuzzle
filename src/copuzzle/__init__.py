"""CoPuzzle: independent goal-conditioned puzzles and offline data."""
from gymnasium.envs.registration import register, registry

from copuzzle.games.lights_out import PuzzleEnv, PuzzleRule, graph_rule, grid_rule
from copuzzle.games.lights_out import get_rule as _lights_out_rule, make_env as _lights_out_env
from copuzzle.games.lights_out.presets import TASK_PRESETS
from copuzzle.games.lights_out.structure import PublicStructure, export_structure, load_structure, save_structure

__version__ = "0.1.0.dev3"


def get_rule(task_id="grid-3x3"):
    """Resolve a Lights Out, Pancake or fixed-position TopSpin task."""
    if isinstance(task_id, str) and task_id.startswith(("sliding-", "hanoi-")):
        from copuzzle.games.classic.rules import get_rule as classic_rule
        return classic_rule(task_id)
    if isinstance(task_id, str) and task_id.startswith(("pancake-", "topspin-")):
        from copuzzle.games.permutation.rules import get_rule as permutation_rule
        return permutation_rule(task_id)
    return _lights_out_rule(task_id)


def make_env(task_id="grid-3x3", **kwargs):
    """Create a goal-conditioned environment with the family's own contract."""
    if isinstance(task_id, str) and task_id.startswith(("sliding-", "hanoi-")):
        from copuzzle.games.classic import make_env as classic_env
        return classic_env(task_id, **kwargs)
    if isinstance(task_id, str) and task_id.startswith(("pancake-", "topspin-")):
        from copuzzle.games.permutation import make_env as permutation_env
        return permutation_env(task_id, **kwargs)
    return _lights_out_env(task_id, **kwargs)


def make_physics_env(task_id="grid-3x3", **kwargs):
    """Load optional MuJoCo support only when a physical environment is created."""
    if isinstance(task_id, str) and task_id.startswith(("pancake-", "topspin-")):
        from copuzzle.games.permutation.physics import make_physics_env as factory
    else:
        from copuzzle.games.lights_out.physics import make_physics_env as factory
    return factory(task_id, **kwargs)


def make_manipulation_env(task_id, **kwargs):
    """Create the direct-contact tabletop models, separately from old fixtures.

    TopSpin uses the physical 20/4 track and turntable; Pancake uses a real
    contact clamp on four supported rigid sorting discs. Their actions and
    trajectory contracts differ from the archived three-rack fixture.
    """
    if task_id == "topspin-20-4":
        from copuzzle.games.topspin.physics import make_env as factory
    elif task_id == "pancake-4":
        from copuzzle.games.pancake.physics import make_physics_env as factory
    else:
        raise ValueError("Direct manipulation supports topspin-20-4 and pancake-4")
    return factory(**kwargs)


for _preset in TASK_PRESETS:
    if _preset.gym_id not in registry:
        register(id=_preset.gym_id, entry_point="copuzzle:make_env",
                 kwargs={"task_id": _preset.task_id})
    _physics_id = _preset.gym_id.replace("CoPuzzle/", "CoPuzzle/Physical")
    if _physics_id not in registry:
        register(id=_physics_id, entry_point="copuzzle:make_physics_env",
                 kwargs={"task_id": _preset.task_id})

for _task_id, _gym_id in (
    ("pancake-4", "CoPuzzle/Pancake4-v0"),
    ("pancake-6", "CoPuzzle/Pancake6-v0"),
    ("pancake-8", "CoPuzzle/Pancake8-v0"),
    ("topspin-8-4", "CoPuzzle/TopSpin8x4-v0"),
    ("sliding-2x3", "CoPuzzle/Sliding2x3-v0"),
    ("sliding-3x3", "CoPuzzle/Sliding3x3-v0"),
    ("hanoi-3", "CoPuzzle/Hanoi3-v0"),
    ("hanoi-4", "CoPuzzle/Hanoi4-v0"),
    ("hanoi-5", "CoPuzzle/Hanoi5-v0"),
):
    if _gym_id not in registry:
        register(id=_gym_id, entry_point="copuzzle:make_env", kwargs={"task_id": _task_id})

for _task_id, _gym_id in (("topspin-8-4", "CoPuzzle/PhysicalTopSpin8x4-v0"),
                          ("pancake-4", "CoPuzzle/PhysicalPancake4-v0")):
    if _gym_id not in registry:
        register(id=_gym_id, entry_point="copuzzle:make_physics_env", kwargs={"task_id": _task_id})

for _task_id, _gym_id in (("topspin-20-4", "CoPuzzle/TopSpin20x4Contact-v0"),
                          ("pancake-4", "CoPuzzle/Pancake4Contact-v0")):
    if _gym_id not in registry:
        register(id=_gym_id, entry_point="copuzzle:make_manipulation_env", kwargs={"task_id": _task_id})

__all__ = ["PuzzleEnv", "PuzzleRule", "get_rule", "graph_rule", "grid_rule", "make_env",
           "PublicStructure", "export_structure", "load_structure", "save_structure", "make_physics_env",
           "make_manipulation_env"]
