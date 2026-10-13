"""Optional physical Lights Out; importing this namespace does not load MuJoCo."""
from importlib import import_module


def make_physics_env(task_id="grid-3x3", **kwargs):
    """Create a contact-controlled, goal-conditioned UR5e environment."""
    from ..rules import get_rule
    from .env import PhysicalPuzzleEnv
    return PhysicalPuzzleEnv(get_rule(task_id), **kwargs)


_EXPORTS = {
    "PhysicalPuzzleEnv": ".env",
    "ButtonSequencePolicy": ".scripted",
    "PhysicsTrajectory": ".data",
    "PhysicsGoalSampler": ".data",
    "load_physics_trajectory": ".data",
    "record_episode": ".data",
    "relabel_physics_transitions": ".data",
    "replay_trajectory": ".replay",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


__all__ = ["make_physics_env", *_EXPORTS]
