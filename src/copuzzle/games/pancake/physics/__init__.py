"""Contact-driven rigid Pancake manipulation; optional MuJoCo dependency."""


def make_physics_env(**kwargs):
    from .env import PancakeManipulationEnv
    return PancakeManipulationEnv(**kwargs)


__all__ = ['make_physics_env']
