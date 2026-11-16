"""Optional MuJoCo indexed fixtures; import physics only when requested."""

def make_physics_env(task_id='topspin-8-4',**kwargs):
    from .env import PhysicalPermutationEnv
    return PhysicalPermutationEnv(task_id,**kwargs)

__all__=['make_physics_env']
