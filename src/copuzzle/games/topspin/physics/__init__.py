"""Passive planar TopSpin 20/4 tabletop prototype, independently versioned."""


def make_env(**kwargs):
    from .env import TopSpinPhysicalEnv
    return TopSpinPhysicalEnv(**kwargs)


__all__=['make_env']
