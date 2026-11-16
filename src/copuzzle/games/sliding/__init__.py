"""Bounded sliding tiles; actions move the blank up, down, left, right."""

from typing import Any

from ..classic.env import ClassicEnv
from ..classic.oracle import ExactOracle
from ..classic.rules import ClassicRule, get_rule as _get_rule


def get_rule(task_id: str) -> ClassicRule:
    rule = _get_rule(task_id)
    if rule.family != "sliding":
        raise ValueError("expected a sliding-RxC task")
    return rule


def make_env(task_id: str = "sliding-2x3", **kwargs: Any) -> ClassicEnv:
    return ClassicEnv(get_rule(task_id), **kwargs)


__all__ = ["ClassicRule", "ClassicEnv", "ExactOracle", "get_rule", "make_env"]
