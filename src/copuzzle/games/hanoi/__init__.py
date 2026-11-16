"""Tower of Hanoi; disk index zero is smallest and peg labels are 0, 1, 2."""

from typing import Any

from ..classic.env import ClassicEnv
from ..classic.oracle import ExactOracle
from ..classic.rules import ClassicRule, get_rule as _get_rule


def get_rule(task_id: str) -> ClassicRule:
    rule = _get_rule(task_id)
    if rule.family != "hanoi":
        raise ValueError("expected a hanoi-N task")
    return rule


def make_env(task_id: str = "hanoi-3", **kwargs: Any) -> ClassicEnv:
    return ClassicEnv(get_rule(task_id), **kwargs)


__all__ = ["ClassicRule", "ClassicEnv", "ExactOracle", "get_rule", "make_env"]
