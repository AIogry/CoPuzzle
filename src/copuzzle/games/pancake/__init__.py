"""Unburnt Pancake: action a reverses the first a+2 of N tokens."""

from typing import Any

from ..permutation.env import PermutationEnv
from ..permutation.oracle import ExactOracle
from ..permutation.rules import PermutationRule, get_rule as _get_rule


def get_rule(task_id: str) -> PermutationRule:
    rule = _get_rule(task_id)
    if rule.family != "pancake":
        raise ValueError("expected a pancake-N task")
    return rule


def make_env(task_id: str = "pancake-6", **kwargs: Any) -> PermutationEnv:
    return PermutationEnv(get_rule(task_id), **kwargs)


__all__ = ["PermutationRule", "PermutationEnv", "ExactOracle", "get_rule", "make_env"]
