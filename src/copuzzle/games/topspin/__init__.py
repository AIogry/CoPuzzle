"""TopSpin: left/right ring shifts and reversal of a fixed prefix window."""

from typing import Any

from ..permutation.env import PermutationEnv
from ..permutation.oracle import ExactOracle
from ..permutation.rules import PermutationRule, get_rule as _get_rule


def get_rule(task_id: str) -> PermutationRule:
    rule = _get_rule(task_id)
    if rule.family != "topspin":
        raise ValueError("expected a topspin-N-K task")
    return rule


def make_env(task_id: str = "topspin-8-4", **kwargs: Any) -> PermutationEnv:
    return PermutationEnv(get_rule(task_id), **kwargs)


__all__ = ["PermutationRule", "PermutationEnv", "ExactOracle", "get_rule", "make_env"]
