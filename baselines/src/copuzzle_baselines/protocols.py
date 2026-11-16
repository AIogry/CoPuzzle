"""Explicit protocol dispatch; the training path imports public data readers only."""
from __future__ import annotations

import json
from pathlib import Path


def protocol(directory):
    value = json.loads((Path(directory) / "metadata.json").read_text()).get("protocol")
    if value not in ("copuzzle.rule-composition.rc1", "copuzzle.rule-composition.v1"):
        raise ValueError("unsupported learner protocol")
    return value


def public_api(directory):
    if protocol(directory) == "copuzzle.rule-composition.v1":
        from copuzzle.benchmark import data
    else:
        from copuzzle.research import data
    return data


def evaluator_api(bundle):
    if protocol(Path(bundle) / "learner") == "copuzzle.rule-composition.v1":
        from copuzzle.benchmark import evaluation, generation
    else:
        from copuzzle.research import evaluation, bundle as generation
    return generation, evaluation
