"""Compatibility exports for Lights Out; use copuzzle.games.lights_out.evaluation."""

from .games.lights_out.evaluation import (
    TASK_SCHEMA,
    RESULT_SCHEMA,
    write_json,
    generate_task_suite,
    validate_task_suite,
    load_task_suite,
    RandomPolicy,
    OraclePolicy,
    evaluate_policy,
    audit_overlap,
)

__all__ = [
    "TASK_SCHEMA",
    "RESULT_SCHEMA",
    "write_json",
    "generate_task_suite",
    "validate_task_suite",
    "load_task_suite",
    "RandomPolicy",
    "OraclePolicy",
    "evaluate_policy",
    "audit_overlap",
]
