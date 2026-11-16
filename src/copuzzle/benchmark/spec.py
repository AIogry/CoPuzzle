"""Frozen scope and scoring semantics; artifact inventories identify concrete data."""
from dataclasses import asdict

from copuzzle.research.spec import INSTANCES, REGIMES, canonical, digest, instance, integer
from . import PROTOCOL


MAX_PER_CELL = 32
BANK_LIMIT = 256
EXHAUSTIVE_LIMIT = 4096
FINAL_DATA_SEEDS = (101, 202, 303)
FIT_SEEDS = (0, 1, 2)


def specification():
    return {
        "protocol": PROTOCOL,
        "question": "Infer reusable transition structure from offline experience, then compose predictions and control goals.",
        "instances": [{**asdict(s), "control_budget": s.control_budget} for s in INSTANCES],
        "conditions": list(REGIMES), "split_buckets": [70, 15, 15],
        "partition_recipe": "copuzzle.partition.v1",
        "sampling_recipe": "copuzzle.reference_pairs.v1",
        "max_per_cell": MAX_PER_CELL, "bank_limit": BANK_LIMIT, "exhaustive_limit": EXHAUSTIVE_LIMIT,
        "training": "every retained one-step attempt exactly once; collector cutoffs are not terminals",
        "information": "board and opaque action IDs; G adds goal, remaining budget and previous attempted-action legality",
        "R": "all retained held-out queries; primary = mean joint exact accuracy across present legality classes",
        "C": "shortest words; supported primitives in transition holdout, unseen endpoints in state holdout; primary = equal-length mean whole-prefix exact accuracy",
        "G": "closed-loop control with fixed per-instance attempt budget; primary = equal-tier mean success including policy errors",
        "population": "all unordered pairs with at least one reference endpoint; all endpoints if pool <= 4096, otherwise 256 hash-chosen references",
        "selection": "take min(32, eligible population) per cell by hash priority, without replacement; no required cell may be empty",
        "pair_split": "C/G share a dev/test unordered-pair assignment in transition holdout; state holdout uses disjoint endpoint pools",
        "operator_effects": "audited, not held out between dev/test",
        "policy_errors": "retain case, record exception/type, assign failure and full control budget cost; malformed datasets remain fatal",
        "model_budget": "candidate state-action predictions, including unknown/illegal and batched requests; charge before prediction",
        "planning_conditions": {"direct_policies": [0], "model_planners": [32, 128]},
        "planner": {"search": "bounded best-first, slot-mismatch heuristic", "beam_width": 32, "max_depth": 8,
                    "expansion": "batch all remaining action candidates at a popped state up to remaining query budget"},
        "aggregation": "equal length/tier within instance; average Lights Out instances, then weight five families equally",
        "uncertainty": "paired hierarchical resampling of data and fitting seeds; conditional on fixed instances/case inventory",
        "claim_scope": "within-instance deterministic fully observed finite-component generalization; no unseen rules/sizes/families or perception claim",
        "physical_scope": "separate continuous execution extension; witness lengths are not exact abstract difficulty",
    }
