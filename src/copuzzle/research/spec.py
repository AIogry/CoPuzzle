"""Executable scope of the research protocol; independent of legacy presets."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import operator


PROTOCOL = "copuzzle.rule-composition.rc1"
REGIMES = ("transition_holdout", "state_holdout")
SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class Instance:
    task: str
    family: str
    component_states: int
    diameter: int
    # Inclusive exact shortest-path distances, in atomic legal operations.
    easy: tuple[int, int]
    medium: tuple[int, int]
    hard: tuple[int, int]
    composition_lengths: tuple[int, ...]

    @property
    def control_budget(self) -> int:
        # Fixed for the whole instance: never reveal a case's hidden distance.
        return 2 * self.diameter


INSTANCES = (
    Instance("grid-3x3", "lights_out", 512, 9, (1, 3), (4, 6), (7, 9), (2, 4, 8)),
    Instance("grid-4x4", "lights_out", 4096, 7, (1, 2), (3, 5), (6, 7), (2, 4, 6)),
    Instance("pancake-6", "pancake", 720, 7, (1, 2), (3, 4), (5, 7), (2, 4, 6)),
    Instance("topspin-8-4", "topspin", 40320, 22, (1, 7), (8, 14), (15, 22), (2, 4, 8)),
    Instance("sliding-2x3", "sliding", 360, 21, (1, 7), (8, 14), (15, 21), (2, 4, 8)),
    Instance("hanoi-5", "hanoi", 243, 31, (1, 10), (11, 20), (21, 31), (2, 4, 8)),
)


def instance(task: str) -> Instance:
    for item in INSTANCES:
        if item.task == task:
            return item
    raise ValueError(f"task must be one of {[item.task for item in INSTANCES]}")


def integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def specification() -> dict:
    return {
        "protocol": PROTOCOL,
        "status": "release-candidate; generated suites are development qualification",
        "question": "Infer reusable transition structure from experience; compose it for prediction and goal control.",
        "instances": [{**asdict(item), "control_budget": item.control_budget} for item in INSTANCES],
        "regimes": list(REGIMES), "split_hash_buckets": [70, 15, 15],
        "training": "exhaustive retained one-step attempt episodes; no goal or oracle labels",
        "R": "exhaustive held-out one-step next-state and legality prediction; report legal/illegal counts separately",
        "C": "open-loop shortest sequences; supported primitives in transition_holdout, unseen endpoints in state_holdout",
        "G": "closed-loop goal control; exact-distance strata; fixed per-instance attempt budget",
        "information": "board and opaque action IDs; G additionally receives goal; no true action mask",
        "invalid_attempt": "in-range illegal action consumes a step, leaves state unchanged, legality=false",
        "primary_claim": "within-instance generalization; not unseen rules, sizes, or game families",
        "physical_extension": "separate execution diagnostics; not part of this exact finite-state core",
        "seed_policy": "report all dataset, fit and evaluation seeds; no best-seed selection",
    }
