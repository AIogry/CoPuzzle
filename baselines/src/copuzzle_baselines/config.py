"""Strict, serializable development configurations; not frozen paper settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import operator


METHODS = ("gcbc", "gciql", "dynamics")


def integer(value, name, minimum=0):
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def number(value, name, low, high=None, *, open_low=False, open_high=False):
    if isinstance(value, (bool, str)) or type(value).__name__ == "bool_":
        raise ValueError(f"invalid {name}")
    result = float(value)
    if not math.isfinite(result) or result < low or (open_low and result == low):
        raise ValueError(f"invalid {name}")
    if high is not None and (result > high or (open_high and result == high)):
        raise ValueError(f"invalid {name}")
    return result


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class Config:
    method: str = "gciql"
    hidden_dims: tuple[int, ...] = (128, 128)
    learning_rate: float = 3e-4
    discount: float = 0.99
    expectile: float = 0.7
    temperature: float = 3.0
    target_rate: float = 0.005
    value_future_probability: float = 0.5
    actor_future_probability: float = 1.0
    batch_size: int = 256
    coverage: float = 1.0
    coverage_seed: int = 0

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if not isinstance(self.hidden_dims, (list, tuple)) or not self.hidden_dims:
            raise ValueError("hidden_dims must be a nonempty sequence")
        object.__setattr__(self, "hidden_dims", tuple(integer(w, "hidden width", 1) for w in self.hidden_dims))
        for name in ("batch_size", "coverage_seed"):
            object.__setattr__(self, name, integer(getattr(self, name), name, 1 if name == "batch_size" else 0))
        for name in ("discount", "target_rate", "value_future_probability", "actor_future_probability"):
            object.__setattr__(self, name, number(getattr(self, name), name, 0, 1))
        object.__setattr__(self, "expectile", number(self.expectile, "expectile", 0, 1, open_low=True, open_high=True))
        object.__setattr__(self, "learning_rate", number(self.learning_rate, "learning_rate", 0, open_low=True))
        object.__setattr__(self, "temperature", number(self.temperature, "temperature", 0))
        object.__setattr__(self, "coverage", number(self.coverage, "coverage", 0, 1, open_low=True))
        if self.method == "gcbc" and self.actor_future_probability != 1:
            raise ValueError("GCBC requires actor_future_probability=1")

    def to_dict(self):
        value = asdict(self)
        value["hidden_dims"] = list(self.hidden_dims)
        return value

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("unknown configuration fields")
        return cls(**value)
