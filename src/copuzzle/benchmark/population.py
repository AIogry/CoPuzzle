"""Exhaust declared endpoint populations; select distinct cases by hash priority."""
from __future__ import annotations

from functools import lru_cache
from copy import deepcopy
import hashlib

import numpy as np

from copuzzle.research.spec import canonical, instance, integer
from .graph import distances, full_distances, partition


SAMPLING_RECIPE = "copuzzle.reference_pairs.v1"


def _salt(task, regime, seed, purpose):
    return int.from_bytes(hashlib.sha256(canonical([SAMPLING_RECIPE, task, regime, seed, purpose])).digest()[:8], "big")


def priority(codes, salt):
    """A deterministic bijection of uint64 keys (SplitMix-style finalizer).

    This is hash-priority sampling, not a claim of independent uniform draws.
    It is for reproducible case selection, not cryptographic access control.
    """
    x = np.asarray(codes, dtype=np.uint64) ^ np.uint64(salt)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


class Selection:
    def __init__(self, cap, salt):
        self.cap, self.salt = cap, salt
        self.count = 0
        self.codes = np.empty(0, dtype=np.uint64)
        self.scores = np.empty(0, dtype=np.uint64)

    def add(self, codes):
        codes = np.asarray(codes, dtype=np.uint64)
        self.count += len(codes)
        if not len(codes):
            return
        scores = priority(codes, self.salt)
        if len(codes) > self.cap:
            keep = np.argpartition(scores, self.cap - 1)[:self.cap]
            codes, scores = codes[keep], scores[keep]
        codes = np.concatenate((self.codes, codes))
        scores = np.concatenate((self.scores, scores))
        keep = np.argsort(scores)[:self.cap]
        self.codes, self.scores = codes[keep], scores[keep]


@lru_cache(maxsize=12)
def _population(task, regime, seed, max_per_cell, bank_limit, exhaustive_limit):
    """Exact counts on the declared bank-pair universe, never guessed capacities.

    If an endpoint pool fits exhaustive_limit, every state is a reference bank
    member. Otherwise bank_limit members are chosen before distance inspection.
    The universe contains all distinct unordered pairs having at least one bank
    endpoint. Transition holdout uses a shared C/G pair split; state holdout uses
    its disjoint endpoint-state partitions directly.
    """
    seed = integer(seed, "seed")
    max_per_cell = integer(max_per_cell, "max_per_cell", 1)
    bank_limit = integer(bank_limit, "bank_limit", 1)
    exhaustive_limit = integer(exhaustive_limit, "exhaustive_limit", 1)
    p, spec = partition(task, regime, seed), instance(task)
    g, size = p.graph, len(p.graph.states)
    selectors = {(split, "C", length): Selection(max_per_cell, _salt(task, regime, seed, ["C", split, length]))
                 for split in (1, 2) for length in spec.composition_lengths}
    selectors.update({(split, "G", tier): Selection(max_per_cell, _salt(task, regime, seed, ["G", split, tier]))
                      for split in (1, 2) for tier in range(3)})
    banks = {}
    passes = (0,) if regime == "transition_holdout" else (1, 2)
    pair_split_salt = _salt(task, regime, seed, "shared_C_G_split")
    for phase in passes:
        pool = np.arange(size, dtype=np.int32) if phase == 0 else np.flatnonzero(p.state_split == phase)
        bank = pool if len(pool) <= exhaustive_limit else pool[np.argsort(priority(pool, _salt(task, regime, seed, ["bank", phase])))[:bank_limit]]
        bank = np.sort(bank)
        banks[str(phase)] = {"pool_states": len(pool), "bank_states": len(bank), "state_indices": bank.tolist(),
                             "population_kind": "all_endpoint_pairs" if len(bank) == len(pool) else "pairs_with_a_reference_endpoint"}
        in_bank = np.zeros(size, dtype=bool)
        in_bank[bank] = True
        for goal in bank:
            # Include each unordered pair exactly once even when both ends are anchors.
            candidates = pool[(pool != goal) & ((pool > goal) | ~in_bank[pool])]
            d = full_distances(task, int(goal))
            train_d = distances(g, int(goal), p.owners == 0, max_depth=max(spec.composition_lengths)) if phase == 0 else None
            codes = np.minimum(candidates, goal).astype(np.uint64) * np.uint64(size) + np.maximum(candidates, goal).astype(np.uint64)
            owners = 1 + (priority(codes, pair_split_salt) & np.uint64(1)).astype(np.int8) if phase == 0 else np.full(len(codes), phase, np.int8)
            for split in (1, 2) if phase == 0 else (phase,):
                owned = owners == split
                for tier, (lo, hi) in enumerate((spec.easy, spec.medium, spec.hard)):
                    keep = owned & (d[candidates] >= lo) & (d[candidates] <= hi)
                    selectors[split, "G", tier].add(codes[keep])
                for length in spec.composition_lengths:
                    keep = owned & (d[candidates] == length)
                    if train_d is not None:
                        keep &= train_d[candidates] == length
                    selectors[split, "C", length].add(codes[keep])
    cells = {}
    for (split, probe, stratum), selection in selectors.items():
        key = f"{probe}_{'dev' if split == 1 else 'test'}_{stratum}"
        cells[key] = {"eligible_pairs": selection.count, "selected_cases": len(selection.codes),
                      "exhaustive": selection.count <= max_per_cell, "pair_codes": [int(x) for x in selection.codes]}
    return {"recipe": SAMPLING_RECIPE, "task": task, "regime": regime, "seed": seed,
            "max_per_cell": max_per_cell, "bank_limit": bank_limit, "exhaustive_limit": exhaustive_limit,
            "banks": banks, "cells": cells}


def population(task, regime, seed, max_per_cell=32, bank_limit=256, exhaustive_limit=4096):
    """Return an independent audit, with validation outside the reusable cache."""
    values = [integer(seed, "seed"), integer(max_per_cell, "max_per_cell", 1),
              integer(bank_limit, "bank_limit", 1), integer(exhaustive_limit, "exhaustive_limit", 1)]
    return deepcopy(_population(task, regime, *values))


def oriented_pair(task, regime, seed, code, size):
    first, second = divmod(int(code), size)
    swap = int(priority(np.asarray([code]), _salt(task, regime, seed, "orientation"))[0] & np.uint64(1))
    return (second, first) if swap else (first, second)
