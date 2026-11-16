"""Generate and fully verify immutable v1 learner/evaluator data bundles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from copuzzle.research.bundle import _effect, _path, _read_npz, _rule_probe
from copuzzle.research.graph import training_components
from . import PROTOCOL
from .graph import distances, full_distances, partition
from .population import _salt, oriented_pair, population, priority
from .spec import (BANK_LIMIT, EXHAUSTIVE_LIMIT, FINAL_DATA_SEEDS, MAX_PER_CELL, canonical, digest,
                   instance, integer, specification)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ["__init__.py", "benchmark/__init__.py", "benchmark/spec.py", "benchmark/graph.py",
             "benchmark/population.py", "benchmark/generation.py", "benchmark/collection.py", "research/spec.py", "research/graph.py",
             "research/bundle.py", *(f"games/{family}/rules.py" for family in ("lights_out", "permutation", "classic"))]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rng(task, regime, seed, purpose):
    return np.random.default_rng(int(digest([PROTOCOL, task, regime, seed, purpose])[:16], 16))


def materialize(task, regime, seed, *, purpose="development", max_per_cell=MAX_PER_CELL, bank_limit=BANK_LIMIT, exhaustive_limit=EXHAUSTIVE_LIMIT):
    seed = integer(seed, "seed")
    if purpose not in ("development", "benchmark"):
        raise ValueError("unknown artifact purpose")
    if purpose == "benchmark" and (seed not in FINAL_DATA_SEEDS or (max_per_cell, bank_limit, exhaustive_limit) != (MAX_PER_CELL, BANK_LIMIT, EXHAUSTIVE_LIMIT)):
        raise ValueError("benchmark data require all prespecified population settings and a final data seed")
    p, spec = partition(task, regime, seed), instance(task)
    g, size = p.graph, len(p.graph.states)
    populations = population(task, regime, seed, max_per_cell, bank_limit, exhaustive_limit)
    if any(cell["eligible_pairs"] == 0 for cell in populations["cells"].values()):
        empty = [name for name, cell in populations["cells"].items() if cell["eligible_pairs"] == 0]
        raise ValueError(f"required population cells are empty: {empty}; revise the protocol, do not replace a prespecified seed")
    queries = p.queries(0)
    _rng(task, regime, seed, "train_order").shuffle(queries)
    transitions = [p.successor(int(s), int(a)) for s, a in queries]
    public = {"train": {
        "observations": np.stack((g.states[queries[:, 0]], g.states[[s for s, _ in transitions]]), axis=1),
        "actions": queries[:, 1, None],
        "legal": np.asarray([flag for _, flag in transitions], np.uint8)[:, None],
    }}
    truth, effects = {}, {}
    for split, label in ((1, "dev"), (2, "test")):
        public[f"R_{label}"], r_truth = _rule_probe(p, split, _rng(task, regime, seed, ["R", label]))
        truth.update({f"R_{label}_{key}": value for key, value in r_truth.items()})
        for probe, strata in (("C", spec.composition_lengths), ("G", range(3))):
            rows = [(code, stratum) for stratum in strata for code in populations["cells"][f"{probe}_{label}_{stratum}"]["pair_codes"]]
            order = np.argsort(priority([code for code, _ in rows], _salt(task, regime, seed, ["row_order", probe, label])))
            rows = [rows[int(i)] for i in order]
            boards, goals, words, prefixes, lengths, empirical, unseen, fingerprints = [], [], [], [], [], [], [], []
            for code, stratum in rows:
                source, target = oriented_pair(task, regime, seed, code, size)
                boards.append(g.states[source])
                goals.append(g.states[target])
                full = full_distances(task, target)
                if probe == "G":
                    lengths.append(int(full[source]))
                    empirical.append(int(distances(g, target, p.owners == 0)[source]))
                    continue
                allowed = p.owners == 0 if regime == "transition_holdout" else None
                d = distances(g, target, allowed, max_depth=stratum) if allowed is not None else full
                if int(full[source]) != stratum or int(d[source]) != stratum:
                    raise ValueError("selected composition is not a supported shortest path")
                actions, states = _path(p, source, d, allowed)
                word = np.full(max(spec.composition_lengths), -1, dtype=np.int32)
                prefix = np.full((len(word), g.states.shape[1]), -1, dtype=np.int32)
                word[:stratum], prefix[:stratum] = actions, states
                words.append(word)
                prefixes.append(prefix)
                lengths.append(stratum)
                previous, missing = source, 0
                for action, state in zip(actions, states):
                    missing += p.owners[previous, p.action_map[action]] != 0
                    previous = g.locate(state)
                unseen.append(int(missing))
                fingerprints.append(_effect(p, actions))
            base = {"board": np.asarray(boards, np.int32)}
            truth[f"{probe}_{label}_pair_codes"] = np.asarray([code for code, _ in rows], np.int64)
            if probe == "C":
                public[f"C_{label}"] = {**base, "actions": np.asarray(words, np.int32), "lengths": np.asarray(lengths, np.int32)}
                truth[f"C_{label}_prefix_boards"] = np.asarray(prefixes, np.int32)
                truth[f"C_{label}_unseen_transitions"] = np.asarray(unseen, np.int32)
                effects[label] = fingerprints
            else:
                public[f"G_{label}"] = {**base, "goal": np.asarray(goals, np.int32)}
                truth[f"G_{label}_distance"] = np.asarray(lengths, np.int32)
                truth[f"G_{label}_tier"] = np.asarray([stratum for _, stratum in rows], np.int32)
                truth[f"G_{label}_empirical_distance"] = np.asarray(empirical, np.int32)
    metadata = {
        "protocol": PROTOCOL, "purpose": purpose, "task": task, "regime": regime, "state_width": g.states.shape[1], "n_actions": g.n_actions,
        "control_budget": spec.control_budget, "training_episode_length": 1,
        "training_episode_end": "collector_cutoff; not a terminal or goal success",
        "invalid_attempt": "unchanged board; legal=false; consumes one step",
        "action_encoding": "opaque integer IDs; mapping belongs to evaluator", "probes": ["R", "C", "G"],
        "rule_evaluation_population": "all retained queries in each held-out split",
        "composition_condition": "supported_primitives" if regime == "transition_holdout" else "unseen_endpoints",
        "sampling": {key: populations[key] for key in ("max_per_cell", "bank_limit", "exhaustive_limit")},
    }
    audit = {
        "population": populations, "canonical_component_states": size,
        "query_counts": {label: int(np.sum(p.owners == i)) for i, label in enumerate(("train", "dev", "test"))},
        "state_partition_counts": {label: int(np.sum(p.state_split == i)) for i, label in enumerate(("train", "dev", "test"))},
        "discarded_cross_state_queries": int(np.sum(p.owners < 0)), "training_source_states": len(set(queries[:, 0].tolist())),
        "training_legal_components": training_components(p), "opaque_to_native": p.action_map.tolist(),
        "composition_effect_sha256": effects,
        "composition_dev_test_effect_overlap": len(set(effects["dev"]) & set(effects["test"])),
        "composition_primitive_effect_overlap": len(set(effects["dev"] + effects["test"]) & {_effect(p, [a]) for a in range(g.n_actions)}),
        "goal_empirically_reachable": {label: int(np.sum(truth[f"G_{label}_empirical_distance"] >= 0)) for label in ("dev", "test")},
        "claims": {"unseen_rule": False, "unseen_size": False, "unseen_family": False,
                   "C_G_dev_test_pair_disjoint": True, "C_G_effect_disjoint": False,
                   "R_state_disjoint": regime == "state_holdout", "G_unseen_endpoints": regime == "state_holdout",
                   "G_entire_path_unseen": False},
    }
    return public, truth, metadata, audit


def write_bundle(output, task, regime, seed, *, purpose="development", max_per_cell=MAX_PER_CELL, bank_limit=BANK_LIMIT, exhaustive_limit=EXHAUSTIVE_LIMIT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source = source_hashes()
    public, truth, metadata, audit = materialize(task, regime, seed, purpose=purpose, max_per_cell=max_per_cell, bank_limit=bank_limit, exhaustive_limit=exhaustive_limit)
    if source_hashes() != source:
        raise RuntimeError("generator source changed during materialization")
    output.mkdir(parents=True)
    (output / "learner").mkdir()
    (output / "evaluator").mkdir()
    for name, arrays in public.items():
        np.savez_compressed(output / "learner" / f"{name}.npz", **arrays)
    np.savez_compressed(output / "evaluator/answers.npz", **truth)
    for name, value in (("learner/metadata.json", metadata), ("evaluator/audit.json", audit)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    files = sorted(p for p in output.rglob("*") if p.is_file())
    manifest = {"protocol": PROTOCOL, "specification_sha256": digest(specification()),
                "task": task, "regime": regime, "seed": int(seed), "purpose": purpose,
                "sampling": metadata["sampling"], "generator_sources": source,
                "files": {p.relative_to(output).as_posix(): file_hash(p) for p in files}}
    if source_hashes() != source:
        raise RuntimeError("generator source changed during bundle writing")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def validate_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if set(manifest) != {"protocol", "specification_sha256", "task", "regime", "seed", "purpose", "sampling", "generator_sources", "files"}:
        raise ValueError("invalid manifest fields")
    if manifest["protocol"] != PROTOCOL or manifest["specification_sha256"] != digest(specification()):
        raise ValueError("benchmark protocol differs")
    if manifest["generator_sources"] != source_hashes():
        raise ValueError("generator source differs; use the frozen source snapshot")
    if set(manifest["sampling"]) != {"max_per_cell", "bank_limit", "exhaustive_limit"}:
        raise ValueError("invalid population settings")
    public, truth, metadata, audit = materialize(manifest["task"], manifest["regime"], manifest["seed"], purpose=manifest["purpose"], **manifest["sampling"])
    expected = {f"learner/{name}.npz" for name in public} | {"learner/metadata.json", "evaluator/answers.npz", "evaluator/audit.json"}
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()} - {"manifest.json"}
    if set(manifest["files"]) != expected or actual != expected:
        raise ValueError("bundle inventory differs")
    for name, checksum in manifest["files"].items():
        if file_hash(directory / name) != checksum:
            raise ValueError(f"checksum mismatch: {name}")
    for name, arrays in [(f"learner/{key}.npz", value) for key, value in public.items()] + [("evaluator/answers.npz", truth)]:
        recorded = _read_npz(directory / name)
        if set(recorded) != set(arrays) or any(recorded[k].dtype != arrays[k].dtype or not np.array_equal(recorded[k], arrays[k]) for k in arrays):
            raise ValueError(f"semantic mismatch: {name}")
    for name, value in (("learner/metadata.json", metadata), ("evaluator/audit.json", audit)):
        if canonical(json.loads((directory / name).read_text())) != canonical(value):
            raise ValueError(f"semantic metadata mismatch: {name}")
    return {"valid": True, "protocol": PROTOCOL, "task": manifest["task"], "regime": manifest["regime"],
            "seed": manifest["seed"], "training_episodes": len(public["train"]["actions"]),
            "evaluation_cases": {key: len(value["board"]) for key, value in public.items() if key != "train"},
            "cell_populations": {key: {field: value[field] for field in ("eligible_pairs", "selected_cases", "exhaustive")}
                                 for key, value in audit["population"]["cells"].items()}}
