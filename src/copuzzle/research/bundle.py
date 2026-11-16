"""Generate and validate evaluator bundles for rule, composition and goal probes.

Only ``learner/`` belongs in a learner's data mount. Everything else is privileged.
This is a reproducible development protocol, not a security sandbox.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .graph import (checked_diameter, distances, exact_distances, immutable,
                    partition, training_components)
from .spec import PROTOCOL, SPLITS, canonical, digest, instance, integer, specification


def _rng(task, regime, seed, purpose):
    entropy = int(digest([PROTOCOL, task, regime, seed, purpose])[:16], 16)
    return np.random.default_rng(entropy)


def _path(p, source, distance, allowed=None):
    """Deterministic shortest path in opaque action order."""
    boards, actions = [], []
    while distance[source] > 0:
        for action, native in enumerate(p.action_map):
            target = int(p.graph.successors[source, native])
            if target >= 0 and distance[target] == distance[source] - 1:
                if allowed is None or allowed[source, native]:
                    actions.append(action)
                    boards.append(p.graph.states[target])
                    source = target
                    break
        else:
            raise ValueError("shortest path has no admissible descending edge")
    return actions, boards


def _effect(p, actions):
    # Fingerprint the *partial* composed operator on the ENTIRE component.
    # Invalid prefixes stay undefined, even if later attempted moves would recover.
    current = np.arange(len(p.graph.states), dtype=np.int32)
    valid = np.ones(len(current), dtype=bool)
    for action in actions:
        target = p.graph.successors[current, p.action_map[action]]
        valid &= target >= 0
        current = np.where(valid, target, 0)
    return hashlib.sha256(np.where(valid, current, -1).astype("<i4").tobytes()).hexdigest()


def _rule_probe(p, split, rng):
    # Exhaust the actual held-out population. Small state partitions may have
    # very few legal edges; fixed-count subsampling would reject valid seeds.
    selected = p.queries(split)
    if not len(selected):
        raise ValueError("empty held-out query population")
    rng.shuffle(selected)
    sources, actions = selected.T
    successors = [p.successor(int(s), int(a)) for s, a in selected]
    return ({"board": p.graph.states[sources], "action": actions},
            {"next_board": p.graph.states[[s for s, _ in successors]],
             "legal": np.asarray([legal for _, legal in successors], dtype=np.uint8)})


def _goals(p, split, count, rng, occupied):
    spec, g = instance(p.graph.task), p.graph
    pool = np.arange(len(g.states)) if p.regime == "transition_holdout" else np.flatnonzero(p.state_split == split)
    starts, goals, depths, tiers, empirical_depths = [], [], [], [], []
    for tier, (lo, hi) in enumerate((spec.easy, spec.medium, spec.hard)):
        accepted = 0
        for _ in range(max(1000, count * 100)):
            goal = int(rng.choice(pool))
            d = exact_distances(g.task, goal)
            candidates = pool[(d[pool] >= lo) & (d[pool] <= hi)]
            candidates = [int(s) for s in candidates if tuple(sorted((int(s), goal))) not in occupied]
            if not candidates:
                continue
            source = int(rng.choice(candidates))
            occupied.add(tuple(sorted((source, goal))))
            starts.append(g.states[source])
            goals.append(g.states[goal])
            depths.append(d[source])
            tiers.append(tier)
            empirical_depths.append(distances(g, goal, p.owners == 0)[source])
            accepted += 1
            if accepted == count:
                break
        if accepted != count:
            raise ValueError("cannot fill exact-distance goal strata; no partial suite was returned")
    # File order must not identify the hidden difficulty tier to the learner.
    order = rng.permutation(len(starts))
    return ({"board": np.asarray(starts, dtype=np.int32)[order], "goal": np.asarray(goals, dtype=np.int32)[order]},
            {"distance": np.asarray(depths, dtype=np.int32)[order], "tier": np.asarray(tiers, dtype=np.int32)[order],
             "empirical_distance": np.asarray(empirical_depths, dtype=np.int32)[order]})


def _composition(p, split, count, rng, occupied):
    spec, g = instance(p.graph.task), p.graph
    supported = p.regime == "transition_holdout"
    allowed = p.owners == 0 if supported else None
    endpoints = np.arange(len(g.states)) if supported else np.flatnonzero(p.state_split == split)
    starts, words, prefixes, lengths, effects, unseen = [], [], [], [], [], []
    width = max(spec.composition_lengths)
    for length in spec.composition_lengths:
        accepted = 0
        for _ in range(max(1000, count * 100)):
            goal = int(rng.choice(endpoints))
            true_d = exact_distances(g.task, goal)
            train_d = distances(g, goal, allowed) if supported else true_d
            # A nominally long word that cancels or has a shortcut is not long-horizon evidence.
            pool = endpoints[(true_d[endpoints] == length) & (train_d[endpoints] == length)]
            pool = [int(s) for s in pool if tuple(sorted((int(s), goal))) not in occupied]
            if not pool:
                continue
            source = int(rng.choice(pool))
            actions, boards = _path(p, source, train_d, allowed)
            word = np.full(width, -1, dtype=np.int32)
            prefix = np.full((width, g.states.shape[1]), -1, dtype=np.int32)
            word[:length], prefix[:length] = actions, boards
            starts.append(g.states[source])
            words.append(word)
            prefixes.append(prefix)
            lengths.append(length)
            effects.append(_effect(p, actions))
            previous = source
            missing = 0
            for action, board in zip(actions, boards):
                missing += p.owners[previous, p.action_map[action]] != 0
                previous = g.locate(board)
            unseen.append(missing)
            occupied.add(tuple(sorted((source, goal))))
            accepted += 1
            if accepted == count:
                break
        if accepted != count:
            raise ValueError("insufficient training-supported shortest compositions; no partial suite returned")
    return ({"board": np.asarray(starts, dtype=np.int32), "actions": np.asarray(words, dtype=np.int32),
             "lengths": np.asarray(lengths, dtype=np.int32)},
            {"prefix_boards": np.asarray(prefixes, dtype=np.int32),
             "unseen_transitions": np.asarray(unseen, dtype=np.int32)}, effects)


def materialize(task: str, regime: str, seed: int = 0, per_cell: int = 8):
    """Evaluator-only generation. All retained training queries are included once.

    The regimes have separate training corpora and different C conditions.
    Never borrow records or fitted models across regimes.
    """
    seed, per_cell = integer(seed, "seed"), integer(per_cell, "per_cell", 1)
    p = partition(task, regime, seed)
    g, spec = p.graph, instance(task)
    checked_diameter(task)
    q = p.queries(0)
    _rng(task, regime, seed, "train_order").shuffle(q)
    targets = [p.successor(int(s), int(a)) for s, a in q]
    train = {
        "observations": np.stack((g.states[q[:, 0]], g.states[[s for s, _ in targets]]), axis=1),
        "actions": q[:, 1, None],
        "legal": np.asarray([legal for _, legal in targets], dtype=np.uint8)[:, None],
    }
    public = {"train": train}
    answers, effect_sets, goal_pairs, composition_pairs = {}, {}, set(), set()
    for split in (1, 2):
        label = SPLITS[split]
        rng = _rng(task, regime, seed, label)
        public[f"R_{label}"], truth = _rule_probe(p, split, rng)
        answers.update({f"R_{label}_{key}": value for key, value in truth.items()})
        public[f"G_{label}"], truth = _goals(p, split, per_cell, rng, goal_pairs)
        answers.update({f"G_{label}_{key}": value for key, value in truth.items()})
        public[f"C_{label}"], truth, effects = _composition(p, split, per_cell, rng, composition_pairs)
        answers.update({f"C_{label}_{key}": value for key, value in truth.items()})
        effect_sets[label] = effects
    metadata = {
        "protocol": PROTOCOL, "task": task, "regime": regime,
        "state_width": g.states.shape[1], "n_actions": g.n_actions,
        "control_budget": spec.control_budget, "training_episode_length": 1,
        "training_episode_end": "collector_cutoff; not a terminal or goal success",
        "invalid_attempt": "unchanged board; legal=false; consumes one step",
        "action_encoding": "opaque integer IDs; mapping belongs to evaluator",
        "probes": ["R", "C", "G"],
        "rule_evaluation_population": "all retained queries in each held-out split",
        "composition_condition": "supported_primitives" if regime == "transition_holdout" else "unseen_endpoints",
    }
    components = training_components(p)
    audit = {
        "canonical_component_states": len(g.states), "diameter": checked_diameter(task),
        "query_counts": {label: int(np.sum(p.owners == i)) for i, label in enumerate(SPLITS)},
        "discarded_cross_state_queries": int(np.sum(p.owners < 0)),
        "state_partition_counts": {label: int(np.sum(p.state_split == i)) for i, label in enumerate(SPLITS)},
        "training_source_states": len(set(q[:, 0].tolist())),
        "training_legal_components": components,
        "opaque_to_native": p.action_map.tolist(),
        "goal_sampling": "uniform component/partition goal, then uniform eligible start; retry empty or duplicate pairs",
        "composition_effects": {label: {"cases": len(v), "unique": len(set(v)), "sha256": v} for label, v in effect_sets.items()},
        "composition_dev_test_effect_overlap": len(set(effect_sets.get("dev", [])) & set(effect_sets.get("test", []))),
        "composition_primitive_effect_overlap": len(set(sum(effect_sets.values(), [])) & {_effect(p, [a]) for a in range(g.n_actions)}),
        "goal_empirically_reachable": {label: int(np.sum(answers[f"G_{label}_empirical_distance"] >= 0)) for label in ("dev", "test")},
        "composition_unseen_transition_counts": {label: answers[f"C_{label}_unseen_transitions"].tolist() for label in ("dev", "test")},
        "claims": {
            "unseen_rule": False, "unseen_family": False, "unseen_size": False,
            "R_inverse_edge_disjoint": regime == "transition_holdout",
            "R_state_disjoint": regime == "state_holdout",
            "G_unseen_endpoints": regime == "state_holdout",
            "G_entire_path_unseen": False,
            "C_all_primitive_transitions_seen": regime == "transition_holdout",
            "C_dev_test_operator_effect_disjoint": False,
        },
    }
    return public, answers, metadata, audit


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = [root / "__init__.py", *sorted((root / "research").glob("*.py")),
             *(root / "games" / family / "rules.py" for family in ("lights_out", "permutation", "classic"))]
    return {path.relative_to(root).as_posix(): _hash_file(path) for path in files}


def write_bundle(output, task: str, regime: str, *, seed: int = 0, per_cell: int = 8) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    public, answers, metadata, audit = materialize(task, regime, seed, per_cell)
    output.mkdir(parents=True)
    (output / "learner").mkdir()
    (output / "evaluator").mkdir()
    for name, arrays in public.items():
        np.savez_compressed(output / "learner" / f"{name}.npz", **arrays)
    np.savez_compressed(output / "evaluator" / "answers.npz", **answers)
    for name, value in (("learner/metadata.json", metadata), ("evaluator/audit.json", audit)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in output.rglob("*") if p.is_file())
    manifest = {
        "protocol": PROTOCOL, "specification_sha256": digest(specification()),
        "generator_source_sha256": _source_hashes(),
        "purpose": "development-qualification; not an untouched publication test",
        "task": task, "regime": regime, "seed": integer(seed, "seed"), "per_cell": integer(per_cell, "per_cell", 1),
        "files": {p.relative_to(output).as_posix(): _hash_file(p) for p in files},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def _read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: immutable(archive[key]) for key in archive.files}


def validate_bundle(path) -> dict:
    """Recompute semantics and partitions, not just caller-editable checksums."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    expected_keys = {"protocol", "specification_sha256", "generator_source_sha256", "purpose", "task", "regime", "seed", "per_cell", "files"}
    if set(manifest) != expected_keys or manifest["protocol"] != PROTOCOL or manifest["specification_sha256"] != digest(specification()):
        raise ValueError("unsupported or changed protocol manifest")
    if manifest["generator_source_sha256"] != _source_hashes():
        raise ValueError("generator source differs; validate with the artifact's recorded source revision")
    if manifest["purpose"] != "development-qualification; not an untouched publication test":
        raise ValueError("this candidate generator cannot certify a final publication test")
    public, answers, metadata, audit = materialize(manifest["task"], manifest["regime"], manifest["seed"], manifest["per_cell"])
    expected_files = {f"learner/{name}.npz" for name in public} | {"evaluator/answers.npz", "learner/metadata.json", "evaluator/audit.json"}
    actual_files = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()} - {"manifest.json"}
    if set(manifest["files"]) != expected_files or actual_files != expected_files:
        raise ValueError("bundle file inventory mismatch")
    for name, expected_hash in manifest["files"].items():
        if _hash_file(path / name) != expected_hash:
            raise ValueError(f"checksum mismatch: {name}")
    for name, expected in [(f"learner/{key}.npz", value) for key, value in public.items()] + [("evaluator/answers.npz", answers)]:
        actual = _read_npz(path / name)
        if set(actual) != set(expected) or any(actual[k].dtype != expected[k].dtype or not np.array_equal(actual[k], expected[k]) for k in expected):
            raise ValueError(f"semantic or split mismatch: {name}")
    for name, expected in (("learner/metadata.json", metadata), ("evaluator/audit.json", audit)):
        if canonical(json.loads((path / name).read_text())) != canonical(expected):
            raise ValueError(f"metadata mismatch: {name}")
    return {"valid": True, "protocol": PROTOCOL, "task": manifest["task"], "regime": manifest["regime"],
            "training_episodes": len(public["train"]["actions"]), "audit": audit}
