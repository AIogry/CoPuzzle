"""Prepare comparable fixed-rule development data at several task sizes.

Run after installation:
    python examples/generate_task_scales.py --output artifacts/task-scales-v0

This generates benchmark data and evaluator-only task labels. It does not
train an agent or establish a held-out state, combination, or graph protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import gymnasium
import numpy as np

import copuzzle
from copuzzle.games.lights_out.data import generate_dataset
from copuzzle.games.lights_out.evaluation import generate_task_suite, write_json
from copuzzle.games.lights_out.presets import TASK_PRESETS, get_task_preset


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', default=[p.task_id for p in TASK_PRESETS])
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--val-episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=30)
    parser.add_argument('--per-depth', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task-seed', type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    if len(set(args.tasks)) != len(args.tasks):
        parser.error('Task names must be distinct')
    try:
        presets = [get_task_preset(task) for task in args.tasks]
    except ValueError as error:
        parser.error(str(error))
    if min(args.episodes, args.val_episodes, args.horizon, args.per_depth) < 1:
        parser.error('Episode counts, horizon, and per-depth must be positive')
    if min(args.seed, args.task_seed) < 0:
        parser.error('Seeds must be nonnegative')
    if any(max(preset.evaluation_depths) > args.horizon for preset in presets):
        parser.error('Horizon must cover the selected preset depths')
    package_root = Path(copuzzle.__file__).resolve().parent
    source_hashes = {str(path.relative_to(package_root)): file_hash(path)
                     for path in sorted(package_root.rglob('*.py'))}
    manifest = {
        'schema_version': 'copuzzle.task_scale_collection.v0',
        'scope': 'Development data and exact-distance tasks for independently trained fixed rules; no claim of unseen states, pairs, combinations, or graphs',
        'versions': {'copuzzle': copuzzle.__version__, 'numpy': np.__version__,
                     'gymnasium': gymnasium.__version__, 'python': platform.python_version()},
        'package_source_sha256': source_hashes,
        'recipe_sha256': file_hash(Path(__file__)),
        'parameters': {key: value for key, value in vars(args).items() if key not in {'output', 'tasks'}},
        'tasks': {},
    }
    args.output.mkdir(parents=True)
    for preset in presets:
        rule = copuzzle.get_rule(preset.task_id)
        directory = args.output / preset.task_id
        directory.mkdir()
        for split, episodes in [('train', args.episodes), ('val', args.val_episodes)]:
            data = generate_dataset(rule, episodes, args.horizon, args.seed, split)
            data.save(directory / f'{split}.npz')
        suite = generate_task_suite(rule, depths=preset.evaluation_depths,
                                    per_depth=args.per_depth, seed=args.task_seed,
                                    max_episode_steps=args.horizon)
        write_json(directory / 'tasks.json', suite)
        entry = {'rule_id': rule.rule_id, 'buttons': rule.n_buttons,
                 'gym_id': preset.gym_id, 'goal_cases': len(suite['cases']),
                 'evaluation_depths': list(preset.evaluation_depths),
                 'files_sha256': {path.name: file_hash(path) for path in sorted(directory.iterdir())}}
        manifest['tasks'][preset.task_id] = entry
        print(json.dumps({'task': preset.task_id, 'train_transitions': args.episodes * args.horizon,
                          'goal_cases': entry['goal_cases']}, sort_keys=True), flush=True)
    # Presence of the final manifest identifies a completed collection.
    write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
