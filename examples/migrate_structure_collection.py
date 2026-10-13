"""Attach public structure to an existing task-scale collection, without resampling.

python examples/migrate_structure_collection.py \
    --source artifacts/task-scales-v0 --output artifacts/structure-v1

The source tree is read only. Train/val archives and sidecars are copied byte
for byte. Task starts, goals, distance labels and budgets are preserved.
A completed output contains manifest.json; choose a new output after failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import copuzzle
from copuzzle.games.lights_out.bundle import create_offline_bundle, load_offline_bundle
from copuzzle.games.lights_out.evaluation import load_task_suite, save_task_suite, with_structure, write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error('Output must be outside the source collection')
    if args.output.exists():
        parser.error('Output exists; choose a new directory')
    source_manifest_path = args.source / 'manifest.json'
    source = json.loads(source_manifest_path.read_text())
    if source.get('schema_version') != 'copuzzle.task_scale_collection.v0':
        parser.error('Expected a task_scale_collection.v0 source manifest')
    # Verify all declared source files before publishing any output.
    for task, entry in source['tasks'].items():
        if Path(task).name != task or task in ('.', '..'):
            parser.error('Invalid task directory in source manifest')
        for filename, expected in entry['files_sha256'].items():
            if Path(filename).name != filename or filename in ('.', '..'):
                parser.error('Invalid source filename')
            if digest(args.source / task / filename) != expected:
                parser.error(f'Source checksum mismatch: {task}/{filename}')
    package = Path(copuzzle.__file__).parent
    report = {'schema_version': 'copuzzle.structure_collection.v1',
              'copuzzle_version': copuzzle.__version__, 'observation_profile': 'structure_v1',
              'source_manifest_sha256': digest(source_manifest_path),
              'package_source_sha256': {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob('*.py'))},
              'recipe_sha256': digest(Path(__file__)),
              'scope': 'Development collection; source trajectories and cases retained; no frozen train/dev/test split',
              'tasks': {}}
    args.output.mkdir(parents=True)
    for task in source['tasks']:
        old, new = args.source / task, args.output / task
        rule = copuzzle.get_rule(task)
        create_offline_bundle(new, train_path=old / 'train.npz', val_path=old / 'val.npz', rule=rule)
        original_suite = load_task_suite(old / 'tasks.json')
        save_task_suite(with_structure(original_suite, rule=rule), new / 'tasks.json')
        migrated_suite = load_task_suite(new / 'tasks.json')
        if migrated_suite['cases'] != original_suite['cases']:
            raise ValueError('Migration changed evaluation cases')
        bundle = load_offline_bundle(new)
        validation = load_offline_bundle(new, split='val')
        entry = {'structure_sha256': bundle.structure.structure_sha256, 'rule_id': rule.rule_id,
                 'train_transitions': bundle.dataset.num_episodes * bundle.dataset.horizon,
                 'val_transitions': validation.dataset.num_episodes * validation.dataset.horizon,
                 'goal_cases': len(original_suite['cases']),
                 'files_sha256': {str(p.relative_to(new)): digest(p) for p in sorted(new.rglob('*')) if p.is_file()}}
        report['tasks'][task] = entry
        print(json.dumps({'task': task, 'train_transitions': entry['train_transitions'],
                          'goal_cases': entry['goal_cases']}), flush=True)
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
