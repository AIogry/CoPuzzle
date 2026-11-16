"""Reproducible direct-contact development suites and continuous data inspection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import TASK_IDS, SPLITS, load_transitions
from .suite import generate_suite, save_suite, load_suite, suite_sha256, split_overlap


def main(argv=None):
    parser = argparse.ArgumentParser(prog='copuzzle manipulation', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    suite = commands.add_parser('suite', help='Create a deterministic multi-step development task suite')
    suite.add_argument('--output', type=Path, required=True)
    suite.add_argument('--seed', type=int, default=0)
    suite.add_argument('--cases-per-depth', type=int, default=1)
    suite.add_argument('--depths', type=int, nargs='+', default=[1, 2, 3])
    suite.add_argument('--tasks', choices=TASK_IDS, nargs='+', default=list(TASK_IDS))
    collect = commands.add_parser('collect', help='Execute and replay native continuous trajectories')
    collect.add_argument('--suite', type=Path, required=True)
    collect.add_argument('--output', type=Path, required=True)
    collect.add_argument('--policy', choices=['reference', 'zero'], default='reference')
    collect.add_argument('--splits', choices=SPLITS, nargs='+')
    collect.add_argument('--tasks', choices=TASK_IDS, nargs='+')
    collect.add_argument('--max-tasks', type=int)
    collect.add_argument('--max-steps', type=int, help='Collection cap; does not change the task horizon')
    inspect = commands.add_parser('inspect', help='Check a collection and summarize one continuous task/split')
    inspect.add_argument('--collection', type=Path, required=True)
    inspect.add_argument('--task', choices=TASK_IDS, required=True)
    inspect.add_argument('--split', choices=SPLITS, default='train')
    inspect.add_argument('--bootstrap-mode', choices=['terminated_only', 'finite_horizon'], default='terminated_only')
    args = parser.parse_args(argv)
    if args.command == 'suite':
        value = generate_suite(seed=args.seed, cases_per_depth=args.cases_per_depth,
                               depths=tuple(args.depths), task_ids=tuple(args.tasks))
        path = save_suite(value, args.output)
        result = {'suite': str(path), 'suite_sha256': suite_sha256(value),
                  'tasks': len(value['tasks']), 'overlap': split_overlap(value)}
    elif args.command == 'collect':
        from .runner import run_collection
        report = run_collection(load_suite(args.suite), args.output, policy=args.policy,
            splits=args.splits, task_ids=args.tasks, max_tasks=args.max_tasks, max_steps=args.max_steps)
        result = {'manifest': str(args.output.resolve() / 'manifest.json'),
                  'status': report['status'], 'attempted_tasks': len(report['episodes']),
                  'verified_goal_successes': sum(bool(row['goal_success']) for row in report['episodes'])}
    else:
        data = load_transitions(args.collection, task_id=args.task, split=args.split,
                                bootstrap_mode=args.bootstrap_mode)
        result = {'schema_version': data.provenance['schema_version'], 'task_id': args.task,
                  'split': args.split, 'episodes': len(data.episode_ids), 'control_transitions': len(data),
                  'state_features': data.observations.shape[1], 'goal_features': data.goals.shape[1],
                  'action_dimensions': data.actions.shape[1], 'terminated': int(data.terminated.sum()),
                  'truncated': int(data.truncated.sum()), 'collection_cutoffs': int(data.collection_cutoff.sum()),
                  'bootstrap_mode': args.bootstrap_mode,
                  'configuration_sha256': data.provenance['configuration_sha256'],
                  'provenance': data.provenance}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
