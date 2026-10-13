"""Public offline inputs and frozen-policy evaluation for Grid and Graph.

Run after installation: python examples/structure_workflow.py
This interface check uses an untrained random policy; it makes no claim of
rule learning. Preparation and evaluation use executable rules. The learner
section only reads a completed bundle and samples recorded experience.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from copuzzle import get_rule
from copuzzle.games.lights_out.bundle import BundleGoalSampler, load_offline_bundle, write_offline_bundle
from copuzzle.games.lights_out.data import generate_dataset
from copuzzle.games.lights_out.evaluation import (
    RandomPolicy, evaluate_policy, generate_task_suite, load_task_suite,
    save_task_suite, with_structure,
)


def demonstrate(directory, task):
    # Benchmark-side preparation; an offline learner does not run this section.
    directory.mkdir(parents=True)
    rule = get_rule(task)
    for split in ('train', 'val'):
        generate_dataset(rule, 4, 8, seed=0, split=split).save(directory / f'{split}.npz')
    write_offline_bundle(directory, train_path=directory / 'train.npz',
                         val_path=directory / 'val.npz', rule=rule)
    suite = with_structure(generate_task_suite(rule, per_depth=2, depths=(1, 2)), rule=rule)
    save_task_suite(suite, directory / 'tasks.json')

    # Learner boundary: only read fixed artifacts. No rule or solver query.
    bundle = load_offline_bundle(directory, split='train')
    batch = BundleGoalSampler(bundle, seed=7).sample(16)
    # Static structure is shared by the batch, without a leading batch axis.
    policy_inputs = {'board': batch['observations'], 'goal': batch['goals'],
                     'previous_event': batch['previous_events'], **batch['structure']}
    expected = bundle.structure.observation_fields()
    for key in expected:
        np.testing.assert_array_equal(policy_inputs[key], expected[key])

    # Evaluation never passes a suite's answers to the policy.
    class CheckedRandomPolicy(RandomPolicy):
        def __call__(self, observation):
            assert set(observation) == {'board', 'goal', 'previous_event'} | set(expected)
            for key, value in expected.items():
                np.testing.assert_array_equal(observation[key], value)
            return super().__call__(observation)

    report = evaluate_policy(CheckedRandomPolicy(seed=3), load_task_suite(directory / 'tasks.json'))
    return {'task': task, 'structure_sha256': bundle.structure.structure_sha256,
            'training_input_shapes': {key: list(value.shape) for key, value in policy_inputs.items()},
            'evaluation': report['summary'], 'is_learned_policy': False}


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='copuzzle-structure-example-') as path:
        for task in ('grid-4x5', 'graph-cycle20'):
            print(json.dumps(demonstrate(Path(path) / task, task), sort_keys=True))
