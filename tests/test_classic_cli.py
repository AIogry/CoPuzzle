"""Saved collection integrity and command-line data-to-evaluation workflow."""
import hashlib
import json

import pytest

from copuzzle.games.classic.cli import main
from copuzzle.games.classic.data import load_dataset
from copuzzle.games.classic.evaluation import load_task_suite
from copuzzle.games.classic.rules import get_rule


def test_saved_collection_complete_disjoint_and_audited(tmp_path, capsys):
    output = tmp_path / 'collection'
    main(['collection', '--output', str(output), '--tasks', 'sliding-2x3', 'hanoi-3',
          '--episodes', '12', '--val-episodes', '4', '--per-depth', '2'])
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert 'development' in manifest['purpose']
    for name, checksum in manifest['files'].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == checksum
    for task in ('sliding-2x3', 'hanoi-3'):
        folder = output / task
        train = load_dataset(folder / 'train.npz')
        rule = get_rule(task)
        train.validate(rule)
        assert train.states.shape == (12, 31, rule.state_size)
        assert train.action_masks.shape == (12, 31, rule.n_actions)
        dev, test = (load_task_suite(folder / f'{split}_tasks.json') for split in ('dev', 'test'))
        pairs = lambda suite: {(tuple(c['state']), tuple(c['goal'])) for c in suite['cases']}
        assert not pairs(dev) & pairs(test)
        for split in ('dev', 'test'):
            report = json.loads((folder / f'{split}_oracle.json').read_text())
            assert report['summary']['success_rate'] == 1
            assert report['summary']['mean_excess_steps_on_success'] == 0
            assert 'training_overlap' in report
    before = (output / 'manifest.json').read_bytes()
    with pytest.raises(SystemExit):
        main(['collection', '--output', str(output)])
    assert (output / 'manifest.json').read_bytes() == before
    assert capsys.readouterr().out


def test_cli_standalone_generation_validation_and_exclusion(tmp_path, capsys):
    folder = tmp_path / 'data'
    main(['generate', '--task', 'hanoi-2', '--output', str(folder),
          '--episodes', '6', '--val-episodes', '3', '--horizon', '10'])
    main(['validate', str(folder / 'train.npz'), '--check-rule'])
    for name in ('dev', 'test'):
        args = ['tasks', '--task', 'hanoi-2', '--depths', '1,2,3',
                '--per-depth', '2', '--output', str(folder / f'{name}.json')]
        if name == 'test':
            args.extend(['--exclude-suite', str(folder / 'dev.json')])
        main(args)
    main(['evaluate', '--tasks', str(folder / 'test.json'), '--policy', 'oracle',
          '--training-data', str(folder / 'train.npz'), '--output', str(folder / 'oracle.json')])
    assert json.loads((folder / 'oracle.json').read_text())['summary']['success_rate'] == 1
    assert capsys.readouterr().out


def test_collection_rejects_infeasible_configuration_before_writing(tmp_path):
    output = tmp_path / 'bad'
    with pytest.raises(SystemExit):
        main(['collection', '--output', str(output), '--horizon', '4'])
    assert not output.exists()
    with pytest.raises(SystemExit):
        main(['collection', '--output', str(output), '--tasks', 'hanoi-2', '--per-depth', '100'])
    assert not output.exists()


def test_default_collection_depths_are_feasible(tmp_path):
    from copuzzle.games.classic.cli import DEFAULT_TASKS, PRESETS
    from copuzzle.games.classic.evaluation import generate_task_suite
    for task in DEFAULT_TASKS:
        rule = get_rule(task)
        dev = generate_task_suite(rule, depths=PRESETS[task]['depths'], tasks_per_depth=16)
        test = generate_task_suite(rule, depths=PRESETS[task]['depths'], tasks_per_depth=16,
                                   exclude_pairs=dev['cases'])
        assert len(dev['cases']) == len(test['cases'])
