import json
import subprocess
import sys

import pytest

from copuzzle.cli import main


def test_cli_complete_workflow(tmp_path, capsys):
    data = tmp_path / 'data'
    tasks = tmp_path / 'tasks.json'
    main(['generate', '--output', str(data), '--episodes', '12', '--val-episodes', '4', '--horizon', '10'])
    main(['validate', str(data / 'train.npz'), '--check-rule'])
    main(['tasks', '--output', str(tasks), '--per-depth', '2', '--depths', '1,2,3'])
    report = tmp_path / 'oracle.json'
    main(['evaluate', '--tasks', str(tasks), '--policy', 'oracle', '--output', str(report),
          '--training-data', str(data / 'train.npz')])
    result = json.loads(report.read_text())
    assert result['summary']['success_rate'] == 1.0
    assert result['summary']['mean_excess_presses_on_success'] == 0.0
    assert result['privileged_rule_access'] is True
    assert 'training_overlap' in result
    main(['diagnose', '--train', str(data / 'train.npz'), '--test', str(data / 'val.npz'),
          '--output', str(tmp_path / 'diagnostics.json'), '--lengths', '2,4', '--samples', '8'])
    diagnostic = json.loads((tmp_path / 'diagnostics.json').read_text())
    assert diagnostic['is_rl'] is False
    assert diagnostic['privileged_rule_access'] is False
    assert capsys.readouterr().out


def test_cli_refuses_overwrite_and_bad_inputs(tmp_path):
    out = tmp_path / 'data'
    main(['generate', '--output', str(out), '--episodes', '1', '--val-episodes', '1'])
    before = (out / 'train.npz').read_bytes()
    with pytest.raises(SystemExit) as err:
        main(['generate', '--output', str(out)])
    assert err.value.code == 2
    assert (out / 'train.npz').read_bytes() == before
    with pytest.raises(SystemExit):
        main(['tasks', '--output', str(tmp_path / 'bad.json'), '--depths', '1,0'])


def test_module_cli_runs_outside_checkout(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'copuzzle', '--help'], cwd=tmp_path,
                            check=True, capture_output=True, text=True)
    assert 'generate' in result.stdout and 'diagnose' in result.stdout
