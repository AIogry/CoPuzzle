import json
import subprocess
import sys

from copuzzle.manipulation.cli import main
from copuzzle.manipulation.suite import load_suite


def test_cli_builds_portable_suite_without_importing_physics(tmp_path):
    output = tmp_path / 'suite.json'
    code = """
import sys
class NoPhysics:
    def find_spec(self, fullname, *args):
        if fullname == 'mujoco':
            raise AssertionError('Suite generation must not import MuJoCo')
sys.meta_path.insert(0, NoPhysics())
from copuzzle.cli import main
main(['manipulation', 'suite', '--output', sys.argv[1], '--seed', '12', '--depths', '1', '2'])
"""
    run = subprocess.run([sys.executable, '-c', code, str(output)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    suite = load_suite(output)
    assert len(suite['tasks']) == 18
    assert json.loads(run.stdout)['tasks'] == 18


def test_cli_does_not_overwrite_suite(tmp_path):
    import pytest
    path = tmp_path / 'suite.json'
    main(['suite', '--output', str(path), '--tasks', 'pancake-4', '--depths', '1'])
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        main(['suite', '--output', str(path), '--tasks', 'pancake-4', '--depths', '1'])
    assert path.read_bytes() == before
