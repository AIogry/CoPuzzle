import json
import pytest

from copuzzle.manipulation.cli import main
from copuzzle.manipulation.play import load_play_trace


def test_cli_collects_new_schema_and_replays_without_overwriting(tmp_path, capsys):
    pytest.importorskip('mujoco')
    path = tmp_path / 'play.npz'
    arguments = ['play', '--task', 'grid-3x3', '--operations', '0', '8',
                 '--max-steps', '2', '--output', str(path)]
    assert main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['replay']['max_absolute_observation_error'] == 0
    trace = load_play_trace(path)
    assert trace.metadata['reset_count'] == 1 and trace.metadata['requested_segments'] == 2
    assert trace.collection_cutoff[-1]
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        main(arguments)
    assert path.read_bytes() == before


def test_cli_rejects_noninteger_button_before_simulation(tmp_path):
    with pytest.raises(SystemExit):
        main(['play', '--task', 'grid-3x3', '--operations', 'button', '--output', str(tmp_path/'play.npz')])
    assert not list(tmp_path.iterdir())
