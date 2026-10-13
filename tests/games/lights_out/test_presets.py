"""Exercise the catalog through the installed Gym and CLI entry points."""
import json

import gymnasium as gym
import pytest

import copuzzle
from copuzzle.cli import main
from copuzzle.games.lights_out.evaluation import load_task_suite
from copuzzle.games.lights_out.presets import TASK_PRESETS, get_task_preset


def test_catalog_and_cli_discover_same_unique_tasks(capsys):
    assert len({preset.task_id for preset in TASK_PRESETS}) == 12
    assert len({preset.gym_id for preset in TASK_PRESETS}) == 12
    main(['list'])
    rows = json.loads(capsys.readouterr().out)
    assert [row['task_id'] for row in rows] == [preset.task_id for preset in TASK_PRESETS]
    assert {row['buttons'] for row in rows} == {9, 16, 20, 24}


@pytest.mark.parametrize('preset', TASK_PRESETS, ids=lambda item: item.task_id)
def test_registered_preset_and_exact_depths(preset, tmp_path):
    rule = copuzzle.get_rule(preset.task_id)
    env = gym.make(preset.gym_id)
    try:
        obs, _ = env.reset(seed=1)
        assert env.action_space.n == rule.n_buttons
        assert obs['goal'].shape == (rule.n_buttons,)
        assert set(obs) == {'board', 'goal', 'previous_event'}
    finally:
        env.close()
    # Generate and reload one exactly verified task in every suggested stratum.
    target = tmp_path / 'tasks.json'
    main(['tasks', '--task', preset.task_id, '--depths', 'preset',
          '--per-depth', '1', '--output', str(target)])
    suite = load_task_suite(target)
    assert tuple(case['optimal_steps'] for case in suite['cases']) == preset.evaluation_depths
    assert not suite['reveal_graph']


def test_legacy_default_depths_and_custom_rule_explicit_depths(tmp_path):
    default = tmp_path / 'default.json'
    main(['tasks', '--task', 'grid-4x6', '--per-depth', '1', '--output', str(default)])
    assert [case['optimal_steps'] for case in load_task_suite(default)['cases']] == [1, 2, 3, 4]
    custom = tmp_path / 'custom.json'
    main(['tasks', '--task', 'graph-path8', '--depths', '1,2', '--per-depth', '1', '--output', str(custom)])
    assert [case['optimal_steps'] for case in load_task_suite(custom)['cases']] == [1, 2]
    with pytest.raises(ValueError, match='specify explicit depths'):
        get_task_preset('graph-path8')
    with pytest.raises(SystemExit) as error:
        main(['tasks', '--task', 'graph-path8', '--depths', 'preset', '--output', str(tmp_path / 'bad.json')])
    assert error.value.code == 2
    assert not (tmp_path / 'bad.json').exists()
