"""Goal views do not turn initial equality or monitor-only dwell into actor labels."""
import numpy as np
import pytest

from copuzzle.manipulation.play.collector import collect_play
from copuzzle.manipulation.play.data import goal_episode, future_episode


@pytest.fixture(scope='module')
def prefixes():
    pytest.importorskip('mujoco')
    return {task: collect_play(task, seed=8, num_segments=1, max_steps=4)
            for task in ('grid-3x3', 'pancake-4', 'topspin-20-4')}


@pytest.mark.parametrize('task', ['grid-3x3', 'pancake-4'])
def test_initially_achieved_goal_has_no_actor_supervision(prefixes, task):
    trace = prefixes[task]
    view = goal_episode(trace, 0, trace.observations['board'][0])
    assert view['initially_satisfied'] and not view['actor_supervision_valid']
    assert len(view['actions']) == 0 and len(view['terminated']) == 0
    assert view['shortest_distance'] is None


def test_initial_topspin_confirmation_is_not_a_fake_three_step_demonstration(prefixes):
    trace = prefixes['topspin-20-4']
    view = goal_episode(trace, 0, trace.observations['board'][0])
    assert view['initial_goal_confirmation_pending'] and not view['initially_satisfied']
    assert not view['actor_supervision_valid'] and len(view['actions']) == 0
    assert 'success_streak' not in {item['name'] for item in view['feature_spec']}
    assert 'settled_board_steps' in {item['name'] for item in view['feature_spec']}


@pytest.mark.parametrize('task', ['grid-3x3', 'pancake-4', 'topspin-20-4'])
def test_unachieved_target_retains_cutoff_and_original_goal_diagnostics(prefixes, task):
    trace = prefixes[task]
    view = goal_episode(trace, 0, trace.behavior_goals[0])
    np.testing.assert_array_equal(view['actions'], trace.actions)
    assert view['collection_cutoff'][-1] and not view['terminated'].any()
    assert view['bootstrap_masks'][-1] == 1
    assert view['actor_supervision_valid']
    assert view['feature_profile'] == 'physical_play_goal_free_v1'
    np.testing.assert_array_equal(view['source_goal_reached'], trace.goal_reached)


@pytest.mark.parametrize('start,end', [(True, 1), (-1, 1), (0, True), (3, 2), (0, 100)])
def test_invalid_window_is_rejected(prefixes, start, end):
    trace = prefixes['grid-3x3']
    with pytest.raises(ValueError):
        goal_episode(trace, start, trace.behavior_goals[0], end)


def test_future_target_must_be_strictly_later(prefixes):
    with pytest.raises(ValueError):
        future_episode(prefixes['grid-3x3'], 1, 1)
