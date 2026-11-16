"""Small real-contact collection smoke; full development collection is separate."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('mujoco')

SCRIPT = Path(__file__).resolve().parents[1] / 'examples/generate_physical_permutation_collection.py'
spec = importlib.util.spec_from_file_location('copuzzle_physical_collection_example', SCRIPT)
collector = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = collector
spec.loader.exec_module(collector)


def test_physical_collection_saves_replays_and_preserves_abort_recovery(tmp_path, capsys):
    folder = tmp_path / 'collection'
    result = collector.build_collection(
        folder, tasks=('pancake-4',), train_episodes=1, val_episodes=1,
        conditions=('noisy',), min_operations=1, max_operations=1,
        max_steps=500, evaluation_cases_per_depth=0)
    assert result['status'] == 'complete'
    assert json.loads((folder / 'manifest.json').read_text())['source_sha256'] == result['source_sha256']
    for path, checksum in result['files'].items():
        assert hashlib.sha256((folder / path).read_bytes()).hexdigest() == checksum
    rows = result['tasks']['pancake-4']['episodes']
    assert len({row['reset_seed'] for row in rows}) == 2
    for row in rows:
        assert row['actual_operations'] == row['intended_operations']
        assert row['aborted_attempts'] == 1 and row['recovery_completed']
        assert row['collection_cutoff'] and not row['goal_success']
        assert row['replay']['max_absolute_observation_error'] == 0
        restored = collector.load_physics_trajectory(folder / row['trajectory'])
        assert restored.metadata['user']['condition'] == 'noisy'
        assert restored.metadata['user']['action_noise_amplitude'] == .025
        assert restored.metadata['user']['controller_failure'] is None
    with pytest.raises(FileExistsError):
        collector.build_collection(folder)
    assert capsys.readouterr().out


def test_controller_timeout_preserves_recorded_prefix(tmp_path, monkeypatch):
    env = collector.PhysicalPermutationEnv('pancake-4', max_episode_steps=100)
    policy = collector.CollectionPolicy([0], action_scale=env.action_scale,
                                        condition='play', noise_seed=0)
    original = collector.FixtureSequencePolicy.__call__
    calls = 0
    def failing_after_prefix(self, observation):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError('injected controller timeout after three actions')
        return original(self, observation)
    monkeypatch.setattr(collector.FixtureSequencePolicy, '__call__', failing_after_prefix)
    try:
        state = np.arange(1, 5, dtype=np.int32)
        episode = collector.record_trajectory(env, policy, seed=0,
                    options={'state': state, 'goal': env.rule.transition(state, 0)})
        assert episode.num_steps == 3
        assert episode.collection_cutoff[-1] and not episode.truncated[-1]
        assert policy.error == 'RuntimeError: injected controller timeout after three actions'
        episode.metadata['user']['controller_failure'] = policy.error
        path = episode.save(tmp_path / 'partial.npz')
        restored = collector.load_physics_trajectory(path)
        assert restored.num_steps == 3
        assert restored.metadata['user']['controller_failure'] == policy.error
        assert collector.replay_trajectory(env, restored)['max_absolute_observation_error'] == 0
    finally:
        env.close()


def test_expert_preparation_uses_exact_distance_and_play_goal_is_off_path():
    env = collector.PhysicalPermutationEnv('pancake-4', max_episode_steps=500)
    try:
        oracle = collector.ExactOracle(env.rule)
        start, goal, operations, depth = collector._prepare(env, oracle, 3, 'expert', 4, 6)
        assert depth == 4 and len(operations) == 4 and oracle.distance(start, goal) == 4
        start, goal, operations, depth = collector._prepare(env, oracle, 3, 'play', 4, 6)
        assert depth is None and 4 <= len(operations) <= 6
        current = start
        assert not np.array_equal(current, goal)
        for action in operations:
            current = env.rule.transition(current, action)
            assert not np.array_equal(current, goal)
    finally:
        env.close()


@pytest.mark.parametrize('options', [dict(train_episodes=0), dict(noise_amplitude=.2),
                                    dict(min_operations=3, max_operations=2), dict(tasks=('pancake-6',))])
def test_invalid_collection_fails_before_creating_output(tmp_path, options):
    output = tmp_path / 'bad'
    with pytest.raises(ValueError):
        collector.build_collection(output, **options)
    assert not output.exists()
