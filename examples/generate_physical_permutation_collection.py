"""Collect and replay contact-actuated UR5e permutation development data.

Run from an installed checkout:
  python examples/generate_physical_permutation_collection.py --output artifacts/permutation-physics-v1

All trajectories use continuous robot control. The indexed fixture's ideal weld
clamps and geared carriers are explicitly declared by its environment config.
This scripted development collection is not a learned-policy benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np

import copuzzle
from copuzzle.games.permutation.oracle import ExactOracle
from copuzzle.games.permutation.physics.data import (
    load_physics_trajectory, record_trajectory, replay_trajectory,
)
from copuzzle.games.permutation.physics.env import PhysicalPermutationEnv
from copuzzle.games.permutation.physics.scripted import FixtureSequencePolicy

TASKS = ('topspin-8-4', 'pancake-4')
CONDITIONS = ('play', 'noisy', 'expert')
EVALUATION_DEPTHS = {'topspin-8-4': (1, 3, 5, 7), 'pancake-4': (1, 2, 3, 4)}
SCHEMA = 'copuzzle.permutation_physics_collection.v1'


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _seed(seed, task, split, index, purpose):
    digest = hashlib.sha256(_json_bytes([SCHEMA, seed, task, split, index, purpose])).digest()
    return int.from_bytes(digest[:4], 'big')


def _sources():
    """Snapshot the complete package's source/assets and this collector script."""
    package = Path(copuzzle.__file__).resolve().parent
    suffixes = {'.py', '.xml', '.obj', '.json', '.md'}
    result = {'src/copuzzle/' + str(path.relative_to(package)): _hash(path)
              for path in sorted(package.rglob('*')) if path.is_file() and path.suffix in suffixes}
    result['examples/generate_physical_permutation_collection.py'] = _hash(__file__)
    return result


def _versions():
    return {'python': platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ('copuzzle', 'numpy', 'gymnasium', 'mujoco')}}


class CollectionPolicy:
    """Public-observation executor with a bounded noisy abort/retry condition.

    Only task preparation uses the abstract rule/oracle. The low-level executor
    follows its intended operation IDs using public robot/fixture observations.
    Runtime controller failures set completion before the next environment step;
    the recorder retains the existing prefix with a collector cutoff.
    """
    def __init__(self, operations, *, action_scale, condition, noise_seed, noise_amplitude=.025):
        self.operations = tuple(int(value) for value in operations)
        self.action_scale = float(action_scale)
        self.condition, self.noise_seed = condition, int(noise_seed)
        self.noise_amplitude = float(noise_amplitude) if condition == 'noisy' else 0.
        self.reset()

    def reset(self):
        self.driver = FixtureSequencePolicy(list(self.operations), action_scale=self.action_scale)
        self.rng = np.random.default_rng(self.noise_seed)
        self.error = None
        self.recovery_started = self.recovery_completed = self._recovering = False
        self._recovery_steps = 0

    @property
    def completed(self):
        return self.error is not None or (self.driver.completed and not self._recovering)

    def __call__(self, observation):
        try:
            if (self.condition == 'noisy' and not self.recovery_started
                    and observation['mechanism_phase'][0] == 1):
                self.recovery_started = self._recovering = True
            if self._recovering:
                self._recovery_steps += 1
                if self._recovery_steps > 400:
                    raise RuntimeError('Noisy abort/retry did not return to ready within 400 control steps')
                if observation['mechanism_phase'][0] == 0:
                    self._recovering = False
                    self.recovery_completed = True
                    self.driver.reset()
                    action = self.driver(observation)
                else:
                    rest = observation['rack_rest_positions'][self.operations[0]]
                    target = rest + [0, 0, .12]
                    action = (target - observation['end_effector_position']) / self.action_scale
            else:
                action = self.driver(observation)
            if self.noise_amplitude:
                action = action + self.rng.uniform(-self.noise_amplitude, self.noise_amplitude, size=3)
            return np.clip(action, -1, 1).astype(np.float32)
        except RuntimeError as error:
            self.error = f'{type(error).__name__}: {error}'
            return np.zeros(3, dtype=np.float32)


def _prepare(env, oracle, seed, condition, min_operations, max_operations, exact_depth=None):
    rng = np.random.default_rng(seed)
    state = rng.permutation(np.arange(1, env.rule.n_tokens + 1, dtype=np.int32))
    if condition == 'expert':
        depth = int(rng.integers(min_operations, max_operations + 1)) if exact_depth is None else exact_depth
        depth = min(depth, int(oracle.distances.max()))
        relatives = oracle.reachable_states[oracle.distances == depth]
        goal = state[relatives[int(rng.integers(len(relatives)))] - 1]
        operations = oracle.solve(state, goal)
        if operations is None or len(operations) != depth:
            raise RuntimeError('Exact reference failed to prepare the requested distance')
        return state, goal, list(map(int, operations)), depth
    length = int(rng.integers(min_operations, max_operations + 1))
    operations = rng.integers(0, 3, size=length).tolist()
    visited = {tuple(state)}
    current = state.copy()
    for action in operations:
        current = env.rule.transition(current, action)
        visited.add(tuple(current))
    # Finite choice from the reachable component, avoiding intermediate task
    # termination. These are collection goals, not successful play labels.
    eligible = [relative for relative in oracle.reachable_states
                if tuple(state[relative - 1]) not in visited]
    if not eligible:
        raise ValueError('Requested play path covers every reachable state; no off-path goal remains')
    goal = state[eligible[int(rng.integers(len(eligible)))] - 1]
    return state, goal, operations, None


def _episode_record(env, folder, *, task, split, index, condition, seed,
                    min_operations, max_operations, noise_amplitude, oracle,
                    exact_depth=None, excluded_pairs=()):
    preparation_seed = _seed(seed, task, split, index, 'preparation')
    for retry in range(1000):
        state, goal, operations, depth = _prepare(
            env, oracle, preparation_seed + retry, condition, min_operations, max_operations, exact_depth)
        pair = (tuple(map(int, state)), tuple(map(int, goal)))
        if pair not in excluded_pairs:
            break
    else:
        raise ValueError('Could not sample an evaluation pair outside prior reset pairs')
    reset_seed = _seed(seed, task, split, index, 'environment')
    noise_seed = _seed(seed, task, split, index, 'action_noise')
    policy = CollectionPolicy(operations, action_scale=env.action_scale, condition=condition,
                              noise_seed=noise_seed, noise_amplitude=noise_amplitude)
    identifier = f'{split}-{index:04d}-{condition}'
    metadata = {
        'episode_id': identifier, 'task': task, 'split': split, 'condition': condition,
        'preparation_seed': preparation_seed + retry, 'action_noise_seed': noise_seed,
        'intended_operations': operations, 'exact_initial_distance': depth,
        'controller': 'FixtureSequencePolicy with observed completion and optional abort/retry',
        'controller_uses_rule_or_oracle': False,
        'preparation_uses_analytic_rule': True, 'preparation_uses_exact_oracle': True,
        'operation_plan_uses_exact_oracle': condition == 'expert',
        'action_noise_distribution': 'uniform_independent_xyz_before_clipping',
        'action_noise_amplitude': policy.noise_amplitude,
        'inserted_partial_stroke_abort': condition == 'noisy',
        'goal_selection': 'exact_distance_pair' if condition == 'expert' else 'reachable_goal_outside_planned_visited_states',
        'split_semantics': 'independent seeded trajectories; state overlap is expected',
    }
    began = time.perf_counter()
    episode = record_trajectory(env, policy, seed=reset_seed,
                                options={'state': state, 'goal': goal}, metadata=metadata)
    episode.metadata['user'].update(controller_failure=policy.error,
                                    recovery_started=policy.recovery_started,
                                    recovery_completed=policy.recovery_completed)
    path = episode.save(folder / f'{identifier}.npz')
    restored = load_physics_trajectory(path)
    replay = replay_trajectory(env, restored)
    _write(path.with_suffix('.replay.json'), replay)
    actual = restored.event_operations.tolist()
    attempts = [attempt for info in restored.infos for attempt in info['attempt_log']]
    row = {
        'episode_id': identifier, 'task': task, 'split': split, 'condition': condition,
        'trajectory': path.name, 'reset_seed': reset_seed,
        'state': state.tolist(), 'goal': goal.tolist(),
        'intended_operations': operations, 'actual_operations': actual,
        'planned_operations_completed_in_order': actual == operations,
        'control_steps': restored.num_steps,
        'simulation_seconds': float(restored.infos[-1]['elapsed_time']),
        'collection_and_replay_wall_seconds': time.perf_counter() - began,
        'goal_success': bool(restored.terminated[-1]), 'environment_timeout': bool(restored.truncated[-1]),
        'collection_cutoff': bool(restored.collection_cutoff[-1]),
        'stop_reason': restored.metadata['collector']['stop_reason'], 'controller_failure': policy.error,
        'mechanism_failure': restored.infos[-1]['failure_reason'],
        'attempts': attempts, 'aborted_attempts': sum(a.get('result') == 'aborted_and_returned' for a in attempts),
        'recovery_started': policy.recovery_started, 'recovery_completed': policy.recovery_completed,
        'exact_initial_distance': depth, 'replay': replay,
        'valid_board_fraction': float(restored.observations['board_valid'].mean()),
        'intermediate_board_observations': int((restored.observations['board_valid'] == 0).sum()),
        'configuration_sha256': restored.metadata['configuration_sha256'],
    }
    _write(path.with_suffix('.report.json'), row)
    return row, restored


def _summary(rows):
    if not rows:
        return {'episodes': 0}
    operations = sum(len(row['actual_operations']) for row in rows)
    aborts = sum(row['aborted_attempts'] for row in rows)
    return {
        'episodes': len(rows), 'control_steps': sum(row['control_steps'] for row in rows),
        'simulation_seconds': sum(row['simulation_seconds'] for row in rows),
        'completed_operations': operations, 'aborted_attempts': aborts,
        'planned_sequence_success_rate': sum(row['planned_operations_completed_in_order'] for row in rows) / len(rows),
        'planned_sequence_failure_rate': sum(not row['planned_operations_completed_in_order'] for row in rows) / len(rows),
        'goal_success_rate': sum(row['goal_success'] for row in rows) / len(rows),
        'mechanism_failure_rate': sum(row['mechanism_failure'] is not None for row in rows) / len(rows),
        'controller_failure_rate': sum(row['controller_failure'] is not None for row in rows) / len(rows),
        'environment_timeout_rate': sum(row['environment_timeout'] for row in rows) / len(rows),
        'aborted_attempt_fraction': aborts / (operations + aborts) if operations + aborts else None,
        'operation_counts': dict(sorted(Counter(str(a) for row in rows for a in row['actual_operations']).items())),
        'all_episodes_replayed': True,
    }


def _observed_states(episode):
    observation = episode.observations
    valid = (observation['board_valid'][:, 0] == 1) & (observation['mechanism_phase'][:, 0] == 0)
    return [tuple(map(int, board)) for board in observation['board'][valid]]


def build_collection(output, *, tasks=TASKS, train_episodes=24, val_episodes=6,
                     min_operations=4, max_operations=6, max_steps=1500,
                     seed=0, noise_amplitude=.025, evaluation_cases_per_depth=1,
                     conditions=CONDITIONS):
    output = Path(output).expanduser().absolute()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite output: {output}')
    tasks, conditions = tuple(tasks), tuple(conditions)
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in TASKS for task in tasks):
        raise ValueError('tasks must be distinct supported physical presets')
    if not conditions or len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        raise ValueError('conditions must be distinct play/noisy/expert names')
    for name, value, lower in [('train_episodes', train_episodes, 1), ('val_episodes', val_episodes, 1),
                               ('min_operations', min_operations, 1), ('max_operations', max_operations, 1),
                               ('max_steps', max_steps, 1), ('seed', seed, 0),
                               ('evaluation_cases_per_depth', evaluation_cases_per_depth, 0)]:
        if type(value) is not int or value < lower:
            raise ValueError(f'{name} must be an integer >= {lower}')
    if min_operations > max_operations:
        raise ValueError('min_operations must not exceed max_operations')
    if isinstance(noise_amplitude, bool) or not np.isfinite(noise_amplitude) or not 0 <= noise_amplitude <= .1:
        raise ValueError('noise_amplitude must lie in [0, 0.1]')
    source = _sources()
    output.mkdir(parents=True)
    records = {}
    for task in tasks:
        env = PhysicalPermutationEnv(task, max_episode_steps=max_steps)
        oracle = ExactOracle(env.rule)
        rows, eval_rows, reset_pairs, training_episodes = [], [], set(), []
        try:
            for split, count in [('train', train_episodes), ('val', val_episodes)]:
                folder = output / task / split
                for index in range(count):
                    condition = conditions[index % len(conditions)]
                    row, episode = _episode_record(
                        env, folder, task=task, split=split, index=index, condition=condition, seed=seed,
                        min_operations=min_operations, max_operations=max_operations,
                        noise_amplitude=float(noise_amplitude), oracle=oracle)
                    row['trajectory'] = str((folder / row['trajectory']).relative_to(output))
                    reset_pairs.add((tuple(row['state']), tuple(row['goal'])))
                    if split == 'train':
                        training_episodes.append(_observed_states(episode))
                    rows.append(row)
                    print(json.dumps({'task': task, 'episode': row['episode_id'], 'steps': row['control_steps'],
                                      'operations': row['actual_operations'], 'replayed': True}), flush=True)
            folder = output / task / 'evaluation'
            for depth in EVALUATION_DEPTHS[task]:
                for replicate in range(evaluation_cases_per_depth):
                    index = len(eval_rows)
                    row, _ = _episode_record(
                        env, folder, task=task, split='evaluation', index=index, condition='expert', seed=seed,
                        min_operations=min_operations, max_operations=max_operations,
                        noise_amplitude=0., oracle=oracle, exact_depth=depth, excluded_pairs=reset_pairs)
                    row['trajectory'] = str((folder / row['trajectory']).relative_to(output))
                    reset_pairs.add((tuple(row['state']), tuple(row['goal'])))
                    pair = tuple(row['state']), tuple(row['goal'])
                    row['case_id'] = hashlib.sha256(_json_bytes([env.rule.rule_id, pair])).hexdigest()[:24]
                    row['training_overlap'] = {
                        'start_seen': any(pair[0] in states for states in training_episodes),
                        'goal_seen': any(pair[1] in states for states in training_episodes),
                        'forward_pair_seen': any(pair[0] in states and pair[1] in states
                            and states.index(pair[0]) < len(states) - 1 - states[::-1].index(pair[1])
                            for states in training_episodes),
                    }
                    eval_rows.append(row)
            evaluation = {
                'schema_version': 'copuzzle.permutation_physics_goal_evaluation.v1',
                'purpose': 'development oracle-plan execution; not learned performance or a frozen benchmark',
                'task': task, 'rule_id': env.rule.rule_id, 'logical_difficulty': 'exact_minimum_atomic_operations',
                'policy': 'privileged_exact_oracle_plan_then_observation_only_contact_executor',
                'privileged_rule_access': True, 'uses_solver': True,
                'observation_distances_or_solutions': False,
                'pair_semantics': 'reset pairs excluded from train/val and prior evaluation cases; trajectory overlap separately audited',
                'training_overlap_definition': 'phase-zero valid board states; forward pair occurs in order within one training episode',
                'summary': _summary(eval_rows), 'cases': eval_rows,
            }
            _write(output / task / 'goal_evaluation.json', evaluation)
            records[task] = {
                'configuration': env.configuration(), 'configuration_sha256': env.configuration_sha256,
                'summary': _summary(rows), 'by_condition': {c: _summary([row for row in rows if row['condition'] == c]) for c in conditions},
                'by_split': {s: _summary([row for row in rows if row['split'] == s]) for s in ('train', 'val')},
                'training_unique_ready_states': len(set(state for states in training_episodes for state in states)),
                'episodes': rows, 'goal_evaluation': _summary(eval_rows),
            }
            _write(output / task / 'collection_report.json', records[task])
        finally:
            env.close()
    if source != _sources():
        raise RuntimeError('Package source/assets or collector changed during generation; collection remains incomplete')
    manifest = {
        'schema_version': SCHEMA, 'status': 'complete', 'versions': _versions(),
        'purpose': 'development continuous-control dataset with scripted exploration and privileged expert references',
        'configuration': {'tasks': list(tasks), 'conditions': list(conditions), 'train_episodes': train_episodes,
                          'val_episodes': val_episodes, 'min_operations': min_operations, 'max_operations': max_operations,
                          'max_steps': max_steps, 'seed': seed, 'noise_amplitude': float(noise_amplitude),
                          'evaluation_cases_per_depth': evaluation_cases_per_depth},
        'split_semantics': 'independent seeded train/val trajectories, not state-disjoint; no novel-state claim',
        'seed_derivation': 'first32bits_sha256(schema,seed,task,split,index,purpose)',
        'goal_success_interpretation': 'play/noisy goals deliberately outside planned path; use planned_sequence_success_rate for primitive execution',
        'recovery_interpretation': 'noisy trajectories intentionally abort one partial stroke, release, and retry; this is scripted recovery',
        'limitations': ['ideal docking welds and rack/gear linkages', 'no frictional grasping',
                        'state observations, no image training corpus', 'no real-robot or domain-randomization validation'],
        'source_scope': 'package .py/.xml/.obj/.json/.md files and this collector script',
        'source_sha256': source, 'tasks': records,
        'files': {str(path.relative_to(output)): _hash(path) for path in sorted(output.rglob('*')) if path.is_file()},
    }
    _write(output / 'manifest.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument('--train-episodes', type=int, default=24)
    parser.add_argument('--val-episodes', type=int, default=6)
    parser.add_argument('--min-operations', type=int, default=4)
    parser.add_argument('--max-operations', type=int, default=6)
    parser.add_argument('--max-steps', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noise-amplitude', type=float, default=.025)
    parser.add_argument('--evaluation-cases-per-depth', type=int, default=1)
    args = parser.parse_args(argv)
    try:
        manifest = build_collection(args.output, tasks=args.tasks, conditions=args.conditions,
                                    train_episodes=args.train_episodes, val_episodes=args.val_episodes,
                                    min_operations=args.min_operations, max_operations=args.max_operations,
                                    max_steps=args.max_steps, seed=args.seed, noise_amplitude=args.noise_amplitude,
                                    evaluation_cases_per_depth=args.evaluation_cases_per_depth)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({'status': manifest['status'], 'manifest': str(args.output / 'manifest.json'),
                      'tasks': list(manifest['tasks'])}, indent=2))


if __name__ == '__main__':
    main()
