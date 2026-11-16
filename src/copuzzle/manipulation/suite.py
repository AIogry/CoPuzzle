"""Deterministic native-manipulation tasks with bounded reference witnesses.

Reference operations are privileged scripted-collector inputs, not learner
observations. A reference length is a known solution-length upper bound, never
an exact distance. Only unordered endpoint pairs are partitioned; states and
witness paths may overlap across train/dev/test. No physics engine is imported.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
from itertools import combinations, permutations
import json
from numbers import Integral
import os
from pathlib import Path
import random
import re
import tempfile


SCHEMA_VERSION = 'copuzzle.manipulation_suite.v1'
TASK_IDS = ('grid-3x3', 'pancake-4', 'topspin-20-4')
SPLITS = ('train', 'dev', 'test')
MAX_CASES_PER_DEPTH = 32
MAX_SEED = 2**63 - 1
MAX_SAMPLING_ATTEMPTS = 512
DEFAULT_CONTROL_STEPS = {'grid-3x3': 1500, 'pancake-4': 8000, 'topspin-20-4': 5000}
REFERENCE_SEMANTICS = (
    'reference_operations are a privileged scripted-collector witness; '
    'reference_length is a solution-length upper bound, not a shortest distance'
)
SPLIT_SEMANTICS = (
    'unordered start/goal pairs are unique within each task_id across the entire suite; '
    'initial, goal, and witness states may overlap between splits'
)
_TOP_KEYS = {'schema_version', 'seed', 'depths', 'cases_per_depth', 'task_ids',
             'reference_semantics', 'split_semantics', 'tasks', 'sha256'}
_CASE_KEYS = {'id', 'task_id', 'split', 'reset_seed', 'state', 'goal',
              'reference_operations', 'reference_length', 'max_control_steps'}
_OPERATIONS = {'grid-3x3': tuple(range(9)), 'pancake-4': (2, 3, 4),
               'topspin-20-4': ('reverse', 'shift_left', 'shift_right')}


def _integer(value, name, minimum=0, maximum=MAX_SEED):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}], not a boolean')
    return value


def _argument_integer(value, name, minimum=0, maximum=MAX_SEED):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer, not a boolean')
    return _integer(int(value), name, minimum, maximum)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=True, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as error:
        raise ValueError('Suite must contain finite JSON-compatible values') from error


def _digest(suite):
    return hashlib.sha256(_canonical({key: value for key, value in suite.items()
                                      if key != 'sha256'})).hexdigest()


def _derived_seed(seed, *domain):
    return int.from_bytes(hashlib.sha256(_canonical([SCHEMA_VERSION, seed, *domain])).digest()[:8],
                          'big') & MAX_SEED


@lru_cache(maxsize=1)
def _lights_rule():
    # The canonical pure Lights Out forward rule; no environment or solver.
    from copuzzle.games.lights_out.rules import grid_rule
    return grid_rule(3, 3)


def _forward(task_id, state, operation):
    if task_id == 'grid-3x3':
        return tuple(_lights_rule().transition(list(state), operation).tolist())
    if task_id == 'pancake-4':
        return tuple(reversed(state[:operation])) + tuple(state[operation:])
    if operation == 'reverse':
        return tuple(reversed(state[:4])) + tuple(state[4:])
    if operation == 'shift_left':
        return tuple(state[1:]) + tuple(state[:1])
    return tuple(state[-1:]) + tuple(state[:-1])


def _board(value, task_id, name):
    n = {'grid-3x3': 9, 'pancake-4': 4, 'topspin-20-4': 20}[task_id]
    if type(value) is not list or len(value) != n or any(type(item) is not int for item in value):
        raise ValueError(f'{name} must be a JSON list of {n} integer entries, not booleans')
    if task_id == 'grid-3x3':
        if any(item not in (0, 1) for item in value):
            raise ValueError(f'{name} must be a nine-bit Lights Out state')
    elif sorted(value) != list(range(1, n + 1)):
        raise ValueError(f'{name} must contain each token 1..{n} exactly once')
    return tuple(value)


def _operation(value, task_id):
    expected_type = str if task_id == 'topspin-20-4' else int
    if type(value) is not expected_type or value not in _OPERATIONS[task_id]:
        raise ValueError(f'Invalid reference operation for {task_id}: {value!r}')
    return value


def _witness_path(task_id, state, operations):
    path = [tuple(state)]
    seen = {path[0]}
    for operation in operations:
        operation = _operation(operation, task_id)
        following = _forward(task_id, path[-1], operation)
        if following in seen:
            raise ValueError('Reference witness revisits a state or reaches its final goal prematurely')
        path.append(following)
        seen.add(following)
    return tuple(path)


def _pair_key(state, goal):
    # Sorting the endpoints makes the reverse ordered pair the same split unit.
    return tuple(sorted((tuple(state), tuple(goal))))


def _slots(task_id, depths, cases_per_depth):
    return [(f'{task_id}-{split}-w{depth}-{index:03d}', split, depth)
            for split in SPLITS for depth in depths for index in range(cases_per_depth)]


@lru_cache(maxsize=16)
def _pancake_witness_candidates(depths):
    """Enumerate bounded simple paths for capacity/allocation, not distances."""
    result = {depth: {} for depth in depths}
    maximum = max(depths)

    def walk(path, operations):
        depth = len(operations)
        if depth in result:
            key = _pair_key(path[0], path[-1])
            result[depth].setdefault(key, []).append((path[0], operations, path[-1]))
        if depth == maximum:
            return
        for operation in _OPERATIONS['pancake-4']:
            following = _forward('pancake-4', path[-1], operation)
            if following not in path:
                walk(path + (following,), operations + (operation,))

    for state in permutations(range(1, 5)):
        walk((state,), ())
    return result


def _pancake_allocations(slots, depths, cases_per_depth, rng):
    candidates = _pancake_witness_candidates(tuple(depths))
    # All subset capacity checks cover competition between witness-length pools.
    for size in range(1, len(depths) + 1):
        for subset in combinations(depths, size):
            available = set().union(*(set(candidates[depth]) for depth in subset))
            requested = len(SPLITS) * cases_per_depth * len(subset)
            if requested > len(available):
                raise ValueError(f'pancake-4 endpoint-pair capacity exceeded for witness lengths {subset}: '
                                 f'requested {requested}, available {len(available)}')
    choices = []
    for _, _, depth in slots:
        keys = sorted(candidates[depth])
        rng.shuffle(keys)
        choices.append(keys)
    owner, assigned = {}, {}

    def augment(slot, visited):
        for pair in choices[slot]:
            if pair in visited:
                continue
            visited.add(pair)
            previous = owner.get(pair)
            if previous is None or augment(previous, visited):
                owner[pair] = slot
                assigned[slot] = pair
                return True
        return False

    for slot in sorted(range(len(slots)), key=lambda i: (len(choices[i]), i)):
        if not augment(slot, set()):
            raise ValueError('pancake-4 cannot allocate unique endpoint pairs for the requested witness lengths')
    return [rng.choice(candidates[depth][assigned[index]])
            for index, (_, _, depth) in enumerate(slots)]


def _random_candidate(task_id, depth, rng):
    if task_id == 'grid-3x3':
        state = tuple(rng.randrange(2) for _ in range(9))
    else:
        values = list(range(1, 21))
        rng.shuffle(values)
        state = tuple(values)
    operations = tuple(rng.choice(_OPERATIONS[task_id]) for _ in range(depth))
    try:
        path = _witness_path(task_id, state, operations)
    except ValueError:
        return None
    return state, operations, path[-1]


def generate_suite(seed=0, cases_per_depth=1, depths=(1, 2, 3), task_ids=TASK_IDS):
    """Build three splits with unique unordered endpoint pairs per native task.

    ``depths`` are witness lengths 1..4, not shortest-path distances. Generated
    control budgets are fixed per task family. Validation also permits explicitly
    shortened positive budgets for genuine timeout tests, without changing physics.
    """
    seed = _argument_integer(seed, 'seed')
    cases_per_depth = _argument_integer(cases_per_depth, 'cases_per_depth', 1, MAX_CASES_PER_DEPTH)
    if not isinstance(depths, (tuple, list)) or not depths:
        raise ValueError('depths must be a nonempty tuple/list of witness lengths')
    depths = [_argument_integer(depth, 'depth', 1, 4) for depth in depths]
    if len(set(depths)) != len(depths):
        raise ValueError('depths must be unique')
    depths.sort()
    if (not isinstance(task_ids, (tuple, list)) or not task_ids
            or any(type(task) is not str or task not in TASK_IDS for task in task_ids)
            or len(set(task_ids)) != len(task_ids)):
        raise ValueError(f'task_ids must be a nonempty unique subset of {TASK_IDS}')
    tasks = []
    for task_id in task_ids:
        rng = random.Random(_derived_seed(seed, 'generation', task_id))
        slots = _slots(task_id, depths, cases_per_depth)
        used = set()
        allocations = (_pancake_allocations(slots, depths, cases_per_depth, rng)
                       if task_id == 'pancake-4' else None)
        for slot_index, (case_id, split, depth) in enumerate(slots):
            if allocations is not None:
                state, operations, goal = allocations[slot_index]
            else:
                for _ in range(MAX_SAMPLING_ATTEMPTS):
                    candidate = _random_candidate(task_id, depth, rng)
                    if candidate is None:
                        continue
                    state, operations, goal = candidate
                    if _pair_key(state, goal) not in used:
                        break
                else:
                    raise ValueError(f'Bounded sampling exhausted for {case_id} after '
                                     f'{MAX_SAMPLING_ATTEMPTS} attempts; reduce cases or change seed')
            used.add(_pair_key(state, goal))
            tasks.append({'id': case_id, 'task_id': task_id, 'split': split,
                'reset_seed': _derived_seed(seed, 'reset', case_id),
                'state': list(state), 'goal': list(goal), 'reference_operations': list(operations),
                'reference_length': depth, 'max_control_steps': DEFAULT_CONTROL_STEPS[task_id]})
    suite = {'schema_version': SCHEMA_VERSION, 'seed': seed, 'depths': depths,
             'cases_per_depth': cases_per_depth, 'task_ids': list(task_ids),
             'reference_semantics': REFERENCE_SEMANTICS, 'split_semantics': SPLIT_SEMANTICS,
             'tasks': tasks, 'sha256': '0' * 64}
    suite['sha256'] = suite_sha256(suite)
    validate_suite(suite)
    return suite


def _validate(suite, *, check_hash):
    if type(suite) is not dict or set(suite) != _TOP_KEYS:
        raise ValueError('Unexpected or missing suite fields')
    if suite['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported manipulation suite schema')
    _integer(suite['seed'], 'seed')
    count = _integer(suite['cases_per_depth'], 'cases_per_depth', 1, MAX_CASES_PER_DEPTH)
    depths = suite['depths']
    if type(depths) is not list or not depths:
        raise ValueError('depths must be a nonempty JSON list')
    for depth in depths:
        _integer(depth, 'depth', 1, 4)
    if depths != sorted(set(depths)):
        raise ValueError('depths must be unique and increasing')
    task_ids = suite['task_ids']
    if (type(task_ids) is not list or not task_ids
            or any(type(task) is not str or task not in TASK_IDS for task in task_ids)
            or len(set(task_ids)) != len(task_ids)):
        raise ValueError('Invalid, unknown, or duplicate task_ids')
    if suite['reference_semantics'] != REFERENCE_SEMANTICS or suite['split_semantics'] != SPLIT_SEMANTICS:
        raise ValueError('Suite reference/split semantics differ from the declared protocol')
    digest = suite['sha256']
    if type(digest) is not str or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('sha256 must be a lowercase SHA256 digest')
    tasks = suite['tasks']
    expected_count = len(task_ids) * len(depths) * len(SPLITS) * count
    if type(tasks) is not list or len(tasks) != expected_count:
        raise ValueError(f'tasks must contain exactly {expected_count} declared cases')
    ids, pairs, paths = set(), set(), set()
    groups, reconstructed = Counter(), []
    for task in tasks:
        if type(task) is not dict or set(task) != _CASE_KEYS:
            raise ValueError('Unexpected or missing task fields')
        case_id, task_id, split = task['id'], task['task_id'], task['split']
        if type(case_id) is not str or not re.fullmatch('[a-z][a-z0-9._-]{0,127}', case_id):
            raise ValueError('Task id must be a safe nonempty lowercase identifier')
        if case_id in ids:
            raise ValueError(f'Duplicate task id: {case_id}')
        ids.add(case_id)
        if type(task_id) is not str or task_id not in task_ids:
            raise ValueError('Task belongs to an undeclared task_id')
        if type(split) is not str or split not in SPLITS:
            raise ValueError('Task split must be train, dev, or test')
        _integer(task['reset_seed'], 'reset_seed')
        depth = _integer(task['reference_length'], 'reference_length', 1, 4)
        if depth not in depths:
            raise ValueError('Task reference_length is not declared in depths')
        _integer(task['max_control_steps'], 'max_control_steps', 1, DEFAULT_CONTROL_STEPS[task_id])
        state, goal = _board(task['state'], task_id, 'state'), _board(task['goal'], task_id, 'goal')
        operations = task['reference_operations']
        if type(operations) is not list or len(operations) != depth:
            raise ValueError('reference_operations length differs from reference_length')
        path = _witness_path(task_id, state, operations)
        if path[-1] != goal:
            raise ValueError('Recorded goal does not match the forward reference witness')
        path_key = (task_id, path)
        if path_key in paths:
            raise ValueError('Duplicate reference witness path')
        paths.add(path_key)
        pair = (task_id, _pair_key(state, goal))
        if pair in pairs:
            raise ValueError('Duplicate unordered start/goal pair, including a reversed pair')
        pairs.add(pair)
        groups[(task_id, split, depth)] += 1
        reconstructed.append(path)
    expected_groups = {(task_id, split, depth): count for task_id in task_ids
                       for split in SPLITS for depth in depths}
    if dict(groups) != expected_groups:
        raise ValueError('Task counts do not match every task/depth/split declaration')
    expected_digest = _digest(suite)
    if check_hash and digest != expected_digest:
        raise ValueError('Suite SHA256 checksum mismatch')
    return reconstructed


def validate_suite(suite):
    """Validate schema, witnessed transitions, split uniqueness, and checksum."""
    _validate(suite, check_hash=True)


def suite_sha256(suite):
    """Hash validated content, excluding the existing ``sha256`` value.

    This permits re-binding an intentionally shortened control budget. The
    existing hash must still have valid digest syntax; its value is ignored.
    """
    _validate(suite, check_hash=False)
    return _digest(suite)


def save_suite(suite, path):
    """Publish complete JSON without replacing an existing file or symlink."""
    validate_suite(suite)
    path = Path(path).expanduser().absolute()
    if path.suffix != '.json':
        raise ValueError('Suite path must have a .json suffix')
    if os.path.lexists(path):
        raise FileExistsError('Refusing to overwrite a manipulation suite')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(suite, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'Non-finite JSON number: {value}')


def load_suite(path):
    path = Path(path).expanduser().absolute()
    if path.suffix != '.json':
        raise ValueError('Suite path must have a .json suffix')
    with path.open(encoding='utf-8') as stream:
        suite = json.load(stream, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    validate_suite(suite)
    return suite


def split_overlap(suite):
    """Report exact endpoint/witness-state intersections for a valid suite.

    Pair disjointness is guaranteed; unseen-state separation is deliberately
    not claimed, even if a particular sampled split has zero observed overlap.
    """
    paths = _validate(suite, check_hash=True)
    comparisons = []
    for task_id in suite['task_ids']:
        by_split = {split: {'initial': set(), 'goal': set(), 'witness': set(), 'pairs': set(), 'cases': 0}
                    for split in SPLITS}
        for task, path in zip(suite['tasks'], paths):
            if task['task_id'] != task_id:
                continue
            entry = by_split[task['split']]
            entry['cases'] += 1
            entry['initial'].add(path[0])
            entry['goal'].add(path[-1])
            entry['witness'].update(path)
            entry['pairs'].add(_pair_key(path[0], path[-1]))
        for left_name, right_name in combinations(SPLITS, 2):
            left, right = by_split[left_name], by_split[right_name]
            initial = left['initial'] & right['initial']
            goal = left['goal'] & right['goal']
            endpoints = (left['initial'] | left['goal']) & (right['initial'] | right['goal'])
            witness = left['witness'] & right['witness']
            comparisons.append({
                'task_id': task_id, 'splits': [left_name, right_name],
                'unordered_pair_overlap': len(left['pairs'] & right['pairs']),
                'initial_state_overlap': len(initial), 'goal_state_overlap': len(goal),
                'left_initial_right_goal_overlap': len(left['initial'] & right['goal']),
                'left_goal_right_initial_overlap': len(left['goal'] & right['initial']),
                'endpoint_state_overlap': len(endpoints), 'witness_state_overlap': len(witness),
                'left_counts': {'cases': left['cases'], 'initial_states': len(left['initial']),
                                'goal_states': len(left['goal']), 'witness_states': len(left['witness'])},
                'right_counts': {'cases': right['cases'], 'initial_states': len(right['initial']),
                                 'goal_states': len(right['goal']), 'witness_states': len(right['witness'])},
                'shared_initial_states': [list(value) for value in sorted(initial)],
                'shared_goal_states': [list(value) for value in sorted(goal)],
                'shared_witness_states': [list(value) for value in sorted(witness)],
            })
    return {'schema_version': 'copuzzle.manipulation_split_overlap.v1',
            'suite_sha256': suite['sha256'], 'split_semantics': SPLIT_SEMANTICS,
            'unordered_pairs_disjoint': all(item['unordered_pair_overlap'] == 0 for item in comparisons),
            'unseen_state_split': False, 'comparisons': comparisons}


__all__ = ['SCHEMA_VERSION', 'TASK_IDS', 'SPLITS', 'MAX_CASES_PER_DEPTH', 'DEFAULT_CONTROL_STEPS',
           'generate_suite', 'validate_suite', 'suite_sha256', 'save_suite', 'load_suite', 'split_overlap']
