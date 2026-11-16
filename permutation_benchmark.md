# Pancake and TopSpin development benchmark

These abstract environments introduce noncommuting position permutations beside
Lights Out's XOR operations. The first implementation covers 2–8 tokens; the
standard development collection uses Pancake 4/6/8 and TopSpin 8-4. It is an
unreleased addition to the current checkout, not a frozen public benchmark.

## State, actions and observations

Every board and goal is an `int32[N]` permutation of tokens **1 through N**.
Positions are indexed from zero. A goal is any reachable complete permutation;
it need not be sorted. Goals compare all positions exactly.

| Game | Task ID | Action encoding | Event width |
|---|---|---|---|
| Pancake | `pancake-N` | `a=0..N-2` reverses the first `a+2` positions | N−1 |
| TopSpin | `topspin-N-K` | 0: left by one, 1: right by one, 2: reverse first K | 3 |

TopSpin uses a fixed position marker. A rotation costs one operation and is not
an equivalent goal. All three operations cost one step; rotation-plus-reversal
macros are not part of this action encoding. `2 <= K <= N`.

```python
import copuzzle

env = copuzzle.make_env("topspin-8-4", max_episode_steps=30)
observation, info = env.reset(seed=7)
observation, reward, terminated, truncated, info = env.step(0)
env.close()
```

Gym IDs: `CoPuzzle/Pancake4-v0`, `CoPuzzle/Pancake6-v0`,
`CoPuzzle/Pancake8-v0`, `CoPuzzle/TopSpin8x4-v0`.

The policy receives `board`, `goal`, `previous_event: uint8[A]`, and
`action_mask: uint8[A]`. Every action is legal, so masks are all ones. Previous
event is zero at reset, then one-hot for the last executed operation. Operation
maps, exact distances and solutions are not observation fields. The task's
declared name/action encoding is configuration information, not a rule-learning
claim. Algorithms using analytic mechanics must disclose that separately.

Success after an operation gives reward 0 and `terminated=True`; other steps
give −1. The budget gives `truncated=True` only if unsolved. Success takes
precedence at the final step. Explicit equal start/goal resets are already
solved and cannot be stepped. Explicit unreachable goals are rejected. Reset
without options samples a uniform initial permutation and a uniform reachable
nontrivial goal. Returned observations are independent copies.

## Exact difficulty and reachability

`ExactOracle(rule)` exhausts the identity component with BFS, caching immutable
relative states, distances and shortest paths. Relabeling the source's tokens
reduces arbitrary start/goal pairs to this table. This handles disconnected
TopSpin configurations; it does not assume all N! permutations communicate.

The maximum supported N is 8 and the search ceiling is 40,320 states. If a
lower requested `max_states` is exceeded, search raises instead of returning
partial or mislabeled distances. Larger games require a separately specified
solver and difficulty protocol. Scramble length is never a shortest-distance
label. The oracle is a privileged correctness reference, not a learned policy.

## Offline storage and goal sampling

`copuzzle.permutation_trajectory.v1` has:

| Array | Shape | Dtype |
|---|---|---|
| states | E × (T+1) × N | int32 |
| actions | E × T | int64 |
| events | E × T × A | uint8, one-hot |

The JSON sidecar binds schema/generator versions, canonical rule specification
and identity, token/action dimensions, seeds, collection distribution, NumPy
version, array shapes/dtypes and archive SHA256. Saving refuses existing paths.
Loading disables pickle and validates permutations, action/event alignment and
artifact metadata. `dataset.validate(rule)` additionally audits recorded
transitions. Ordinary loading and goal sampling do not execute transitions or
query an oracle.

Collection samples uniform initial permutations and uniform actions. Stable
SHA256-derived initial/action streams depend on seed and split label. Different
split labels give independent trajectories; they do not create disjoint states.
Both trains and validation retain every successor, including the cutoff state.

```python
from copuzzle.games.permutation.data import load_dataset, GoalSampler

dataset = load_dataset("artifacts/permutation-v1/pancake-6/train.npz")
sampler = GoalSampler(dataset, seed=7)
batch = sampler.sample(256)
policy_inputs = {
    "board": batch["observations"], "goal": batch["goals"],
    "previous_event": batch["previous_events"],
    "action_mask": batch["action_masks"],
}
```

Future goals are later indices of the same trajectory, excluding boards equal
to the current state and retaining repeated-board multiplicity. The sampler
uses a snapshot of its dataset. It returns episode/step/goal indices for audit.
Reward and success are recomputed from successor equality with the chosen goal.
`bootstrap_masks = 1 - terminated`; the raw collection cutoff does not remove
bootstrapping. `relabel_transitions` accepts explicit goals but does not assert
their reachability. Use a persistent sampler so its RNG advances between calls.

## Reproducible collection and evaluation

```bash
copuzzle permutation collection --output artifacts/permutation-v1
```

Per preset: 1,000 train and 100 validation trajectories, 30 operations each,
32 evaluator cases per exact distance in each of dev and test. Pancake distances
are 1/2/3/4 for N=4, 1/3/5/7 for N=6, and 1/3/5/7/9 for N=8. TopSpin 8-4 uses
1/4/8/12/16/20/22. Dev and test use distinct seeds and have disjoint directed
start/goal pairs. Each suite's overlap with training states and same-trajectory
ordered pairs is reported; exclusion between dev/test does not imply exclusion
from training. Do not tune against the test suite.

Each task directory contains train/val archives and sidecars, `generation.json`,
`dev_tasks.json`, `test_tasks.json`, random and oracle reports for both suites,
and `lookup.json`. The collection's final `manifest.json` records configuration,
dependency versions, exact source hashes, output checksums and component sizes.
No final manifest means an interrupted or failed collection. Use a new directory
for a fresh run. Artifacts are Git-ignored and distributed separately.

For a smaller or separate workflow:

```bash
copuzzle permutation generate --task pancake-6 --output artifacts/pancake-small --episodes 100 --val-episodes 20
copuzzle permutation tasks --task pancake-6 --output artifacts/pancake-small/dev.json --seed 100
copuzzle permutation tasks --task pancake-6 --output artifacts/pancake-small/test.json --seed 200 --exclude-suite artifacts/pancake-small/dev.json
copuzzle permutation evaluate --tasks artifacts/pancake-small/test.json --policy oracle --output artifacts/pancake-small/oracle.json --training-data artifacts/pancake-small/train.npz
```

Python evaluation uses `evaluate_policy(policy, suite)`, where
`policy(observation) -> int`; an optional `reset()` runs before each case. Reports
record success, timeout, operation counts, excess steps on success, inference
time, per-distance metrics and task identity. Evaluator-only suite files contain
exact answers and must not become training labels for the offline GCRL protocol.

The position-map diagnostic fits each action's map solely from recorded unique
tokens. It predicts successors and ordered compositions on validation windows.
It is a fixed-instance lookup, not RL, learned cross-game transfer, or a test of
unseen compositions. It supplies a useful correctness ceiling for prediction.

## Remaining milestones

This addition supplies environments, offline artifacts and evaluator interfaces.
Large permutation sizes, state/combination-disjoint benchmark distributions,
modular dials and Rubik families remain future work. Sliding Puzzle and Hanoi
now have a [separate legal-operation interface](classic_benchmark.md), and
Pancake-4/TopSpin-8-4 have [physical fixture prototypes](physical_permutation_benchmark.md).
Learned algorithms
and their frozen cross-game experiments belong in CoPuzzleLab; environment and
oracle correctness alone do not demonstrate learning performance.
