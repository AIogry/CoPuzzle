# Sliding Puzzle and Tower of Hanoi development interface

CoPuzzle includes two additional **abstract discrete simulation** families: Sliding
Puzzle and Tower of Hanoi. They add state-dependent legal actions, while retaining
arbitrary goal states, exact bounded reference solvers, offline trajectories, and
goal-conditioned Gym observations. These modules do not claim robot manipulation
or contact dynamics for either family.

Supported presets are `sliding-2x3`, `sliding-3x3`, and `hanoi-2` through `hanoi-6`.
Registered Gym presets are `CoPuzzle/Sliding2x3-v0`, `CoPuzzle/Sliding3x3-v0`, and
`CoPuzzle/Hanoi3-v0` through `CoPuzzle/Hanoi5-v0`; Hanoi-2/6 are available through
`copuzzle.make_env`. Environments and exact solvers
accept arbitrary valid goals. Sliding start and goal states must be in the same
reachable parity component; the environment rejects unreachable reset pairs.

## States, actions, and task semantics

| Family | `board` and `goal` | Action indices | Legality |
| --- | --- | --- | --- |
| Sliding Puzzle | Row-major `int32[N]` permutation of `0..N-1`; `0` is the blank | `0` blank up, `1` down, `2` left, `3` right | Blank must remain on the board |
| Hanoi | `int32[N]` peg assignments `0..2`, indexed smallest disk first | `0`: 0→1, `1`: 0→2, `2`: 1→0, `3`: 1→2, `4`: 2→0, `5`: 2→1 | Source nonempty; its top disk must be smaller than destination's top disk |

Each observation contains only `board`, `goal`, `previous_event`, and
`action_mask`. The mask is a `uint8[A]` vector for the **current state**. Events
are one-hot vectors of width `A`; reset has a zero previous event. A chosen
illegal action is rejected instead of being silently projected to a legal action
or inserted into the offline dataset. Policies must obey the supplied mask.

Reward is `0` when the successor equals the goal and `-1` otherwise. Success
terminates the episode; exhausting the step budget truncates it. An already solved
reset pair is marked successful and cannot be stepped until reset again. The environment reports evaluation information separately
from the policy observation. Reset and step return copies, so modifying an
observation cannot mutate the underlying puzzle.

Exact BFS is bounded to the supported sizes. Sliding 2×3 has 360 states in a
reachable component, and Sliding 3×3 has 181,440. All `3**N` Hanoi peg assignments
are valid. The solver supports arbitrary goals; Sliding normalization only
relabels nonblank tiles and preserves the blank's location. The reported
`max_distance` is the **canonical goal's eccentricity**, not an assertion that all
arbitrary goals have identical distance distributions.

## Offline trajectory contract

The separate schema `copuzzle.classic_trajectory.v1` stores:

| Array | Shape | Type |
| --- | --- | --- |
| `states` | `(episodes, horizon + 1, state_size)` | `int32` |
| `actions` | `(episodes, horizon)` | `int64` |
| `events` | `(episodes, horizon, n_actions)` | `uint8` |
| `action_masks` | `(episodes, horizon + 1, n_actions)` | `uint8` |

All successors and the final state's mask are retained. Initial states are sampled
uniformly from valid state representations; actions are uniform among currently
legal actions. Sliding training data include both parity components. Every
individual trajectory stays in its own component. Independent seeded train/val
streams are **trajectory splits, not state-disjoint splits**.

Each NPZ has a sibling JSON file containing its SHA256, exact array schema,
canonical rule specification, generator version, seed, split, and sampling
semantics. Loading disables pickle and rejects checksum, dtype, shape, identity,
event/action, and stored legality inconsistencies. Publishing refuses to overwrite
either existing file. These checks verify the representation and stored mask
alignment; `validate(rule)` additionally recomputes all transitions and all masks
when explicitly requested. Collection generation audits the saved files with the
rule before marking completion.

Normal loading and goal relabeling never query transitions, masks, reachability,
or shortest paths. `GoalSampler` snapshots its dataset, owns a persistent RNG, and
samples strictly later goal indices whose state differs from the current state.
It returns both `action_masks` and `next_action_masks`, the complete successor,
previous/current events, labels, and episode/step/goal indices. Labels depend on
whether the successor equals the selected goal, even if that goal also occurs at
another later index. Terminal transitions have bootstrap mask `0`; nonsuccessful
time truncations retain bootstrap mask `1`. Its JSON-serializable RNG checkpoint
is bound to the full dataset content identity. Returned arrays are independent.
Explicit relabeling is available for arbitrary goals and timeout checks.

```python
from copuzzle.games.classic.data import load_dataset, GoalSampler

train = load_dataset("artifacts/classic-v1/hanoi-4/train.npz")
sampler = GoalSampler(train, seed=0)
batch = sampler.sample(256)
# Use batch["next_action_masks"] when maximizing a next-state action value.
```

## Exact development evaluation

Suites contain evaluator-only exact shortest distances. A policy receives copied
environment observations, never the case record, solver output, or distance. The
random reference samples legal actions; the exact oracle declares privileged rule
and solver access. Unknown policy declarations stay `null`. Arbitrary Python
policies are not sandboxed, and parameter freezing remains the caller's
responsibility.

The current task generator uses a fixed seeded pool of up to 16 distinct goals
from the canonical reachable component. **Sliding goals keep the blank at its
canonical cell while varying all other tile positions.** This restricted target
distribution is recorded explicitly and permits one BFS table to serve the 3×3
development suite. The environment and solver have broader arbitrary-goal support.
Within the recorded pool, the generator samples directed `(start, goal)` pairs
uniformly without replacement at each requested exact distance. Finite candidate
populations are checked before sampling; impossible requests fail.

Development and test suites exclude each other's directed pairs. This does not
establish training-state or training-pair novelty. Each evaluation with training
data reports the fraction of observed starts, goals, and pairs for which start
occurs strictly before goal in a single training trajectory. Oracle success is a
pipeline/reference check, not learned performance. Baseline reports are development
artifacts, not an untouched final benchmark.

## Commands and shipped development collection

```sh
copuzzle classic list
copuzzle classic generate --task hanoi-4 --output artifacts/my-hanoi-data
copuzzle classic validate artifacts/my-hanoi-data/train.npz --check-rule
copuzzle classic tasks --task hanoi-4 --depths 1,4,8,12,15 \
  --per-depth 8 --output artifacts/my-dev.json
copuzzle classic tasks --task hanoi-4 --depths 1,4,8,12,15 \
  --per-depth 8 --exclude-suite artifacts/my-dev.json --output artifacts/my-test.json
copuzzle classic evaluate --tasks artifacts/my-test.json --policy oracle \
  --training-data artifacts/my-hanoi-data/train.npz --output artifacts/my-oracle.json
copuzzle classic collection --output artifacts/my-classic-collection
```

`python -m copuzzle.games.classic.cli` exposes the same commands. Output locations
must be new. Collection defaults to Sliding 2×3 and Hanoi 3/4/5, with 1,000 train
and 100 validation trajectories per task, horizon 30, and 16 cases per exact depth
per evaluation split. Optional small Hanoi-2 collections may require a lower
`--per-depth` because their finite pair population is smaller.

The generated `artifacts/classic-v1/manifest.json` marks a complete development
collection: **4,000 train trajectories, 400 validation trajectories, 132,000
transitions, and 576 dev/test goal cases**. Every oracle case passed at its exact
minimum action count. Test random-reference success with seed 0 is:

| Task | Exact depth strata | Dev / test cases | Random test success |
| --- | --- | --- | --- |
| Sliding 2×3 | 1, 3, 6, 10 | 64 / 64 | 34.375% |
| Hanoi 3 | 1, 3, 5, 7 | 64 / 64 | 48.4375% |
| Hanoi 4 | 1, 4, 8, 12, 15 | 80 / 80 | 25% |
| Hanoi 5 | 1, 5, 10, 15, 23 | 80 / 80 | 18.75% |

Training visits every valid state in these small tasks (including both Sliding
parity components); training-pair overlap is measured in each report. These data
are therefore suitable for development and compositional testing, not a claim of
unseen-state generalization. The collection manifest hashes every produced file
and the seven Python sources under `src/copuzzle/games/classic/`; this source
snapshot does not include the top-level package registry or unrelated game modules.
The final manifest is written only after generation, reload validation,
random/oracle evaluation, and a source-stability check all finish successfully.
