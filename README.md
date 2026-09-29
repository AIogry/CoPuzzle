# CoPuzzle

**Development prototype: `0.1.0.dev1`.** Abstract Lights Out and Graph Lights Out, with operation trajectories for offline goal-conditioned reinforcement learning. The public benchmark task distribution and release protocol are not yet frozen.

CoPuzzle is an independent package. Its runtime dependencies are NumPy and Gymnasium. It neither imports nor requires another research checkout, simulator, training framework, or local dataset. All rules and reference algorithms are implemented here.

## Install and try

From this checkout, using Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

Generate a small complete example:

```bash
copuzzle generate --task grid-3x3 --output artifacts/0.1.0.dev1/grid3 --episodes 1000 --val-episodes 100 --horizon 30 --seed 0
copuzzle validate artifacts/0.1.0.dev1/grid3/train.npz --check-rule
copuzzle tasks --task grid-3x3 --output artifacts/0.1.0.dev1/grid3/tasks.json --depths 1,2,3,4 --per-depth 16 --seed 100
copuzzle evaluate --tasks artifacts/0.1.0.dev1/grid3/tasks.json --policy random --seed 0 --output artifacts/0.1.0.dev1/grid3/random.json --training-data artifacts/0.1.0.dev1/grid3/train.npz
copuzzle evaluate --tasks artifacts/0.1.0.dev1/grid3/tasks.json --policy oracle --output artifacts/0.1.0.dev1/grid3/oracle.json
copuzzle diagnose --train artifacts/0.1.0.dev1/grid3/train.npz --test artifacts/0.1.0.dev1/grid3/val.npz --output artifacts/0.1.0.dev1/grid3/lookup.json
```

Output files are not overwritten. For another run, select a new output directory. Replace `grid-3x3` with `graph-cycle9` to run the graph version. `graph-path9` and `grid-RxC` are also supported. These graph instances are initial fixtures, not a finalized graph generalization suite.

## Offline goal-conditioned learning

The main benchmark task is to reach a supplied goal board from a supplied start
using an offline-trained policy. Goals can be any reachable board, including
boards with lights on. The policy is conditioned on both state and goal:
`action ~ policy(board, goal, previous_event)`.

The interfaces form one goal-conditioned workflow:

```text
Fixed exploration trajectories (state, action, event, next_state)
  -> GoalSampler: attach a future goal and recompute reward/success
  -> Training batch (state, action, next_state, goal, reward, bootstrap_mask)
  -> External offline GCRL learner
  -> Frozen policy evaluated on fixed (start, goal) cases
```

Raw exploration need not pursue a collection goal. The same transition can be
reused with different training goals: a successor equal to the selected goal
gets reward `0` and terminates; for another goal it gets reward `-1` and
continues, subject to the operation limit. Rule/effect diagnostics supplement
the goal-reaching evaluation. They do not replace it.

Run the [goal-conditioning example](examples/goal_conditioning.py) from a source
checkout after installation:

```bash
python examples/goal_conditioning.py
```

It checks goal-dependent labels against the environment on all four presets and
shows how to map batch fields to policy inputs. This is an interface example;
it does not train a model. During offline training, use only the declared fixed
data. Environment interaction belongs to benchmark data preparation and frozen
policy evaluation.

## Environment contract

```python
import copuzzle

env = copuzzle.make_env("grid-3x3", max_episode_steps=30)
observation, info = env.reset(seed=0)
action = 4
observation, reward, terminated, truncated, info = env.step(action)
```

The [Gymnasium API](https://gymnasium.farama.org/api/env/) uses separate task termination and time truncation. CoPuzzle follows that API and supports `render_mode="ansi"`.

| Item | Meaning |
|---|---|
| `action` | Integer button ID in `[0, N)`; one ideal, successful press |
| `observation["board"]` | Current binary board, `uint8[N]` |
| `observation["goal"]` | Target binary board, `uint8[N]` |
| `observation["previous_event"]` | Press that produced the current board; all zeros at reset |
| `reward` | `0` if the **successor** equals the goal, otherwise `-1` |
| `terminated` | Success after the action |
| `truncated` | Operation limit reached without success |
| `info` | Success, elapsed operation count, opaque rule identity |

At the last allowed operation, success takes precedence over timeout. Reset is required after either ending. An explicitly supplied already-solved start is reported as successful at reset and cannot be stepped. Default resets sample a nontrivial reachable goal.

```python
observation, info = env.reset(options={"state": start_board, "goal": goal_board})
```

Explicit unreachable goals are rejected. A fixed graph is used throughout an environment instance. **Adjacency and coordinates are absent from default observations and training batches.** `reveal_graph=True` is a separately labeled information condition that adds adjacency; it is not the default protocol. Environment/reference objects hold the true rule for simulation and scoring, so hidden-graph experiments must not query those internals from the learning algorithm.

After importing `copuzzle`, Gymnasium registrations are available:

- `CoPuzzle/Grid3x3-v0`
- `CoPuzzle/Grid4x4-v0`
- `CoPuzzle/GraphPath9-v0`
- `CoPuzzle/GraphCycle9-v0`

## Rules and operations

Each press flips its own light and its graph neighbors. Grids use orthogonal neighbors and row-major button IDs. Graphs are simple, undirected, and have no self-loops in their adjacency matrix.

For board `b`, event `e`, and true toggle matrix `M = I + adjacency` over GF(2):

```text
next_board = b XOR effect
effect = M @ e mod 2
```

An event marks **pressed buttons**; an effect marks **flipped lights**. One trajectory step in this version contains exactly one press. The pure rule API also supports binary press-parity vectors. Aggregated physical intervals would need their event sequences, counts, and durations preserved separately.

```python
from copuzzle.games.lights_out.rules import get_rule, graph_rule

rule = get_rule("grid-4x4")
next_board = rule.transition(board, action)
custom_rule = graph_rule(adjacency, name="my-fixed-graph")
```

Rules are immutable. `rule_id` is a SHA256 of the ordered logical graph and rule version; display names and coordinates do not change it. Permuting node identities generally does.

## Offline dataset contract

`copuzzle generate` samples independent uniform binary initial states and uniform button actions. The default trajectory contains **30 operations and 31 states**. This matches the approximate operation count observed in OGBench play trajectories; it does not reproduce its physical controller distribution. [OGBench](https://seohong.me/projects/ogbench/) motivated the initial task family.

Each archive contains only:

| Array | Shape | Type |
|---|---|---|
| `states` | `[episodes, length + 1, buttons]` | `uint8` |
| `actions` | `[episodes, length]` | `int64` |
| `events` | `[episodes, length, buttons]` | `uint8`, one-hot |

The sibling JSON stores schema/generator versions, task and rule identity, generation parameters, NumPy version, and archive SHA256. It contains no adjacency or toggle matrix. Loading disables pickle and validates schema, checksum, dimensions, binary values, and action/event alignment. `dataset.validate(rule)` additionally performs an explicitly privileged audit of every transition.

All successor states are retained. Raw exploration has no active goal; its final boundary is a collection cutoff. Goals, rewards, and success labels are assigned when forming goal-conditioned training batches.

```python
from copuzzle.games.lights_out.data import load_dataset, GoalSampler

dataset = load_dataset("artifacts/0.1.0.dev1/grid3/train.npz")
sampler = GoalSampler(dataset, seed=7)
batch = sampler.sample(256)  # Reuse the sampler: RNG advances between calls.
```

Batch keys are `observations`, `next_observations`, `actions`, `events`, `previous_events`, `goals`, `rewards`, `terminated`, `truncated`, and `bootstrap_masks`. The current policy input uses current state, goal, and previous event. Its next-state input uses `next_observations`, the same goal, and `events` as the new previous event.

The sampler takes an independent snapshot and caches eligible future goals. A future goal is sampled from later indices in the same trajectory, excluding goals equal to the current state. Repeated future boards retain their multiplicity. Success is checked by state equality. `bootstrap_masks = 1 - terminated`; a collection cutoff alone does not remove bootstrapping.

For explicit goals use `relabel_transitions(dataset, episode_indices, step_indices, goals)`. It does not assert arbitrary supplied goals are reachable. One-shot `sample_goal_transitions(..., seed=...)` remains available; repeatedly supplying the same seed deliberately reproduces the same batch.

Independent split seeds produce **independent trajectories**, not necessarily unseen states. In particular, 3×3 has only 512 boards. Train/validation state overlap is expected. The current generator does not implement a state-disjoint generalization split.

## Evaluation and diagnostics

Goal tasks are fixed JSON artifacts with verified reachable `(start, goal)` pairs, an operation budget, and **exact minimum press counts**. Distance is determined by independent GF(2) elimination and minimum-Hamming solution enumeration, with a nullity budget. Generation fails explicitly if it cannot populate a requested stratum. Random scramble length is not used as a substitute for shortest distance.

```python
from copuzzle.games.lights_out.evaluation import load_task_suite, evaluate_policy

suite = load_task_suite("artifacts/0.1.0.dev1/grid3/tasks.json")
report = evaluate_policy(my_policy, suite)
```

`my_policy(observation)` returns one button ID. Optional `my_policy.reset()` is called before each case. The evaluator passes no solution, shortest distance, rule object, or future event to the policy. Reports include success, timeouts, press counts, excess presses on successful tasks, inference time, policy calls, and results by exact distance. Reports identify the task artifact and graph visibility condition.

Included checks and baselines:

- **Random policy:** untrained control baseline.
- **Exact-rule oracle:** privileged correctness reference, not a learned result.
- **Effect lookup:** fits per-button effects using only offline transitions, then predicts single-step and sequence outcomes. It is a supervised fixed-instance diagnostic, not RL or evidence of cross-size rule learning.
- **Prediction metrics:** complete successor/effect accuracy, overall bit accuracy, and accuracy on changed versus unchanged bits.
- **Composition examples:** contiguous event sequences within a trajectory, with their start and end boards; no automatic claim of unseen combinations.
- **Overlap audit:** checks whether evaluation starts, goals, and ordered start-goal pairs occur in a training archive.

Evaluator-only task files include shortest distances and must not be used as training labels for the main offline protocol. Freeze learning before task evaluation. Any true-rule query, additional simulator training interaction, online adaptation, or alternative information condition must be reported separately.

For cross-size evaluation, generate target-size tasks but train only on the declared source data. Input/action size support is the algorithm's responsibility. Current task presets and overlap reports provide building blocks; a frozen cross-size/cross-graph release protocol and dedicated operation-representation probes remain future work.

## Source layout and additional games

The project uses a Python `src` layout. Current game-specific implementations live under their family namespace:

```text
CoPuzzle/
  pyproject.toml
  MANIFEST.in
  README.md
  src/copuzzle/
    __init__.py              # Public entry points and Gym registrations
    cli.py                   # CLI; currently supports the Lights Out family
    games/
      __init__.py
      lights_out/
        rules.py             # Binary grid and graph rules
        env.py               # Lights Out Gymnasium environment
        data.py              # Lights Out trajectory schema and relabeling
        oracle.py            # GF(2) reachability and shortest solutions
        diagnostics.py       # XOR effects and sequence diagnostics
        evaluation.py        # Lights Out tasks and evaluation
    rules.py / env.py / ...  # Compatibility exports of Lights Out APIs
  examples/
    goal_conditioning.py    # Executable offline GCRL interface walkthrough
  tests/
    test_cli.py
    games/lights_out/        # Family-specific behavior and data tests
  docs/                      # Local research notes; ignored, not distributed
  artifacts/                 # Generated data/reports, organized by version
  dist/                      # Built distributions
  .venv/                     # Local Python environment
  .pytest_cache/             # Test cache
```

`copuzzle.games.lights_out` is the canonical implementation namespace. The existing `copuzzle.rules`, `copuzzle.env`, `copuzzle.data`, `copuzzle.oracle`, `copuzzle.diagnostics`, and `copuzzle.evaluation` imports remain compatible. Their `PuzzleRule`/`PuzzleEnv` names currently refer specifically to Lights Out; they are not cross-game base classes.

Additional families such as pancake, sliding puzzle, TopSpin, and Rubik's Clock can be implemented as sibling game packages. They are not implemented or registered yet. Each family must define its state encoding, action space and legality, goal predicate, transition/effect semantics, sampling, and diagnostic solver. The current assumptions that states are binary, action count equals state width, effects compose by XOR, and shortest solutions are found by GF(2) are specific to Lights Out.

The current trajectory schema is also family-specific. New games must not silently reinterpret its binary arrays. A future shared schema should separate state shape from action/event shape and declare each representation explicitly. Task difficulty must record whether a distance is exact, a bound, or unknown; an efficient exact solver is not a prerequisite for every future game.

Shared lifecycle, storage integrity, and evaluation utilities will be extracted when a second concrete game establishes the actual common requirements. No placeholder game implementations or speculative common base class are required now.

No neural architecture is imposed. The proposed state-evaluation/operation module can later be compared through the same data and policy interfaces. No learned RL baseline or physical executor is included in this first benchmark implementation.

## Versions and naming

Public source, configuration, and documentation filenames use stable descriptive names. Software releases and reusable artifacts use version identifiers rather than calendar dates. Dated working notes belong in the ignored `docs/` directory or an external research-notes workspace. Public API and reproducibility instructions remain in tracked package documentation.

### Edits, commits, and releases

A source edit does not require a new package version. A Git commit records a
coherent change; a release identifies a selected, tested snapshot that others
can install and reproduce. A release can include many commits.

CoPuzzle uses Python's [version specification](https://packaging.python.org/en/latest/specifications/version-specifiers/)
for package identifiers, including `.devN`, and the compatibility distinctions
of [Semantic Versioning](https://semver.org/) for release planning:

- `0.1.0.dev1` is a development snapshot before `0.1.0`. Advance `.devN` when
  issuing another named development snapshot, rather than after every edit.
- During `0.x`, compatible fixes use patch releases and feature additions or
  incompatible API revisions use minor releases. Describe any incompatibility
  explicitly; the API is still under development. This is our project policy.
- After `1.0.0`, compatible fixes increment PATCH, compatible features increment
  MINOR, and incompatible public API changes increment MAJOR.
- Published releases and retained reproducibility snapshots are immutable.
  Changed contents require a new release identifier before distribution.

Working changes may accumulate toward the next release. Record the exact commit
and whether local changes exist for experiments, alongside the package version,
configuration, dependencies and data/task hashes. When Git history is not yet
available, archive the actual source used and its checksum. A package version
alone cannot identify an edited checkout.

### Benchmark contracts

These identifiers version different contracts:

| Identifier | Current value | Change when |
|---|---|---|
| Software package | `0.1.0.dev1` | The software is released or a development snapshot is identified |
| Gym environment IDs | `CoPuzzle/*-v0` | Relevant environment behavior or observation/action/reward semantics change |
| Trajectory format | `copuzzle.trajectory.v1` | Stored fields, encodings, or their meaning change |
| Data generator | `copuzzle.uniform_operations.v1` | Generation/distribution or reproducible RNG semantics change |
| Task/result JSON formats | `copuzzle.goal_tasks.v0`, `copuzzle.evaluation.v0` | Their respective serialization contracts change |
| Frozen benchmark suite | Not released | Task/data splits, difficulty, metrics, and evaluation budgets are formally fixed or revised |

Moving implementation files does not change game behavior, data schemas, generator versions, rule identities, or Gym IDs. Source filenames need no version suffix. Changing the generator/schema identifiers currently changes seed derivation, so they must not be mechanically synchronized with the package version.

Example local artifact path: `artifacts/0.1.0.dev1/grid3/`. Existing development results remain attributed to the version that generated them. A task artifact's hash identifies its concrete contents; its JSON schema version alone does not identify a frozen benchmark release.

## Local development directories

| Directory | Created by | Purpose | Commit to Git? |
|---|---|---|---|
| `.venv/` | Python `venv` | Project interpreter entry points and installed dependencies | No |
| `.pytest_cache/` | pytest | Cache used across test runs, including failure information | No |
| `__pycache__/` | Python | Compiled bytecode cache | No |
| `*.egg-info/` | Packaging tools | Local package metadata, including editable installations | No |
| `build/`, `dist/` | Packaging tools | Build intermediates and distributable wheel/source archives | No; publish selected distributions separately |
| `artifacts/` | CoPuzzle commands | Generated datasets and reports | No; distribute frozen datasets separately |
| `docs/` | Project contributors | Local discussions and experiment notes | No, under this project's convention |

The dependency declaration is `pyproject.toml`. Recreate `.venv` on another machine using the installation instructions; do not copy it as part of the benchmark. `.pytest_cache` and `__pycache__` are disposable caches, not Python or puzzle environments. Ignored directories can still be visible locally. `MANIFEST.in` also excludes local notes and artifacts from source distributions.

## Release status

Environment IDs, schema versions, seeds, artifact hashes, and dependency versions are recorded so experiments can identify their protocol. Compatibility changes will require corresponding version changes. Before a public benchmark release, the project still needs a frozen task/data distribution, generalization splits, learned baseline results, and a license/distribution decision. The package has not been published or assigned a public leaderboard.
