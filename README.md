# CoPuzzle

**Development snapshot: `0.1.0.dev3`.** Abstract and optional MuJoCo Lights Out / Graph Lights Out, with trajectories for offline goal-conditioned reinforcement learning. The public benchmark task distribution and release protocol are not yet frozen.

CoPuzzle is an independent package. The base installation uses NumPy and Gymnasium. The optional physical environment adds MuJoCo and packaged, licensed UR5e assets. It requires no other research checkout, training framework, or local dataset. Rules, button scenes and reference controllers are implemented here.

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
copuzzle generate --task grid-3x3 --output artifacts/0.1.0.dev3/grid3 --episodes 1000 --val-episodes 100 --horizon 30 --seed 0 --observation-profile structure_v1
copuzzle validate artifacts/0.1.0.dev3/grid3/train.npz --check-rule
copuzzle tasks --task grid-3x3 --output artifacts/0.1.0.dev3/grid3/tasks.json --depths 1,2,3,4 --per-depth 16 --seed 100 --observation-profile structure_v1
copuzzle evaluate --tasks artifacts/0.1.0.dev3/grid3/tasks.json --policy random --seed 0 --output artifacts/0.1.0.dev3/grid3/random.json --training-data artifacts/0.1.0.dev3/grid3/train.npz
copuzzle evaluate --tasks artifacts/0.1.0.dev3/grid3/tasks.json --policy oracle --output artifacts/0.1.0.dev3/grid3/oracle.json
copuzzle diagnose --train artifacts/0.1.0.dev3/grid3/train.npz --test artifacts/0.1.0.dev3/grid3/val.npz --output artifacts/0.1.0.dev3/grid3/lookup.json
```

Output files are not overwritten. For another run, select a new output directory. Use `copuzzle list` to discover standard task sizes and suggested goal distances. Parameterized `grid-RxC`, `graph-pathN` (N >= 1), and `graph-cycleN` (N >= 3) are supported. These fixed graph instances are development tasks; the cross-size and cross-graph generalization protocol is not yet frozen.

## Optional physical environment

Install and run a real-contact diagnostic:

```bash
python -m pip install -e '.[physics]'
python examples/physical_lights_out.py --task grid-4x6 --output artifacts/physics-v1/grid4x6.npz
```

For an animation on a Linux EGL server:

```bash
python -m pip install -e '.[render]'
MUJOCO_GL=egl python examples/physical_lights_out.py --task grid-3x3 --output artifacts/physics-v1/grid3x3.npz --gif artifacts/physics-v1/grid3x3.gif
```

```python
env = copuzzle.make_physics_env("grid-4x5")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
# Also registered: gymnasium.make("CoPuzzle/PhysicalGrid4x5-v0")
```

The simulated UR5e moves a fixed pressing tool through three continuous XYZ
commands. Button contact and compression produce measured press events; the
shared logical rule then updates `board`. Goals remain binary lamp states.
Default Graph observations include adjacency; Grid observations include logical
coordinates. Physical position does not define graph connectivity.

The example records a short scripted episode, verifies action replay, and
samples goal-conditioned transitions. It is a contact diagnostic, not a learned
policy or a complete offline training distribution. See
[physical interfaces, data and limitations](physical_benchmark.md).

## Public structure protocol

New research runs should explicitly select `observation_profile="structure_v1"`.
The existing Python default and `CoPuzzle/*-v0` Gym IDs retain their original
hidden observations. The package version does not change the old trajectory
schema, RNG stream or environment defaults.

```python
import copuzzle

env = copuzzle.make_env("grid-4x5", observation_profile="structure_v1")
obs, info = env.reset(seed=0)
assert obs["grid_coordinates"][7].tolist() == [1, 2]
```

| Profile | Added observation fields |
|---|---|
| `legacy_hidden` (default) | None |
| `legacy_adjacency` | `adjacency` for both Grid and Graph; equivalent to old `reveal_graph=True` |
| `structure_v1`, Grid | `grid_shape: int32[2]`, `grid_coordinates: int32[N,2]` |
| `structure_v1`, Graph | `adjacency: uint8[N,N]` |

Every profile includes `board`, `goal`, and `previous_event`. Standard Grid
coordinates are zero-based **(row, column)** with the origin at the upper left:
`i = row * cols + column`. The old `rule.coordinates` remains **(column, row)**
display data. Grid family identity comes from explicit constructor metadata,
not a task-name parser. Graph adjacency is binary, symmetric, and has zero
diagonal; it is not the self-inclusive toggle matrix. Display layouts are not
part of the Graph learner input.

Use the new profile keyword separately from `reveal_graph`. Conflicting legacy
flags and explicit profiles fail. `reset`, `step`, and `observation_space` agree
on fields, shapes and dtypes. Returned arrays cannot change the environment.
A single environment has fixed N; create another environment for another size.

### What the structure condition measures

With adjacency and the known self-plus-neighbors mechanism, the transition is
analytically determined. This protocol studies learning updates, operation
representations and goal solving **conditioned on supplied structure**. Report
any hard-coded update formula, solver, local-neighborhood construction or real
environment query used by a method. Structure access is permitted; oracle
answers and shortest distances remain outside policy inputs.

Logical row/column coordinates and graph connectivity are independent of
physical button poses. The optional physical environment preserves this logical
context and separately supplies calibrated button positions and robot state.
It is a state-based task; learning perception from images remains future work.

### Offline structure bundles

The structured `generate` command writes:

```text
train.npz / train.json
val.npz / val.json
structure.npz / structure.json
bundle.json
generation.json
```

Structure is saved once per instance. `structure_sha256` covers its values,
identity, node order and semantic conventions; archive and manifest hashes
also bind the actual files. Loaders reject inconsistent dimensions, structure,
identities and checksums. Structure objects expose arrays and identity, without
transition or solver methods. Reading a bundle does not construct an environment
or query an executable rule.

```python
from copuzzle.games.lights_out.bundle import load_offline_bundle, BundleGoalSampler

bundle = load_offline_bundle("artifacts/0.1.0.dev3/grid3", split="train")
sampler = BundleGoalSampler(bundle, seed=7)
batch = sampler.sample(256)
inputs = {
    "board": batch["observations"],
    "goal": batch["goals"],
    "previous_event": batch["previous_events"],
    **batch["structure"],
}
```

Dynamic batch arrays have a leading batch dimension. The static structure has
one instance shape, with no repeated batch axis. External learners decide how
to broadcast, encode, bucket or pad it. The original `load_dataset` and
`GoalSampler` remain available for the legacy trajectory-only interface.

Run `python examples/structure_workflow.py` to check complete Grid and Graph
paths from saved offline data to policy evaluation. It uses an untrained random
policy and does not claim learned transfer.

### Existing data and task migration

Attach structure without changing or resampling source trajectories:

```bash
copuzzle bundle --train old/train.npz --val old/val.npz --output artifacts/structured-grid
copuzzle migrate-tasks --tasks old/tasks.json --output artifacts/structured-grid/tasks.json
copuzzle validate artifacts/structured-grid --check-rule
```

The destination must be new. Standard `train.npz`/`val.npz` names stay at its
root. Other source filenames are retained inside split subdirectories so their
original checksum manifests remain byte-for-byte compatible. Read paths through
`bundle.json`. Task migration retains starts, goals, exact distances and budgets;
its new observation condition has a distinct suite identity.

For an existing standard task-scale collection:

```bash
python examples/migrate_structure_collection.py --source artifacts/task-scales-v0 --output artifacts/structure-v1
```

The migration verifies the source manifest and records source, code and output
hashes. It does not create new held-out splits or establish zero-shot learning.
Artifacts are ignored by Git and must be distributed separately when publishing
a dataset; source checkout examples can reproduce the workflow.

### Custom graphs and evaluation

A JSON adjacency matrix can become a portable custom instance:

```bash
copuzzle structure --adjacency adjacency.json --name my-graph --output artifacts/custom/structure.npz
copuzzle generate --structure artifacts/custom/structure.npz --output artifacts/custom-data --observation-profile structure_v1
copuzzle tasks --structure artifacts/custom/structure.npz --output artifacts/custom-data/tasks.json --observation-profile structure_v1 --depths 1 --per-depth 8
copuzzle evaluate --tasks artifacts/custom-data/tasks.json --policy oracle --output artifacts/custom-data/oracle.json
```

For a named instance, use `copuzzle structure --task grid-4x5 --output ...`.
The Python structure API is `export_structure(rule)`, `save_structure(snapshot,
path)` and `load_structure(path)`. Benchmark preparation/evaluation may explicitly
use `rule_from_structure`; learners should use the public snapshot.

Structured suites resolve relative structure references and support custom graph
names without a source checkout registry. `save_task_suite` reuses matching
structure files in the same directory. Keep the referenced NPZ and JSON when
copying a suite. `policy(observation) -> int` receives the target instance's
structure with no task answers. Reports identify the profile, fields, structure
hash and method declarations. Cross-size mode describes the caller's protocol;
the evaluator cannot prove that an arbitrary callable has frozen its parameters.

An external training project needs to pass this context in both training and
inference, use the **target** node count for action masks, and restore a model
that actually supports that size. Keep source training identities in checkpoint
provenance while allowing explicitly declared target structures. Frozen
parameters and training statistics are distinct from inference RNG, recurrent
state or per-case search caches. The benchmark does not modify an external
learner or its checkpoint format automatically.

## Offline goal-conditioned learning

The main benchmark task is to reach a supplied goal board from a supplied start
using an offline-trained policy. Goals can be any reachable board, including
boards with lights on. The policy is conditioned on both state and goal:
`action ~ policy(board, goal, previous_event, public_structure)` for the new
`structure_v1` protocol. Legacy observations omit public structure.

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

It checks goal-dependent labels against the environment on all standard presets and
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

After importing `copuzzle`, all twelve standard presets have Gymnasium registrations:

- `CoPuzzle/Grid3x3-v0`, `CoPuzzle/Grid4x4-v0`, `CoPuzzle/Grid4x5-v0`, `CoPuzzle/Grid4x6-v0`
- `CoPuzzle/GraphPath9-v0`, `CoPuzzle/GraphPath16-v0`, `CoPuzzle/GraphPath20-v0`, `CoPuzzle/GraphPath24-v0`
- `CoPuzzle/GraphCycle9-v0`, `CoPuzzle/GraphCycle16-v0`, `CoPuzzle/GraphCycle20-v0`, `CoPuzzle/GraphCycle24-v0`

## Task sizes and goal distances

Grid and graph presets align at 9, 16, 20, and 24 buttons. Path edges connect
consecutive node IDs; cycles additionally connect the last node to node 0.
Each environment instance has a fixed rule. `structure_v1` exposes its Grid
geometry or Graph adjacency; the legacy default keeps structure hidden.

| Task | Buttons | GF(2) rank | Maximum shortest distance |
|---|---:|---:|---:|
| `grid-3x3` | 9 | 9 | 9 |
| `grid-4x4` | 16 | 12 | 7 |
| `grid-4x5` | 20 | 20 | 20 |
| `grid-4x6` | 24 | 24 | 24 |
| `graph-path9` | 9 | 9 | 9 |
| `graph-path16` | 16 | 16 | 16 |
| `graph-path20` | 20 | 19 | 13 |
| `graph-path24` | 24 | 24 | 24 |
| `graph-cycle9` | 9 | 7 | 4 |
| `graph-cycle16` | 16 | 16 | 16 |
| `graph-cycle20` | 20 | 20 | 20 |
| `graph-cycle24` | 24 | 22 | 12 |

The last column concerns reachable pairs. Rank is an upper bound on distance,
not necessarily the diameter. State-space size is `2**buttons`; from a fixed
start only `2**rank` states are reachable. For example, 4x6 has 16,777,216
possible boards, while Path20 reaches half of all boards from any one start.

List presets or explicitly request their wider evaluation ranges:

```bash
copuzzle list
copuzzle tasks --task grid-4x6 --depths preset --per-depth 64 --seed 100 --output artifacts/grid4x6-goals.json
```

The catalog is available as `TASK_PRESETS` and `get_task_preset(task_id)` in
`copuzzle.games.lights_out.presets`. `--depths preset` spans short and longer
exact distances, reaching 18 for Grid4x5 and 22 for Grid4x6. Existing commands
that omit `--depths` keep the original `1,2,3,4` default. Other valid sizes can
use explicit comma-separated distances.

For full-rank rules, distance N has just one target effect: pressing every
button exactly once. It is a useful boundary check, but a weak sole measure
of goal-dependent choices, so the suggested full-rank presets omit that
boundary. It remains available through explicit `--depths`. Actual results
should be reported by distance; more buttons or a longer shortest solution
alone do not establish that one task separates learning algorithms better.

To prepare matching development data for all twelve presets from a source
checkout:

```bash
python examples/generate_task_scales.py --output artifacts/task-scales-v0
```

Defaults are 1000 training and 100 validation trajectories per task, 30
operations per trajectory, and 64 goal cases per suggested distance. A new
output directory is required. `--tasks grid-4x5 grid-4x6` prepares a subset.
Each task directory contains `train.npz`, `val.npz`, their checksum sidecars,
and evaluator-only `tasks.json`. The final `manifest.json` records parameters,
versions, package source hashes and output hashes. An interrupted run without
that final manifest is incomplete; use a new directory for a fresh run.

This recipe prepares comparable fixed-instance development runs. It does not
create a held-out state/combination protocol or separate tuning/final task
suites. At the same seed and button count, different rules reuse the initial-state
and button-action random streams, providing paired collection conditions.
This recipe uses the legacy hidden protocol; train those instances separately.
Use the collection migration example above to attach `structure_v1` context for
new experiments. Old data files remain compatible and need not be regenerated.

## Rules and operations

Each press flips its own light and its graph neighbors. Grids use orthogonal neighbors and row-major button IDs. Graphs are simple, undirected, and have no self-loops in their adjacency matrix.

For board `b`, event `e`, and true toggle matrix `M = I + adjacency` over GF(2):

```text
next_board = b XOR effect
effect = M @ e mod 2
```

An event marks **pressed buttons**; an effect marks **flipped lights**. One abstract trajectory step contains exactly one press. The pure rule API also supports binary press-parity vectors. The separate physical trajectory format retains continuous actions, event counts, ordered substep events and timing configuration.

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

dataset = load_dataset("artifacts/0.1.0.dev3/grid3/train.npz")
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

suite = load_task_suite("artifacts/0.1.0.dev3/grid3/tasks.json")
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

For cross-size evaluation, train only on declared source data and pass the target
instance's `structure_v1` observation to the same frozen policy. The evaluator
accepts different node counts and rule identities across suites; variable-size
model inputs, action outputs, checkpoint restoration and parameter freezing are
the external algorithm's responsibility. A fixed-width model must report an
unsupported size instead of being silently expanded or retrained. A frozen
train/dev/test benchmark distribution and dedicated representation probes remain
future work.

### Model prediction callback

```python
from copuzzle.games.lights_out.prediction import evaluate_predictor

validation = load_offline_bundle("artifacts/0.1.0.dev3/grid3", split="val")
report = evaluate_predictor(
    my_predictor, validation.dataset, structure=validation.structure,
    sequence_length=4, num_examples=128, seed=0,
)
```

`my_predictor(observation, action)` returns the next binary board. Its input is
only `board` and public structure; the second argument is a candidate button ID.
It never receives goal, true successor or effect labels. This prevents future
goals that happen to equal the successor from leaking prediction answers.
Later sequence steps use earlier predictions. Reports include intermediate and
endpoint errors, sampled episode/start indices, input identities, call counts
and inference time. Windows come from recorded trajectories; this is not an
automatic held-out composition split. A separately fitted representation probe
must be labeled as such, with its data and training budget recorded.

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
        presets.py           # Standard task IDs, Gym IDs and suggested depths
        env.py               # Lights Out Gymnasium environment
        structure.py         # Public geometry/topology snapshots and artifacts
        bundle.py            # Offline trajectories bound to public structure
        data.py              # Lights Out trajectory schema and relabeling
        oracle.py            # GF(2) reachability and shortest solutions
        diagnostics.py       # XOR effects and sequence diagnostics
        prediction.py        # Model callback for single/composed predictions
        evaluation.py        # Lights Out tasks and evaluation
    rules.py / env.py / ...  # Compatibility exports of Lights Out APIs
  examples/
    goal_conditioning.py    # Executable offline GCRL interface walkthrough
    generate_task_scales.py # Comparable legacy data and goals for standard sizes
    structure_workflow.py   # Public offline inputs through Grid/Graph evaluation
    migrate_structure_collection.py # Attach structure without resampling
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

No neural architecture is imposed. The proposed state-evaluation/operation module can later be compared through the same data and policy interfaces. A scripted physical executor is included for contact diagnostics; learned RL baselines belong in a separate algorithm project.

## Versions and naming

Public source, configuration, and documentation filenames use stable descriptive names. Software releases and reusable artifacts use version identifiers rather than calendar dates. Dated working notes belong in the ignored `docs/` directory or an external research-notes workspace. Public API and reproducibility instructions remain in tracked package documentation.

### Edits, commits, and releases

A source edit does not require a new package version. A Git commit records a
coherent change; a release identifies a selected, tested snapshot that others
can install and reproduce. A release can include many commits.

CoPuzzle uses Python's [version specification](https://packaging.python.org/en/latest/specifications/version-specifiers/)
for package identifiers, including `.devN`, and the compatibility distinctions
of [Semantic Versioning](https://semver.org/) for release planning:

- `0.1.0.dev3` is a development snapshot before `0.1.0`. Advance `.devN` when
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
| Software package | `0.1.0.dev3` | The software is released or a development snapshot is identified |
| Gym environment IDs | `CoPuzzle/*-v0` | Relevant environment behavior or observation/action/reward semantics change |
| Physical configuration / trajectories | `copuzzle.physics_config.v1` / `copuzzle.physics_trajectory.v1` | Physical configuration or continuous episode semantics change |
| Trajectory format | `copuzzle.trajectory.v1` | Stored fields, encodings, or their meaning change |
| Data generator | `copuzzle.uniform_operations.v1` | Generation/distribution or reproducible RNG semantics change |
| Public structure | `copuzzle.structure.v1` | Geometry/topology fields, indexing or meaning change |
| Offline bundle | `copuzzle.offline_bundle.v1` | Structure-to-trajectory binding changes |
| Task/result JSON formats | Legacy `copuzzle.goal_tasks.v0` / `copuzzle.evaluation.v0`; structured `copuzzle.goal_tasks.v1` / `copuzzle.evaluation.v1` | Their respective serialization contracts change |
| Prediction reports | `copuzzle.prediction.v1` | Predictor inputs or metric serialization change |
| Frozen benchmark suite | Not released | Task/data splits, difficulty, metrics, and evaluation budgets are formally fixed or revised |

Moving implementation files does not change game behavior, data schemas, generator versions, rule identities, or Gym IDs. Source filenames need no version suffix. Changing the generator/schema identifiers currently changes seed derivation, so they must not be mechanically synchronized with the package version.

Example local artifact path: `artifacts/0.1.0.dev3/grid3/`. Existing development results remain attributed to the version that generated them. A task artifact's hash identifies its concrete contents; its JSON schema version alone does not identify a frozen benchmark release.

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
