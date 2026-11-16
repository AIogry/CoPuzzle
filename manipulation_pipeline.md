# Direct-contact task suites and continuous data

`copuzzle manipulation` turns the supported Lights Out board, Pancake contact
clamp and TopSpin pin/socket model into reproducible development tasks. This
layer keeps their native trajectory formats, continuous actions and termination
contracts. It does not change the mechanical models or legacy data.

## Create and run tasks

```sh
copuzzle manipulation suite --seed 41 --depths 1 2 3 \
  --cases-per-depth 1 --output artifacts/manipulation-suite-v1.json
MUJOCO_GL=egl copuzzle manipulation collect \
  --suite artifacts/manipulation-suite-v1.json \
  --output artifacts/manipulation-collection-v1 --policy reference
copuzzle manipulation inspect \
  --collection artifacts/manipulation-collection-v1 \
  --task pancake-4 --split train --bootstrap-mode finite_horizon
```

The default suite has 27 cases: three games, three reference operation lengths,
and three splits (`train`, `dev`, `test`). `--tasks` and `--splits` restrict a
collection without changing the suite. Existing outputs are never overwritten.
Simulation needs the optional `copuzzle[physics]` dependency; suite generation
and offline data loading do not construct a simulator.

Each task contains its reset state, goal, seed, control horizon and a known
reference sequence. That sequence demonstrates logical reachability. Its length
is an **upper bound in elementary puzzle operations**, not an exact shortest
path or a guarantee that the physical controller will succeed. TopSpin uses
20 tokens and does not inherit the abstract eight-token distance table.
Pancake task sequences use prefixes 2, 3 and 4; the single-disc identity
operation remains a separate execution diagnostic.

Start/goal pairs, including their reversed pairs, are disjoint across splits
within each game. Initial states, goals and intermediate states may overlap;
the suite and executed collection report this explicitly. A small development
suite must not be described as an unseen-state or geometry-generalization
benchmark. All three splits use the same declared mechanical model.

## Reference execution and policy evaluation

The `reference` collector uses the task's supplied operation sequence with the
existing observation-based robot executors. It is a privileged scripted
reference, not a learned planner. Every actual movement still comes from the
robot actuators and contact dynamics.

For an external policy, use `run_collection(..., policy_factory=factory)`.
The factory receives only the task ID, and each policy call receives a copied
public environment observation. Reference operations, future states, controller
phase logs and episode metadata are not passed to it. A new policy instance is
created for each task. The runner saves its identity in the collection report.

`--policy zero` provides a stationary negative control. `--max-steps` limits
collection without changing the environment horizon. To test true timeouts,
create a separately checksummed suite with a shorter `max_control_steps`.
These are different stopping conditions and remain different in the data.
After deliberately editing a suite budget in memory, explicitly rebind it with
`suite["sha256"] = suite_sha256(suite)` before `save_suite` or collection. Ordinary
validation and loading reject an edited suite with its old checksum.

The runner retains unsuccessful trajectories and includes all attempted tasks
in its reports. It distinguishes native goal termination, environment timeout,
collector cutoff, controller failure and infrastructure errors. An expected
safe abort can pass a mechanism diagnostic while still being a failed puzzle
solution. Collection or replay errors are visible; they are not discarded to
improve success rates. If no valid trajectory can be saved, the report retains
the error instead of fabricating one.

Reports include actual control steps, simulated seconds, and measured logical
operations when available. Operation counts come from actual press events or
released, supported, stable arrangements. Planned sequence length is never
substituted for physical cost. Unclassified rearrangements have unknown cost.
Three separate one-cell shifts count as three operations.

## Load continuous transitions

```python
import numpy as np
from copuzzle.manipulation import load_transitions

data = load_transitions(
    "artifacts/manipulation-collection-v1",
    task_id="pancake-4", split="train", bootstrap_mode="finite_horizon")
batch = data.sample(64, rng=np.random.default_rng(7))
```

Each dataset contains exactly one task, split and complete environment
configuration. Lights Out/TopSpin commands remain `float32[T, 3]`; Pancake
commands remain `float32[T, 7]`, including the absolute jaw command. Model
profiles, task modes and action dimensions are not silently mixed.

`observations` and `next_observations` flatten the numeric public observation
fields in sorted name order, with `goal` kept separately as `goals`. The
`feature_spec` records every original name, shape, dtype and flattened slice.
Lights Out's static public structure and button layout are restored from its
native record. No normalization is fitted on dev/test, and no reference plan,
phase or diagnostic info is included in training batches.

The original goals and rewards are retained. There is **no hindsight relabeling**:
a physically achieved goal requires support, release and possibly a stability
history. A stale decoded board during airborne motion is insufficient. In
particular, older TopSpin mixed-operation traces used an off-path collection
goal and are not automatically relabeled as successful demonstrations.

`episode_offsets` and `episode_ids` preserve boundaries. Each sampled next state
comes from the same saved transition, never from the following episode.
`terminated`, `truncated` and `collection_cutoff` remain distinct. Their union
is available as `episode_end`.

Bootstrap convention is explicit: `terminated_only` masks true environment
termination only; `finite_horizon` also masks environment timeouts. A collection
cutoff alone is not an absorbing environment state in either mode. A learner
treating the observed `steps_remaining` as a finite task budget should choose
`finite_horizon`. Both the selected convention and the original flags are
retained, rather than assuming one training objective for every algorithm.

All declared files, native sidecars, suite bindings, configurations and ending
flags are validated. A selected split containing a missing trajectory or an
unresolved collection/replay error is rejected rather than silently filtered.

## Scope

This is the continuous data and evaluation boundary. CoPuzzleLab's abstract-game
learners use discrete actions and are not compatible with these batches. Its
separate OGBench Puzzle research path does not establish native CoPuzzle batch
compatibility. Learning from these batches requires an explicit adapter/learner
contract and a declared training protocol. A script's success
rate and a batch-loading check are not learned-policy results. The public
benchmark distribution and scientific training protocol remain unfrozen.
