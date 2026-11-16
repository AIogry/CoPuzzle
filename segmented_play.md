# Segmented direct-contact play

This protocol adds continuing local-operation data alongside the existing
fixed-goal suites. A physical trace has one reset and an uninterrupted control
clock. Its behavior goal changes only between completed, physically safe
operation segments. It is not a demonstration for one global final goal.

```sh
MUJOCO_GL=egl copuzzle manipulation play --task grid-3x3 --seed 17 \
  --operations 0 0 8 --output artifacts/lights-play-v1.npz
MUJOCO_GL=egl copuzzle manipulation play --task pancake-4 --segments 3 \
  --seed 17 --output artifacts/pancake-play-v1.npz
MUJOCO_GL=egl copuzzle manipulation play --task topspin-20-4 \
  --seed 17 --operations shift_right shift_left reverse --output artifacts/topspin-play-v1.npz
```

Without `--operations`, each next local operation is sampled uniformly using a
recorded NumPy RNG. Explicit operations are diagnostics, not random coverage.
Pancake supports prefixes 2/3/4 here; single-disc identity diagnostics stay in
their existing separate mode. Every output is a new NPZ/JSON pair; the command
loads it again and independently replays recorded actions and goal changes.

## Lifecycle and boundaries

The native model, actuator controller, contact dynamics and event detectors are
unchanged. `PlayBridge` implements a separate lifecycle: it releases the native
success lock to finish withdrawal, changes only goal/monitor bookkeeping, and
keeps the same physical state, IK targets, event history and global step count.
It never continues after a native error or the global horizon. This bridge
uses explicitly tested native observer fields; future native API changes must
be reviewed against its contract.

Reaching a goal and completing a safe segment are different events. Lights Out
can satisfy its lamp goal while a button is still pressed; the executor must
finish release/rearming and withdraw before switching goals. Pancake requires
a supported, stationary stack with the tool released and withdrawn. TopSpin
requires an aligned, stationary mechanism, tool clearance and three consecutive
stable observations of the target board. `controller_completed` alone never
establishes success.

The global horizon ends play even if the native environment reports success
on its final step. A segment limit or a collection cap is a collection cutoff.
A failed controller ends the trace with its complete measured prefix; the
collector does not reset and splice in another rollout or fabricate a failed
action's missing successor. An exception before any transition produces an
explicit error because there is no trajectory to save.

## Archive contract

`copuzzle.segmented_physical_play.v1` stores:

- Goal-independent numeric `observations[T+1]` and original continuous
  `actions[T,A]`; native `goal` and goal-dependent `success_streak` are excluded.
- Action-aligned `behavior_goals[T,N]`, `segment_ids[T]`, native rewards,
  `goal_reached`, global `timeouts` and `collection_cutoff`.
- Contiguous segment records `[start_step, end_step)` with requested local
  operation, behavior goal, first measured goal hit, controller completion,
  actual safe-boundary flag and success/failure/cutoff outcome.
- Native post-action diagnostics, initial reset parameters, native configuration,
  source identities, RNG policy and checked file hashes.

At boundary frame b, action b-1 and infos[b] still belong to the old goal;
action b uses the new goal. Goal changes add no artificial control actions or
zero-time physical frames. `infos` are audit records, not learner features.
Loading validates the exact single-reset declaration, uninterrupted time and
budget, target changes, physical evidence, segment coverage and end flags.
It requires NumPy and the package's base dependencies, not MuJoCo.
Native rewards describe the active behavior goal and are retained for audit.
They must not be treated as rewards for one fixed global goal across a switch;
use the explicitly recomputed goal views for that training objective.

## First-hit training views

```python
from copuzzle.manipulation.play import load_play_trace, goal_episode, future_episode

trace = load_play_trace('artifacts/lights-play-v1.npz')
segment = trace.segments[0]
view = goal_episode(trace, segment['start_step'], segment['goal'], segment['end_step'])
# Future targets must be physically achieved; cross-segment paths require opt-in.
future = future_episode(trace, 0, segment['end_step'], within_segment=True)
```

Views recompute rewards, goal termination and bootstrap masks for their selected
goal, then stop at its **first physical success**. They preserve source flags
under `source_*` names. A behavior-goal switch is not an environment terminal.
A future target frame may lie after the first hit; that does not extend the
resulting training episode beyond success. Cross-segment views are valid
observed paths, not shortest-path or expert-quality labels.

An already achieved starting goal returns zero actor transitions. For TopSpin,
if the starting board is physically ready and already equals the goal but the
three-frame confirmation is pending, the view separately reports
`initial_goal_confirmation_pending=True` and supplies no directed actor
supervision. It does not claim that the raw goal monitor has succeeded or
turn monitor-only waiting into a navigation/manipulation demonstration.

TopSpin state features include an independently recomputed, goal-independent
`settled_board_steps` counter (initial value 0, saturated at 3). It resets when
the board changes or physical readiness is lost. The original behavior goal's
success counter is never reused. Pancake cached board labels during airborne
or still-clamped states cannot certify future goals.

A goal view uses successful goal termination in preference to timeout when its
first hit is exactly at the final horizon; the raw play timeout remains visible
in `source_timeouts`. Bootstrap mode is explicit: `finite_horizon` masks true
goal termination and environment timeout; `terminated_only` masks true goal
termination only. Neither treats a mere collection cutoff as an absorbing state.
Goal views start new logical tasks at their selected physical frames. Their
observed hitting times are not exact shortest distances.

## Scope of this version

This is a collection, replay and label-contract implementation. Fixed-goal task
suites and evaluation remain unchanged. Play data and fixed-goal demonstrations
are separate artifacts; no mixing ratio or learned performance is asserted.
Choosing uniform operations improves neither coverage nor efficiency by itself:
loops, failed actions, state overlap and control costs must still be reported.
The supplied views do not train an actor or enforce external model freezing.
