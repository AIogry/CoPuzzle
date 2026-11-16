# Physical permutation fixtures

> Archived experimental apparatus: this three-rack/weld-clamp model is retained
> for reproducing its development data. It is not a reconstruction of physical
> TopSpin or a contact-grasped pancake stack. Use the separately versioned
> [direct-contact tabletop models](direct_manipulation.md) for the rebuilt facilities.

The optional MuJoCo backend supports `topspin-8-4` and `pancake-4`. A UR5e uses
the existing fixed probe to compress spring-loaded racks. Rack/gear constraints
rotate carriers, and captured rigid tokens move under MuJoCo integration. An
operation is decoded from measured token positions and audited against the rule.

This is a **reduced mechanical fixture**, using ideal docking clamps, gear
linkages and interlocks. It does not implement frictional gripper grasping, a
CAD replica of a commercial toy, deformable pancakes, or hardware transfer.
The arm uses three continuous XYZ commands with fixed downward orientation.
Tokens themselves move continuously; `step` never assigns their poses.

## Mechanisms and game contract

| Task | Token arrangement | Atomic operations |
|---|---|---|
| TopSpin-8-4 | Eight rigid tokens at indexed ring slots above an open base | Rotate the ring left/right by one eighth; reverse the first four around their bisecting radial axis |
| Pancake-4 | Four rigid tokens captured at vertically spaced fixture slots | Rotate the first 2, 3, or 4 tokens 180 degrees about their prefix midpoint |

TopSpin keeps a fixed reference position; rotation is not goal equivalence.
Pancake is unsigned: token face orientation does not affect the goal. Racks
select the mechanically allowed operations; arbitrary token pickup is absent.

On measured probe contact and initial rack travel, the interlock transfers
affected tokens from docking clamps to a carrier. Switchable weld references
capture the **current** relative poses without changing positions or velocities.
The coupled rack and carrier then move under contact forces. At a valid completed
permutation, docking clamps capture the moved tokens and seat the small residual
error through constraint forces. The unloaded carrier returns under the preloaded
spring and settles before becoming ready. Other racks are interlocked meanwhile.

This idealizes clamps, transmissions and interlocks. Rack contact, carrier/token
motion, reaction forces and passive return are simulated. Carrier/arm gravity
compensation, rack preload, geometry and model hashes are recorded; token gravity
remains active. Layout and calibration are fixed in this first version.

## Environment and observation

```python
import copuzzle
env = copuzzle.make_physics_env("topspin-8-4", max_episode_steps=3000)
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step([0., 0., -0.5])
env.close()
```

Gym IDs: `CoPuzzle/PhysicalTopSpin8x4-v0`, `CoPuzzle/PhysicalPancake4-v0`.
Install `copuzzle[physics]`; rendering needs the existing render extra and a
supported GL backend, e.g. `MUJOCO_GL=egl` on a compatible Linux host.

Actions are XYZ increments scaled by 0.012 m relative to the measured probe tip.
Physics/control intervals default to 0.002/0.05 seconds. IK edits scratch data;
the arm moves through joint actuators.

Observations include board/goal, board validity, mechanism phase, active operation,
robot joints and tool pose, token poses/velocities, indexed slots, rack positions
and velocities, carrier angles and velocities, previous events and remaining
control steps. Token arrays are indexed by known identity: perception is supplied.

`board` is the last committed permutation during motion. `board_valid=0` marks
in-flight configurations. Phases are 0 ready, 1 carrying, 2 returning, 3 fault.
Success requires phase 0, a valid measured target board, token world velocity
components below 0.01, carrier speeds below 0.1 rad/s, and rack speeds below
0.01 m/s. The arm itself need not stop. Reward is 0 on success, otherwise −1 per
control interval; success takes precedence over the timeout.

Holding a completed rack produces one event and waits for release. An incomplete
operation can return to its original configuration when released; the aborted
attempt is recorded without a successful operation event. A mechanism mismatch
enters a fault phase instead of inventing a corrected board.

## Trajectories, replay and goals

`copuzzle.permutation_physics_trajectory.v1` stores T continuous actions and T+1
complete observations, ordered events with control/substep indices and durations,
rewards/end flags, and all info dictionaries including aborted attempts. JSON
binds configuration, model/source identities, reset and collector declarations;
NPZ checksums, shapes and dtypes are validated. Existing paths are not replaced.

Event order is retained because operations do not commute. A pure array audit
applies measured events in order and checks every committed board. Loading and
sampling neither import MuJoCo nor query a solver.

```python
from copuzzle.games.permutation.physics.data import (
    load_physics_trajectory, PhysicsGoalSampler, replay_trajectory,
)
episode = load_physics_trajectory("episode.npz")
batch = PhysicsGoalSampler(episode, seed=0).sample(128)
```

Future goals come from later ready, stable boards. Robot/mechanism state remains
attached to each transition. Goal equality during carrying/return does not imply
termination. Rewards and terminal labels are recomputed. Collection cutoffs and
environment timeouts are distinct; only relabeled success removes bootstrapping.

Replay resets the recorded configuration and executes saved continuous actions,
comparing observations, events, info and end labels. It never regenerates actions
from a plan. Reset reconstructs mutable weld references and interlocks. Arbitrary
mid-episode snapshot restoration is not supplied. Cross-platform or cross-version
bitwise replay is not promised.

## Data collection and validation

`examples/generate_physical_permutation_collection.py` separately labels play,
perturbed execution and privileged oracle-plan episodes. Intended plans are
provenance; actual events are measured. Interrupted attempts/failures/cutoffs are
retained, and every saved episode is reloaded and action-replayed.

Separate fixed goals use exact logical plans and the scripted physical executor.
This tests mechanisms and execution, not a learned policy. Report logical distance,
actual events, control steps and simulated seconds separately. Shortest logical
paths are not minimum-time physical plans. Development data are not a frozen
benchmark or evidence of learned control.

Tests cover all six real-contact operations, arbitrary resets, ordered sequences
and inverses, zero action, disabled-contact rejection, invalid intermediate states,
abort recovery, mutable-model reset and stationary success. Instrumented substeps
verify that Python callbacks do not edit live qpos/qvel between integrations.
Independent force-driven mechanism probes in local `docs/physics_mechanism_probes/`
are explicitly separate from UR5e-collected trajectories.

Further work includes realistic clamp/contact geometry, free grippers and pose
control, variable layouts and mechanics, visual perception, larger distributions,
learned execution and a frozen evaluation protocol.
