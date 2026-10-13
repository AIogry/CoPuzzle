# Physical Lights Out

The physical prototype connects a continuous-control UR5e arm to the same
Lights Out rules used by the abstract environments. It supports all 12 standard
Grid, Path and Cycle presets. This is a development interface for offline GCRL;
the physical training distribution and evaluation suite are not frozen.

## Installation and ownership

Install `copuzzle[physics]` for simulation or `copuzzle[render]` for the example
GIF exporter. Simulation and trajectory collection need no display. Rendering
requires a supported OpenGL backend; on Linux servers, set `MUJOCO_GL=egl`
before starting Python if EGL is available. The package does not change this
process-wide setting. `render_mode="rgb_array"` returns a diagnostic camera
image; this release does not define an image-observation training profile.

The robot comes from [MuJoCo Menagerie UR5e](https://github.com/google-deepmind/mujoco_menagerie/tree/c6c202003c72e87d079db69160e499fb0d83c535/universal_robots_ur5e),
at commit `c6c202003c72e87d079db69160e499fb0d83c535`.
The unchanged XML, meshes, BSD-3-Clause license and file hashes are included in
`physics/assets/ur5e/`. The button board, fixed pressing tool, controller and
environment are CoPuzzle code. No local OGBench or other project is imported.

## Environment contract

```python
import copuzzle
env = copuzzle.make_physics_env("grid-4x6", max_episode_steps=1500)
obs, info = env.reset(seed=0)
# Each component is a normalized Cartesian displacement command.
obs, reward, terminated, truncated, info = env.step([0.0, 0.0, -0.5])
configuration = env.configuration()
env.close()
```

Each command targets the measured tip position plus `0.02 * action` metres,
clipped to the workspace. Tool orientation remains downward. Damped Jacobian
IK produces joint targets; position actuators then move the robot during 25
MuJoCo substeps. IK only edits scratch kinematic data. It never teleports the
live robot during `step`. The arm uses gravity compensation; spring buttons
retain gravity and contact dynamics.

| Setting | Default |
|---|---|
| Action | `float32[3]`, components in `[-1, 1]` |
| Physics / control interval | 0.002 s / 0.05 s |
| Button travel | 0.012 m |
| Press / release thresholds | 0.008 m / 0.003 m |
| Horizon | 1500 control steps, 75 simulated seconds |
| Goal | `uint8[N]` desired lamp state |
| Reward | 0 after reaching goal, otherwise −1 per control step |
| Termination | Goal equality after the complete control interval |
| Truncation | Horizon reached without success |

After every physics substep, a button fires only if it is armed, has crossed
the compression threshold, and is in contact with the tool. It must release
before firing again. Holding a button down therefore does not repeatedly flip
lamps. The logical transition is driven by measured events, never the intended
button target. Events at the same substep are ordered by button index. The
shared Lights Out updates commute, but order and multiplicity are still saved.

`info["event_log"]` is a list of `{substep, button}` pairs with zero-based
indices within the last control step. Success takes precedence over a timeout.
An explicit unreachable goal is rejected; an already solved reset cannot be
stepped. After termination or truncation, call `reset`.

### Observation fields

| Field | Shape / meaning |
|---|---|
| `board`, `goal` | `uint8[N]`, lamp bits |
| Grid: `grid_shape`, `grid_coordinates` | `int32[2]`, `int32[N,2]`, logical row/column |
| Graph: `adjacency` | `uint8[N,N]`, public logical graph |
| `joint_positions`, `joint_velocities` | `float64[6]`, radians and radians/s |
| `end_effector_position` | `float64[3]`, tip world XYZ in metres |
| `end_effector_orientation` | `float64[4]`, world quaternion wxyz |
| `button_positions` | `float64[N,3]`, moving cap top positions |
| `button_rest_positions` | `float64[N,3]`, calibrated uncompressed top positions |
| `button_displacements`, `button_velocities` | `float64[N]`, compression metres and metres/s, positive downward |
| `button_armed` | `uint8[N]`, event detector's release/press state |
| `previous_event_counts` | `int32[N]`, measured counts in the previous interval |
| `steps_remaining` | `int32[1]`, remaining control-step budget |

Grid rows run along world +x, columns along +y. Graph buttons use a compact
rectangular arrangement based on node count; their physical proximity does
not determine graph edges. Physical XYZ and logical structure have separate
meanings and are stored separately. This is a calibrated state-input condition:
it does not measure whether a model can discover button positions or graph
structure from images. The original abstract hidden-structure profiles remain
available for those rule-discovery comparisons.

## Offline trajectories and goals

`copuzzle.physics_trajectory.v1` stores one variable-length episode. It has
T continuous actions, T measured event-count vectors, T rewards/end flags and
T+1 observations, including the final state. Ordered events retain their
control-step index, physics-substep index and button ID. Static public structure
and button rest positions are each stored once. Metadata includes reset seed
and options, the physical configuration and model hashes, and controller
collection declarations. NPZ data never require pickle; JSON binds file hashes
and array shapes/dtypes. Saving refuses to overwrite existing artifacts.

```python
from copuzzle.games.lights_out.physics import (
    load_physics_trajectory, PhysicsGoalSampler, replay_trajectory,
)
episode = load_physics_trajectory("artifacts/physics-v1/grid3x3.npz")
batch = PhysicsGoalSampler(episode, seed=0).sample(256)
policy_inputs = {
    **batch["observations"],
    **batch["structure"],
    "button_rest_positions": batch["button_rest_positions"],
}
```

Dynamic batch fields have a leading batch dimension. Static structure and
layout do not; the algorithm chooses how to broadcast or encode them. Loading
and sampling use NumPy and do not import MuJoCo. The abstract trajectory
format still means one discrete press per action; do not feed physical files
to its loader.

Future goals are later **board states**, excluding goals equal to the current
board. Robot states remain attached to the sampled transition. No-event frames
are eligible when a later board differs. An episode containing no board change
cannot provide this kind of future goal; explicit goal relabeling is available.
The sampler recomputes reward and success. A collector stopping early is a
`collection_cutoff`, distinct from the environment's `truncated` timeout.
Only success at the relabeled goal makes `bootstrap_masks` zero, following the
existing continuing-task time-limit convention. Algorithms implementing a
strict finite-horizon value instead must also mask environment timeouts.

`replay_trajectory(env, episode)` checks matching configuration and then
replays actual continuous actions, comparing every observation, measured
event, reward and end flag. It does not regenerate actions from a plan.
Identical configuration includes the MuJoCo version; cross-platform numerical
identity is not guaranteed.

## Diagnostics and next development steps

`examples/physical_lights_out.py` creates a short explicit goal, executes a
supplied button sequence through real contact, records/reloads it, checks
replay, and samples goals. `ButtonSequencePolicy` uses public robot/button
observations and previous events. It is a scripted execution reference, with
a caller-supplied operation plan; it is not a learned GCRL policy. The example
is intended for ordinary distinct-button sequences whose prefixes do not
already reach the final goal. The environment ends immediately on success.

The tests check contact, release/repress, data integrity, goal semantics,
determinism and standard-size reachability. These checks establish an executable
interface, not success guarantees for a learned policy. A three-operation demo
is not sufficient physical training data.

The next benchmark work is to define diverse play/noisy/expert collectors,
freeze separate train/validation/evaluation seeds and goals, and report both
goal success and execution reliability. Physical budgets use control steps
and simulated time; logical difficulty uses press distance. Compare rule
prediction, operation planning and contact execution separately before
attributing an end-to-end failure. Perception, randomized geometry, different
robots, free grippers and hardware transfer require additional protocols and
validation.
