# Direct-contact tabletop models

These models replace the physical interpretation of the earlier indexed
permutation experiment. The archived experiment used three remote rack inputs
and ideal weld clamps. It remains reproducible, but does not reproduce either
game's physical apparatus. Do not mix its trajectories with these new models.

```python
import copuzzle

topspin = copuzzle.make_manipulation_env("topspin-20-4")
pancake = copuzzle.make_manipulation_env("pancake-4")
```

Registered Gym IDs are `CoPuzzle/TopSpin20x4Contact-v0` and
`CoPuzzle/Pancake4Contact-v0`. These are explicitly versioned contact-manipulation
prototypes. They are different from the archived `PhysicalTopSpin8x4-v0` and
`PhysicalPancake4-v0` fixtures. The original abstract games and data are unchanged.

## TopSpin: a track and a horizontal turntable

The model follows the physical arrangement in
[US4871173A, figures 1 and 2](https://patents.google.com/patent/US4871173A/en): a
closed, filled track and a straight four-token channel on a turntable rotating
about a vertical axis. It uses 20 numbered rigid tokens. It is not the previous
eight-token ring whose pieces flipped above and below the table.

The default `engaged_v2` robot directly pushes the turntable handle or inserts
a cylindrical pin into a token's blind socket to move the train. The socket
has a solid floor and separate colliding wall sectors. Lateral force passes
through the socket wall; it does not rely on dragging across the numbered face.
The earlier friction-probe geometry remains available as `legacy_v1` for exact
historical replay. Tokens have no actuators or welds. Fixed channel walls, rotor
banks and the support surface have collision geometry. The closed rotor banks
block entry from the inactive track interfaces during rotation. The rotating
support plate has a matching opening in the fixed floor, avoiding intersecting
solid surfaces.

Token horizontal motion, yaw and small vertical motion are simulated. Their
weight is supported by normal floor contact; restricted tilt and vertical travel
represent the toy's retaining lips. This is an explicit rigid-body approximation
of retention, not an unconstrained tabletop puck task. The robot and all tokens
retain their actual collision bodies. Face numerals are visual-only markings.

Actions are continuous XYZ increments for the robot's probe. The observation
includes robot state, actual token positions/motion, turntable angle and speed,
handle position, the last stable token order and board validity. Version 2 adds
contact identity, normal force, contact-plane slip speed, pin-relative velocity,
tool clearance and measured atomic-operation count. The logical
order is decoded around the track relative to the turntable window. Real gaps
may redistribute; nominal reset slots are not treated as invisible target motors.
An operation is accepted from the full measured permutation after withdrawal
and settling. Repeating three one-cell shifts costs three atomic operations;
there is no new one-action three-cell shortcut in the puzzle rules.

This physical 20/4 task does not have the small abstract 8/4 oracle's difficulty
table. Do not reuse the latter's distance 22, goals, datasets or checkpoints.
Physical operation validation uses measured shifts/reversals and actual contact,
not a claimed exact 20-token shortest-path solver.

See [TopSpin model, controls and native trace contract](topspin_physical.md) for
the current bearing/contact parameters and operation/replay checks.

## Pancake: supported sorting discs and a contact clamp

Four rigid discs with different diameters form a gravity-supported stack on a
plate. They have symmetric central spacing bosses that leave a narrow insertion
gap for a two-fork clamp. These are rigid laboratory sorting discs, not a model
of flexible food or fluid cooking. Their two faces are equivalent under the
unsigned pancake rule.

The robot inserts the lower forks beneath a chosen prefix, closes the upper
forks, lifts the selected discs, turns the clamped group, places it onto the
remaining stack, releases it and withdraws. The discs are carried through normal
and frictional contacts. There are no token welds or token actuators, and no
pose assignments during `step`. The plate, support, tool, arm and discs all
participate in collision checking.

Actions are seven components: normalized XYZ increments, world-frame rotation
increments, and an absolute normalized jaw-closing command. The first six lie
in `[-1,1]`; the seventh lies in `[0,1]`. Robot, target/controller state, jaw
state, disc poses and velocities, and measured clamp contacts are observable.
Persistent IK command state is reset and recorded for reproducible replay.

The board is decoded from the supported disc stack. A correct ordering while
the discs are in the air is not success. Success requires the target stack to
be stable at its support, with the clamp released and withdrawn. The known
prefix sequence used by a scripted demonstrator is an engineering reference,
not a learned planning result.

The contact executor has been checked on all 24 initial disc orders and all
three nontrivial prefix lengths (2, 3 and 4), and can return the empty clamp to a safe pose between
operations. This is a finite engineering coverage check of the fixed geometry
and controller, not a claim of robustness to new materials, dimensions or robots.

Physical grasp count is selectable from **1, 2, 3, 4**. The single-disc case has
its own execution diagnostic because flipping one unsigned disc leaves the
permutation unchanged:

```python
from copuzzle.games.pancake.physics.scripted import PancakeFlipPolicy

env = copuzzle.make_manipulation_env(
    "pancake-4", task_mode="operation", operation_prefix=1)
observation, info = env.reset()
policy = PancakeFlipPolicy(prefix=1)
while True:
    observation, reward, terminated, truncated, info = env.step(policy(observation))
    if terminated or truncated:
        break
env.close()
```

The diagnostic requires measured grasp/lift of the requested group, airborne
inversion, supported restacking, release and withdrawal. An unselected disc
moving over 10 mm or tilting beyond the declared normal-vector tolerance
invalidates it. Phase labels cannot declare success. The
configuration/trajectory schemas are `copuzzle.pancake_operation_config.v1`
and `copuzzle.pancake_operation_trajectory.v1`. Normal goal-mode configurations
and observations remain unchanged; the abstract game still exposes only prefix
lengths 2, 3 and 4. All 24 single-disc initial orders have been recorded and
replayed, alongside fresh 3- and 4-disc demonstrations in
`artifacts/pancake-grasp-counts-v1`.

## Fidelity and data boundary

The [continuous data and evaluation pipeline](manipulation_pipeline.md) provides
versioned multi-operation task suites, reference/observation-only policy runs,
native trajectory collection and per-task continuous transition batches. Its
development splits separate start/goal pairs and report remaining state overlap.
It does not turn reference-controller demonstrations into learned-policy results.

Static renders establish appearance only. Acceptance additionally checks:

- grounded support and real collision geometry for visible functional parts;
- full operation paths, including robot self-collision and robot/fixture contact;
- actual tool contact and measured object motion;
- zero live object pose edits outside reset;
- release, settling, and the physical goal predicate;
- saved continuous-action replay against the exact recorded model/configuration.

Small retained validation episodes belong to their own native schemas and model
hashes. They do not constitute a new frozen offline training benchmark. Bulk
collection, systematic material/geometry randomization, image training,
hardware transfer and learned physical control remain separate research work.
The model/configuration and validation report, not a successful animation alone,
define the tested scope.
