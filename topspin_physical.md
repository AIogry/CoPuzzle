# Direct-contact TopSpin 20/4 prototype

This is a new physical environment with a planar closed track, twenty independent
numbered tokens, a four-token straight groove in a rotating insert, and a real
UR5e contact tool. It is separate from both abstract TopSpin-8-4 and the legacy
three-button permutation fixture. No legacy dataset or environment is replaced.

The mechanism follows the physical arrangement in
[US4871173A, original TopSpin patent](https://patents.google.com/patent/US4871173A/en):
tokens circulate in a filled endless track, and a straight portion turns 180°
about an axis perpendicular to the tabletop. The model is an engineered
simulation prototype, not a CAD reconstruction or parameter identification of a
commercial toy.

## Physical model

- The default `engaged_v2` UR5e directly pushes a visible crank to rotate the
  insert. It circulates the train by inserting a 3.2 mm radius cylindrical pin
  into a 5 mm radius blind socket on a token in the lower straight track. Real
  wall contacts transfer lateral force. Twenty-four convex wall sectors per
  token preserve the open entrance; a solid floor closes the bottom. The 1.8 mm
  nominal radial clearance is finite, so insertion/take-up motion is expected.
  There are no remote puzzle buttons or puzzle actuators.
- The earlier broad friction probe is retained as `facility_profile="legacy_v1"`.
  An audit of its saved mixed sequence confirmed changes of contact identity
  during both shifts; a correct final order was insufficient evidence of
  reliable engagement. Its old configuration, geometry and data remain available
  for exact replay and comparison. The new profile uses separate version-2
  configuration and trajectory schemas.
- Fixed supported walls contain the loop. Solid banks on the rotating insert
  close the two inactive track entrances while it turns. The stationary deck has
  a circular opening around the colliding rotating disk, preventing overlapping
  support geometry.
- Tokens have x/y/z translation and yaw degrees of freedom. Gravity and actual
  deck/disk contacts carry their weight; roll/pitch constraints and ±3 mm vertical
  retaining limits approximate the original toy's retaining groove lips. These
  limits do not carry the resting weight.
- The insert has a passive vertical revolute bearing, 0°/180° mechanical stops,
  and declared dry friction of 0.3 N m. This engineered bearing resistance keeps
  the groove aligned during circulation; the robot overcomes it by pushing the
  crank. MuJoCo's elliptic friction `impratio=10` and friction constraint settings
  are part of the saved model, not an unrecorded controller or state update.
- There are six actuators, all on the UR5e joints, and no equality constraints or
  welds. Token/insert coordinates are initialized only at reset. Every subsequent
  coordinate change comes from simulation integration and contact forces.

The model uses the project's verified, licensed MuJoCo Menagerie UR5e assets.
Configuration metadata records their commit, license and every asset checksum.

## Control and measured state

```python
from copuzzle.games.topspin.physics.env import TopSpinPhysicalEnv
from copuzzle.games.topspin.physics.scripted import TurntablePolicy

env = TopSpinPhysicalEnv(max_episode_steps=1000)
observation, info = env.reset(seed=17)
policy = TurntablePolicy(action_scale=env.action_scale)
while True:
    observation, reward, terminated, truncated, info = env.step(policy(observation))
    if terminated or truncated:
        break
env.close()
```

Actions are `float32[3]` vectors in `[-1, 1]`, requesting measured-tool Cartesian
increments. The default increment scale is 12 mm, physics timestep 2 ms, and
control timestep 50 ms. The fixed-orientation damped IK controller computes robot
actuator targets in scratch data; it does not overwrite live generalized state.

The state observation includes the twenty-token board/goal, measured robot joint
positions and velocities, token positions, token velocities, insert angle and
velocity, handle position, and a validity flag. Token velocity columns are
`xdot, ydot, zdot, yawdot`, with units m/s, m/s, m/s, rad/s. `slot_positions` is the
nominal reset layout; it is not a command to reposition tokens.

Version 2 additionally records a 20-element positive-load contact mask, summed
normal force per token, maximum relative contact-plane speed per token, and the
tool-site velocity relative to each token (including yaw contribution). The
mask identifies actual force-bearing contacts, not the controller's intended
target. Contact-plane speed and center-relative displacement are distinct
measurements. `tool_clear`, `success_streak` and `atomic_operation_count` expose
the measured completion state.

Real track gaps redistribute during circulation. Consequently, board decoding
uses measured clockwise track order, starting at the leftmost of the four tokens
fully inside the straight insert. It does not require every token to remain at an
equally spaced nominal slot. `board_valid` requires a settled, aligned insert and
a valid four-token window. In version 2 it also requires the tool above the
150 mm clearance plane and no token/handle contact. During manipulation, `board` retains the previous
committed permutation and is explicitly marked invalid.

Success requires the complete measured goal arrangement to remain settled with
the tool withdrawn for three control intervals. The insert must be within
0.005 rad of an aligned endpoint.
Reward is 0 on success and -1 otherwise. Success terminates, while a step-budget
limit truncates. Events are classified after observing a changed valid
permutation: `reverse_four`, `shift_left`, `shift_right`, or explicitly
`unclassified_measured_rearrangement`. Intended controller operations never set
the board or generate an event.
Each classified one-cell shift or four-token reversal has atomic cost 1.
An unclassified rearrangement has unknown cost (`None`) and is never recorded
as a successful one-cell operation. Three successive shifts are three verified
operations, even if a caller groups them into one macro. The continuous action
space and the puzzle's elementary operation rules are unchanged.

`TurntablePolicy` pushes clockwise or counterclockwise between the two endpoints.
`TrackPushPolicy` approaches the selected socket, aligns, inserts, transports,
withdraws and verifies the complete expected 20-token permutation. During
transport it monitors contact identity, radial clearance and relative velocity.
Loss of engagement triggers withdrawal and reacquisition of the same token;
exhausting the retry limit leads to an explicit failure after withdrawal.
The old friction policy is used only with a legacy observation contract.
`SequencePolicy` composes `reverse`, `shift_left`, and `shift_right` without reset.
These are scripted state-observation controllers, not learned policies.

## Validation traces and native visualization

```sh
MUJOCO_GL=egl .venv/bin/python examples/topspin_physical.py \
  --output artifacts/topspin-engaged-v2
.venv/bin/python -m pytest tests/games/topspin/physics -q
```

By default the example records eight cases from a seeded arbitrary permutation:
three single operations, one four-operation sequence without reset, three
successive one-cell shifts, a noisy shift, recovery after an injected tool-lift
disturbance, and safe failure when reacquisition is disabled. The failure is an
expected diagnostic outcome, not a successful puzzle operation. It saves exact float32 actions,
all T+1 observations and info records, measured contacts/events, environment
configuration, source hashes, and a checksum-protected NPZ/JSON pair per case.
It then instantiates a fresh matching environment and replays only the recorded
continuous actions, comparing every observation, event, reward and ending.
Existing output files are never overwritten. A final manifest is written only
when all expected measured outcomes and independent replays pass. `--cases`
selects a subset; `--profile legacy_v1` explicitly selects the historical model.

With `--render`, the example also saves native oblique/top/overview initial renders, actual
90°/180° insert states, the circulated loop, and a GIF of the mixed sequence.
The GIF samples every eight control intervals and displays each frame for
100 ms: explicitly **4× simulation speed**. These images come from the live
MuJoCo renderer, not a generated illustration or a separately animated board.

For color and contact inspection, export an original-speed native MP4 directly
from any saved trace (never by converting a GIF):

```sh
MUJOCO_GL=egl .venv/bin/python examples/render_physical_videos.py \
  --tasks topspin --trajectory artifacts/topspin-engaged-v2/mixed.npz \
  --output artifacts/topspin-video-review --ffmpeg /path/to/ffmpeg
```

This reconstructs the recorded profile, checks every replayed observation and
event, and writes H.264/BT.709 video plus original/decoded PNG color comparisons.
The exporter refuses to overwrite an existing output directory.

`examples/contact_engagement_review.py TRACE --engaged --output NEW_REPORT`
independently audits recorded contact identity, geometric token order and
operation costs without constructing a simulator. The audit implementation and
its test fixtures are repository files; tests do not depend on the ignored local
`docs/` directory. Historical v1 audits require a checksum-matching policy
snapshot supplied with `--policy-source`.

The current trace schema is `copuzzle.topspin20_contact_trace.v2`; the archived
`copuzzle.topspin20_contact_trace.v1` remains readable with `legacy_v1`.
`load_trace` performs checksum, array shape/dtype, provenance, event/state,
reward and episode-ending validation without constructing a simulator.
`replay_trace(env, trace)` additionally requires matching source and configuration.
A narrowly enumerated historical source migration permits the preserved v1
sources to replay only against the identical legacy model/configuration; it
does not authorize arbitrary source mismatches.
These traces are small mechanism-validation artifacts. A broad exploration
corpus, learned policy evaluation, domain randomization, and real-robot transfer
are separate future work. The 20-token physical task does not use the bounded
8-token exact solver, and arbitrary reset-goal reachability is not globally
certified; the validation goals are constructed from known operations.
