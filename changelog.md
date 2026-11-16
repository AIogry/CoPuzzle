# Changelog

## Unreleased

- Add checksum-bound readable result tables and optional scientific PNG/SVG
  figures. Label synthetic rendering fixtures, preserve renderer/source versions,
  and require interpretation of the actual formal analysis in the study package.
- Add checked artifact relocation, deterministic distribution staging and
  installed-wheel physical history qualification with NPZ/sidecar bindings.
  Preserve selected checkpoints, recovery evidence and all attempt costs in the
  study package; local drafts remain separate from public release acceptance.
- Add complete-cohort analysis with paired data/fit-seed intervals, equal-family
  weighting, retained failed runs, checked attempt costs and per-case R/C archives
  that reproduce saved scores. Register the analysis plan before final testing,
  and document the frozen dataset and baseline scope in public-facing cards.
- Register the formal 972-fit study with immutable source/data/config bindings,
  resumable attempts, kernel-backed leases and complete-cohort final-test sealing.
  Preserve failure histories and controller revisions. Add CI and installed-wheel
  validation; license proposals remain pending owner confirmation.
- Freeze a separate v1 benchmark protocol and prespecified data inventory with
  exact finite populations, exhaustive small cells, shared C/G pair holdouts,
  equal-length/tier metrics, batched query accounting and retained policy errors.
  Add data-fitted structural and matched-planner oracle controls, composition
  support diagnostics and checkpoint-bound final-test access records.
- Add a separately installable JAX baseline component with train-only goal
  relabeling, nested coverage subsets, GCBC, GCIQL, generic categorical dynamics,
  budgeted planning, exact checkpoint continuation and frozen R/C/G evaluation.
  Keep unattempted-action legality unknown and preserve failure/cutoff semantics.
- Define `copuzzle.rule-composition.rc1`: five families, six exact finite-state
  instances, distance tiers, separate inverse-edge/state holdouts, and rule,
  open-loop composition and budgeted goal-control probes. Add deterministic
  data generation, strict semantic audits, learner-only loaders and calibration
  controls. Keep qualification data distinct from a final publication test.
- Add a separate segmented physical-play protocol with one reset, safe behavior
  goal changes, native action replay and first-hit goal relabeling. Keep physical
  episode endings distinct from local goal completion and retain failed prefixes.
- Add direct-contact multi-operation development suites with disjoint task pairs,
  native physical collection/replay and failure-inclusive evaluation reports.
- Add checked continuous transition batches with explicit episode boundaries,
  original goals and configurable timeout bootstrap semantics; keep privileged
  reference plans outside policy inputs and learner features.
- Replace the TopSpin friction-drag interface with versioned physical pin/socket
  engagement, measured contact diagnostics and post-withdrawal index verification;
  preserve the earlier model as `legacy_v1` for historical replay.
- Add Pancake single-disc grasp/flip execution diagnostics alongside existing
  2/3/4-disc goal operations, without adding an identity move to the abstract game.
- Allow native MP4 export from a selected physical trajectory and reconstruct
  its recorded facility/task mode before checking every replayed observation.
- Rebuild TopSpin as a numbered 20-token closed track with a horizontal four-token
  turntable, directly contacted by the UR5e. Rebuild Pancake with different-size
  gravity-supported sorting discs and a contact-only fork clamp for lift/flip/place.
  Expose these separately through `make_manipulation_env`; retain earlier indexed
  fixtures as archived experiments instead of relabeling their data.
- Rebuild the default Lights Out button assembly with real stems, hollow collision
  guides and supported bases; retain the exact earlier facility as `legacy_v1`.
  Add full-stroke geometry, contact and native-render acceptance checks.
- Add UR5e contact-driven TopSpin-8-4 and Pancake-4 indexed fixtures, with explicit
  ideal clamps/gears, measured token permutations, stable success, ordered event
  trajectories, action replay, future goals, and physical development collection.
- Add Sliding Puzzle and Hanoi with state-dependent action masks, bounded exact
  oracles, checked offline data, dev/test goals, and `copuzzle classic` workflows.
- Add bounded Pancake and fixed-position three-action TopSpin environments,
  independent permutation trajectory/goal formats, exact BFS and four Gym presets.
- Add offline future-goal sampling with action widths independent of token count,
  strict artifact checks, and ordered position-map diagnostics fitted from data.
- Add `copuzzle permutation` commands and a reproducible development collection
  with disjoint dev/test goal pairs, training-overlap audits, random/oracle
  references and source/output checksums.
- Preserve existing Lights Out data, RNG, environment IDs and defaults.

These working changes are not a new published package release. Learned baseline
performance and a frozen cross-game benchmark remain separate milestones.

## 0.1.0.dev3

Development prototype of state-based physical Lights Out for offline GCRL.

- Add optional MuJoCo UR5e environments for all 12 standard Grid/Path/Cycle tasks, with separate physical Gym IDs.
- Package pinned official robot assets with their BSD license and SHA256 provenance; no OGBench or dm_control dependency.
- Add continuous XYZ control, actual contact-gated button events, press/release hysteresis, board goals and public logical/physical observations.
- Add complete continuous episodes with measured event logs, strict NPZ/JSON integrity checks, future-goal sampling and action replay.
- Add an observation-only scripted button executor and a recording/rendering example.
- Keep the abstract environment, data and structure contracts unchanged.

The prototype uses a fixed pressing tool, calibrated state inputs and arm gravity
compensation. Training distributions, physical evaluation suites, learned control
and perception remain open work; scripted completion does not establish RL performance.

## 0.1.0.dev2

Development snapshot for public structure and variable-size algorithm inputs.
The benchmark distribution remains under development.

- Add `structure_v1`: Grid shape and integer row/column coordinates, or Graph adjacency.
- Preserve legacy hidden observations, `reveal_graph`, v0 Gym defaults, rule identities, trajectory formats and random streams.
- Add immutable public structure snapshots with explicit semantics and checked NPZ/JSON artifacts.
- Add offline bundles, byte-preserving trajectory migration and goal sampling with static structure context.
- Add structured goal suites, migration, custom graph loading and structure-aware policy evaluation.
- Add single-step and autoregressive prediction evaluation with goal-free model inputs.
- Add CLI commands and executable Grid/Graph workflow and collection-migration examples.
- Include the expanded standard 9/16/20/24-button Grid, Path and Cycle presets.

This release supplies benchmark interfaces. It does not implement a learned RL
algorithm, guarantee a model supports variable input sizes, or add physical
simulation. External training projects must adapt their inputs and checkpoint
contracts to the declared observation profile.
