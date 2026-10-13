# Changelog

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
