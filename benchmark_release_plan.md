# CoPuzzle v1 benchmark release plan

Status: **active goal; release acceptance is incomplete**.
Started: 2026-11-17 (Asia/Shanghai).
Starting source revision: `1332193ed53abffdc5a6144ebd347977968702e1`.

The outcome is a publicly usable, independently reproducible benchmark with
real learned-baseline evidence for experience-based rule induction,
composition prediction and goal control. The benchmark identity `v1` is separate
from the software package version. The current package remains an unreleased
development snapshot; this plan is not a release announcement.

The current executable candidate is described in [research_protocol.md](research_protocol.md).
The machine-readable work ledger is [release/plan.json](release/plan.json).
Mark acceptance gates complete only with inspectable evidence. A successful
smoke run, a large dataset, or a planned experiment does not complete a gate.

## Scope

The core comprises `grid-3x3`, `grid-4x4`, `pancake-6`, `topspin-8-4`,
`sliding-2x3` and `hanoi-5`, with separate transition-holdout and state-holdout
conditions. The native Lights Out, Pancake-4 and TopSpin-20/4 physical models
remain an execution extension with their own costs, failure outcomes and
mechanical validation. Neither TopSpin instance substitutes for the other.

Unseen rule families, cross-size learning and image perception are outside the
first release's primary claim. Scientific conclusions must distinguish observed
performance from hypotheses about an agent's internal understanding. Negative
results are acceptable; favorable performance is not a release gate.

## Acceptance gates

| Gate | Current state | Required completion evidence |
|---|---|---|
| A: protocol and population audit | In progress | Reviewed information contract, exact difficulty, coverage, sampling populations, metrics, budgets and final data inventory; test artifacts frozen before method selection |
| B: standalone learning pipeline | Not complete | Train-only adapters, GCBC, offline goal-conditioned RL and learned dynamics with planning; checkpoint resume, frozen evaluation, input/provenance tests and a clean-install workflow |
| C: formal baseline matrix | Not started | All six instances and both splits, three prespecified data seeds and three fitting seeds, fixed method-selection rules, checkpoints and all per-case outcomes |
| D: scientific comparisons | Not started | Coverage and planning-budget comparisons, paired uncertainty, per-instance/tier results, family macro averages, measured computational cost and explicit limitations |
| E: engineering qualification | Prior candidate qualified; final run pending | Appropriate CI, full final regression, independent package installation, deterministic data checks and three-family historical physical replay |
| F: public release materials | Not complete | Installation/reproduction guide, dataset and method cards, citation metadata, licenses and third-party attribution, downloadable release artifacts with checksums, independent acceptance report |

Progress through these gates is recorded independently. All are required before
the goal is declared complete. Any unresolved rights, access or publication
decision remains explicit in the final acceptance report.

## Experiment design to freeze before formal fitting

The three planned final data seeds are **101, 202, 303**. The three fitting seeds
are **0, 1, 2**, applied to every data partition. These choices precede their
population audit and method fitting. Do not replace a seed because it is difficult
or yields too few convenient examples. Existing data seeds 0, 1 and 2, including
their inspected test files, remain development resources.

The minimum primary matrix is six instances × two conditions × three data seeds
× three fitting seeds × three learned method classes: **324 fitted runs**.
Budgets and training hyperparameters remain unset until correctness and runtime
calibration on development data; they must be resolved and hashed before formal
fitting. Do not advertise this matrix as completed merely because all jobs were
submitted. Failed runs remain in the inventory with reasons and retry history.

The initial method classes are:

- Goal-conditioned behavior cloning (GCBC): native G results; unsupported native
  R/C probes remain explicitly unsupported.
- Offline goal-conditioned RL: GCIQL is the initial implementation candidate,
  subject to loss/goal-label verification against primary algorithm sources.
- Learned categorical dynamics with planning: R, C and G, with explicit learned
  legality prediction and measured model-query budgets.

Random control, empirical transition lookup/graph search and privileged exact
oracles are separately labeled controls. An oracle must not become a learned
method's training target or planning service. Architecture priors and any
representation readouts require their own declarations and controls.

The planned coverage levels are **25%, 50%, 100%** of retained training queries,
with nested, deterministic selection using training records only. The planned
search budgets are **0, 32, 128** predicted state-action transitions per decision.
The exact comparison matrix, coverage unit and method-selection policy are to be
frozen in gate A. Recompute training support after subsampling: C examples whose
primitive transitions were present in the full corpus may lose that property in
a coverage ablation. Batched model calls must charge for every predicted candidate.

Population audits precede final C/G cardinalities. RC1 has very small legal R
populations for some state partitions, and some C strata cannot supply eight
distinct shortest compositions. Do not pad with duplicates or silently drop a
seed. Any revision of population/sampling semantics receives a new protocol
identity; old candidate artifacts and results keep their original identity.

## Initial implementation audit

The sibling research workspace contains JAX/Flax/Optax agents and categorical
dynamics implementations that may be reusable after review. It is not the public
CoPuzzle package and is not a required installed dependency. Preserve the existing
JAX framework choice. Prepare a separately installable baseline component inside
this repository so the base environment stays lightweight and an external user
does not need an unpublished sibling checkout. Review provenance before copying
implementation code; do not publish unrelated research artifacts.

The current legacy Classic adapter supplies full recorded action masks during
training and true environment masks during evaluation. Its categorical dynamics
trainer also supervises complete applicability vectors. Those interfaces do not
match the new experience-only protocol. The new adapter must:

- Read only `learner/` arrays and static public metadata; no simulator, true rule,
  solver, evaluator labels or action permutation mapping.
- Preserve all attempted actions and the observed legality of the selected
  action. Unobserved actions have **unknown**, not implicitly false, legality.
- Use the full action-ID range for an unmasked policy. A vector of ones denotes
  available action IDs, not a claim that every move is legal.
- Treat one-step episode endings as collection cutoffs. Goal equality determines
  success and bootstrapping for relabeled goals; an already-satisfied initial
  goal provides no directed actor demonstration.
- Form goals, encodings, normalization and training subsets from train only;
  distinguish future-successor goals from independent observed training goals.
- Keep per-case memory separate from frozen model parameters, and account for
  every planner transition prediction, including cached lookup and batches.

These are adaptation requirements, not claims that legacy experiments violated
their own separately declared information conditions.

## Resource and execution policy

A read-only host check found two RTX 4090 GPUs, both with existing compute
processes. Sandboxed device visibility differs from host availability. Verify
availability when scheduling, use controlled memory allocation, and measure one
development run's time/memory before expanding the matrix. Do not stop or alter
unrelated jobs. Training runs need resumable checkpoints and immutable manifests;
external paid compute is not part of the initial plan.

Resolve and record the actual Python/JAX/Flax/Optax/CUDA versions used for each
run. A process exit without a finished checkpoint/evaluation record is not a
successful experiment. Preserve failure-inclusive results and compute costs.

## Starting evidence and release gaps

At the starting revision, the RC1 implementation had 1,340 passing tests, twelve
qualified development bundles with 235,177 single-step training episodes and
69,997 evaluation cases, independent native-rule/oracle checks, a project-external
wheel smoke test and three historical physical replays with zero error. These
are candidate qualification results, not final learned benchmark results.

No project-level license or CI workflow is present at the starting revision.
Bundled UR5e assets do carry their own pinned provenance and BSD-3-Clause license;
that does not license unrelated project code or all derived distributions.
Review code/data ownership, adaptation notices, dependency requirements and
artifact contents before preparing final release files. Keep necessary owner
decisions separate from completed engineering work.

The next deliverable is the train-only baseline adapter and a tested CPU
training/evaluation/checkpoint path on development data, followed by measured
host GPU calibration and the final protocol/data freeze.
