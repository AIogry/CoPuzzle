# CoPuzzle rule-composition benchmark v1

Protocol identity: **`copuzzle.rule-composition.v1`**. This specification freezes
the core task, population and metric semantics. A concrete data inventory hash
identifies the evaluation artifacts. The learned study, package release and
license/distribution gates are separate; a data freeze is not a completed public
benchmark release. RC1 development artifacts retain their previous identity and
commands, described in [research_protocol.md](research_protocol.md).

## Question, scope and difficulty

The question is whether finite offline experience supports reusable transition
prediction, composition and goal control in unseen state contexts. This is a
behavioral test, not a claim about human-like understanding or a required neural
architecture. All primary instances are deterministic, fully observed and
reversible within the selected finite components.

| Instance | Component states | Easy distance | Medium | Hard | C lengths | G control budget |
|---|---:|---|---|---|---|---:|
| Lights Out 3×3 | 512 | 1–3 | 4–6 | 7–9 | 2, 4, 8 | 18 |
| Lights Out 4×4 | 4,096 | 1–2 | 3–5 | 6–7 | 2, 4, 6 | 14 |
| Pancake 6 | 720 | 1–2 | 3–4 | 5–7 | 2, 4, 6 | 14 |
| TopSpin 8/4 | 40,320 | 1–7 | 8–14 | 15–22 | 2, 4, 8 | 44 |
| Sliding 2×3 | 360 | 1–7 | 8–14 | 15–21 | 2, 4, 8 | 42 |
| Hanoi 5 | 243 | 1–10 | 11–20 | 21–31 | 2, 4, 8 | 62 |

Distances are exact shortest legal-operation counts to the actual goal, not
scramble lengths. The canonical components and state/action semantics match the
native games: zero-board Lights Out, identity Pancake/TopSpin, canonical
Sliding/Hanoi goals. Goal positions are not fixed to the canonical goal. Only
4,096 of 65,536 binary Lights Out 4×4 boards and one 360-state Sliding parity
component are included. TopSpin positions are not quotiented by rotation.

The control budget is fixed at twice the instance diameter and reveals no
per-case optimum. Tiers do not assert equal computational hardness across games.
Cross-size, unseen-rule/family, stochastic/irreversible dynamics and image
perception claims are outside v1. The existing Lights Out, Pancake-4 and
TopSpin-20/4 physical facilities remain an execution extension. Their continuous
costs and witness lengths are not exact labels for this abstract core.

## Experience and information permissions

Training contains every retained query once, in independently shuffled one-step
episodes: observations `[E,2,N]`, actions `[E,1]`, legality `[E,1]`. A boundary is
a collector cutoff, not a terminal. There is no behavior goal, optimal path,
expert label or oracle value target. These are controlled finite-state datasets,
not claims about realistic long-trajectory exploration or data-collection cost.

Only `learner/` data and declared static metadata may be used for fitting.
Fit encodings, normalization, goals, coverage subsets and auxiliary representations
using training records only. The learner may observe the legality of the action
actually attempted; it does not receive unattempted-action or next-state masks.
An in-range illegal attempt leaves the board unchanged and consumes one control
step. Native ClassicEnv error behavior is unchanged; this is the benchmark's
attempt/rejection wrapper.

No true-rule, solver, simulator-training query or dev/test answer is a fitting
input. Structure formulas, pretraining and symbolic priors must be declared as
distinct information conditions. Opaque action IDs discourage literal action-name
lookup but do not make public puzzle definitions secret. Code/data separation is
an auditable contract, not a security sandbox against a malicious participant.

Goal-conditioned rewards may be computed by equality with observed successor
states. A non-successful collector cutoff still bootstraps. An initially achieved
goal supplies no directed actor demonstration. The baseline sampling recipes,
including independent/future mixtures and rejected attempts, are described in
[baselines/README.md](baselines/README.md); they are method settings, not extra
environment information.

## Independent generalization conditions

The stable partition recipe is `copuzzle.partition.v1`, independent of RC1's
namespace. SHA-256 buckets assign approximately 70/15/15 percent to train/dev/test;
actual counts are recorded. A task/data seed also fixes its opaque action mapping.
Methods never transfer weights or records between the two conditions.

- **Transition holdout:** all actions connecting the same unordered legal state
  pair, including inverse moves, share a partition. Rejected `(state, action)`
  queries are split separately. States may overlap. R is transition transfer;
  C uses entirely observed primitive transitions; G is a same-state control
  diagnostic and does not claim unseen training goal pairs.
- **State holdout:** source and successor must belong to the same partition for
  a query to be retained. Cross-partition queries are discarded and counted.
  Training inputs and labels contain no dev/test states. R queries have both
  ends in their respective held-out partition. C/G endpoints are held out;
  intermediate states may belong to other partitions. This is not an entirely
  unseen-path condition.

Prespecified benchmark data seeds are **101, 202, 303**. Fitting seeds are
**0, 1, 2**. Previously inspected seeds 0/1/2 remain development data. Do not
replace a data seed based on model performance or convenient sample availability.

## Exact evaluation populations and case selection

R evaluates every retained query in the held-out split. It does not subsample
rare legal queries to force equal cell sizes.

C/G use the versioned `copuzzle.reference_pairs.v1` population:

1. The endpoint pool is the complete canonical component for transition holdout,
   or the corresponding held-out state partition for state holdout.
2. If this pool contains at most 4,096 states, every endpoint is a reference
   member. Otherwise choose **256** members by deterministic hash priority before
   inspecting distances. The large-pool case currently affects TopSpin.
3. The declared universe consists of every distinct **unordered** endpoint pair
   having at least one reference member. Enumerate it without double counting.
   This is not uniform sampling of all possible TopSpin state pairs.
4. For transition holdout, C and G share the same hashed dev/test pair assignment.
   For state holdout, disjoint endpoint pools already determine the split.
   Thus no C-dev/G-test or G-dev/C-test unordered pair overlap is permitted.
5. G cells use the exact-distance intervals above. C cells require the stated
   exact length. Supported C also requires a path of that same length using
   retained training edges. A cancelling/redundant word cannot fake a long case.
6. Select **min(32, exact eligible population)** distinct pairs in each cell by
   hash priority. Small cells are exhaustive; no padding with duplicates or
   rejection of inconvenient seeds. The manifest records population and selected
   counts, reference members and the selection recipe. A required empty cell
   prevents freezing and requires an explicit protocol revision.
7. Pair orientation and final row order use independent hash priorities. Hidden
   G tiers cannot be read from file order. C uses one deterministic shortest word
   per selected oriented pair, breaking path ties in opaque-action order.

Small supported Hanoi composition cells can contain only six distinct cases in
the prespecified partitions. Report their counts and uncertainty. Each word's
partial operator is fingerprinted on the full component; dev/test effect aliases
are audited rather than excluded. Different words are not automatically different
operators, especially in commuting Lights Out. Do not claim effect-disjoint
composition generalization.

## Metrics and controls

| Probe | Inputs | Primary score |
|---|---|---|
| R | Board, action | Mean joint exact successor-and-legality accuracy over present legality classes |
| C | Initial board, action word | Equal-length mean of whole-prefix exact accuracy, without true intermediate feedback |
| G | Current board, goal, remaining budget, previous legality feedback | Equal-tier mean success under the fixed control budget |

Report per-class/per-length/per-tier results and micro averages as well. The
primary macros prevent larger finite cells from dominating after exhaustive
small-cell evaluation. Across instances, average the two Lights Out instances
first, then weight the five families equally. Unsupported native R/C interfaces
are marked unsupported, not assigned zero or replaced by an unlabeled readout.

G also reports all-case attempts, failure-budget-charged cost, successful-case
excess over optimum, illegal-attempt rate and policy-error rate. Policy exceptions,
bad action types and out-of-range actions retain the case and receive failure
plus full budget cost. In-range illegal attempts are a different outcome.
Malformed benchmark data remain fatal errors. User cancellation propagates and
does not become a falsely completed evaluation.

Planning conditions are **0 explicit transition queries** for direct policies,
and **32/128 candidate state-action predictions per decision** for model planners.
The matched reference planner uses bounded best-first search, slot mismatch as
heuristic, beam width 32 and depth at most 8. A popped node's remaining action
candidates are evaluated in a batch up to the remaining budget. Charge every
candidate before prediction, including illegal/unknown candidates and unused
members of an early-success batch. Report Python/model call counts separately;
one batch is not one transition. Direct-policy inference is not cost-free merely
because its explicit transition-query count is zero.

Use random control, training lookup and privileged optimal-oracle calibration.
Additionally compare exact dynamics through the **same bounded planner** to
separate prediction errors from search limitations. Data-fitted constant-XOR
and constant-position-map controls are labeled structural-prior conditions;
their effects come from training records, not the true rules. They are not
applicable to all families. An unbounded optimal oracle has no comparable
model-query budget and must not appear as a learned zero-query method.

The baseline evaluator separately diagnoses one-step errors on true C-prefix
contexts and whether each primitive query is present in the **actual selected
training subset**. Those teacher-forced queries never replace the open-loop C
inputs. This matters when C paths cross state partitions or 25/50 percent
coverage removes previously observed primitives. Training-graph connectivity
and empirical G reachability are reported; disconnected coverage is not an
implementation failure by itself.
The frozen G reachability labels describe the original full retained corpus.
Coverage analysis additionally reports observed-graph reachability in each actual
selected subset, retaining both definitions explicitly. These support-stratum
diagnostics do not replace the primary equal-tier goal score or establish that
an unobserved path is impossible in the underlying game.

## Freezing, selection and reproduction

```sh
copuzzle benchmark spec
copuzzle benchmark population --task hanoi-5 --regime transition_holdout --seed 101
copuzzle benchmark build --task grid-3x3 --regime state_holdout --seed 0 \
  --output artifacts/v1-development/grid3
copuzzle benchmark freeze-data --output artifacts/benchmark-v1-frozen
copuzzle benchmark validate-inventory artifacts/benchmark-v1-frozen --semantic
```

`build` creates development bundles. Nonstandard case caps are explicitly
recorded and cannot receive benchmark purpose. `freeze-data` writes all 36
prespecified combinations with standard parameters, exact populations, checked
arrays, generator/metric source hashes, dependency versions and a Python source
snapshot. The snapshot supports abstract regeneration; it is not a substitute for
the packaged physical assets. Keep outputs/checkpoints outside the frozen data
directory. Python bytecode caches are ignored; data/source modifications are not.

Learner APIs are in `copuzzle.benchmark.data`; generation and scoring remain
separate evaluator modules. Validation recomputes transitions, path lengths,
population membership, selected cases and hidden labels, not just file hashes.
The inventory's label-free public descriptor can be tracked in Git without
publishing individual answers there. Generated data are distributed separately.

Freeze this data inventory before accuracy-driven method/hyperparameter
selection. Use development data or allowed dev splits for selection, then lock
method code, configurations, fitting/checkpoint rules and checkpoint hashes
before learned final-test evaluation. The baseline CLI rejects benchmark test
prediction/evaluation without a selection-lock binding to the exact checkpoint,
data, source and configuration. The sealed-study executor must account for every
registered fit, startup failure and evaluation failure; dispatch alone is not
completion. This protocol does not claim those experiments have already run.

The primary comparison requires three data seeds and three fitting seeds for
each of three learned method classes across six instances and two conditions:
324 primary fits, plus declared coverage/search-budget comparisons. Report all
seeds and failed runs. Use paired hierarchical uncertainty over data/fitting
seeds, conditional on the frozen instances and case inventory, and publish the
underlying per-case records. Three outer seeds are limited evidence; avoid
overconfident significance or claims about unseen game families.

The [release plan](benchmark_release_plan.md) tracks the remaining experiments,
CI, final independent installation/replay, licenses, attribution and public
distribution. Benchmark v1 is ready for scientific comparison only when those
acceptance gates have evidence.
