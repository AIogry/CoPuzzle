# Experience-based rule induction and compositional goal control

Protocol: **`copuzzle.rule-composition.rc1`**. This tracked specification, the
registry in `copuzzle.research.spec`, generators, validators and scoring APIs
define a concrete release candidate. Generated artifacts are **development
qualification**, including files named `test`; they are not an untouched final
publication test. The package version remains `0.1.0.dev3` with unreleased work.

## Scientific question and falsifiable claims

**Can a learner infer reusable transition structure from finite offline
experience and apply it to novel contexts, action compositions and goals?**
This is a behavioral definition of rule induction. It neither establishes
human-like understanding nor requires a model to have three separate modules.
“Solves a game with a robot” alone does not answer this question.

The candidate separates three measurements:

| Probe | Learner receives | Prediction or behavior | Primary measurement |
|---|---|---|---|
| R: rule transfer | State and one opaque action ID | Next state and whether the action is legal | Joint exact accuracy, separately for legal/illegal attempts |
| C: composition | Initial state and a complete action sequence | Every successive state, without intermediate truth feedback | Whole-prefix and final-state exact accuracy by length |
| G: goal control | Current state, goal, remaining attempt budget, previous attempt legality | One action at a time | Success by exact-distance tier, failure-inclusive cost and illegal-attempt rate |

R distinguishes a transition lookup from prediction at held-out inputs. C
tests error accumulation without feedback correction. G tests whether prediction
supports decisions under a control budget. The pattern “high R, low C” is
consistent with composition errors; “high C, low G” motivates a search/control
diagnosis. Neither pattern by itself identifies a unique internal cause.

Report the vector of results, not only a single success rate. Compare a direct
goal-conditioned policy, a learned transition model plus planning, a training-only
transition lookup, and privileged known-rule calibration. A method can succeed
without exposing a symbolic rule; no explicit rule representation is compulsory.

A policy without a predictive interface may report G and mark native R/C as
unsupported. To study its representation, fit a separately declared, capacity-
matched readout on frozen policy features using training transitions only. Report
those readout scores separately from native model predictions, select the readout
on dev, and never interpret a successful standalone auxiliary model as evidence
that the policy itself uses the inferred rules.

This framing follows the distinction between random splits and systematic
generalization in [Lake & Baroni, ICML 2018](https://arxiv.org/abs/1711.00350),
and capability-oriented offline goal-conditioned evaluation in
[OGBench, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ecd92623ac899357312aaa8915853699-Abstract-Conference.html).
These motivate the controls; they do not establish novelty of this benchmark.

## Core task scope and exact difficulty

Each row is trained and evaluated independently. All states use the existing
native game's integer encoding. The component contains the zero board for
Lights Out, identity permutation for Pancake/TopSpin, and canonical goal for
Sliding/Hanoi. Goals vary across the component, including Sliding blank locations.
TopSpin uses fixed positions, not equivalence up to rotation.

| Instance | Reachable states in the selected component | Easy | Medium | Hard | C lengths | G attempt budget |
|---|---:|---|---|---|---|---:|
| Lights Out 3×3 | 512 | 1–3 | 4–6 | 7–9 | 2, 4, 8 | 18 |
| Lights Out 4×4 | 4,096 | 1–2 | 3–5 | 6–7 | 2, 4, 6 | 14 |
| Pancake 6 | 720 | 1–2 | 3–4 | 5–7 | 2, 4, 6 | 14 |
| TopSpin 8/4 | 40,320 | 1–7 | 8–14 | 15–22 | 2, 4, 8 | 44 |
| Sliding 2×3 | 360 | 1–7 | 8–14 | 15–21 | 2, 4, 8 | 42 |
| Hanoi 5 | 243 | 1–10 | 11–20 | 21–31 | 2, 4, 8 | 62 |

Difficulty is the exact shortest number of legal atomic operations to the actual
goal, not scramble length. Pancake flips 2–6 items; a single un-oriented item
flip is an identity and is not an abstract move. TopSpin has left shift, right
shift and four-item reversal. Classic game legality remains state-dependent.

The diameters are checked by exhaustive BFS. Lights Out and permutation graphs
have a transitive symmetry, so one root suffices. Sliding and Hanoi are checked
over **all** roots; a canonical goal's eccentricity is not assumed to be the
diameter. Only 4,096 of the 65,536 binary 4×4 boards belong to the selected Lights
Out component; only 360 of 720 Sliding permutations belong to its parity component.
The protocol makes no claim about other components or unreachable goals.

The tiers are relative to each instance, not equal computational hardness across
games. Branching, legality, commutativity, algebraic shortcuts and training-graph
connectivity also matter. These six bounded instances make exact audits possible;
they do not establish scaling to large combinatorial problems.

## Two independent generalization conditions

Both conditions use deterministic SHA-256 partitioning with 70/15/15 hash buckets.
Actual counts vary by seed and are recorded. Native action IDs are deterministically
permuted into opaque IDs, consistently within a bundle. The evaluator retains the
mapping. Action permutation discourages literal action-name lookup; it does not
hide family semantics from a reader of the public benchmark.

| Condition | Training experience | R evaluation | C evaluation | G interpretation |
|---|---|---|---|---|
| `transition_holdout` | Retained state–action attempts | Held-out legal edges or illegal queries | **Supported composition**: every primitive transition was in training | Same-state control diagnostic; empirical path availability is reported |
| `state_holdout` | Both ends of every retained attempt are training states | Both query state and next state belong to dev/test respectively | **Transfer composition**: start and final state are in dev/test respectively | Generalization to unseen endpoint states |

For transition holdout, all actions connecting the same unordered state pair,
including inverse actions, share one partition. Illegal `(state, action)` queries
are partitioned separately. Thus the reverse of an R test transition cannot appear
in training. This condition **allows overlapping states**.

For state holdout, states are partitioned first. Transitions across partitions are
discarded, with their counts recorded. Training inputs **and labels** exclude
dev/test states. C/G intermediate states may be training states or states from
another held-out partition: this is endpoint generalization, not an entirely
unseen-path claim. Do not add the discarded transitions, mix corpora, transfer
weights between the two conditions, or normalize using held-out boards.

These are within-instance splits. **Unseen rule variants, unseen sizes, cross-game
transfer, image perception and online system identification are outside rc1.**
Each deserves a separate protocol with training rule instances and held-out rule
instances. Calling these current tests “meta-learning an unseen game” is incorrect.

## Experience, trajectory boundaries and composition controls

The training corpus contains every retained query exactly once, shuffled into
independent one-step episodes: observations `[E, 2, N]`, actions `[E, 1]`, legality
`[E, 1]`. There is no behavior goal, shortest path, expert-action label, or value
target. An episode ends because collection stops, **not** because the environment
terminates. A Bellman learner must preserve bootstrap across this cutoff when its
goal has not been reached. Goals/rewards may be derived from observed training
states; oracle goals, distances and dev/test answers are not training labels.

These short episodes deliberately isolate experience composition. They are not a
replacement for the existing fixed-goal or segmented physical-play corpora, and
their scores do not settle which long-trajectory collection strategy is best.
The cost of preparing exhaustive finite-state experience must be disclosed;
scalable interaction-budget studies are a later experiment.

In this research wrapper only, an **in-range illegal attempt** consumes a step,
leaves the board unchanged and reports legality false. Existing ClassicEnv APIs
still raise for illegal moves. An out-of-range or non-integer policy action is a
protocol error, not an ordinary failed attempt. R includes legal and illegal
strata when they exist; all-actions-legal families have no illegal stratum.

All C words are shortest paths of their stated length in the full graph. Supported
C additionally has the same distance in the training graph. Thus repetitions,
inverse cancellations and shortcuts cannot create artificial “long” examples.
True intermediate boards are evaluator-only. Prefixes are scored together, so a
wrong intermediate prediction followed by a correct final board is not a wholly
correct composition.

The audit fingerprints each C word's partial operator over the entire component,
with undefined illegal prefixes. Different words can implement the same operation
(especially commuting Lights Out presses). Operator aliases, including dev/test
effect overlap, are recorded. C does **not** claim effect-disjoint dev/test
operators. Supported C is intentionally solvable by a training lookup; success
there alone is not evidence of rule induction. Transfer C and R provide the
context-generalization tests. Unseen primitive-transition counts are recorded per C case.

The empirical training graph can be disconnected. Reports include its component
sizes and whether each G pair is connected by observed legal transitions. G
reports are also stratified by empirical reachability. A learner's failure on
unseen states must not be equated with an implementation defect or insufficient
search of the empirical graph.

## Sampling, information access and budgets

G samples a goal uniformly from the permitted component/partition, then a start
uniformly from eligible states in the requested distance interval, retrying empty
or duplicate pairs. This is **not uniform sampling of all eligible pairs**.
Unordered endpoint pairs are disjoint across G dev/test and separately across C
dev/test. G rows are shuffled so file order does not disclose difficulty tiers.
R enumerates **all** retained queries in each split and reports legal/illegal
strata separately, with a balanced mean over nonempty strata. Empty strata are
reported as null with count zero; an illegal-only result cannot establish legal
transition prediction. State holdout may leave very few legal R queries in a
small instance (Sliding seed 1 has six in dev and four in test). Exhaustive R
avoids rejecting such seeds just to obtain a fixed sample count. For C/G the
generator fails rather than silently returning fewer requested cases when a
stratum cannot be filled.

Only the `learner/` directory is an algorithm input. The evaluator manifest,
source hashes, action mapping, exact graph, distances, C prefixes, legality answers
and audit reports stay in a separate evaluator process/data mount. Training code
must not call true game transitions, an exact solver, or additional simulator
interaction. The public protocol and code are not a security boundary; declare
pretraining, hard-coded game rules, structural priors and all auxiliary data.
The older `structure_v1` geometry/adjacency condition is separately labeled and
must not be silently added to this experience-only condition.

Fit on train and select hyperparameters on dev. Freeze weights before evaluation;
reset episode memory/caches for every G case. Within-case observation history and
planning caches are allowed. No evaluation rollouts become training experience.
`policy_factory()` creates that per-case policy; its observations contain only
board, goal, remaining budget and previous legality feedback.

The G budget is fixed per instance at twice its diameter, **not twice each case's
hidden distance**. Report success, mean attempts over all cases, budget-charged
cost for failures, illegal-attempt rate, and successful-case excess over optimum.
Do not report efficiency on successes alone without the failure-inclusive metrics.

For planning comparisons, use **0, 32 and 128 predicted transitions per control
decision** as separate conditions. Count each state–action candidate, including
table lookup and batched predictions. `ModelBudget.predict` enforces this when
all single-transition model queries are routed through it; code review is still
needed to detect bypasses or hidden batching. Disclose model forward passes,
training cost, wall time and offline precomputation separately. A direct policy
has no explicit transition query; that does not make its inference cost zero.
Privileged oracle calibration is exempt and explicitly named as such.

## Reproducible workflow

```sh
copuzzle research spec
copuzzle research build --task topspin-8-4 --regime state_holdout \
  --seed 0 --per-cell 8 --output artifacts/research-rc1/topspin-state
copuzzle research validate artifacts/research-rc1/topspin-state
copuzzle research references artifacts/research-rc1/topspin-state \
  --output artifacts/research-rc1/topspin-state-controls.json
python examples/research_qualification.py --output artifacts/research-rc1-qualification
```

`per-cell` is the number of C cases per length and G cases per tier, separately
for dev/test. R always uses its full held-out population. Generation never overwrites an existing
directory. The full qualification script checks all six instances and both
conditions, including test controls, so those artifacts are explicitly development
data. This is a functional acceptance run, not a statistically powered comparison.

```python
from copuzzle.research import load_training, load_probe
train = load_training("artifacts/research-rc1/topspin-state/learner")
queries = load_probe("artifacts/research-rc1/topspin-state/learner", "C", "dev")
# Fit using train only; predict all prefixes recursively without evaluator truth.
```

R prediction NPZ fields are `next_board` and `legal`; C uses `prefix_boards` with
the same padded shape as the requested word lengths and state width. Evaluate
frozen outputs with `copuzzle research score --probe R|C --predictions FILE ...`.
G uses `copuzzle.research.evaluation.evaluate_policy`; call `validate_bundle`
once before direct Python scoring. The CLI does that validation automatically.
Validation reconstructs the full artifact and checks source identities, all file
hashes, split membership, transitions, actual distances and hidden answers.
To validate historical artifacts, use their recorded source revision. Offline
learner loading neither reads the evaluator directory nor constructs a true graph.

## Qualification versus a publication claim

For formal method comparisons, freeze a new artifact inventory and seeds before
testing learned methods. Use at least three dataset partitions and three fitting
seeds per partition, shared evaluation cases across methods, paired uncertainty
estimates, and per-instance/per-tier results. Publish raw per-case records. The
family macro average first averages the two Lights Out instances, then weights
the five families equally; a pooled transition average would overweight TopSpin.
Inspect seed variance rather than selecting the most favorable split.

Audit population capacity across **all prespecified dataset seeds** before
choosing the common C/G sample count. Some small held-out partitions cannot supply
eight distinct shortest compositions of a requested length. In that case reduce
the common count or revise the task scope for all seeds before freezing artifacts;
do not omit the failing seed or duplicate examples to reach a nominal count.

Before a benchmark release, complete the learned baseline matrix and coverage/
budget ablations, choose the final data scale and untouched evaluation inventory,
check the claimed scientific distinction against prior work, and finish licensing
and distribution. RC qualification does not complete those steps.

The existing physical Lights Out, Pancake-4 and TopSpin-20/4 environments remain a
separate **execution extension** with continuous control costs and recorded
mechanical success. Reference sequence length there is generally an upper bound,
not an exact difficulty label. TopSpin-8/4 and TopSpin-20/4 are different instances;
their scores are not paired abstraction-to-physics measurements. A matched physical
transfer protocol and learned low-level controllers remain future work.
