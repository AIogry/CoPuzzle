# CoPuzzle learning baselines

This is a separately installable JAX/Flax/Optax component in the CoPuzzle
repository. The base environment does not depend on it. It uses the
`copuzzle.rule-composition.v1` and legacy RC1 learner arrays, not the sibling CoPuzzleLab
checkout or its legacy masks. The v1 data inventory, method configurations and
formal matrix are frozen. The registered 972-fit experiment is running; final
comparative results remain pending. See the [method card](METHOD_CARD.md).

## Installation and commands

From the repository root, in a dedicated Python environment:

```sh
python -m pip install -e '.[dev,render]' -e './baselines[dev]'
JAX_PLATFORMS=cpu python -m pytest -q baselines/tests
copuzzle benchmark build --task sliding-2x3 --regime state_holdout \
  --seed 0 --max-per-cell 4 --output artifacts/baseline-development/data
copuzzle-baselines train --learner artifacts/baseline-development/data/learner \
  --config baselines/configs/development.json --method gciql --seed 0 \
  --steps 1000 --output artifacts/baseline-development/gciql
copuzzle-baselines evaluate \
  --checkpoint artifacts/baseline-development/gciql/checkpoint-00001000.zip \
  --bundle artifacts/baseline-development/data \
  --output artifacts/baseline-development/gciql-dev.json
```

Run the same training command with `--method gcbc` or `--method dynamics` and a
new output directory. The method flag is an explicit override of the JSON config;
the resolved configuration is saved. For the dynamics checkpoint, evaluation
defaults to 32 predicted transitions per decision; `--model-query-limit 128`
selects a different condition. Direct policies use zero explicit transition
queries. Model-only planning has no zero-query actor and rejects that setting.

`JAX_PLATFORMS=cpu python baselines/examples/qualify.py --output
artifacts/baseline-pipeline-v1` exercises all six instances, both conditions and
all three methods with short fits, saved checkpoints and frozen dev evaluation.
Its 36 runs are pipeline qualification and do not enter the formal 324-fit inventory.

For CPU operation set `JAX_PLATFORMS=cpu` before importing JAX. For a locally
configured CUDA JAX installation, select a device and memory policy before
launching the command; for example `CUDA_VISIBLE_DEVICES=1
XLA_PYTHON_CLIENT_PREALLOCATE=false`. Check existing jobs and measure memory/time
before scheduling a matrix. The package does not change device allocation or
terminate other processes.

## Information and sampling contract

Training reads only `learner/train.npz` and `learner/metadata.json`. It does not
read R/C/G queries, the evaluator directory, rules, exact distances, or action
permutation maps. Encodings and goal pools are fit on selected training records.
The vocabulary has an input-only unknown category for unseen symbols; inference
never expands the vocabulary from held-out boards.

Every attempted action has one observed legality label. Other actions' legality
is unknown. The policy action set is the whole declared ID range; no true action
mask is accepted by the update interface. The transition model predicts a
categorical distribution for each output slot and a single legality logit for
the queried state/action. It does not impose XOR, token conservation, parity,
Hanoi constraints, reversibility, or a known transition formula.

Coverage subsets are selected by a stable hash on **observed** undirected legal
edges, grouping inverse/parallel actions together, or rejected `(state, action)`
queries. Fractions are nested at a fixed coverage seed. Actual retained counts
are recorded; there is no backfilling to achieve an exact fraction or guarantee
that every action survives. Both vocabulary and unique goal-state pool are
recomputed from the selected subset. An empty subset fails explicitly.
In a coverage ablation, C's training-support description initially refers to the
full corpus. Audit primitive support against the actual selected subset before
interpreting a formal comparison; subsampling can remove previously seen steps.

| Method | Training samples and goals | Native diagnostics |
|---|---|---|
| GCBC | Uniform nontrivial observed transitions, with the observed successor as goal | G; native R/C unsupported |
| GCIQL | Uniform attempted transitions, with separate value/actor goal mixtures | G; native R/C unsupported |
| Dynamics + planning | Uniform attempted transitions, next-state categories and attempted-action legality | R, C, G |

The GCBC filter excludes self-loops that supply no directed action toward an
unsatisfied goal; their count remains visible in the corpus identity. IQL keeps
rejected attempts. Its independent goals are uniform over distinct observed
training states excluding the current state. Its future branch uses the observed
successor when that successor differs from the current state, otherwise falls
back to the independent branch. Value and actor branches draw independently.
Sampling counters record rejected attempts, branch selections and value-goal
successes; a future-branch probability is not the same as the realized success rate.

Value rewards are zero when the observed successor equals the relabeled goal,
otherwise minus one. Bootstrap is zero only at that goal success. The one-step
collection cutoff does not disable bootstrap. Initially achieved goals are
excluded from action/value transition sampling. Consequently, a rejected attempt
receives reward -1 and bootstrap 1 under an unsatisfied relabeled goal.

## Learning and planning semantics

GCBC minimizes categorical negative log likelihood. GCIQL uses twin Q estimates,
an upper-expectile value objective and advantage-weighted actor regression, with
stopped targets/weights and post-gradient target-network averaging. The actor and
value losses may use different sampled goals. This implementation uses the AWR
actor, not a continuous-action DDPG+BC objective. The definitions follow
[IQL](https://arxiv.org/abs/2110.06169) and the
[OGBench goal-conditioned formulation](https://arxiv.org/abs/2410.20092);
see [source attribution](THIRD_PARTY_NOTICES.md).

All models use categorical one-hot inputs and MLPs with configurable hidden
widths. The generic dynamics model minimizes mean per-slot categorical cross
entropy plus binary cross entropy for the attempted action's observed legality.
No unattempted action receives a negative legality target.

The model planner is a bounded best-first search with a slot-mismatch heuristic,
beam width 32, and maximum depth 8 (also bounded by remaining control steps).
In v1, each popped node expands a batch of remaining action candidates through
`ModelBudget.predict_batch`; every candidate is charged, including unused batch
members if a goal is found early. RC1 retains its single-query path. The planner
skips model-predicted illegal edges/self-loops and
returns a first action toward a model-predicted goal or the best explored
mismatch/depth candidate. If no candidate progresses, it falls back to an action
ID using a deterministic within-case rotation. This heuristic is neither an
exact distance nor a complete solver. Invalid hypothetical boards are not repaired
using puzzle rules. Budgets that cannot cover every first action remain reported
as supplied; the primary planned comparisons use 32 and 128.

R predicts complete successor states and legality. C recursively feeds each
prediction into the next step without receiving a true intermediate board.
Direct policies mark unsupported R/C results explicitly rather than reporting
zero scores or attaching an untrained auxiliary predictor. G uses a fresh policy
object per case. The complete learner/optimizer/step state is hashed before and
after prediction/evaluation to verify that it was not updated.

## Checkpoints, provenance and failures

Each new run requires a fresh directory. A run stores resolved configuration,
training hashes, selected-row identity, source-file snapshots, dependency versions,
sampling counters, JSONL metrics and immutable numbered checkpoints. Checkpoints
contain no pickle; a ZIP carries a checked header and Flax MessagePack state.
Load requires the recorded baseline source identity. A run's `source/` directory
can be placed on `PYTHONPATH` to restore its exact implementation.

```sh
copuzzle-baselines train --learner artifacts/baseline-development/data/learner \
  --resume artifacts/baseline-development/gciql/checkpoint-00001000.zip \
  --steps 2000 --output artifacts/baseline-development/gciql-resumed
```

`steps` is the total update count, including the checkpoint's prior updates.
Unless overridden, the CLI restores its method, config and fitting seed. A
changed corpus, coverage, sampling configuration or fitting seed is rejected.
Restore preserves Adam, target networks, exact sampler RNG state and counters.
The resumed run records its parent checkpoint hash; it does not overwrite it.

Training exceptions retain available checkpoints and mark the run failed.
Numerically nonfinite updates are rejected before modifying learner state.
The source guard refuses checkpoints if implementation files changed mid-run.
`train_seconds` measures the update loop, JIT compilation, logging and checkpoint
I/O for that run segment; it excludes initial data loading/network construction.
Full-process and device costs must additionally be measured for formal comparisons.

Inference verifies the checkpoint against the bundle's original training and
metadata hashes, which also bind opaque action identities. Full bundle semantic
validation belongs to evaluation, never to the training process. Evaluation
reports preserve every G case and retain the protocol's failure-inclusive costs.

Checkpoint schema v2 records the numerical learner/runtime source identity;
the full module tree is also saved. CLI/reporting additions alone no longer
invalidate model weights. Historical v1 checkpoints use their saved source
snapshots, without silent migration. Run schema v2 records checkpoint/logging
schedules, actual backend/devices, setup/function time and an optional bound
formal-study context. The ordinary CLI still creates development fits.

For benchmark-purpose v1 data, `predict` and `evaluate --split test` require
`--test-lock FILE` tying the exact checkpoint, configuration, source, seed and
training-data identity to a sealed selection record. Development-purpose bundles
remain usable for engineering validation. Lock creation from the complete
registered study, retries/failure accounting and final aggregation are separate
executor responsibilities; a manually labeled context is not evidence of a
completed formal experiment.

Data-only lookup, fitted XOR/position operators and privileged exact-dynamics
controls are available in `copuzzle_baselines.controls`. They use the same
bounded planner where applicable. `python baselines/examples/v1_controls.py
--output artifacts/benchmark-v1-development` runs these controls on seed-0 dev
data only. Structural assumptions and privilege are reported explicitly. V1
model evaluation additionally reports teacher-forced C-context diagnostics and
primitive support against the actual coverage subset; primary C stays open loop.

Formal matrix execution, completed selection, statistical aggregation,
CI and public distribution remain release gates, tracked in the
repository's [release plan](https://github.com/AIogry/CoPuzzle/blob/main/benchmark_release_plan.md).

The registered development-selection controller is launched from the repository
root after generating the v1 development control data:

```sh
JAX_PLATFORMS=cpu python -m copuzzle_baselines.study.pilot \
  --plan release/method_selection_plan.json \
  --data artifacts/benchmark-v1-development/data \
  --frozen-data artifacts/benchmark-v1-frozen \
  --output artifacts/benchmark-v1-pilot --workers 2
```

It copies source/configuration snapshots, executes 48 registered development
fits, records startup/fit/evaluation failures and selects the GCIQL goal mixture
using only dev scores with the declared family weighting/tie break. An incomplete
or duplicated matrix cannot write a successful selection. This development
selection file is not a final-test authorization lock. The [formal executor](STUDY.md)
adds registration, kernel-backed job leases, immutable attempts, checkpoint
recovery and complete-cohort test sealing. The [analysis workflow](ANALYSIS.md)
adds verified per-case evidence and paired statistics. Executed formal analysis
and public acceptance remain separate gates.
