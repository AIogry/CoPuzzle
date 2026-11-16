# Complete-cohort analysis

The analysis contract is registered in
[`release/analysis_plan.json`](../release/analysis_plan.json), before final-test
evaluation. It supplements the frozen metric definitions without changing the
data, learners, evaluation cases or selected training settings.

After every fit and evaluation has a terminal receipt, preserve per-case rule
and composition predictions and verify their scores:

```sh
JAX_PLATFORMS=cpu python -m copuzzle_baselines.study.prediction_archive \
  --study artifacts/benchmark-v1-study --output artifacts/benchmark-v1-predictions
JAX_PLATFORMS=cpu python -m copuzzle_baselines.study.prediction_archive \
  --study artifacts/benchmark-v1-study --output artifacts/benchmark-v1-predictions --verify
python -m copuzzle_baselines.study.analysis \
  --study artifacts/benchmark-v1-study --predictions artifacts/benchmark-v1-predictions \
  --plan release/analysis_plan.json \
  --output artifacts/benchmark-v1-analysis
```

This reads the sealed final selection, never refits a learner, and reproduces R/C
predictions using the selected checkpoint or registered reference model. Archive
export can resume within the same source/runtime binding. Search budgets share
R/C archives; their G rollouts remain separate. An archive cannot be accepted if
its predictions do not reproduce both saved metric reports exactly. G case
records are already retained in the original evaluation files.

The analyzer verifies checkpoint, data, source, configuration, seed, split and
test-authorization bindings. It recomputes G aggregates from all case records,
checks frozen distance/tier labels and failure penalties, and checks R/C group
weights and archive scores. Missing, running and interrupted jobs block a final
analysis. A failed job stays present with zero supported performance and full
normalized goal cost. Unsupported native R/C interfaces use `null`; they do not
become evidence of failed rule understanding. Unknown crash duration is not zero.

`rows.json` retains every registered evaluation, failure, stratum and available
composition-support diagnostic. `goal_cases.jsonl` retains every expected goal
outcome, marking unavailable evaluations without inventing trajectories.
`statistics.json` contains per-instance, per-family and core family-macro
estimates and paired contrasts. `costs.json` counts original and retry attempts
once, so evaluating a checkpoint under two budgets does not double-count its
training. The manifest binds these outputs and the analysis source snapshot.

## Estimands and uncertainty

R balances present legality classes; C balances sequence lengths; G balances
difficulty tiers. These are three separate probes. G's cost companion divides
failure-inclusive attempts by the instance's fixed horizon. Conditions are
reported separately. Average the two Lights Out instances within their family,
then weight the five game families equally. Controls defined on a subset of
families receive only matching instance/family summaries, not an all-core score.

For each estimate, average the three fitting seeds within each of the three data
seeds. The paired hierarchical bootstrap uses 10,000 draws and RNG seed 0: draw
three data-seed IDs with replacement, then draw three fit-seed IDs inside each
sampled cluster. The same indices apply across methods, fixed tasks, coverage
levels and budgets. Repeated outer clusters get independent inner draws.
Deterministic references broadcast across the fitting axis only to support
pairing; they are not counted as three independent fits.

The 2.5/97.5 percentiles describe uncertainty conditional on these fixed
instances, cases, partition recipe and selected method settings. There is no
case/family bootstrap and no claim about unseen games. Only three outer seeds
make uncertainty estimates fragile; publish all nine values alongside intervals,
avoid significance declarations, and do not select a favorable seed or budget.
This follows the general motivation to report interval evidence in few-run RL
evaluation described by [Agarwal et al.](https://arxiv.org/abs/2108.13264); the
specific hierarchy and weights here are CoPuzzle's declared design, not a claim
that this exact procedure was prescribed by that paper.

Coverage comparisons hold gradient updates fixed, not epochs or the number of
distinct samples seen. Subsampling can remove primitive-query support from C;
report its support diagnostics with the composition score. Comparing learned
dynamics with a budget-matched true model distinguishes model error from limits
of this particular planner/heuristic. The optimal oracle is privileged and has
no comparable query budget. None of these contrasts alone establishes an
agent's internal rule representation or a causal account of its errors.
