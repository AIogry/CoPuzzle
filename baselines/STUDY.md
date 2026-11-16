# Registered formal execution

The executor registers 324 primary fits and 648 coverage-ablation fits before
training: 36 datasets × three fitting seeds × three methods × three coverage
levels. The 972 models imply 1,296 learned evaluations per split because dynamics
uses both 32/128-query budgets. References add 576 evaluations per split.
Evaluation counts are not additional fitted models.

Use an environment matching `release/selected_methods.json`, from the repository
root after installing the core and baseline components:

```sh
JAX_PLATFORMS=cpu python -m copuzzle_baselines.study.runner register \
  --data artifacts/benchmark-v1-frozen --methods release/selected_methods.json \
  --output artifacts/benchmark-v1-study
JAX_PLATFORMS=cpu python -m copuzzle_baselines.study.runner run \
  --study artifacts/benchmark-v1-study --phase fit --workers 2
python -m copuzzle_baselines.study.runner status \
  --study artifacts/benchmark-v1-study --phase fit
```

Each job is bound to its actual training-view identity, configuration, coverage,
seed, update count, schedules, numerical source and dependencies. Data location
is kept separately in `locations.json`; relocation is allowed only to the same
inventory hash. Workers import the registered source snapshots, so checkout
edits do not silently change an active experiment. Other jobs/environments are
not stopped or modified.

The public binding for the current formal registration is
[`release/study_registration.json`](../release/study_registration.json).
Controller-only recovery fixes do not change model/data identities: each new
controller invocation preserves its actual source, start and terminal records in
`controllers/`. The initial simultaneous-worker-completion scheduling fault and
its early exit are retained there; completed fits were reused without refitting.

## Recovery and liveness

Every job has immutable attempt records and an atomic receipt. A receipt never
points to a record being rewritten. Reports/checkpoints have independent hashes.
Startup failure without a `run.json` is recorded as failure, never success.

Controllers and workers use POSIX kernel advisory locks. File existence or a
stored PID is not proof of liveness; the held kernel lock is probed. This avoids
mistaking namespace-local PIDs for host IDs. One controller owns the study and
each worker separately holds its job lease throughout work.

Completed fits are verified and skipped. Interrupted work resumes the highest
valid checkpoint for the same registered job, preserving optimizer, targets,
RNG and counters. A valid final checkpoint can be recovered after a receipt
failure without retraining. Corrupt/mismatched candidates are recorded/rejected.
Failed jobs require an explicit diagnosed retry:

```sh
python -m copuzzle_baselines.study.runner stop --study artifacts/benchmark-v1-study
JAX_PLATFORMS=cpu python -m copuzzle_baselines.study.runner run \
  --study artifacts/benchmark-v1-study --phase fit --workers 2 --clear-stop
# After diagnosing failed jobs, retry identical settings if appropriate:
JAX_PLATFORMS=cpu python -m copuzzle_baselines.study.runner run \
  --study artifacts/benchmark-v1-study --phase fit --workers 2 --retry-failed
```

Stop requests propagate to this study's workers. Updates after the last saved
checkpoint may be recomputed; their earlier process costs remain recorded.
Never select favorable retries/seeds or overwrite prior attempts. Cancellation
does not become a falsely completed evaluation.

## Final-test gate

Optional `--phase dev` evaluates the registered final checkpoints on dev. Their
selection remains the predeclared last update; results do not authorize changing
the sealed method settings.

```sh
JAX_PLATFORMS=cpu python -m copuzzle_baselines.study.runner seal \
  --study artifacts/benchmark-v1-study
JAX_PLATFORMS=cpu python -m copuzzle_baselines.study.runner run \
  --study artifacts/benchmark-v1-study --phase test --workers 2
JAX_PLATFORMS=cpu python -m copuzzle_baselines.study.runner run \
  --study artifacts/benchmark-v1-study --phase control-test --workers 2
```

Sealing checks the entire registered fit cohort, restores completed checkpoints,
and verifies data/config/source/seed/step bindings. A barrier prevents new fitting
or retries during sealing. Cohort/authorization records are published atomically.
Pending, active, interrupted or missing jobs cannot seal. A handwritten completion
flag is insufficient: test access also verifies the registered matrix and full
cohort manifest. No fitting or retries are permitted after the seal.

Failed fits normally prevent sealing and require diagnosis. If a documented
algorithm failure legitimately remains under the fixed protocol, explicit
`seal --include-failed` retains it. It is unavailable and receives zero primary
performance in failure-inclusive analysis; it is never dropped. Evaluation
startup/model failures likewise stay in the denominator. A broken implementation
is not evidence of a successfully completed benchmark.

The seal binds checkpoint hashes and reference-control settings to exact data
manifests. Structural assumptions and privileged references remain labeled; the
optimal oracle has no comparable query budget. Qualification studies used in
tests cannot authorize final benchmark tests.

Raw outcomes and all attempt costs must feed paired uncertainty and family-macro
analysis. Terminal counts alone do not establish scientific conclusions or
release readiness. Aggregation, CI, final installation/replay, licensing,
attribution and distribution remain separate acceptance gates.
