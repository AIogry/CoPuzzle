# Local distribution staging and independent verification

Staging creates local drafts. It neither uploads artifacts nor grants a license.
The complete formal experiment, CI/install/replay evidence, rights and citation
decisions, and public hosting remain separate release gates. Current licensing
status is recorded in [`release/license_proposal.md`](../release/license_proposal.md).

The frozen dataset can be staged independently of fitting:

```sh
python -m copuzzle_baselines.study.packaging data \
  --data artifacts/benchmark-v1-frozen --output artifacts/distribution-draft-v1/data
python -m copuzzle_baselines.study.packaging verify \
  --output artifacts/distribution-draft-v1/data
```

The archive manifest binds every member and every compressed part by SHA-256 and
byte count. Members have deterministic path ordering and zero metadata timestamps;
gzip headers omit machine paths and timestamps. Parts default to at most 1 GiB
of uncompressed input. Verification checks regular-file types, paths, duplicates,
member hashes and completeness without extracting files or following links.
The exact builder source is retained alongside each draft archive manifest.

Each staging command accepts optional `--notices /path/to/manifest.json`.
The notice manifest uses schema `copuzzle.distribution_notices.v1`, a `state`
of `proposal` or `owner_confirmed`, and a `files` mapping from relative paths to
SHA-256 digests. Notices are copied to a sibling `notices/` directory, outside
the immutable `data/` and `study/` trees. This preserves frozen inventories.
The state is supplied decision metadata; the packager does not establish rights
or obtain approval. A proposal remains a proposal. Confirm ownership, the chosen
code/data/checkpoint terms and citation names before an actual public release,
and retain all applicable third-party notices with the distributions.

After all registered fits and evaluations finish, the cohort is sealed, R/C case
archives are verified, and the [analysis](ANALYSIS.md) is complete:

```sh
python -m copuzzle_baselines.study.packaging study \
  --study artifacts/benchmark-v1-study \
  --predictions artifacts/benchmark-v1-predictions \
  --analysis artifacts/benchmark-v1-analysis \
  --report artifacts/benchmark-v1-report \
  --output artifacts/distribution-draft-v1/study
```

This rejects an active study, incomplete or changed outcomes, mismatched case
archives, changed attempt costs, or statistics that fail recomputation. It stages
the selected final checkpoints, recovery checkpoints referenced by attempt
records, full registered source snapshots, every original/retry attempt record,
logs, training manifests, metrics, evaluation reports, case evidence, analysis,
readable tables, optional figures and the evidence-based interpretation.
Unselected intermediate checkpoints and duplicate per-fit source copies are
omitted explicitly. This is sufficient to restore and evaluate the selected
models; it is not an archive of every intermediate training state.

The staged study has no machine-specific `locations.json`. After verifying and
extracting the data and study parts, install the declared package/dependency
versions and bind the new location:

```sh
python -m copuzzle_baselines.study.runner relocate --study /new/root/study --data /new/root/data
python -m copuzzle_baselines.study.runner validate --study /new/root/study
python -m copuzzle_baselines.study.prediction_archive --verify \
  --study /new/root/study --output /new/root/predictions
```

Relocation verifies the same frozen inventory and refuses a live controller or
worker. It does not modify the experiment, checkpoint or selection identities.
Recorded numerical sources and runtime versions remain mandatory; bitwise
cross-platform learning equivalence is not promised.

An independent installed-package check reruns a fixed subset of the complete
final study. It selects data seed 101 and learned fit seed 0 at full coverage,
all twelve task/condition combinations and both learned model budgets: 48
learned evaluations. It also checks 56 reference evaluations with reference seed
0 and 32-query planning where applicable. Selection uses no performance values.

```sh
JAX_PLATFORMS=cpu python -m copuzzle_baselines.study.reproduce \
  --study /new/root/study --output /new/root/independent-reproduction \
  --forbid-source /path/to/source/checkout
```

Both imported packages must be outside the forbidden checkout. Sibling research
and optional physics imports are blocked. The checker restores the fixed
checkpoints and compares all metrics, cases, budgets and provenance with the
retained reports; only measured timing fields are excluded. Fresh reports and
any differences are retained. It refuses an active/incomplete study and makes
no claim of retraining or identical execution on another platform.

## Physical compatibility evidence

Historical physical inputs have separate NPZ and JSON-sidecar checksums in
[`release/physical_history_inputs.json`](../release/physical_history_inputs.json).
Run the installed core from outside the checkout:

```sh
MUJOCO_GL=egl python /path/to/source/scripts/verify_physical_history.py \
  --manifest /path/to/source/release/physical_history_inputs.json \
  --inputs /path/to/physical-history \
  --output /path/to/new/physical-history-report.json \
  --forbid-source /path/to/source
```

The check replays every saved continuous action under its original configuration,
comparing observations, events and episode endings through the native APIs. It
retains failures and actual runtime/source identities. Its result is historical
compatibility evidence, not a learned-policy score or a promise of identical
floating-point behavior on another simulator version.

After that qualification passes, stage the original NPZ/JSON pairs and evidence:

```sh
python -m copuzzle_baselines.study.packaging physical \
  --manifest release/physical_history_inputs.json --inputs artifacts \
  --qualification artifacts/release-engineering-v1/physical-history-with-metadata.json \
  --script scripts/verify_physical_history.py \
  --output artifacts/distribution-draft-v1/physical
```
