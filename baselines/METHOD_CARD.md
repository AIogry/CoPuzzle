# CoPuzzle v1 baseline method card

Status: the registered formal fits, final evaluations and technical qualification
are complete; public release approval remains pending. This card describes the
methods; artifact evidence is indexed in the release ledger. The authoritative settings are
[`release/selected_methods.json`](../release/selected_methods.json), and the
matrix binding is [`release/study_registration.json`](../release/study_registration.json).

## Capability and information contract

| Method | Learned object | Native probes | Explicit model queries per G decision |
|---|---|---|---|
| GCBC | Categorical goal-conditioned actor | G | 0 |
| GCIQL | Twin Q, expectile value, advantage-weighted actor | G | 0 |
| Dynamics + planning | Per-slot categorical successor and attempted-action legality | R, C, G | 32 and 128 |

All encodings, vocabularies and goal pools are fitted on retained training
records. Inputs use categorical state slots and opaque action IDs. There is no
true action mask, oracle goal relabeling, exact-distance supervision or simulator
query during fitting. The generic transition model does not hard-code puzzle
operators or repair predicted states with known rules. GCBC/GCIQL do not expose
a native R/C predictor; unsupported results stay null and do not diagnose an
absence of internal rule information.

GCBC uses nontrivial observed successors as goals. GCIQL keeps attempted illegal
transitions and samples future-successor and independent observed goals with
separate actor/value mixtures. A failed attempt toward an unsatisfied relabeled
goal has reward −1 and bootstraps; collection cutoffs are not terminals. Full
sampling/loss details and source references are in [README.md](README.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Frozen fitting and evaluation choices

- Two 128-unit hidden layers, batch 256, learning rate 0.0003, 10,000 updates.
- GCIQL discount 0.99, expectile 0.7, advantage temperature 3, target rate 0.005;
  actor future probability 0.5 and value future probability 0.5.
- Last registered update only; checkpoint every 1,000 updates, log every 100.
- Three data seeds (101/202/303), three fit seeds (0/1/2), coverage 25/50/100%.
- CPU backend, at most two formal workers, exact recorded dependency versions.

The 48-fit development selection chose the GCIQL actor mixture from the declared
options. Development scores are not final comparison results. The complete
matrix has 324 full-coverage primary fits and 648 coverage fits: 972 total, each
with 10,000 updates. Dynamics' two G budgets yield 1,296 learned evaluations per
split, not additional fitted models. Every failed fit/evaluation remains recorded.

The bounded planner uses slot mismatch, beam width 32 and depth at most 8. It
charges every predicted state/action candidate, including batched candidates,
unknown transitions and predicted illegal attempts. It is not an exact solver.
Models stay frozen during R/C/G evaluation; state/optimizer/counter hashes are
checked. Direct policies act greedily with no explicit learned-model queries.

## References and interpretation

Random actions, empirical lookup, fitted XOR/position operators, a true dynamics
model with the same bounded planner, and an optimal oracle are separate controls.
Structural controls are declared only on their supported game families and
carry explicit algebraic priors. Oracle controls are privileged; the optimal
oracle has no comparable model-query budget. They are not ordinary learned
competitors in a single undifferentiated ranking.

All results must include per-instance/condition estimates, failure counts,
coverage and search contrasts, and measured work from every attempt. The
[analysis workflow](ANALYSIS.md) specifies paired seed uncertainty, family
weighting and raw-case evidence. Equal update counts do not mean equal epochs
or equal hardware cost. A few seeds and one small MLP family support limited
claims; failure can reflect model capacity, optimization, relabeling, planning,
or their interaction. No favorable score or inferred internal representation is
a release requirement.
