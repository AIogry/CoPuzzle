# CoPuzzle v1 dataset card

Status: the v1 data inventory is frozen; learned comparisons and public release
acceptance are in progress. This card describes `copuzzle.rule-composition.v1`,
not the earlier game-specific trajectory collections or RC1 qualification data.
The exact contract is [benchmark_protocol.md](benchmark_protocol.md).

## Intended use and scope

The dataset tests whether finite offline transition experience supports
held-out transition prediction (R), open-loop composition prediction (C), and
closed-loop goal control (G). It comprises fully observed, deterministic, finite
state spaces. Every model is fitted within one instance and partition. Test
action IDs keep that dataset's mapping. This is not a test of learning unseen
game families, changing rules, transfer to larger boards, visual perception,
or contact-rich robot control.

| Instance | Reachable component states | Diameter | Easy / medium / hard exact distance | Control attempts |
|---|---:|---:|---|---:|
| Lights Out 3×3 | 512 | 9 | 1–3 / 4–6 / 7–9 | 18 |
| Lights Out 4×4 | 4,096 | 7 | 1–2 / 3–5 / 6–7 | 14 |
| Pancake 6 | 720 | 7 | 1–2 / 3–4 / 5–7 | 14 |
| TopSpin 8/4 | 40,320 | 22 | 1–7 / 8–14 / 15–22 | 44 |
| Sliding 2×3 | 360 | 21 | 1–7 / 8–14 / 15–21 | 42 |
| Hanoi 5 | 243 | 31 | 1–10 / 11–20 / 21–31 | 62 |

Counts refer to the declared reachable component, not every syntactically
possible observation. Distances count atomic legal abstract operations. The
fixed control horizon is twice the instance diameter and does not reveal an
individual query's distance. The separate physical extension uses Lights Out,
Pancake-4 and TopSpin-20/4; its execution/witness costs are not these distances.

## Construction and inventory

Six instances × two holdout conditions × three data seeds (101, 202, 303)
produce 36 bundles. They contain **703,076 one-step training episodes** and
**220,434 evaluation queries**:

| Probe | Development | Final test |
|---|---:|---:|
| R: transition prediction | 103,456 | 103,696 |
| C: composition prediction | 3,190 | 3,180 |
| G: goal control | 3,456 | 3,456 |

The inventory SHA-256 is
`c43ba9209ecbfccf44edd28fce9d77c93a553d7a227fd3179e84d5bf907abb35`.
The [label-free freeze descriptor](release/data_freeze.json) provides individual
bundle/file hashes, cell populations and counts. It was frozen before the
registered method-selection experiment. Generator and metric sources are saved
with the full inventory; do not regenerate old data under a new protocol label.

Training enumerates every retained attempted one-step query once. Legal inverse
edges and parallel actions share a partition assignment; rejected actions keep
an unchanged state and a false legality observation. Transition holdout separates
these query groups and permits state reuse. State holdout separates endpoint
pools, retaining transitions within a pool. The two conditions therefore ask
different questions and are reported separately.

R exhausts its retained held-out queries. C/G select distinct eligible pairs
using deterministic hash priority and a shared dev/test unordered-pair assignment.
Their candidate population uses all endpoints for pools up to 4,096 states, or
256 hash-selected reference endpoints for larger pools. Each required cell uses
`min(32, eligible population)` cases without duplication; small cells are
exhaustive. All required cells are nonempty. The unequal counts are intentional,
and primary metrics balance length/tier rather than pooling all cases.

## Files and access boundaries

Each bundle separates `learner/` from `evaluator/`:

| File | Meaning and shape |
|---|---|
| `learner/metadata.json` | Public task ID, widths, opaque action count, fixed horizon and schema conventions |
| `learner/train.npz` | `observations[N,2,D]`, `actions[N,1]`, `legal[N,1]`; one attempted transition per episode |
| `learner/R_{dev,test}.npz` | Input boards and opaque action IDs |
| `learner/C_{dev,test}.npz` | Input boards, action words and lengths; unused action positions padded with −1 |
| `learner/G_{dev,test}.npz` | Initial boards and goals |
| `evaluator/` | Privileged answers, exact distances, partition/mapping information and validation evidence |
| `manifest.json` | Bundle identity, generation settings, file checksums and population accounting |

Fitting reads only training arrays and public metadata. Even probe inputs are
not fitting data. Unattempted action legality is unknown; full true masks, action
semantics, solver paths and distances are not learner inputs. The public metadata
names an instance, so compliance also requires that a learner implementation not
dispatch to a known rule table using that name. Physical/legacy Gym interfaces
may expose other diagnostics and are not substitutes for this learner adapter.

The one-step episode end is a **collection cutoff**, not an MDP terminal or a
successful goal. Goals can be relabeled from observed training states, subject
to the declared method's sampling policy. The core corpus has no long behavior
trajectory whose intended goal silently switches. Earlier physical segmented-play
datasets explicitly store segment boundaries and goals under their own schema.

## Limitations, maintenance and rights

The corpus deliberately covers small finite components and synthetic exact
states. Exhaustive retained one-step coverage differs from naturally collected
offline robot experience. Coverage ablations are nested observed-query subsets,
not a new family or size generalization test. They may remove primitive support
from C; report the actual subset's support diagnostics. A finite reference bank
does not represent all possible long-range pairs in a large state space.

No human-subject, personal, or scraped behavioral records are part of this
generated core dataset. Robot meshes and external reference code have separate
provenance and licenses. The proposed project code/data licensing and public
credit remain [pending owner confirmation](release/license_proposal.md); this
card grants no new license. Download locations, final result/checkpoint hashes
and a permanent citation will be supplied only after their release gates pass.
Report issues through the [repository](https://github.com/AIogry/CoPuzzle/issues).
