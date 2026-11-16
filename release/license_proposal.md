# Proposed public licensing — owner decision pending

**This is a review proposal, not an adopted project license.** The repository
currently has no project-level license. Do not treat this file as granting rights
to CoPuzzle's original contributions.

The proposed split is:

| Material | Proposed treatment |
|---|---|
| Original CoPuzzle and baseline code | MIT; proposed notice: “Copyright (c) 2026 CoPuzzle contributors” |
| Original generated state/action datasets, annotations and result tables | CC BY 4.0, with attribution to the project and exact artifact/version |
| Bundled UR5e XML/mesh assets | Keep their existing BSD-3-Clause license and ROS Industrial Consortium notice |
| OGBench-derived/reference baseline material | Retain the OGBench Authors' MIT notice and source attribution |
| Separately installed dependencies | Retain their own package licenses; do not claim CoPuzzle licenses those packages |

MIT permits broad software reuse subject to preservation of its notice; the
proposed text is in [proposals/LICENSE.code.txt](proposals/LICENSE.code.txt).
See the [official MIT text](https://opensource.org/license/mit).

CC BY 4.0 permits sharing/adaptation, including commercial use, subject to its
attribution and other terms. The proposed dataset notice is in
[proposals/DATA_LICENSE.md](proposals/DATA_LICENSE.md); the controlling terms
would be the [Creative Commons legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en).
This would cover only rights held in the project's own contributions. It would
not relicense third-party assets or imply a grant of rights the project does not hold.

Before adoption, the owner needs to confirm authority over the original code and
data, including any institutional/coauthor requirements, and accept or replace
this proposed license split. The public copyright/citation name also needs to be
confirmed. “CoPuzzle contributors” is a proposed collective label, not an invented
list of individual authors or an assertion of institutional endorsement.

The asset/reference inventory is [third_party_inventory.json](third_party_inventory.json).
Original third-party files and notices are retained; no approval is being sought
to remove or replace them. Code, training and validation work can continue while
the owner's licensing/credit decision is pending.
