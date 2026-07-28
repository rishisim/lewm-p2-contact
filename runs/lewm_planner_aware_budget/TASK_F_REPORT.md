# Task F PushT critic feasibility screen

Status: complete under the Tasks A–E verified runtime: Python 3.10, PyTorch
2.12.0/MPS, Stable WorldModel cache checkpoint
`0be0611227823a00b2f9acac299375a013bb23cd2a33b8ffa94a3bdb5e19b737`, and
the 2,336,736-row `pusht_expert_train.lance` dataset.

## Result and decision

**Fail. Retain the qualified native depth-0/population-300 planner cost and
stop this planner/refinement branch.** No critic-in-CEM integration,
closed-loop confirmation, dispatcher, or refiner retraining is authorized.

The frozen selection role chose native: mean bank regret was 3,073.0 for
native versus 3,116.6 for the linear head. The untouched evaluation contained
8 banks (4 entirely held-out source starts × 2 fresh candidate seeds), each
with 300 valid simulator-labelled candidates. For transparency, the tested
linear head is retained in the aggregate even though it was not selected.
Its held-out linear-minus-native top-choice-regret difference was **+897.1**
(whole-start paired bootstrap 95% CI **+461.1 to +1,372.2**). It was harmful
for both candidate seeds (+1,718.1 and +76.1).

The linear head made 8 choice changes: 3 helpful and 5 harmful, with
regret-weighted benefit -7,176.5. Its catastrophic selected-regret fraction
was 0.125 versus 0.0 native, and its selected-regret 90th percentile was
0.860 versus 0.595 native. These independently fail the frozen benefit,
cross-seed, helpful/harmful, and safety gates. The 52-parameter candidate
local head itself was fast (median/p95 0.0188/0.0199 ms per 300-candidate
bank); the full native-rollout-plus-head timing was 172.7/227.0 ms. Latency
therefore does not rescue the failed scientific endpoint.

The critic remains restricted to final-bank reranking; no early-CEM stage or
closed-loop generalization is claimed. Native/linear secondary ranking metrics,
sealed role/bank identities, replay checks, and hash-pinned artifact provenance
are in `task_f_results.json`, `task_f_manifest.json`, and
`task_f_artifact_manifest.json`. Calibration is not meaningful for the frozen
ranking-only objective and was not fitted.
