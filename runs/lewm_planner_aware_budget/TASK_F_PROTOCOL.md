# Task F: PushT simulator-aligned candidate critic

Status: frozen before Task F candidate generation, labels, fitting, selection,
or evaluation outcomes.

## Question and bounded claim

Can a lightweight, inference-safe scalar candidate cost improve simulator
top-choice regret when reranking a new, fixed, final depth-0/population-300
CEM bank? This is **not** an iterative-CEM integration or closed-loop claim.
It is restricted to final-iteration banks. Earlier CEM proposal distributions
are deliberately out of scope, so a pass cannot justify scoring earlier CEM
iterations without a new protocol.

Task E banks, labels, results, starts, and seeds are consumed discovery
evidence only: none can enter fitting, normalization, model selection,
thresholds, or evaluation. The qualified depth-0/population-300 dynamics and
native cost path are retained exactly. No refiner or dispatcher is trained.

## Sealed design and data handling

`freeze_task_f.py` seals entirely new source episodes/start rows and candidate
seeds. Roles are pilot, fit, selection, and evaluation; source episode and
candidate-bank seed never cross roles. Each role uses two new bank seeds. The
canonical runner uses Task C's `call_seed(seed,row_id,0)` innovation contract
and the Task E reset → `_set_state` → `_set_goal_state` replay semantics.
Every candidate is executed once from its sealed state/goal. Model-visible and
environment-executed actions must be exact; a preregistered 12-candidate
replay subset is bitwise exact. All raw candidates, labels, action features,
predictions, logs, checkpoints, and temporary manifests belong only in ignored
`runs/lewm_planner_aware_budget/work/task_f/`.

The mechanical pilot is exactly four banks (two starts × two seeds). It may
check schema, finite values, exact replay, runtime and storage only. It cannot
change cohort, seeds, bank size, feature set, model family, loss, thresholds,
or analysis. Sealed data contains 22 banks (11 starts × 2 seeds), 300
candidates each; the role sizes and ceilings are in the frozen config.

## Inference-safe critic contract

One candidate is scored independently with shape `[B,5,10]` actions and a
native depth-0 cost `[B]`; the batched output is scalar `[B]`. Features are
the native latent goal cost plus the candidate's 50 model-visible action
coordinates. These are causal functions of the current encoded observation,
goal encoding, depth-0 rollout, and candidate actions. Simulator state,
geometry/contact, task cost/success, future observations, outcome labels,
bank/start/seed/role identifiers, ranks, and bank-composition normalization
are forbidden and rejected by the schema. Normalization mean/std is fitted on
fit candidates only, then frozen; zero/invalid standard deviations are errors.
The linear head has 52 parameters (51 weights + bias) and uses no bank-wise
operation. Non-finite features or output fail the bank. This establishes
candidate-local, permutation-equivariant scoring suitable in principle for a
future replacement of a CEM cost, while this task measures only final-bank
reranking.

## Frozen heads, fitting, and selection

Exactly two heads are compared: native latent cost (no fitted parameters) and
one linear scalar head. A nonlinear head is not prospectively justified and is
not permitted. The linear head is fitted only on fit banks with a single
top-tail (`30`) pairwise logistic ranking loss for simulator cumulative task
cost ordering. Its fixed optimizer/epochs/seed are in config. Candidate-level
MSE and global pair totals do not select a head. Selection computes bank-level
top-choice regret, chooses the lower mean regret, and breaks exact ties in
favor of native. It writes a hash-pinned artifact manifest before evaluation;
the evaluator refuses to open evaluation banks before that manifest exists.

## Endpoints, uncertainty, and decision

The inferential unit is a start with both seeds retained; candidate pairs are
not independent. Primary endpoint is critic minus native simulator-referenced
top-choice regret, with a 9,999-draw whole-start paired bootstrap CI. Secondary
endpoints are top-k recall/overlap, concordance, rank correlation, calibration,
choice changes (helpful/tied/harmful at `1e-7`), per-seed direction, selected
candidate catastrophic regret fraction (`>=0.8` of bank range), worst-tail
selected regret (90th percentile), stage coverage, and critic/end-to-end
fixed-bank median/p95 latency. No multiplicity claim is made for secondary
endpoints. Stable index order resolves scoring ties only; ties remain ties for
set/rank metrics.

Proceed to one small closed-loop critic-in-CEM confirmation only if the frozen
selected head has a strictly negative 95% bootstrap upper bound for primary
regret difference, negative mean regret difference for both bank seeds, more
helpful than harmful changes with positive regret-weighted benefit, no increase
in catastrophic fraction or 90th-percentile selected regret, finite held-out
results across all evaluation starts, latency overhead at or below both frozen
limits, and valid final-bank-only feature coverage. Otherwise retain native
depth-0/population-300 and stop this planner/refinement branch. Evaluation is
never used to retrain, recalibrate, or alter the decision.
