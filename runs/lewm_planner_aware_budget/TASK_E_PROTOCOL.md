# Task E PushT planner-alignment diagnostic protocol

Status: frozen before candidate generation, simulator labels, or diagnostic outcomes.

## Question and scope

Task D completed 576 sealed closed-loop evaluations and found that depth 4
reduced normalized return by 0.0318 (95% CI -0.0493 to -0.0162), while exact
cross-candidate-seed package agreement was 0/24. Task E asks why the Task B
refiner's held-out latent-prediction improvement does not transfer to stable
planner rankings. This is an offline, fixed-bank diagnostic. It does not train
or deploy a dispatcher, refiner, gate, or adaptive controller and does not
repeat the 12-cell grid.

## Frozen cohort and candidate banks

The cohort has two mechanical-pilot and eight sealed development starts, one
row per unique source episode, with the Task A/D 25-step goal offset. Source
episodes used by Tasks A-D, including Task D pilot and sealed cohorts, and Task
B expert fit/selection/evaluation are excluded. The machine-readable manifest
seals rows, states, episode/environment seeds, exclusions, hashes, and
selection seed before any candidate or simulator outcome is generated.

Each start has two independent banks owned by candidate seeds 2026072701 and
2026072702. A bank is the population-300 candidate tensor from the final
(20th) iteration of the depth-0 Task C CEM solver at replan index zero. It is
therefore a final base-CEM distribution bank, not a union of depth-specific
proposals. The bank includes the fitted mean in row zero and 299 transforms of
the final maximum-population innovation stream. The qualified, unclipped Task
C mean and candidate values are preserved. Candidate tensors and predictions
remain under ignored `work/task_e/`; the committed manifest retains hashes and
compact summaries only.

The exact same ordered candidate tensor is scored at depths 0, 1, 2, and 4.
Each unique candidate is simulator-labelled once from the same restored PushT
state and its label is reused for every depth. Candidate identity is the hash
of little-endian float32 model-visible shape `[5,10]`.

## Prospective feasibility and stopping

The pilot is exactly two starts by one seed by 300 candidates (600 open-loop
rollouts). It may be inspected only for mechanics, finite values, schema,
reset/replay equality, and projected runtime/storage. Sealed execution proceeds
if all pilot banks are complete, at least 99% of rollouts are valid, the replay
subset is exact within the tolerances below, projected sealed wall time is at
most six hours, and projected raw storage is at most 2 GiB. One mechanical
repair and deterministic rerun is allowed. No endpoint, ranking, contact, or
depth comparison may alter the cohort, bank, or analysis. There is no silent
expansion beyond 4,800 sealed candidate rollouts.

## Simulator reference and action semantics

Every `[5,10]` model-visible sequence is reshaped chronologically to 25 raw
`[2]` actions and executed open loop for five action blocks of five environment
steps. The environment is reset with the sealed environment seed and Task D's
`_set_state` and `_set_goal_state` callables before every candidate. Both the
model-visible raw action and the exact action passed into the unwrapped PushT
environment are recorded. No Task E clipping or range redefinition is applied;
the environment's native action transform is retained. Candidate validity
requires 25 executed steps and finite state/cost/reward. The inherited
`_set_state` callable advances one physics tick, so the reconstructed post-call
state is required to match a second reset/reconstruction exactly; it is not
incorrectly compared with the pre-call dataset vector.

Primary simulator quality is negative cumulative full-state Euclidean task cost
(higher is better); terminal negative distance is co-reported. Success and
reachability are descriptive because the 25-step open-loop horizon is shorter
than Task D's closed-loop episode. Replay checks use 12 candidates selected
mechanically from each pilot bank (indices from a sealed seed), requiring exact
executed actions and absolute cost/state differences at most 1e-7.

## Primary ranking endpoints

The start/candidate-bank is the clustered unit; candidates within a bank are
never treated as independent inferential units. For each depth and bank:

1. top-choice regret: oracle simulator cost minus the simulator cost of the
   model's minimum-predicted-cost candidate (lower is better);
2. pairwise concordance/AUC over all unordered non-tied candidate pairs, with
   lower model cost and lower simulator cost defined as better;
3. top-k recall and Jaccard overlap at k=10 and k=30;
4. Spearman rho and Kendall tau-b, reported as missing for constant vectors;
5. candidate/action changes from depth 0, their L2/max-absolute difference, and
   whether each change improves, ties, or harms simulator cost.

Simulator ties use absolute tolerance 1e-7; prediction ties use 1e-8. Stable
index order breaks ties only for selecting a single candidate, never for
concordance. Invalid candidates receive no ranking label and are excluded from
continuous metrics; a bank below 297 valid candidates fails. Top-k uses all
candidates tied at the kth boundary and reports the expanded set size. Missing
bank metrics remain missing without replacement.

Primary comparisons are depth 1, 2, and 4 minus depth 0 for top-choice regret,
concordance, top-30 recall, and Kendall tau-b (12 contrasts). Direction is
positive-depth beneficial for lower regret and higher other metrics.
Whole-bank bootstrap resamples start clusters with both seeds retained: 9,999
draws, seed 2026072791. Whole-bank paired sign-flip permutation uses 9,999
draws, seed 2026072792. Holm correction is applied across the 12 primary
contrasts separately for bootstrap sign exclusion and permutation p-values.
All other endpoints and ablations are mechanism diagnostics with intervals but
no confirmatory multiplicity claim.

## Frozen ablations and mechanisms

The scoring boundary supports fixed schedules over the five predicted
transitions. The all-zero schedule must be bitwise equal to depth 0. Frozen
schedules are: every-step depth 1/2/4; depth 1/2/4 applied only at transition
1, 2, 3, 4, or 5; and cumulative prefixes of one through five refined
transitions. History prefixes 1/2/3 are evaluated where defined by retaining
only the most recent prefix in the refiner input. Schedule call ledgers must
equal the requested stage count.

For each candidate transition, the simulator-opened analysis fields are:
agent-to-block proximity/contact proxy, early (1-2) versus late (3-5) phase,
action L2/max magnitude, block displacement, base/refined latent error to the
encoded simulator frame, base cost dispersion, refiner residual/update norm,
and depth disagreement. Simulator state/contact fields are analysis labels
only and never enter model scoring.

The preregistered mechanism mappings are:

- compounding bias: isolated one-step refinement improves latent error or
  ranking while all-step refinement reverses the sign;
- representation/metric misalignment: latent MSE improves at a matched step
  while simulator cost concordance/ranking worsens;
- contact selectivity: bank-level benefit is positive in proximity/contact
  strata and negative in free motion, with the sign repeated across seeds;
- overshoot: depth 1 or 2 improves ranking and depth 4 removes/reverses it;
- candidate-distribution mismatch: a result differs between the final-CEM bank
  and a mechanically sealed 64-candidate iteration-zero proposal prefix scored
  as a secondary bank type. The proposal prefix is simulator-labelled only for
  the two pilot starts and sealed confirmation starts 1-4; it cannot replace
  the primary final bank.

## Seed stability, opportunity, and diagnostic probe

Depth effects are reported by start and bank seed. A random-intercept
method-of-moments summary partitions between-start, between-bank-within-start,
and residual/depth contrast variation when estimable; it is descriptive at
eight starts. Sign reproduction requires the same nonzero base-versus-depth
direction in both seeds. Fixed-bank instability is compared qualitatively with
Task D's closed-loop seed instability; it is not attributed wholly to CEM
updates without evidence.

An optimistic post-outcome oracle chooses depth 0 versus each positive depth
per bank using simulator outcome at identical population/work opportunity and
is labelled non-deployable. Identity control leaves depth labels unchanged;
histogram control permutes the oracle's exact depth allocation across banks
within seed using 9,999 draws and seed 2026072793.

At most one tiny diagnostic probe may predict whether a positive depth beats
base by at least 0.01 normalized simulator-cost units from inference-available
causal signals (update norm, depth disagreement, base planner dispersion,
action magnitude). It uses leave-one-source-episode-out cross-validation,
fixed L2 logistic regression C=1, no architecture search, and reports balanced
accuracy/AUC against 9,999 label permutations (seed 2026072794). Simulator-only
contact is reported as an unattainable upper-bound feature, never deployed.

## Frozen decision mapping

Planner-aware Task F training is recommended only if all hold: latent accuracy
and simulator ranking demonstrably diverge; depth 0 has nontrivial oracle
ranking headroom (mean top-choice regret at least 2% of the within-bank
simulator-cost range); at least one positive depth changes ranking materially;
the targetable error sign is stable across bank seeds or a causal regime is
reproducible; population sampling alone does not explain the sign; and a
bounded fixed-bank dataset is sufficient.

Selective one-axis refinement is preferred only if a positive effect at least
0.01 normalized simulator-cost units has the same sign in both seeds and the
inference-available probe has cross-validated balanced accuracy at least 0.65
with permutation p<0.05. If refinement has no reproducible ranking value but
depth 0 has headroom, continue population-only diagnostics. If depth 0 has
little headroom or no targetable mechanism remains, stop/refactor refinement.

If Task F is selected, the smallest authorized next recipe (not executed here)
uses common final base-CEM candidate banks; simulator cumulative-cost pairwise
ranking targets; a five-step rollout loss; a hinge constraint preventing base
top-choice-regret degradation; and an optional confidence-gated residual that
may output exactly zero. Splits are by source episode and whole bank
(fit/selection/evaluation), and gates require fixed-bank concordance and regret
improvement with stable seed sign plus a small closed-loop safety confirmation
before any grid. No weighted kitchen-sink objective is authorized.
