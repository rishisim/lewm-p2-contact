# LeWM Adaptive Compute V3: Frozen Joint Refinement/Halting Preregistration

Status: **frozen before creating the V3 split, extracting any V3 episode, or
inspecting any V3 train/calibration/test prediction loss**. The accompanying
exact configuration is `full_config.json`; both files are hashed in
`preregistration.json` immediately after creation. V1/V2 results are immutable
development evidence only. No V3 final-test outcome may change this plan.

Pre-data clerical amendment, 2026-07-13T19:59:45-07:00: the first smoke
invocation aborted in the preregistration validator before split generation or
HDF5 access because this plan transcribed the source size as 101,640,570,983
bytes. `stat` reports 101,942,558,720 bytes. The assertion below and validator
are corrected to that value and the amended plan is re-hashed. No split,
outcome, architecture, training, metric, baseline, or decision choice changed.

## Question and primary claim

Can a frozen released LeWM plus a jointly trained shared iterative residual
refiner and causal local stop/continue gate improve the prediction-error versus
compute frontier? The primary claim is adaptive latent computation, not contact
classification. The gate is local: after each refiner call it uses only that
transition's history, action history, current prediction, last update, and call
index. It never ranks or optimizes across test transitions.

## Frozen provenance, exclusions, and splits

- Cube HDF5:
  `/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5`,
  10,000 episodes, 2,010,000 rows, 101,942,558,720 bytes. A 95-GB whole-file
  hash is intentionally not required; the immutable layout/size are asserted
  and every selected episode is bound through the extracted latent-cache hash,
  row keys, action-normalization constants, and split manifest.
- Released config SHA-256:
  `4d446944fe28922cc2c5763f43d4ef9132a457bd89e9a0ce5dbceac183994999`.
- Released weights SHA-256:
  `2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89`.
- Initialize the V3 shared refiner from the V1-selected checkpoint
  `388a82fc30c96921083bfa4f2578cb296e6c3544510953d17578cf766cdf10f1`.
  Unlike V2, the refiner is then trained jointly with the V3 gate.
- Exclude diagnostic episodes 0--29 plus every episode in the V1/V2 full and
  smoke manifests. The four manifest hashes are `3cc9bd4d...ed544d`,
  `f5a4da6c...59b65940`, `ce34fa66...c1c0`, and
  `ad1d2e1...5ffb3`; the mechanically audited union contains 1,242 episodes
  and has sorted-int32-LE SHA-256
  `ac71dbe795efdd0217983cd615a55ed81244303ae1243d559932d2a5def62915`.
- Reserve unused episodes 42--47 for smoke only: train 42--45, calibration 46,
  test 47. Smoke artifacts never enter confirmatory fitting or evaluation.
- Confirmatory selection: from 0--9999 excluding the 1,242 prior episodes and
  V3 smoke, NumPy `default_rng(260913)` draws 600 without replacement in
  returned order: 420 train, 90 calibration/model-selection, 90 final test.
  Each episode yields 38 transitions. All partitions are episode-disjoint.
- Original LeWM-pretraining membership is unavailable. V3 is fresh to all
  listed diagnostics/refiners but is not asserted unseen in LeWM pretraining.

The final-test split may be encoded with the frozen backbone as one mechanical
extraction job, but test targets are not exposed to model training, seed or
threshold selection. At the final boundary, all three frozen seed checkpoints,
calibrated thresholds, and baseline rules are evaluated in one test pass.

## Architecture and causal-feature audit

The released encoder, action encoder, predictor, projector, and prediction
projector are strict-loaded, evaluation-only, and `requires_grad=False`.

The V3 refiner is the V1 335,360-parameter shared residual MLP, initialized from
V1: flattened 3x192 latent history, flattened 3x25 blocked actions, current
192-D prediction, and a learned 16-D call embedding feed two 256-unit GELU
layers and a 192-D residual. The same weights are called up to depth 4:
`z[d+1] = z[d] + R(history, actions, z[d], d)`.

One shared local gate is evaluated after depths 1, 2, and 3. Its inputs are
flattened latent/action history, current prediction, last residual update,
norm/convergence/alignment summaries, and an 8-D call embedding. An MLP
`D -> 128 -> 64 -> 1` predicts the next-call marginal raw-MSE gain. The feature
builder's API accepts no target, future, loss, benefit, label, contact, regime,
phase, episode statistic, split statistic, or other transition. Tests mutate
targets/labels and unrelated batch rows and require identical features and
decisions. Gate normalization is train-only.

## Joint training, compute pricing, and selection

Seeds are 260913, 260914, and 260915. Use AdamW (learning rate 2e-4, weight
decay 1e-4), batch 256, maximum 80 epochs, clip 1.0, patience 15, minimum
calibration improvement 1e-7. Each batch draws a maximum unroll uniformly from
1--4 using the seed-specific stochastic-depth stream. Loss components are:

1. equal-weight raw latent MSE at every realized exit (deep supervision);
2. weight 1.0 monotonic hinge `relu(L[d+1]-L[d])` to penalize degrading calls;
3. weight 1e-4 mean squared residual-update magnitude;
4. weight 0.25 Smooth-L1 regression of standardized one-call marginal gains;
5. weight 0.10 price-conditioned BCE for whether gain exceeds each call price
   in `{0, 2e-5, 5e-5, 1e-4}` raw MSE;
6. weight 0.05 within-batch pairwise logistic ranking.

Marginal-gain labels are training/calibration-only and detached from refiner
gradients. Model selection minimizes the same calibration objective, restores
the best epoch, and never sees test. The calibration-selected primary seed is
the lowest calibration objective (ties use listed seed order). All seeds are
evaluated once for stability; the selected seed is the deployable primary.

## Local policies and operating points

Predeclared target mean depths are 1.25, 1.5, 2.0, and 2.5; 2.0 is primary.
Every sample always executes depth 1. After depths 1--3, continue iff the local
gate score exceeds one scalar threshold for that operating point. Calibration
chooses the threshold from 2,001 deterministic score midpoints minimizing
absolute deviation from target mean depth, then calibration raw MSE, then the
higher threshold. Test thresholds are never retroactively adjusted. Report
calibration-to-test compute transfer and realized test calls.

No global dynamic program, batch rank, percentile, quota, or episode-global
statistic is allowed for adaptive inference. The diagnostic oracle may optimize
test losses only after predictions/allocations freeze and is never a policy.

## Baselines and exact compute accounting

Report depth 0 and V3 fixed depths 1, 2, and 4. At each adaptive realized test
call count, compare:

- the strongest transition-independent mixture over depths `{1,2,4}`, chosen
  solely from calibration fixed-depth mean losses and assigned on test with a
  frozen random order, with its total calls exactly equal to adaptive calls;
- a random local allocation with exactly the adaptive realized depth histogram;
- an independent histogram-preserving permutation control;
- a target-informed oracle over depths 1--4 at the same total calls,
  descriptive upper bound only.

If integer feasibility prevents a `{1,2,4}` mixture from matching a call count,
the closest feasible count within one call is used and marked; primary success
requires exact equality. Dense evaluation calls used to expose exits are
diagnostic accounting and not policy compute. Runtime adaptive execution must
process exactly `sum(depth)` rows. Refiner FLOPs count dense linear multiply-add
as two operations. Gate FLOPs use the same convention. Total latent-inference
FLOPs add a hook-derived released LeWM `predict` estimate (including linear and
attention matrix products) to refiner plus gate FLOPs. Median synchronized MPS
latency over five repeats includes base prediction, refiner, and gate; batch
size and device are fixed across policies.

## Metrics, intervals, robustness, and frontier

Primary error is per-transition raw latent MSE. Calibration-fit whitening
(eigenvalue floor `1e-6 * lambda_max`) supplies whitened MSE robustness. Report
fixed/adaptive loss, paired benefits, degradation rates by call, allocation,
calibration transfer, seed stability, and post-hoc physical regimes. Multi-step
rollout is reported only if the existing frozen evaluator supports V3 exits
without refitting or ambiguous action alignment; otherwise record it as
unsupported rather than inventing a proxy.

All paired intervals are episode-clustered 95% percentile intervals with 2,000
replicates and seed 260919. The primary selected seed determines the verdict.
Seed aggregation reports the per-transition mean comparison across all three
seeds with a clustered CI, plus the number of seeds with the same benefit sign.
A point is nondominated if no preregistered fixed or transition-independent
baseline has both no more calls and no greater raw error, with one strict.

Whitened robustness is `strong` if the primary adaptive comparisons versus the
matched baseline and fixed depth 1 both have positive whitened CI lower bounds;
`directional` if both means are positive; otherwise `failed`. Seed robustness
is `strong` if all three means favor adaptive, `majority` if two do, otherwise
`failed`. These strength labels do not silently replace the raw primary rule.

## Frozen decision taxonomy

Validity is evaluated first. Any split/exclusion leak, target/future/label or
cross-transition gate leak, test-derived fitting, checkpoint/hash/strict-load
failure, base mutation/gradient, nonfinite primary metric, nondeterministic
replay failure, or primary exact-compute mismatch gives
`validity_or_budget_failed`.

Otherwise at the primary 2.0 target:

1. `adaptive_refiner_headroom_failed` if fixed depth 1 does not significantly
   improve on depth 0, or the matched-call descriptive oracle does not have a
   positive raw clustered-CI advantage over the strongest baseline.
2. `adaptive_pareto_gate_passed` only if adaptive has positive raw-MSE mean
   benefits with CI lower bounds above zero versus both the strongest matched
   transition-independent baseline and fixed depth 1; at least one adaptive
   operating point is nondominated; at least two of three seed-specific primary
   mean comparisons favor adaptive for both required baselines; and all audits
   pass.
3. `local_halting_gate_failed` otherwise.

## Post-hoc physical interpretation and artifacts

Only after all test allocations freeze, attach contact, impact,
transport/free, and static regimes using calibration-fit thresholds. Report
allocation and gains by regime without a contact-aware or causal claim.

Required artifacts are source/config hashes, split/cache manifests, all seed
checkpoints and histories, frozen thresholds and pre-outcome allocations,
per-transition outputs, raw/whitened metrics and CIs, strict `decision.json`,
`REPORT.md`, `README.md`, and publication-quality Pareto, allocation,
calibration, seed-stability, and regime figures. Abort only for a genuine
unresolvable resource failure; smoke failures are fixed before confirmatory
execution. No V1/V2 artifact is modified, and no commit or push is authorized.
