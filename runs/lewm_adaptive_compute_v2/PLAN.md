# LeWM Adaptive Compute V2: Frozen Sequential-Gate Preregistration

Status: **frozen at 2026-07-13T19:05:19-07:00 (America/Phoenix)**, before
generating the V2 split, extracting or encoding any V2 episode, running the
frozen refiner on any V2 example, or examining any V2 outcome. V1 results and
the explicitly identified sequential-oracle follow-up are development evidence
only. No V2 test outcome may change this plan. Any implementation correction
made before test outcomes are viewed will be timestamped as a pre-test
amendment here; changes after test inspection are exploratory and cannot alter
the primary decision.

Pre-test amendment, 2026-07-13T19:16:54-07:00 (America/Phoenix): a read-only
implementation audit found that the inherited V1 extractor would recompute the
released action normalizer across the whole Cube HDF5 after the V2 split was
chosen, which would unnecessarily derive preprocessing constants from V2 test
episodes. Before generating any V2 split or accessing any V2 episode, V2 now
freezes the exact released/V1 constants already recorded in the V1 cache and
manifest: mean `[0.010884696617722511, -0.003141433000564575,
0.002646582666784525, 0.00042392866453155875, 0.1592525690793991]`, standard
deviation `[0.28941991925239563, 0.39371708035469055, 0.6431366801261902,
0.3928017318248749, 0.25030744075775146]`, originally computed from 2,000,000
finite released-training action rows with unbiased standard deviation. The V2
extractor uses these constants directly and records their provenance; it does
not scan selected or unselected episode actions to fit preprocessing. No model,
feature, gate, split, comparison, metric, or decision rule changed.

Pre-test statistical clarification, 2026-07-13T19:23:11-07:00
(America/Phoenix): a read-only review found that bootstrapping per-row losses
under one target-optimized realized oracle histogram would condition on that
oracle selection and omit optimizer uncertainty. Before any V2 split was
generated, oracle comparison intervals were therefore specified as procedural
episode-cluster bootstraps: in every replicate, sample test episodes with
replacement, concatenate all transitions from each sampled episode, and rerun
the sequential exact-budget raw-loss oracle at exactly `2N` calls on that
replicate. Compute raw oracle advantage/regret from that re-optimized allocation;
for whitened sensitivity, gather whitened losses under the same raw-selected
replicate oracle. Target-free adaptive, random, and permutation allocations are
frozen and merely resampled. Point estimates remain the full-test exact oracle.
This clarification changes no model, feature, gate, split, comparison, bootstrap
seed/count, threshold, or decision rule; it makes the diagnostic oracle CI
reflect the procedure named in the original plan.

## Confirmatory question and scope

Does information causally available after one shared-refiner call predict which
Cube transitions benefit from further refinement well enough to beat uniform
depth 2 at exactly the same total block-call budget?

Every transition first pays for depth 1. A frozen gate then chooses final depth
from `{1, 2, 4}` under exactly `2N` total calls for `N` transitions. The primary
claim is transition-selective latent computation. Physical contact or grounding
is not a primary claim: V1 did not show contact-specific compute benefit.

Only this LeWM Cube experiment is in scope. No Fetch or sensor-modality work is
part of V2.

## Development evidence, frozen model, and provenance

V1 found depth-0 raw MSE `0.00413054664`, depth-1 MSE `0.00302288660`
(26.8163% lower), and depth-2 MSE `0.00303482824`. Its target-informed
unrestricted oracle beat uniform depth 2 by 2.90747%, while its 12-feature ridge
gate allocated depth 1 to all transitions. An explicitly exploratory
sequentially feasible V1 reanalysis, constrained to first execute depth 1, used
histogram `{1:1764, 2:774, 4:882}` at 6,840 calls and beat uniform depth 2 by
`6.822024e-05` raw MSE (2.24791%, clustered 95% CI
`[6.350197e-05, 7.308338e-05]`). Its whitened sensitivity was negative. These
values motivate V2 but are not V2 evidence.

V2 freezes without retraining or selecting:

- Cube source:
  `/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5`
  (10,000 episodes, 201 rows each).
- Released base config: `runs/lewm_transfer/cube/cache/model/config.json`,
  SHA-256 `4d446944fe28922cc2c5763f43d4ef9132a457bd89e9a0ce5dbceac183994999`.
- Released base weights: `runs/lewm_transfer/cube/cache/model/weights.pt`,
  SHA-256 `2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89`.
- V1-selected refiner: `runs/lewm_adaptive_compute_v1/checkpoints/refiner_seed_260713.pt`,
  SHA-256 `388a82fc30c96921083bfa4f2578cb296e6c3544510953d17578cf766cdf10f1`.
  This was selected in V1 on V1 calibration only (best epoch 10; objective
  `0.003060628029610417`).

The expected base contract is latent dimension 192, history 3, frame skip 5,
raw action dimension 5, blocked-action dimension 25, and image size 224. The
base has 18,034,628 frozen parameters. The refiner is the V1 335,360-parameter
shared recurrent residual block: input dimension 859, two 256-unit GELU hidden
layers, 192-dimensional output, and a 16-dimensional iteration embedding.

The implementation must strictly load exact key sets, verify file hashes before
and after the experiment, keep both models in evaluation mode with all
parameters `requires_grad=False`, verify no gradients, and hash all state
tensors before and after. The copied V2 refiner checkpoint must be byte-identical
to the V1 source. Repeated depth outputs on a fixed audit batch must be bitwise
identical when execution order and batches are identical. Staged depth-1 then
continuation outputs must match dense prefix outputs bitwise on identical rows;
mixed-row pruned execution must pass a documented numerical tolerance of
`2e-5` and exact per-sample call accounting.

The original released LeWM episode-level pretraining manifest is unavailable.
V2 episodes are new to these diagnostics and refiners, but pretraining
membership remains unknown; V2 is not a clean base-model generalization test.

## Episode exclusions and deterministic splits

Prior-manifest hashes are frozen as:

- V1 full manifest `runs/lewm_adaptive_compute_v1/cache/split_manifest.json`:
  `3cc9bd4d7df229b0a4111a49933e98c5b72e916b5b5f98d7d5e023fc30ed544d`.
- V1 smoke manifest
  `runs/lewm_adaptive_compute_v1/smoke_run/cache/split_manifest.json`:
  `f5a4da6c0f37d6222e83085327e30a1b70f0ace6f96e9c97d37c1067d7b65940`.

The prior exclusion union is all diagnostic episodes `0..29`, V1 smoke
episodes `30..35`, and every full-run V1 train/calibration/test episode parsed
from its manifest. It must contain exactly 636 unique ordinals. No other prior
Cube episode manifest was found; Fetch manifests are a different dataset.

Episodes `36..41` are reserved now as the disjoint engineering smoke run:
train `36..39`, calibration `40`, test `41`. The confirmatory eligible set is
`0..9999` minus the 636 prior exclusions minus these six smoke episodes. With
NumPy `default_rng(260813)`, draw 600 ordinals without replacement and preserve
the returned order: first 420 sequential-gate train, next 90 calibration, last
90 untouched final test.

Each episode yields 41 strided embeddings and 38 transitions: 15,960 train,
3,420 calibration, and 3,420 test. Assert exact split counts, unique row keys,
temporal order, source episode indices, pairwise episode disjointness, no
selected/excluded overlap, no smoke/full overlap, and equality between planned
and extracted episode sets. Store the sorted exclusion list, all split lists,
source-manifest paths and hashes, and selection seed. Do not access physical
labels during gate construction or policy freezing.

## Frozen prediction and causal post-call features

For each transition:

```text
z0 = released LeWM prediction
z1 = z0 + R(history, action_history, z0, iteration=0)
z2 = z1 + R(history, action_history, z1, iteration=1)
z3 = z2 + R(history, action_history, z2, iteration=2)
z4 = z3 + R(history, action_history, z3, iteration=3)
```

The first call is executed for every example and must account for exactly one
processed row per example. The gate feature vector is fixed to the following
target-free quantities, in this order:

1. flattened three-token latent history (`3*192` values);
2. flattened three blocked action-history vectors (`3*25` values);
3. full `z0` (`192` values), full `z1` (`192` values), and full first update
   `delta1 = z1-z0` (`192` values);
4. history token norms (3), successive history-difference norms (2), action
   block norms (3), successive action-difference norms (2);
5. norms of `z0`, `z1`, `delta1`; `||delta1||/(||z0||+1e-6)`;
6. cosine alignments (denominators floored at `1e-6`) of `(z0,z1)`,
   `(z0,delta1)`, `(z1,delta1)`, and `(delta1,z0-history[-1])`;
7. distances from `z0` and `z1` to the last history latent and the relative
   reduction `(dist0-dist1)/(dist0+1e-6)`.

No hidden activation is exposed in the primary V2 gate. This avoids expanding
the frozen representation contract. The builder accepts only history,
action-history, z0, and z1; it has no target, future state, loss, physical
kinematics, contact/regime, phase, or split-outcome argument. Feature names are
whitelisted and causality tests must show that target/label mutation cannot
change features. Feature fitting uses train only. Test features are computed
only after the gate seed is frozen and are used to freeze allocations before
test targets are scored.

## One preregistered lightweight gate family

Train labels are marginal terminal benefits, not absolute depth-0 gains:

```text
b12 = L1 - L2
b14 = L1 - L4
```

where `Lk` is raw per-dimension latent MSE at final depth `k`. Positive values
favor continuation. Inputs are standardized by train-only mean and standard
deviation with a `1e-6` floor. Each output target is centered by its train-only
median and scaled by `max(IQR/1.349, 1e-6)`.

The sole architecture is an MLP `D -> 128 -> 64 -> 2`, GELU after each hidden
linear layer, dropout `0.05` after the first GELU, and linear outputs. No
architecture or feature subset is selected. Gate seeds are `260813`, `260814`,
and `260815`. Use AdamW, learning rate `3e-4`, weight decay `1e-4`, batch size
256, maximum 150 epochs, gradient-norm clip 1.0, and deterministic seeded epoch
shuffles.

The training objective is mean Smooth-L1 loss on the two standardized benefits
(`beta=0.5`) plus `0.25` times a within-batch pairwise logistic ranking loss.
For each output, shuffled adjacent row pairs with nonzero target difference
contribute `softplus(-sign(y_i-y_j)*(p_i-p_j))`. The two outputs have equal
weight. Early stopping monitors the same calibration objective with patience
20, minimum improvement `1e-5`, and restores the best state. Calibration then
selects the seed having the largest mean Spearman correlation across `b12` and
`b14`; ties within `1e-12` use smaller calibration objective, then the listed
seed order. No test result selects seed, features, architecture, loss, or
hyperparameters.

Report calibration Pearson and Spearman correlations, RMSE/MAE for both
benefits, objective, selected seed/epoch, and actual benefit by five equal-count
predicted-score quantiles separately for `b12` and `b14`. Save every seed
summary but only the calibration-selected checkpoint is primary.

## Sequential exact-budget policy and baselines

Everyone pays depth 1. Define continuation costs `{0,1,3}` corresponding to
final depths `{1,2,4}` and continuation utilities `{0,predicted_b12,
predicted_b14}`. A deterministic multiple-choice dynamic program maximizes
total predicted utility subject to continuation cost exactly `N`, equivalently
final-depth sum exactly `2N`. Ties resolve toward smaller final depth in the
fixed order `{1,2,4}`. The optimizer API accepts predicted utilities and sample
count only—never targets or observed losses.

The untouched test comparisons are frozen as:

- adaptive sequential gate versus uniform depth 2, both exactly `2N` calls;
- adaptive versus fixed depth 1 (adaptive uses `2N`, fixed uses `N`);
- adaptive versus a target-free random exact-budget allocation generated with
  seed `260816` over continuation costs `{0,1,3}` and exactly `N` continuation
  calls;
- adaptive versus a permutation of its exact final-depth histogram using seed
  `260817`;
- adaptive versus the target-informed sequentially feasible exact-budget
  oracle over `{1,2,4}`, diagnostic only.

The random baseline uses no predictions, targets, or labels. The permutation is
created only after the adaptive histogram freezes. The oracle is computed only
after the adaptive test allocation has frozen and test losses are scored; it is
never a gate input or model-selection device.

## Metrics, uncertainty, compute, and diagnostics

The primary loss is raw per-dimension latent MSE. Fit a symmetric whitening
transform using calibration target latents only, with covariance eigenvalue
floor `1e-6 * largest_eigenvalue`, freeze it, and report calibration-whitened
MSE as a sensitivity analysis. Report for every policy: mean raw and whitened
loss, exact depth histogram, total/mean calls, and paired differences versus
the adaptive policy where applicable.

Use episode-clustered 95% percentile bootstrap intervals with 2,000 replicates
and seed `260818`. The primary benefits are `uniform_d2_loss-adaptive_loss` and
`fixed_d1_loss-adaptive_loss`; positive favors adaptive. Also report raw and
whitened paired intervals for random and permutation comparisons and diagnostic
oracle regret `adaptive_loss-oracle_loss`. Whitened results cannot overturn the
raw-MSE primary decision. Report calibration/test score correlations and
five-quantile benefit curves; test diagnostics are descriptive only.

Compute reporting includes exact first and continuation calls, block
invocations, 669,184 preregistered dense-linear FLOPs per frozen-refiner call
(also recomputed from modules), gate parameter count, gate dense-linear FLOPs
`2*(D*128 + 128*64 + 64*2)` per transition, total refiner/gate FLOPs, and median
of five synchronized batched latency repeats for first call, gate feature plus
MLP allocation overhead, adaptive continuation, and uniform depth-2 execution.
Exact block calls, not latency or gate FLOPs, define budget matching.

After adaptive predictions and allocations are frozen, attach contact, impact,
transport-free, and static labels for interpretation only. Any motion threshold
is fit on calibration labels and then applied to test. Report allocation and
gain summaries by physical regime, explicitly post hoc; they cannot define or
change the primary decision.

## Frozen decisions

Validity is checked first. Any split/exclusion leak, target/label feature leak,
test-derived fitting or selection, nonfinite primary artifact, strict-load/hash
failure, base/refiner mutation or gradient, depth-output inconsistency beyond
the declared check, failure to execute the first call for every example, or
test adaptive/uniform call total other than exactly `2N` yields `invalid`.
A genuinely missing local resource after safe alternatives are exhausted yields
`blocked` and no confirmatory claim.

Otherwise compute the diagnostic sequential oracle advantage over uniform
depth 2 on raw test MSE:

1. `no_sequential_headroom` if its clustered 95% CI lower bound is not above
   zero.
2. `sequential_gate_success` if adaptive beats both uniform depth 2 and fixed
   depth 1 with positive raw-MSE mean benefits whose clustered CI lower bounds
   exceed zero, and adaptive and uniform each have exactly `2N` calls.
3. `sequential_headroom_gate_failed` otherwise (oracle headroom exists, but at
   least one required adaptive comparison fails).

No minimum post-hoc regime effect is required. Physical-regime results cannot
alter the verdict.

## Engineering tests and artifacts

Before the full run, run focused unit tests and the reserved six-episode smoke
pipeline. Required tests cover strict base/refiner keys and hashes, frozen
parameters and no gradients, bitwise repeat outputs, staged continuation and
exact call counts, feature shapes/order/finite values/causality, marginal-label
construction, robust/ranking gate loss, seed selection, sequential exact-budget
optimizer edge cases, random/permutation histogram and budget behavior,
whitening and clustered CIs, strict finite JSON (`allow_nan=False`), exclusion
union and split isolation, 38 transitions per episode, and figure PNG decode,
dimensions, and source-data integrity. Smoke artifacts stay under
`smoke_run/` and cannot enter full training or evaluation.

Required full artifacts under this V2 directory are `README.md`, this frozen
`PLAN.md`, `REPORT.md`, strict `decision.json`, full/smoke configs, provenance
and split manifests, a causal first-call feature cache, frozen refiner and gate
checkpoints, machine-readable CSV/JSON/NPZ metrics, and clear figures for policy
loss/comparisons, gate calibration/quantiles, allocation histogram, and post-hoc
regime interpretation. V1 and unrelated files must remain untouched. No commit,
push, download, external post, or auxiliary worktree is authorized.
