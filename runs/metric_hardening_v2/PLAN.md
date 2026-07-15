# Metric Hardening V2 Preregistered Plan

## Stage 0 Scope And Stop Rule

This is a recompute-only hardening pass. It will not train models, encode pixels,
call checkpoints, collect new data, or modify any existing run directory. The
only write target for the hardening pass is `runs/metric_hardening_v2/`.

Stage 0 ends with this file. No metric recomputation, tests, plots, or reports
will be run until this plan is reviewed.

## Existing Inputs

Cube inputs:

- `runs/lewm_transfer/cube/relabel_motion/step_records_relabel_motion.csv`:
  row labels, split, `row_id`, `episode_ordinal`, `episode_id`,
  `model_step`, `raw_step`, `transition_block`, and regimes
  `interaction`, `transport_free`, `static`.
- `runs/lewm_transfer/cube/residuals.npz`: row-aligned `prev_latents`,
  `target_latents`, `pred_latents`, `residuals`, and `row_id`.
- `runs/lewm_transfer/cube/cache/cube_first30_pixel_subset.lance`: cached raw
  Cube subset for per-step action and kinematic fields. This may be read only;
  it must not be regenerated.
- `/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5`:
  provenance cross-check for action and kinematic joins only, not a source for
  new model outputs.

Fetch inputs:

- `le-wm/diagnostics/fetch_contact_compute/data/swm__FetchPushDense-v3/records.csv.gz`
  and `.../swm__FetchSlideDense-v3/records.csv.gz`: existing state, next-state,
  action, qpos/qvel, contact, and regime records.
- `le-wm/diagnostics/fetch_contact_compute/runs/fullstate_ablation/split_keys.json`:
  frozen train/validation split.
- `le-wm/diagnostics/fetch_contact_compute/runs/ensemble_bv/member_predictions_val.npz`:
  saved validation ensemble predictions and targets.
- `le-wm/diagnostics/fetch_contact_compute/runs/ensemble_bv/val_decomposition.csv.gz`:
  validation metadata aligned to the saved predictions.

If any required input is missing, ambiguous, or not one-to-one joinable, the
affected R-item is `void_provenance`, not a negative result.

## Shared Statistical Rules

- Evaluation rows: Cube uses `split == "heldout"`; Fetch uses validation rows
  from `split_keys.json`.
- Calibration rows: Cube uses `split == "calibration"`; Fetch uses train split
  records. All standardizers, whitening transforms, action scales, radius bins,
  and density bins are fit on calibration/train only unless explicitly stated.
- Bootstrap: episode-clustered percentile bootstrap with `n_bootstrap = 1000`
  and fixed seed `260727`. Cube clusters by `episode_ordinal`; Fetch clusters
  by `(env_id, episode_id)`.
- Reported summaries for every regime and contrast: `n`, eligible episodes,
  mean, median, 10% trimmed mean, clustered 95% CI, and
  `P(superiority) = P(left > right) + 0.5 P(left == right)`.
- All neighbor sets are cross-episode only. Neighbors from the same episode as
  the anchor are forbidden.
- A row is usable for a neighbor-based metric only if at least `k_min = 5`
  cross-episode neighbors are found. Interaction rows with fewer than 5 usable
  neighbors are flagged in every relevant table.
- Primary Cube contrast: `interaction` vs `transport_free`.
- Primary Fetch contrast: `interaction` vs `free`, where interaction is the
  validation rows used in the prior metric-hardening report and free is
  `primary_regime == "free_motion"` or the prior equivalent free label.
- Pass/fail/void labels are assigned per R-item and per domain. A void item does
  not count as passed.

## Headline Metrics

For anchor row `i` with neighbor set `N_i`:

- Input spread: `S_in(i) = median_j distance(x_j, x_i)`.
- Successor spread: `S_out(i) = median_j distance(y_j, y_i)`.
- Residual spread: `S_res(i) = median_j distance(r_j, r_i)`.
- Local expansion: `E(i) = S_out(i) / max(S_in(i), 1e-8)`.
- Model sensitivity: `M(i) = S_res(i) / max(S_in(i), 1e-8)`.
- Neighbor radius: median input distance to the usable neighbors.

Cube successors are existing `target_latents`; Cube residuals are existing
`pred_latents - target_latents`. Fetch successors are saved target/next-state
vectors; Fetch residuals are ensemble-mean prediction minus target. Finite
ensemble variance is descriptive only, not the primary sensitivity metric.

## R1 - Action-Matched Neighbors

Purpose: rule out successor divergence caused by action variability rather than
contact dynamics branching.

Primary neighbor metric:

- Use a block-balanced joint distance, not raw concatenated Euclidean distance.
- Standardize state features and action features with calibration/train means
  and standard deviations.
- Compute `d_state` as RMS Euclidean distance over standardized state features.
- Compute `d_action` as RMS Euclidean distance over standardized action
  features.
- Joint distance: `sqrt(d_state^2 + d_action^2)`.

Cube state/action construction:

- State input is `prev_latents` under the same scale family being evaluated.
- Rejoin actions from the cached Lance subset by `episode_id` and
  `transition_block`.
- For each row, flatten the 5 raw Cube `action` rows corresponding to the
  original prediction action block: raw indices
  `transition_block * 5 : transition_block * 5 + 5` within the episode.
- Expected Cube action dimension is 25. Any row with a missing or duplicate
  action block makes Cube R1 `void_provenance`.
- Cross-check a deterministic sample of rows against the source H5 action array
  and the existing `raw_step == model_step * 5` convention.

Fetch state/action construction:

- State input is the prior normalized full-state prediction input rebuilt from
  existing records without same-row post-step leakage.
- Action input is `action_0` through `action_3` from the same validation row.
- Train-split means/stds are used for both state and action blocks.

Report:

- Per domain, per regime: `n_anchors`, `n_usable`, `n_below_5_neighbors`,
  median/trimmed mean neighbor radius, local expansion, and model sensitivity.
- Per domain, primary contrast: expansion and sensitivity deltas with clustered
  CI and P(superiority).
- Action-distance diagnostics: median action distance to chosen neighbors and
  action-distance contrast by regime.

Pass:

- Cube R1 passes only if both action-matched expansion and action-matched model
  sensitivity have positive `interaction - transport_free` trimmed-mean deltas
  with clustered 95% CI lower bounds greater than zero.
- Fetch R1 passes only if both action-matched expansion and action-matched model
  sensitivity have positive `interaction - free` trimmed-mean deltas with
  clustered 95% CI lower bounds greater than zero.
- If interaction rows drop below 5 usable neighbors for any anchor, the row is
  excluded and counted. If fewer than 30 interaction anchors or fewer than 5
  interaction episodes remain, the domain R1 result is `void_insufficient_coverage`.

## R2 - Cube Whitening Robustness

Purpose: check whether the Cube sensitivity result depends on the prior full
whitening transform whose covariance condition number was approximately
`5.2e5`.

Transforms fit on Cube calibration target latents:

- Full whitening diagnostic: reproduce the prior transform and condition number.
- Shrinkage family: eigenvalue-floor whitening with floors at `1e-3 * lambda_max`
  and `1e-2 * lambda_max`.
- PCA family: PCA-truncated whitening retaining 90% and 99% calibration variance.
  Dropped components are not used in distances or MSEs.

Report:

- Calibration covariance eigenvalue summaries and effective retained dimensions.
- For each transform, Cube interaction-vs-transport_free local expansion and
  model sensitivity contrasts.
- Density diagnostics by transform.

Pass:

- R2 passes if Cube local expansion has a positive `interaction -
  transport_free` trimmed-mean delta with clustered 95% CI lower bound greater
  than zero under both settings of at least one transform family:
  both eigenvalue floors, or both PCA truncation levels.
- Model sensitivity is reported under the same transforms. If sensitivity passes
  but expansion does not, R2 is `fail_expansion_only` rather than passed.

## R3 - Cube Aliasing Discriminator

Purpose: distinguish intrinsic branching from encoder aliasing in the neighbor
definition.

Kinematic neighbor input:

- Rejoin from the cached Cube subset at each row's `raw_step`.
- Use current effector position plus block pose: `proprio_effector_pos`,
  `privileged_block_0_pos`, and normalized `privileged_block_0_quat`. If the
  quaternion field is unavailable, use `sin` and `cos` of
  `privileged_block_0_yaw` and mark the report accordingly.
- Standardize kinematic features on calibration rows only.

Successor/residual space:

- Measure successor spread and residual spread in latent space using the
  preregistered target-std scale fit on calibration target latents.
- Report the kinematic-neighbor result alongside the latent-neighbor
  action-matched result from R1.

Pass:

- R3 passes if kinematic-neighbor Cube local expansion has a positive
  `interaction - transport_free` trimmed-mean delta with clustered 95% CI lower
  bound greater than zero.
- Model sensitivity under kinematic neighbors is reported as a secondary
  discriminator. If expansion fails, the aliasing caveat remains open even if
  sensitivity is positive.

## R4 - k-Sweep

Purpose: check whether the headline result depends on the single previous value
`k = 10`.

Values:

- `k in {5, 10, 20, 40}` with `k_min = min(5, k)`.
- Use the R1 action-matched joint metric for both Cube and Fetch.
- Use the same successor/residual scaling as the primary R1 run.

Report:

- One table per domain with expansion and sensitivity deltas, CI bounds,
  P(superiority), usable anchor counts, and neighbor-radius summaries for each
  `k`.
- Plot `delta_trimmed_mean +/- clustered 95% CI` vs `k` for expansion and
  sensitivity under each domain.

Pass:

- A metric passes the k-sweep in a domain if all four point deltas are positive
  and at least three of four clustered 95% CIs exclude zero on the positive
  side.
- Cube R4 passes only if both expansion and sensitivity pass this rule for
  `interaction - transport_free`.
- Fetch R4 passes only if both expansion and sensitivity pass this rule for
  `interaction - free`.
- If fewer than three `k` values have sufficient coverage, the domain R4 result
  is `void_insufficient_coverage`.

## R5 - Fetch Density Matching

Purpose: check whether Fetch support is an artifact of sparser interaction
neighborhoods. The prior report found normalized interaction radius about
`+3.66` above free.

Procedure:

- Use the Fetch R1 action-matched metric at `k = 10`.
- Fit neighbor-radius decile edges on train-split anchors using the same input
  metric and cross-episode rule.
- Assign validation anchors to these preregistered radius bins.
- Within each radius bin, match each interaction validation anchor to the
  nearest free validation anchor by neighbor radius, from a different episode,
  without replacement when possible.
- If a bin lacks free anchors, interaction anchors in that bin are unmatched and
  reported.
- Compute matched-subset expansion and sensitivity contrasts and paired
  interaction-minus-free deltas. Bootstrap clusters by the interaction anchor's
  episode key, with matched free episodes retained in the resample.

Report:

- Bin-level counts, matched/unmatched interaction anchors, radius gaps, and
  free reuse counts if replacement is unavoidable.
- Matched expansion and sensitivity deltas with clustered CIs and
  P(superiority).
- A clear statement whether Fetch remains convergent support after matching.

Pass:

- R5 passes only if both Fetch expansion and Fetch model sensitivity remain
  positive after radius matching, with clustered 95% CI lower bounds greater
  than zero.
- If less than 70% of interaction anchors can be matched, or fewer than 30
  interaction anchors / 5 interaction episodes remain, R5 is
  `void_insufficient_density_overlap`.
- If R5 fails or is void, Fetch is marked unsupported and excluded from any
  positive support claim, regardless of R1 or R4.

## Final Verdict Rules

The report must include a machine-readable final verdict block with these
fields:

```text
cube_main_status: confirmed | mixed | unsupported | inconclusive
fetch_support_status: supported | unsupported | inconclusive
holes_closed:
  action_conditioning: closed | open | void
  cube_whitening_conditioning: closed | open | void
  single_k_dependence: closed | open | void
  fetch_density_mismatch: closed | open | void
extra_checks:
  cube_aliasing_discriminator: passed | failed | void
```

Verdict assignment:

- `cube_main_status = confirmed` only if Cube R1, R2, R3, and Cube R4 pass.
- `cube_main_status = mixed` if at least one Cube R-item passes and at least one
  non-void Cube R-item fails.
- `cube_main_status = unsupported` if all non-void Cube R-items fail.
- `cube_main_status = inconclusive` if all Cube R-items are void.
- `fetch_support_status = supported` only if Fetch R1, Fetch R4, and R5 pass.
- `fetch_support_status = unsupported` if R5 fails or any non-void Fetch
  hardening item fails.
- `fetch_support_status = inconclusive` if all Fetch hardening items are void.

The four requested robustness holes are closed as follows:

- `action_conditioning` is closed only if Cube R1 passes. Fetch R1 is reported
  separately because Fetch cannot rescue the main Cube claim.
- `cube_whitening_conditioning` is closed only if R2 passes.
- `single_k_dependence` is closed for the main claim only if Cube R4 passes.
  Fetch k-sweep status is required only for `fetch_support_status`.
- `fetch_density_mismatch` is closed only if R5 passes.

## Unit Tests Before Recompute

Add and pass focused unit tests before generating `REPORT.md`:

- Cube action join correctness: synthetic rows with known `episode_id`,
  `transition_block`, `raw_step`, and raw action arrays must rejoin exactly the
  flattened 5-step action block used by `run_cube_transfer.py`; missing,
  duplicate, and off-by-one joins must raise.
- Real Cube action join smoke test: a small deterministic sample from the cached
  subset must match the source H5 action block and produce 25 action dimensions.
- Shrinkage whitening math: a two-dimensional covariance with a tiny eigenvalue
  must floor eigenvalues exactly at the requested fraction of `lambda_max`, keep
  the transform finite, and produce the expected manual scaled squared distance.
- PCA truncation math: retained dimension must be the smallest number of sorted
  components whose cumulative variance reaches the requested threshold.
- Cross-episode neighbor constraint: no neighbor set may include the anchor's
  episode key.
- Local expansion synthetic contraction and branching examples must move in the
  expected directions.
- `P(superiority)` must stay in `[0, 1]` and count ties as half.
- Clustered bootstrap smoke test must return finite CI bounds on a deterministic
  toy table.

## Planned Outputs After Review

- `runs/metric_hardening_v2/PLAN.md` (this file).
- `runs/metric_hardening_v2/test_metric_hardening_v2.py`.
- `runs/metric_hardening_v2/metric_hardening_v2_utils.py`.
- `runs/metric_hardening_v2/analyze_metric_hardening_v2.py`.
- `runs/metric_hardening_v2/cube/r1_action_matched_summary.csv`.
- `runs/metric_hardening_v2/cube/r2_whitening_robustness.csv`.
- `runs/metric_hardening_v2/cube/r3_aliasing_discriminator.csv`.
- `runs/metric_hardening_v2/cube/r4_k_sweep.csv`.
- `runs/metric_hardening_v2/fetch/r1_action_matched_summary.csv`.
- `runs/metric_hardening_v2/fetch/r4_k_sweep.csv`.
- `runs/metric_hardening_v2/fetch/r5_density_matched.csv`.
- `runs/metric_hardening_v2/figures/k_sweep_delta_ci.png`.
- `runs/metric_hardening_v2/REPORT.md`.
- `runs/metric_hardening_v2/decision.json`.

Stop here for review.
