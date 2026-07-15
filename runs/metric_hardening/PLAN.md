# Metric Hardening Diagnostic Pilot Plan

## Scope And Stop Rule

This is a relabel/recompute diagnostic using existing Cube latent artifacts and existing Fetch proxy artifacts. It will not run new benchmarks, recollect data, encode pixels, train models, call checkpoints, or modify any existing script or run directory.

All new scripts, tests, reports, and derived small tables will live under `runs/metric_hardening/`, with subdirectories `cube/` and `fetch/`. Large derived arrays, if needed for temporary computation, will be omitted from git and either not written or written only under `runs/metric_hardening/**/cache/` with a local `.gitignore`.

Stage 0 ends after this plan. No metric computation will be run until the plan is reviewed.

## Existing Inputs To Reuse

### Cube

- `runs/lewm_transfer/cube/step_records.csv`: original row-aligned Cube transfer records with `row_id`, episode ids, split, phase position, original contact/support/free labels, raw/whitened model and persistence MSE, and excess columns.
- `runs/lewm_transfer/cube/residuals.npz`: row-aligned arrays `residuals`, `pred_latents`, `target_latents`, `prev_latents`, and `row_id`. These are sufficient for recomputing raw, whitened, and per-dimension target-normalized model/persistence/excess metrics without model calls.
- `runs/lewm_transfer/cube/relabel_motion/step_records_relabel_motion.csv`: existing motion-relabel records with non-circular `interaction`, `transport_free`, and `static` regimes. This is the primary label table for this pilot.
- `runs/lewm_transfer/cube/relabel_motion/metrics/phase_adjusted_contrast.csv` and `matched_position_summary.csv`: prior +0.0145 phase-adjusted and +0.0229 matched-position reference numbers to reproduce under harder metrics.
- `runs/lewm_transfer/lewm_transfer_stats.py`: reuse `trimmed_mean`, `probability_superiority`, `cluster_bootstrap_*`, `make_phase_decile_contrasts`, `make_matched_position_pairs`, `wilcoxon_greater`, `whitening_matrix`, and `apply_whitened_mse` where the signatures fit.

### Fetch

- `le-wm/diagnostics/fetch_contact_compute/data/swm__FetchPushDense-v3/records.csv.gz` and `.../swm__FetchSlideDense-v3/records.csv.gz`: existing proxy state records with state, next-state, action, contact/regime labels, and qpos/qvel. Same-row qpos/qvel are post-step and will not be used as input-time features.
- `le-wm/diagnostics/fetch_contact_compute/runs/fullstate_ablation/split_keys.json`: frozen 800 train / 200 validation episode split.
- `le-wm/diagnostics/fetch_contact_compute/runs/ensemble_bv/member_predictions_val.npz`: saved validation ensemble predictions and targets, shape `(5, 9400, 28)` for predictions and `(9400, 28)` for targets.
- `le-wm/diagnostics/fetch_contact_compute/runs/ensemble_bv/val_decomposition.csv.gz`: validation metadata in the same order as `member_predictions_val.npz`.
- `le-wm/diagnostics/fetch_contact_compute/runs/ensemble_bv_persistence_control/val_persistence_decomposition.csv.gz`: persistence-control recompute, used as a provenance cross-check and to keep the target-std normalization definition aligned.
- `le-wm/diagnostics/fetch_contact_compute/experiment_utils.py`: reuse `load_records`, `build_examples`, `masks_from_train_keys`, `standardize`, `trimmed_mean`, `probability_superiority`, and clustered bootstrap helpers where possible.

## New Files To Add

- `runs/metric_hardening/analyze_metric_hardening.py`: one read-only analysis driver that writes only under `runs/metric_hardening/`.
- `runs/metric_hardening/metric_hardening_utils.py`: small local utilities for neighbor search, per-dim normalization, double matching, table writing, and synthetic tests. If code is copied from existing helpers, comments will name the source file.
- `runs/metric_hardening/test_metric_hardening.py`: unit tests requested below.
- `runs/metric_hardening/REPORT.md`: final concise report with verdict, assumptions, failure modes, and paths.
- `runs/metric_hardening/cube/*.csv` and `runs/metric_hardening/fetch/*.csv`: small metric tables only.

## Shared Statistical Conventions

- Primary evaluation split: held-out/validation only. Cube uses `split == "heldout"`. Fetch uses validation rows defined by `split_keys.json` and the saved validation ensemble order.
- Primary contrast: `interaction` vs `transport_free` for Cube; `gripper_object_contact == true` vs `primary_regime == "free_motion"` for Fetch. Static/support/secondary regimes are descriptive only unless named in the confirm-data gate.
- Regime labels must be target-independent and not derived from prediction error, residuals, or high-MSE labels.
- Headline statistics for every group/contrast: median, 10% trimmed mean, mean, episode-clustered 95% percentile bootstrap CI, `P(superiority) = P(left > right) + 0.5 P(left == right)`, and Cliff's delta where useful.
- Bootstrap clustering: Cube clusters by `episode_ordinal`; Fetch clusters by `(env_id, episode_id)`. Default `n_bootstrap = 1000` and fixed seed `260727` unless runtime makes this impractical; any change will be reported.
- Paired episode tests: within each eligible episode, collapse matched pair deltas to the episode median, then run one-sided Wilcoxon signed-rank with alternative `interaction > transport_free/free`.
- Missing or insufficient coverage does not become a negative result. It is reported as `void_insufficient_coverage`.

## Part 1: Harden The Cube Matched-Phase Positive

Evaluation table: start from `runs/lewm_transfer/cube/relabel_motion/step_records_relabel_motion.csv`, heldout rows only, with arrays joined by `row_id` from `runs/lewm_transfer/cube/residuals.npz`.

### Metric 1: Whitened Latent Excess

Definition:

- Fit whitening on calibration target latents only: `mean, W = whitening_matrix(target_latents[row_id where split == "calibration"])`.
- Model whitened MSE for row `i`: `mean(((pred_latents[i] - target_latents[i]) @ W)^2)`.
- Persistence whitened MSE for row `i`: `mean(((prev_latents[i] - target_latents[i]) @ W)^2)`.
- Whitened excess: `mse_model_whitened_recomputed - mse_persistence_whitened_recomputed`.

This should reproduce the existing `mse_model_whitened`, `mse_persistence_whitened`, and `excess_whitened` columns within numerical tolerance. If the recompute disagrees, stop and report a provenance failure before interpreting any result.

Report:

- Phase-decile-adjusted interaction-vs-transport_free contrast on `excess_whitened`.
- Matched-position within-episode pairs on `excess_whitened`, using the existing nearest transport_free row in the same episode and normalized-position decile without replacement.

Pass for this metric: phase-decile-adjusted `delta_trimmed_mean > 0` and its clustered 95% CI lower bound `> 0`, plus matched-position `median_episode_delta > 0`, `fraction_episode_positive > 0.5`, and Wilcoxon `p < 0.05`.

### Metric 2: Per-Dimension Target-Std-Normalized Excess

Definition:

- Fit `target_std[d] = std(target_latents[calibration_rows, d])`.
- Replace any `target_std[d] < 1e-6` with `1.0`.
- Target-normalized model MSE for row `i`: `mean(((pred_latents[i] - target_latents[i]) / target_std)^2)`.
- Target-normalized persistence MSE for row `i`: `mean(((prev_latents[i] - target_latents[i]) / target_std)^2)`.
- Target-normalized excess: `mse_model_targetnorm - mse_persistence_targetnorm`.

This is the Cube version of the Fetch target-std control that inverted the raw Fetch excess contrast. It is fit on calibration targets only and evaluated on heldout rows only.

Report:

- Phase-decile-adjusted interaction-vs-transport_free contrast on `excess_targetnorm`.
- Matched-position within-episode pairs on `excess_targetnorm`.

Pass for this metric: same as Metric 1.

### Metric 3: Phase + Persistence-Magnitude Double Matching

Definition:

- Persistence magnitude for row `i` is the normalized latent motion magnitude, not model error:
  - whitened double-match variant: `sqrt(mean(((prev_latents[i] - target_latents[i]) @ W)^2))`.
  - targetnorm double-match variant: `sqrt(mean(((prev_latents[i] - target_latents[i]) / target_std)^2))`.
- Fit persistence-magnitude bins on heldout rows without using labels as follows: use quintile edges of the heldout persistence magnitude distribution, with duplicate edges dropped. If fewer than three non-empty bins remain, the double-match result is `void_insufficient_coverage`.
- Within each episode, normalized-position decile, and persistence-magnitude bin, greedily pair each interaction row to the nearest available transport_free row by absolute persistence-magnitude difference, without replacement.
- Paired delta for row pair `j`: `left_excess_metric[j] - right_excess_metric[j]`, computed separately for `excess_whitened` and `excess_targetnorm`.

Report:

- Number of eligible episodes, pairs, deciles, and bins.
- Distribution of absolute persistence-magnitude pair gaps.
- Per-pair trimmed-mean delta with episode-clustered CI.
- Per-episode median paired delta, positive-episode fraction, and Wilcoxon p-value.

Pass for double matching: for both whitened and targetnorm double-matched excess, `delta_trimmed_mean > 0` with clustered 95% CI lower bound `> 0`, and the per-episode paired Wilcoxon criteria above pass.

### Part 1 Acceptance

The Cube matched-phase signal is called `survives_all_part1_controls` only if Metric 1, Metric 2, and Metric 3 all pass. If it survives some but not all, report `mixed_part1_controls` and name exactly which controls passed.

## Part 2: Reframe Difficulty As Sensitivity / Multimodality

The goal is to test whether contact/interaction regions have higher local expansion or model residual instability after scale normalization. These metrics are not defined from prediction error or high-MSE labels.

### Neighbor Sets

For each anchor row `i`, find `k = 10` nearest neighbors among rows from different episodes only. If fewer than `k_min = 5` cross-episode neighbors exist, the anchor is excluded and coverage is reported.

Cube neighbor input:

- Primary normalized input vector: `prev_latents` transformed with calibration-fit whitening.
- Secondary normalized input vector: `prev_latents / target_std` after subtracting calibration target mean where appropriate. The report will state which of whitening or targetnorm is primary based on numerical stability; the default primary is whitening because raw latent covariance failed the isotropy gate.
- Un-normalized diagnostic input vector: raw `prev_latents`.

Fetch neighbor input:

- Primary normalized input vector: the leak-safe prediction input rebuilt by `build_examples(..., include_shifted_fullstate=True)`, standardized using train-split mean/std only. This includes observation/action/history plus shifted previous-row qpos/qvel and excludes same-row post-step qpos/qvel leakage.
- Secondary state-only normalized input vector: current `state_*` plus `action_*`, standardized on train rows, if needed for interpretability.
- Un-normalized diagnostic input vector: corresponding raw feature vector.

Density diagnostic:

- `neighbor_radius_i = median distance from anchor i to its k neighbors`.
- Report per-regime median/trimmed mean neighbor radius and interaction-vs-transport_free/free contrast. If interaction has much larger radii, sensitivity estimates may reflect sparse support; if much smaller radii, they may reflect oversampled regimes.

### 2a: True-Dynamics Local Expansion

For anchor `i` with neighbor set `N_i`:

- Input spread: `S_in(i) = median_{j in N_i} distance(x_j, x_i)` in the chosen metric.
- Successor vector:
  - Cube: `y = target_latents` transformed by the same whitening/targetnorm transform used for inputs.
  - Fetch: `y = next_state_*`/saved targets transformed by train-split target std or train-split standardization.
- Successor spread: `S_out(i) = median_{j in N_i} distance(y_j, y_i)` in the same normalized metric family.
- Local expansion: `E(i) = S_out(i) / max(S_in(i), eps)`, with `eps = 1e-8`.

Un-normalized diagnostic:

- Compute the same ratio using raw `prev_latents`/raw features and raw successors. This is reported alongside normalized expansion to show whether any apparent effect is motion/scale dominated.

Report:

- Per regime: n anchors, n episodes, median, trimmed mean, mean, clustered CI for `E`, and density diagnostic.
- Interaction-vs-transport_free/free contrast: `delta_trimmed_mean`, CI, `P(superiority)`.

Pass for 2a: normalized `E` has `interaction - transport_free/free` `delta_trimmed_mean > 0` with clustered 95% CI lower bound `> 0`. Un-normalized expansion may agree or disagree but is not part of the pass condition.

### 2b: Model Sensitivity / Residual Instability

For the same anchor and neighbor set:

- Residual vector:
  - Cube: existing single-checkpoint `residuals = pred_latents - target_latents`, transformed by the same whitening/targetnorm scaling used for successors.
  - Fetch: ensemble-mean residual `mean_member_prediction - target`, transformed by train-split target std. Also report finite-ensemble member variance as a secondary descriptive column, not as the primary residual-instability metric.
- Residual spread: `S_res(i) = median_{j in N_i} distance(r_j, r_i)`.
- Model sensitivity: `M(i) = S_res(i) / max(S_in(i), eps)`.

Un-normalized diagnostic:

- Compute the same ratio using raw residuals and raw input distances.

Report:

- Same per-regime and contrast statistics as 2a.
- Include residual magnitude summaries separately so residual-spread results are not silently reinterpreted as raw error magnitude.

Pass for 2b: normalized `M` has `interaction - transport_free/free` `delta_trimmed_mean > 0` with clustered 95% CI lower bound `> 0`.

### Assumptions And Failure Modes To Report

- Cross-episode nearest neighbors reduce temporal leakage but do not guarantee identical hidden state, action intent, or object geometry.
- Cube sensitivity is latent-state based because saved Cube artifacts do not include the full action-conditioned model input; this weakens causal interpretation relative to Fetch's rebuilt state+action inputs.
- Neighbor expansion is sensitive to the distance metric and neighborhood density. The density diagnostic is mandatory and can downgrade interpretation even if the CI passes.
- Local expansion can be high because of stochasticity, unobserved variables, multimodality, bad metric geometry, or sparse neighborhoods; it is not by itself a proof of contact dynamics branching.
- Fetch is a state-space proxy cross-check, not a latent LeWM result.

## Confirm-Data Gate

The final verdict is preregistered as follows:

- `confirmed_motion_free_contact_difficulty` if at least one of these holds with clustered 95% CI excluding zero:
  1. Part 1: Cube matched-phase excess survives both whitening and per-dim target-std normalization, and also survives phase + persistence-magnitude double matching.
  2. Part 2a: normalized true-dynamics local expansion is higher at Cube interaction than transport_free.
  3. Part 2b: normalized model sensitivity is higher at Cube interaction than transport_free.
- `fetch_convergent_support` may be added only if Fetch also shows a positive normalized local-expansion or model-sensitivity contrast with CI lower bound `> 0`; Fetch alone cannot confirm the main gate.
- `no_motion_free_contact_difficulty_found_current_data` if none of the three main Cube conditions hold.
- If all main Cube conditions are void due to coverage/provenance failures, verdict is `inconclusive_current_artifacts`, not negative.

Strength language:

- Part 1 passing all controls is the strongest result for the existing Cube positive.
- Part 2a or 2b passing without Part 1 is only evidence for a motion-free sensitivity signal, not evidence that the original matched-phase excess was real.
- Any positive Fetch result is consistency evidence only.
- If none hold, step 3/richer environment is not warranted on the current hypothesis framing.

## Unit Tests Before Analysis

The following tests must pass before producing the final report:

- Local expansion synthetic contraction: a deterministic map `y = 0.5x` has median expansion `< 1`.
- Local expansion synthetic branching: near-identical inputs with two successor branches has median expansion `> 1` when cross-episode neighbors mix branches.
- Per-dimension target-std normalization: manual two-dimensional example matches the implemented MSE and handles near-zero std by replacing it with `1.0`.
- `P(superiority)` bounds: always in `[0, 1]`, with ties counted as half.
- Cross-episode neighbor constraint: neighbor sets contain no row from the anchor episode.
- Bootstrap smoke test: clustered CI code returns finite bounds on a tiny deterministic table.

## Planned Outputs

- `runs/metric_hardening/PLAN.md` (this file).
- `runs/metric_hardening/cube/part1_metric_contrasts.csv`.
- `runs/metric_hardening/cube/double_matched_pairs.csv`.
- `runs/metric_hardening/cube/sensitivity_regime_summary.csv`.
- `runs/metric_hardening/cube/sensitivity_contrasts.csv`.
- `runs/metric_hardening/fetch/sensitivity_regime_summary.csv`.
- `runs/metric_hardening/fetch/sensitivity_contrasts.csv`.
- `runs/metric_hardening/REPORT.md`.

No existing file outside `runs/metric_hardening/` will be modified.
