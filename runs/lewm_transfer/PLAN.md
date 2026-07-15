# LeWM Transfer Diagnostic Pilot Plan

Status: Stage 0 plan only. Do not edit analysis code or run the new diagnostic until this plan is reviewed.

## Goal And Claim Boundary

Test whether the state-space Fetch proxy finding transfers to the released LeWM JEPA latent predictor on Cube first, then PushT later. This is a diagnostic pilot, not a paper-result claim. The only allowed headline is whether the Cube latent predictor shows an interaction-specific persistence-adjusted excess-error elevation that survives phase control.

Acceptance criterion for "transfers" on Cube:

- On held-out diagnostic episodes, interaction-contact excess error must exceed free excess error, where `excess = mse_model - mse_persistence`.
- The clustered 95% bootstrap CI for the interaction-vs-free excess contrast must exclude zero on the positive side.
- The effect must survive phase control: the phase-decile-adjusted interaction-vs-free excess contrast must have a positive clustered CI, and the matched-position within-episode paired Wilcoxon test must have positive median paired delta with `p < 0.05`.

If the interaction effect vanishes under phase control, the report will say that plainly and call the transfer result negative or unresolved.

## Inspection Summary

Inspected sources:

- `le-wm/diagnose_pusht_latent_contacts.py`
- `le-wm/diagnostics/pusht_latent_contacts/summary.md`
- `le-wm/diagnostics/cube_event_localization/diagnostic.py`
- `le-wm/diagnostics/cube_event_localization/summary.md`
- `le-wm/jepa.py`
- `le-wm/module.py`
- repo dataset/cache policy in `README.md`, `le-wm/diagnostics/README.md`, and `.gitignore`

Relevant facts from inspection:

- PushT harness loads `quentinll/lewm-pusht`, preprocesses pixels with ImageNet normalization, calls `model.encode({"pixels": batch.unsqueeze(0)})`, action-encodes flattened action blocks, then predicts one-step latents with `model.predict(emb_window, act_window)[:, -1]`.
- Cube harness already does the same for `quentinll/lewm-cube` and additionally validates the pixel-ViT config, Cube history size 3, frameskip/action block 5, action encoder input dim 25, Lance row count, sensor-contact precheck, geometry agreement, identity/persistence MSE, and output artifacts.
- Existing Cube summary reports mean latent MSE `0.141483`, identity/persistence MSE `0.309381`, model/persistence ratio `0.457308`, contact/non-contact raw MSE ratio `4.76519`, sensor-contact geometry agreement at 0.04m of `81.542%`, and a strong normalized-position confound.
- Existing PushT summary reports 30 trajectories, 727 prediction records, geometry-from-state contacts, contact/non-contact mean ratio `1.272`.
- `JEPA.encode` stores `info["emb"]`; `JEPA.predict` returns predicted embeddings with the same `(B,T,D)` convention; `JEPA.rollout` is autoregressive and is not needed for this one-step diagnostic.
- `module.SIGReg` is the latent isotropy motivation, but isotropy must still be checked empirically from target latents.
- The old Cube summary points at `/tmp/lewm_cube_stage0/cube_single_expert.h5`, but the current verified provenance source is `/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5`, with verified Lance at `/Users/rishisim/.stable_worldmodel/datasets/cube_single_expert.lance`.
- Generated Lance tables, checkpoints, full datasets, and weights are ignored by repo policy and must stay out of git.
- The current bare `python3` lacks the LeWM diagnostic runtime packages (`torch`, `h5py`, `hdf5plugin`, `lance`), so actual runs must use the existing LeWM/stable-worldmodel environment.

## Scope Order

1. Cube complete first.
2. PushT only after Cube report is reviewed.
3. No LeWM Fetch attempt. There is no Fetch LeWM checkpoint for this pilot.

## Reuse

Cube reuse by import from `le-wm/diagnostics/cube_event_localization/diagnostic.py` using `importlib.util.spec_from_file_location`, because `le-wm` is not a normal Python package name:

- `load_model`
- `validate_training_match`
- `h5_layout`
- `write_lance_subset`
- `encode_episode`
- `edge_indices`
- `raw_edge_to_target_model_step`
- `contact_stats`
- `compare_precheck`
- `wrapped_angle_delta`
- constants for model repo, cache defaults, and ImageNet normalization

Cube code to copy or adapt only with source comments:

- The prediction loop from `predict_episode_records`, because the new diagnostic must also store the full residual vector `pred - target`, target latent vectors for isotropy, and a three-regime label. The existing function only writes scalar MSEs and event-localization columns.
- The figure/output verification pattern from the Cube script, but outputs must go under `runs/lewm_transfer/cube/`.

Statistics reuse:

- Adapt `trimmed_mean`, `probability_superiority`, `bootstrap_ci`, and clustered bootstrap patterns from `le-wm/diagnostics/fetch_contact_compute/experiment_utils.py` with source comments. The existing bootstrap helpers assume `env_id`, `episode_id`, and a hard-coded `mse` column, so the new module needs a local generalized version for Cube episode clusters and metrics such as `mse_model`, `mse_persistence`, `excess`, and residual-structure summaries.

PushT later reuse:

- Reuse `le-wm/diagnose_pusht_latent_contacts.py` model loading, HDF5/subset handling, pixel preprocessing, action blocking, and PushT contact extraction. Do not start PushT until the Cube report is finished and reviewed.

## New Files And Artifacts

All new writes stay under `runs/lewm_transfer/`.

Planned code:

- `runs/lewm_transfer/cube/run_cube_transfer.py`: Cube CLI driver and one-step record collection.
- `runs/lewm_transfer/lewm_transfer_stats.py`: robust stats, episode-clustered bootstrap, P(superiority), phase controls, residual PCA/effective-rank helpers, and report-table helpers.
- `runs/lewm_transfer/lewm_transfer_plots.py`: six required figures with non-empty output checks.
- `runs/lewm_transfer/tests/test_transfer_stats.py`: `unittest` tests.

Planned Cube outputs:

- `runs/lewm_transfer/cube/step_records.csv`: one row per predicted step, with episode id, step ids, normalized position, split, regime labels, `mse_model`, `mse_persistence`, and `excess`.
- `runs/lewm_transfer/cube/residuals.npz`: full residual vectors, target latents, optional predicted latents, and row ids matching `step_records.csv`.
- `runs/lewm_transfer/cube/label_provenance.json`: label sources, thresholds, split policy, and geometry-agreement/sanity rates.
- `runs/lewm_transfer/cube/validity_gates.json`: collapse/persistence gate, latent isotropy, circular-label gate, split/threshold gate, and coverage gate.
- `runs/lewm_transfer/cube/metrics/regime_summary.csv`
- `runs/lewm_transfer/cube/metrics/regime_contrasts.csv`
- `runs/lewm_transfer/cube/metrics/bootstrap_cis.csv`
- `runs/lewm_transfer/cube/metrics/phase_decile_contrasts.csv`
- `runs/lewm_transfer/cube/metrics/matched_position_pairs.csv`
- `runs/lewm_transfer/cube/metrics/within_episode_pairs.csv`
- `runs/lewm_transfer/cube/metrics/residual_structure.csv`
- `runs/lewm_transfer/cube/figures/*.png`
- `runs/lewm_transfer/cube/REPORT.md`

If a temporary Cube Lance subset is needed, place it under `runs/lewm_transfer/cube/cache/` and rely on `*.lance/` ignore rules. Do not write a new Lance table under `le-wm/diagnostics/cube_event_localization/`.

## Data, Split, And Label Plan

Cube input:

- Default source HDF5: `/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5`.
- Default model repo: `quentinll/lewm-cube`.
- Default checkpoint cache: `runs/lewm_transfer/cube/cache/model/` or the existing HF cache if explicitly passed.
- Default episode set: same bounded first-30 Cube prefix used by the existing diagnostic, unless review changes this.

Diagnostic split:

- Calibration episodes: episode ordinals `0` through `9`.
- Held-out reporting episodes: episode ordinals `10` through `29`.
- "Held-out" here means held out from diagnostic threshold/calibration choices. It is not a claim that these episodes were excluded from the original LeWM pretraining unless a separate config audit establishes that.
- Fixed bins such as normalized-position deciles are not fitted from targets. Any empirical threshold, whitening transform, or support-contact geometry band must be calibrated on the calibration episodes only and evaluated on held-out episodes.

Regime labels:

- `interaction_contact`: gripper-object contact from the Cube sensor column `proprio_gripper_contact > 1e-9`, block-reduced over the raw transition rows exactly as the Cube script does for `contact`.
- `support_contact`: object-table/resting contact from explicit MuJoCo/table-contact data if present in the source. If no explicit column exists, use only a geometry proxy based on block support-plane height from `privileged_block_0_pos`, calibrated on calibration episodes and validated by sanity/agreement checks. Do not use block velocity, block orientation change, latent vectors, target error, or high-MSE labels.
- `free`: neither interaction nor support.
- Precedence: interaction wins over support; support wins over free. The three reporting regimes are `interaction`, `support`, and `free`.
- If a non-circular support label cannot be produced and validated, abort the three-way regime contrast and report label unavailability rather than inventing a broad "contact" label.

Label provenance to report:

- Exact column/proxy source for each label.
- Geometry-agreement/sanity rate, including the existing gripper-contact 0.04m agreement check when applicable.
- Counts by episode and regime on calibration and held-out splits.

## Per-Step Computation

For each held-out predicted target step:

- Encode strided Cube pixels with the reused Cube `encode_episode` path.
- Action-encode flattened 5-raw-step action blocks with `model.action_encoder`.
- Predict one-step target latent with `model.predict(emb_window, act_window)[:, -1]`.
- Store `mse_model = mean((pred - target)^2)`.
- Store `mse_persistence = mean((latent_t - latent_{t-1})^2)`.
- Store `excess = mse_model - mse_persistence`.
- Store full residual vector `r = pred - target`.
- Store target latent vector for latent covariance/isotropy checks.
- Store `episode_id`, `episode_ordinal`, `model_step`, `raw_step`, `transition_block`, `normalized_position`, split, and regime labels.

All MSEs are mean over latent dimensions.

## Analyses

Encoder-vs-predictor disentangling:

- Per regime, report median, trimmed mean, and mean for `mse_model`, `mse_persistence`, and `excess`.
- Cluster all 95% CIs by resampling episodes.
- Interpret high model MSE with high persistence MSE as latent motion/representation, and high model MSE with low persistence MSE as predictor difficulty.
- Use `excess`, not raw model MSE, for the headline transfer contrast.

Interaction vs support vs free:

- Sort regimes by held-out excess error.
- Report medians, trimmed means, clustered CIs, and P(superiority) for interaction-vs-free and support-vs-free.
- Also include raw `mse_model` and `mse_persistence` tables so the prior raw 4.765x Cube number can be related to the adjusted result without overclaiming.

Phase control:

- Fixed normalized-position deciles: compute interaction-vs-free excess contrast within each decile and a decile-adjusted aggregate, clustered over episodes.
- Matched-position paired test: within each episode, pair each interaction step to the nearest free step in the same normalized-position decile, without replacement when possible; collapse to per-episode median paired deltas; run paired Wilcoxon on those per-episode deltas.
- Report whether the interaction excess survives these controls. If not, state that phase explains the apparent raw elevation.

Within-episode paired test:

- Per episode, compute median interaction error and median free error for both `mse_model` and `excess`.
- Report fraction of eligible episodes where interaction exceeds free.
- Run paired Wilcoxon over episode-level paired medians.
- Use excess as primary; raw model MSE is secondary.

Residual structure:

- Per regime, compute directional consistency: `norm(mean residual) / mean(norm(residual))`.
- Per regime, center residuals and compute PCA variance fractions for top `k = 1, 3, 5, 10` subject to latent dimension, plus effective rank.
- Use episode-clustered bootstrap CIs for directional consistency and top-k/effective-rank summaries.
- Report explicitly that this is a single-checkpoint residual-structure diagnostic, not a LeWM ensemble bias-variance decomposition. It can be consistent with reducible bias, but cannot prove bias dominance.

Validity gates:

- Collapse/persistence gate: report held-out `mean(mse_model) / mean(mse_persistence)`. If the ratio is in `[0.9, 1.1]` or worse than persistence, mark the latent prediction contrast void.
- Latent isotropy gate: report target-latent covariance condition number and per-dimension variance spread on held-out episodes. If condition number is above `1e4` or variance max/min is above `100`, additionally report whitened-latent MSE and whitened excess using a whitening transform fit on calibration episodes only.
- No circular labels: abort any contrast whose labels depend on block velocity, block orientation change, latent vectors, prediction error, or high-MSE thresholds.
- Threshold/gate split: empirical thresholds and whitening transforms are fit only on calibration episodes and evaluated only on held-out episodes.
- Coverage gate: if interaction or free has fewer than 5 eligible held-out episodes or fewer than 30 held-out rows, mark the primary contrast underpowered and do not claim transfer.

## Figures

Write PNGs under `runs/lewm_transfer/cube/figures/` and embed them in `REPORT.md`:

1. Per-regime bar chart of `mse_model` vs `mse_persistence` with clustered-CI whiskers.
2. Sorted per-regime excess-error bars.
3. Interaction-vs-free latent-error ECDF/violin with P(superiority) annotated.
4. Error vs normalized-position curves split by regime.
5. Within-episode paired scatter, interaction median vs free median, log-log, `y=x`, percent above, Wilcoxon p.
6. Residual effective-rank and directional-consistency bars by regime.

Figure checks:

- Every expected PNG must exist and be non-empty.
- Figures should render from synthetic test data in unit tests.

## Report

`runs/lewm_transfer/cube/REPORT.md` will include:

- One-paragraph diagnostic summary of whether the proxy finding transfers.
- Disentangling result: model MSE, persistence MSE, and excess by regime.
- Regime contrast under phase control.
- Within-episode paired result.
- Residual-structure result with the single-checkpoint caveat.
- Validity gate table with actual pass/fail numbers.
- Scope and caveats: latent-space not state-space, single checkpoint and no LeWM ensemble, Cube not Fetch, no LeWM Fetch checkpoint, and relationship to the prior raw Cube 4.765x number plus its phase confound.
- Every headline number must point to a CSV/JSON/NPZ artifact path.

## Tests

Use `unittest`, consistent with nearby Fetch diagnostics:

- Persistence baseline toy test: `mse_persistence` must equal `mean((latent_t - latent_{t-1})^2)` on a hand-computed latent sequence.
- P(superiority) bounds test: output must be in `[0, 1]`, with ties counted as half.
- Figure generation test: six plotting functions create non-empty PNGs from a small synthetic record table.

Run command after implementation, in the LeWM diagnostics environment:

```bash
python -m unittest discover -s runs/lewm_transfer/tests
```

## Stop Condition

Stop here for review. After approval, implement Cube only, verify tests and artifacts, then write `runs/lewm_transfer/cube/REPORT.md`. Do not start PushT until Cube is complete and reviewed.
