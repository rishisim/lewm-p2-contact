# Minimal figure plan

Use three main-text figures. Do not add an architecture cartoon, generic workflow, decorative trajectory images, or a latency figure; the method is adequately described in text and latency is clearer as a table.

## Figure 1. Confirmed predictive-compute effects and the PushT pilot-to-confirmation chronology

### Purpose

Establish the main result while making the formally negative PushT pilot impossible to miss.

### Panel A — Cube fresh confirmation

- **Marks:** Two horizontal point-and-interval rows: raw and PlanOracle-native-whitened.
- **x-axis:** Benefit as percentage of the corresponding analytic-comparator episode-averaged MSE, `100 × (comparator - adaptive) / comparator`. Zero is no advantage.
- **Point estimates:** Raw `0.1967556362%`; whitened `0.6124633322%`.
- **Uncertainty:** Transform every saved bootstrap replicate by the fixed comparator mean; draw the two-sided 95% percentile interval. Use a solid lower cap to indicate that the same lower endpoint is the Bonferroni simultaneous one-sided lower bound for the two co-primary claims.
- **Annotation:** `n=1,600 episodes`, `4.333e12 counted FLOPs`, terminal label `passed`.
- **Source arrays:**
  - `runs/lewm_v5_readiness_program/v5_package_versions/v004/metrics/v5_confirmation_episode_metrics.npz`: `raw_vs_analytic`, `native_whitened_vs_analytic`.
  - `runs/lewm_v5_readiness_program/v5_package_versions/v004/metrics/bootstrap_replicates.npz`: same keys.
  - Comparator means from `analysis_result.json`: `compute.raw_analytic_mixture.mean_loss`, `compute.native_whitened_analytic_mixture.mean_loss`.

### Panel B — PushT pilot followed by fresh binary confirmation

- **Layout:** Four endpoint rows grouped by study: four-depth pilot raw/fit-whitened, then binary confirmation raw/fit-whitened. Place a labeled divider between studies: “binary hypothesis derived after pilot; fresh cohort below.”
- **x-axis:** Percentage of the study- and metric-specific analytic-comparator MSE, with a zero line.
- **Pilot points:** Raw `3.5123%` (`0.003909796985 / 0.111317190814`); fit-whitened `-0.1229%` (`-0.000309900129 / 0.252149792697`).
- **Binary points:** Raw `2.252051962%`; fit-whitened `0.534598302%`.
- **Uncertainty:** Pilot: exploratory two-sided 95% episode bootstrap intervals. Binary: two-sided 95% episode bootstrap intervals, with simultaneous lower caps as in Panel A.
- **Labels:** Show `pilot not supported` beside the pilot group and `fresh binary confirmation supported` beside the confirmation group.
- **Source arrays:**
  - Pilot `runs/lewm_pusht_replication_pilot/EVALUATION_ARRAYS.npz`: `episode_raw_vs_primary_analytic`, `episode_fit_whitened_vs_primary_analytic`; comparator means from `PILOT_DECISION.json`.
  - Binary `runs/lewm_pusht_binary_confirmation/EVALUATION_ARRAYS.npz`: `episode_raw_vs_exact_compute_analytic`, `episode_fit_whitened_vs_exact_compute_analytic`, `episode_loss_exact_compute_analytic_raw`, `episode_loss_exact_compute_analytic_fit_whitened`.

### Panel C — Allocation and routing signal

- **Marks:** Two compact stacked bars for observed depth proportions (Cube depths 1–4; PushT binary depths 1–2), plus adjacent text for score/gain Spearman.
- **x-axis:** Fraction of transitions; segments ordered from depth 1 upward.
- **Cube histogram:** `[50973, 7267, 1773, 787]`; stagewise Spearman `0.2626/0.1712/0.1412`.
- **PushT binary histogram:** `[3265, 1055]`; stage-1 combined Spearman `0.4577`.
- **Uncertainty:** None; these are complete-cohort descriptive counts/ranks. Do not add error bars.
- **Source:** Cube `metrics/compute_ledger_realized.json` and `metrics/stagewise_ranking.json`; PushT binary `DECISION.json`.

### Claim supported

Learned routing improved fresh-cohort prediction at matched counted compute in Cube and in a separately confirmed pilot-derived PushT binary rule; the initial four-depth PushT pilot itself was negative.

## Figure 2. Frozen Cube zero-shot robustness map

### Purpose

Show that robustness is heterogeneous across metrics and shifts, preventing a universal-generalization reading.

### Panel A — Raw endpoint forest

- **Rows:** MarkovOracle policy structure, action noise `0.1→0.2`, random-action probability `0→0.1`.
- **x-axis:** Benefit as percentage of each regime's raw analytic-comparator MSE. Zero line emphasized.
- **Marks:** Point estimate and individual two-sided 95% interval; overlay a one-sided arrow/cap from the Bonferroni simultaneous lower bound to the point.
- **Status symbols:** Filled only when the simultaneous lower bound is positive.

### Panel B — Fixed-whitened endpoint forest

- Same rows, axis definition, and uncertainty treatment using the frozen PlanOracle-native whitening matrix.

### Panel C — Depth-allocation shift

- **Marks:** Four-segment stacked proportions for the confirmed PlanOracle DGP and each shifted DGP.
- **x-axis:** Fraction of transitions at depths 1–4.
- **Histograms:**
  - confirmed PlanOracle `[50973, 7267, 1773, 787]`;
  - MarkovOracle `[103059, 6748, 3507, 686]`;
  - increased action noise `[96236, 12960, 3261, 1543]`;
  - random-action contamination `[96637, 12788, 3166, 1409]`.
- **Uncertainty:** None for complete-cohort counts.

### Source arrays/artifacts

- Per-regime episode effects: `runs/lewm_v5_generalization/attempts/v005/metrics/{markov_oracle,plan_action_noise_0p2,plan_random_action_0p1}_episode_metrics.npz`.
- Bootstrap replicates: `metrics/bootstrap_replicates.npz` in the same attempt.
- Comparator means, simultaneous bounds, and call histograms: `analysis_result.json` / `decision.json`.

### Claim supported

The same frozen Cube policy has a partial zero-shot robustness envelope: action-noise shift passes both endpoints; policy-structure and random-action shifts pass only fixed-whitened.

## Figure 3. Prediction persists weakly across horizon but does not bridge to planning

### Purpose

Separate multi-step predictive fidelity from downstream decision utility in one evidence chain.

### Panel A — Horizon-wise adaptive-versus-matched prediction

- **x-axis:** Autoregressive horizon `K=1…5`.
- **y-axis:** Adaptive-minus-matched terminal MSE, with separate raw and whitened facets because units differ. Negative favors adaptive.
- **Marks:** Connected point estimates at each K; horizontal zero line.
- **Uncertainty:** No inferential bars. Shade/label the entire panel “exploratory; 3,400 overlapping starts from 100 consumed episodes.” Do not imply independent start-level inference.
- **Source:** `runs/lewm_frozen_gate_planning_bridge/phase_b_metrics.npz`, arrays `condition`, `raw_mse`, `whitened_mse`; compute mean over case axis for `adaptive - matched` at each horizon.

### Panel B — Off-policy candidate rollout transfer

- **x-axis:** Adaptive-minus-matched start-mean terminal MSE.
- **Rows:** Raw and whitened.
- **Marks:** Mean point and paired 95% start-cluster bootstrap interval over 20 starts; zero line.
- **Source:** `runs/lewm_planning_bridge_decomposition/decomposition_metrics.npz`, arrays `adaptive_minus_matched_start_effect_raw` and `adaptive_minus_matched_start_effect_whitened`; bootstrap seed `27182818`, 20,000 replicates.

### Panel C — Why planning is unsupported

- **x-axis:** Start index, 1–20.
- **left y-axis/marks:** Physical candidate-error range in millimeters on a log scale with a visible threshold at `0.1 mm`; use distinct marks for the four exactly constant starts. Avoid plotting numerical zeros on log scale; place them on a labeled floor.
- **right annotation strip:** Number of meaningful physical outcome groups per start; mark the fixed requirement of at least five.
- **Summary callout:** `0/20 informative`; actual-terminal raw latent cost versus physical error mean Spearman `0.0237` on nine rank-defined starts; all-start top-5 overlap `0.040` versus chance `0.078125`.
- **Uncertainty:** None for ranges/group counts; the point is candidate-set inadequacy, not an effect estimate.
- **Source:** `decomposition_metrics.npz`: `physical_outcome_range`, `meaningfully_distinct_outcomes`, `informative_start`, `actual_ceiling_spearman`, `actual_ceiling_top5_overlap`.

### Claim supported

A small adaptive prediction advantage remains visible through five steps and directionally on off-policy candidates, but the consumed candidate pool supplies no supported planning or control bridge.

## Production rules

- Use one color for adaptive benefit and neutral gray for comparators; reserve red only for negative/unsupported status.
- Every panel must state the independent unit and whether uncertainty is simultaneous, individual, exploratory, or absent.
- Never pool raw and whitened values on an unnormalized axis.
- Never use transition count as inferential n when episode or start is the independent unit.
- Do not plot latency as a benefit; report it in Table 7 as a negative diagnostic.
- Keep the post-hoc whitening-floor sensitivity in the supplement unless reviewers specifically ask; if plotted, show it as a dashed sensitivity curve labeled “outcome-cohort metric redefinition, non-confirmatory.”
