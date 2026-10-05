# Manuscript-ready core tables

All effects use comparator MSE minus adaptive MSE unless explicitly labeled otherwise; positive values favor adaptive allocation. MSE is episode-averaged before inference. FLOPs are exact integer counted totals, not wall-clock proxies.

## Table 1. Fresh Cube PlanOracle confirmation

| Endpoint | Adaptive MSE | Strongest analytic-mixture MSE | Benefit | Individual 95% percentile interval | Simultaneous one-sided lower bound | Relative benefit |
|:--|--:|--:|--:|:--|--:|--:|
| Raw latent MSE | 0.003433540759 | 0.003440309763 | 0.000006769003362 | [0.000005423955198, 0.000008162961004] | 0.000005423955198 | 0.1968% |
| PlanOracle-native-whitened latent MSE | 0.09876419994 | 0.09937282203 | 0.0006086220972 | [0.0005135570380, 0.0007084272042] | 0.0005135570380 | 0.6125% |

**Population and design.** `swm/OGBCube-v0`, PlanOracle with `action_noise=0.1`, `p_random_action=0`, 1,600 fresh episodes, 38 transitions/episode, 60,800 total transitions. The two one-sided bounds control familywise alpha `0.05` by Bonferroni (`0.025` per endpoint); 20,000 episode bootstrap replicates. Adaptive counted compute: `4,332,936,120,435` FLOPs. Terminal label: `v5_confirmation_passed`.

**Source.** `runs/lewm_v5_readiness_program/v5_package_versions/v004/decision.json`; `metrics/v5_confirmation_episode_metrics.npz`; `metrics/bootstrap_replicates.npz`.

## Table 2. Zero-shot one-factor Cube shifts with the fully frozen V5 policy

| Shift from confirmed PlanOracle DGP | Episodes (transitions) | Adaptive raw MSE | Raw benefit (simultaneous LB) | Adaptive fixed-whitened MSE | Fixed-whitened benefit (simultaneous LB) | Adaptive counted FLOPs | Regime label |
|:--|--:|--:|:--|--:|:--|--:|:--|
| Policy structure: PlanOracle → MarkovOracle | 3,000 (114,000) | 0.08203089830 | -0.000002513778663 (-0.000005677096742) | 0.5879283392 | 0.0001773618066 (0.00008606212246) | 8,121,837,438,190 | mixed |
| Action noise: 0.1 → 0.2 | 3,000 (114,000) | 0.006131965011 | 0.000006690627725 (0.000005233178062) | 0.1102460678 | 0.0006314822713 (0.0005518158912) | 8,124,093,582,040 | supported |
| Random-action probability: 0 → 0.1 | 3,000 (114,000) | 0.1013982963 | 0.00000009599565688 (-0.000003128983968) | 0.3484651159 | 0.0004263060037 (0.0003605363394) | 8,123,886,122,050 | mixed |

**Multiplicity and interpretation.** Six regime-by-endpoint one-sided claims share familywise alpha `0.05` (`0.0083333` per claim; 20,000 episode bootstraps). The whitening matrix is the unchanged PlanOracle-native transform and is therefore labeled fixed-whitened. Four of six simultaneous lower bounds are positive. Terminal label: `zero_shot_generalization_partial`.

**Source.** `runs/lewm_v5_generalization/attempts/v005/DGP_MATRIX.json`, `decision.json`, and `metrics/*_episode_metrics.npz`.

## Table 3. PushT four-depth pilot and pilot-derived binary confirmation

| Study | Policy tested | Fresh evaluation n | Raw benefit, interval | Fit-whitened benefit, interval | Counted FLOPs per primary arm | Formal result |
|:--|:--|--:|:--|:--|--:|:--|
| Four-depth replication pilot | PushT-specific selected depths 1–4; histogram `[1089, 42, 46, 263]` | 80 episodes; 1,440 transitions | 0.003909796985; exploratory 95% CI [0.002728725277, 0.005088644256] | -0.0003099001292; exploratory 95% CI [-0.001258993863, 0.0006292011461] | 102,905,055,196 | `pusht_replication_pilot_not_supported` |
| Binary fresh confirmation | Pilot-derived: mandatory depth 1, optional depth 2; histogram `[3265, 1055]`; stages 2–3 never consulted | 240 episodes; 4,320 transitions | 0.002733782041; 95% CI [0.002432267825, 0.003039855909]; simultaneous LB 0.002432267825 | 0.001569236827; 95% CI [0.001268753083, 0.001873825556]; simultaneous LB 0.001268753083 | 307,585,049,440 | `pusht_binary_confirmation_supported` |

**Chronology.** The pilot decision was locked at `2026-07-18T19:06:22.440495Z`. A read-only binary simplification on those consumed pilot arrays yielded raw/fit-whitened point benefits `0.002435240881`/`0.001233341903`; it generated the hypothesis but supplied no confirmatory evidence. The binary protocol was locked at `2026-07-18T23:30:23.485504Z`, before the disjoint fresh cohort; the scientific decision was locked at `2026-07-18T23:31:48.777558Z` and independently verified afterward.

**Scope.** Both studies use `swm/PushT-v1` and `WeakPolicy(dist_constraint=100)`. The PushT refiner, whitening, and gate were fitted/selected on PushT roles, so this is environment-specific method replication, not zero-shot Cube-to-PushT transfer.

**Source.** `runs/lewm_pusht_replication_pilot/PILOT_DECISION.json`; `runs/lewm_pusht_binary_confirmation/CONFIG.json`, `DECISION.json`, and `INDEPENDENT_CHECK.json`.

## Table 4. Routing diagnostics in confirmed and pilot studies

| Study | Stage/reached rows | Score-versus-gain Spearman | Allocation summary | Interpretation |
|:--|:--|:--|:--|:--|
| Cube fresh confirmation | Stage 1 / 60,800 | 0.262620548 | Depth histogram `[50973, 7267, 1773, 787]` | Confirmed positive stage-1 ordering signal |
| Cube fresh confirmation | Stage 2 / 9,827 | 0.171193282 | Same policy | Positive on selected reached subset |
| Cube fresh confirmation | Stage 3 / 2,560 | 0.1412291687 | Same policy | Positive on smaller reached subset |
| PushT four-depth pilot | Stages 1/2/3: 1,440/351/309 | 0.429762517 / 0.376714397 / 0.390849708 | Histogram `[1089, 42, 46, 263]` | Strong exploratory routing evidence inside a formally negative pilot |
| PushT binary fresh confirmation | Stage 1 / 4,320 | Combined 0.457739742; raw 0.459258874; fit-whitened 0.395781309 | 3,265 depth-1 and 1,055 depth-2 transitions | Fresh confirmation of the one-extra-step routing signal |

**Sources.** Cube `metrics/stagewise_ranking.json`; PushT pilot `PILOT_DECISION.json`; PushT binary `DECISION.json`.

## Table 5. Multi-step prediction and downstream planning boundary

| Link tested | Population and independent unit | Estimate | Uncertainty / adequacy check | Paper conclusion |
|:--|:--|:--|:--|:--|
| Five-step composed prediction: raw | 100 consumed Cube episodes; 3,400 overlapping starts | K=5 adaptive `0.02967818`, matched `0.02974415`; adaptive-minus-matched `-0.00006597279931` | No confirmatory interval; overlapping starts; consumed data | Small exploratory advantage |
| Five-step composed prediction: whitened | Same | K=5 adaptive `0.155283`, matched `0.156470`; adaptive-minus-matched `-0.001187089240` | Same limitations | Small exploratory advantage |
| Off-policy candidate rollout fidelity: raw | 20 independent Cube starts; 64 fixed candidates/start | Adaptive-minus-matched `-0.000002735299392` | 95% start-cluster interval [-0.00006392055823, 0.00005519269647] | Directional only; interval crosses zero |
| Off-policy candidate rollout fidelity: whitened | Same | Adaptive-minus-matched `-0.0001401812773` | 95% interval [-0.0004938837164, 0.0001551248760] | Directional only; interval crosses zero |
| Actual latent goal cost → physical block error | 20 starts; 9 rank-defined | Raw mean Spearman 0.0237205; all-start top-5 overlap 0.040 | `0/20` starts informative; chance top-5 overlap 0.078125 | Unsupported |
| Downstream planning/control utility | Same consumed pilot | Adaptive and matched raw selections differed on 1/20 starts and 0 informative starts | Candidate pool tie-heavy; no closed-loop study | Unsupported; do not claim |

Negative adaptive-minus-matched MSE values favor adaptive routing. All planning intervals are descriptive.

**Sources.** `runs/lewm_frozen_gate_planning_bridge/RESULTS.json`; `runs/lewm_planning_bridge_decomposition/RESULTS.json`.

## Table 6. Exact compute accounting and comparator type

| Study | Rows | Adaptive call histogram | Adaptive counted FLOPs | Primary comparator counted FLOPs | Equality and comparator type |
|:--|--:|:--|--:|--:|:--|
| Cube fresh confirmation | 60,800 | `[50973, 7267, 1773, 787]` | 4,332,936,120,435 | 4,332,936,120,435 by analytic expectation | Linear fixed-depth analytic mixture; equivalent mean depth 1.252954098. Conservative seeded runnable mixture used 4,332,936,224,000 FLOPs, 103,565 more. |
| Cube MarkovOracle shift | 114,000 | `[103059, 6748, 3507, 686]` | 8,121,837,438,190 | Equal by analytic expectation | Strongest pair can differ by endpoint (raw depths 1/4; fixed-whitened 1/3). |
| Cube action-noise shift | 114,000 | `[96236, 12960, 3261, 1543]` | 8,124,093,582,040 | Equal by analytic expectation | Depth-1/2 analytic mixture for both endpoints. |
| Cube random-action shift | 114,000 | `[96637, 12788, 3166, 1409]` | 8,123,886,122,050 | Equal by analytic expectation | Depth-1/2 analytic mixture for both endpoints. |
| PushT four-depth pilot | 1,440 | `[1089, 42, 46, 263]` | 102,905,055,196 | 102,905,055,196 | Exact integer total; strongest analytic depth-1/2 mixture; equivalent mean depth 1.658270021. |
| PushT binary confirmation | 4,320 | `[3265, 1055]` | 307,585,049,440 | 307,585,049,440 | Exact integer total; strongest pairwise depth-1/2 analytic mixture; equivalent mean depth 1.256074310. |

“Exact integer total” means the analytic mixture's weighted fixed-depth total equals the adaptive integer total; it does not make the fractional mixture an executable schedule. Transition-randomized and seeded comparators provide runnable controls where reported.

**Sources.** Cube `metrics/compute_ledger_realized.json` and shift `decision.json`; PushT pilot/binary decisions.

## Table 7. Synchronized latency diagnostics (milliseconds)

| Study / timed workload | Rows per timed call | Fixed depth 1 median | Fixed depth 2 median | Adaptive median | Other reference | Repeats | Claim boundary |
|:--|--:|--:|--:|--:|:--|--:|:--|
| Cube full latent/action-input path | 1 | 3.4151 | — | 3.8366 | Dense-all-exit shadow 4.1224 | 7 | Adaptive slower than depth 1 |
| Cube full latent/action-input path | 32 | 4.1582 | — | 7.0677 | Dense-all-exit shadow 4.8332 | 7 | No speedup |
| Cube full latent/action-input path | 256 | 13.0494 | — | 14.9812 | Dense-all-exit shadow 14.0034 | 7 | No speedup |
| Cube full latent/action-input path | 1,024 | 48.1154 | — | 49.6928 | Dense-all-exit shadow 49.1027 | 7 | No speedup |
| PushT four-depth pilot, cached base → selected output | 1,440 total rows | 0.7910 | — | 8.6095 | Fixed depth 4: 1.9625; common base predictor: 64.2102 | 5 | Encoding excluded; no speedup |
| PushT binary, cached base → selected output | 4,320 total rows | 1.6025 | 2.7835 | 16.9063 | Common base predictor: 194.0410 | 5 | Pixel encoding excluded; no speedup |

Cube timing includes frozen base prediction, solver blocks, causal features, gate heads, routing/indexing, and synchronization; it excludes simulator rollout and pixel encoding. PushT rows time the named cached-base component separately from the common base predictor. Energy was not measured. These diagnostics are not terminal endpoints.

**Sources.** Cube `runs/lewm_v5_readiness_program/v5_package_versions/v004/metrics/v5_confirmation_latency.json`; PushT pilot `runs/lewm_pusht_replication_pilot/LATENCY.json`; PushT binary `runs/lewm_pusht_binary_confirmation/DECISION.json`.

## Post-hoc PushT whitening sensitivity note (not a core result)

The formal PushT metric uses the pilot-fit whitening transform with floor ratio `1e-3`. Re-estimating the covariance on the confirmation targets after outcomes were available—therefore an invalid replacement endpoint but a useful sensitivity check—gave binary benefits `0.001336506988` at `1e-3`, `0.0008799668240` at `3e-4`, and `-0.0001742822988` at `1e-4`, with calls and comparator construction fixed. Smaller floors can therefore weaken or reverse the whitened effect. Source arrays: `runs/lewm_pusht_binary_confirmation/EVALUATION_ARRAYS.npz`; method: `evidence_check.py`.
