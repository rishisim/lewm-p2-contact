# PushT binary adaptive-computation confirmation

Final scientific label: `pusht_binary_confirmation_supported`.

## Paper-level answer

The simple one-extra-step mechanism replicated on this fixed fresh PushT cohort under both raw and pilot-frozen fit-whitened metrics. The raw episode-averaged benefit over the strongest exact-compute transition-independent analytic comparator was 0.00273378204, with simultaneous one-sided lower bound 0.00243226782. The pilot-frozen fit-whitened benefit was 0.00156923683, with simultaneous lower bound 0.00126875308. This adds a fresh-data PushT confirmation of the binary adaptive-allocation mechanism beyond the already-confirmed Cube PlanOracle result.

The simultaneous bounds use 20,000 fixed-seed episode-cluster bootstrap replicates with Bonferroni alpha/2 = 0.025 per endpoint (familywise one-sided alpha 0.05). The table below gives the ordinary exploratory 95% two-sided intervals from the same fixed bootstrap.

What remains unproven: this is one PushT environment, one WeakPolicy(dist_constraint=100) data-generating process, one frozen base/refiner/gate, and one fresh cohort. It does not establish planning or control improvement, universal generalization, physical-difficulty prediction, or wall-clock speedup.

## Fixed analysis

The binary policy made 3265 depth-1 and 1055 depth-2 decisions across 4320 transitions. The stage-1 score/combined-gain Spearman correlation was 0.457739742 (raw 0.459258874; whitened 0.395781309). Raw episode signs were 216 positive, 0 zero, and 24 negative; the relative raw effect was 2.25205%. Whitened signs were 166 positive, 0 zero, and 74 negative; the relative whitened effect was 0.534598%.

| Comparison (positive favors adaptive) | Mean episode benefit | Exploratory 95% low | Exploratory 95% high |
| --- | ---: | ---: | ---: |
| fit_whitened_vs_exact_compute_analytic | 0.00156923683 | 0.00126875308 | 0.00187382556 |
| fit_whitened_vs_fixed_depth_1 | 0.00331473959 | 0.00296531244 | 0.00367304349 |
| fit_whitened_vs_fixed_depth_2 | -0.00350165195 | -0.00381142588 | -0.00318509677 |
| fit_whitened_vs_within_episode_call_randomization | 0.00162737472 | 0.00138121934 | 0.00188121944 |
| raw_vs_exact_compute_analytic | 0.00273378204 | 0.00243226782 | 0.00303985591 |
| raw_vs_fixed_depth_1 | 0.00372532903 | 0.00332604691 | 0.00413650854 |
| raw_vs_fixed_depth_2 | -0.000146777411 | -0.000440033416 | 0.000154379599 |
| raw_vs_within_episode_call_randomization | 0.00275727824 | 0.00243741628 | 0.0030846903 |

Absolute episode-averaged MSEs were: adaptive raw 0.118656936, fixed depth 1 raw 0.122382265, fixed depth 2 raw 0.118510158; adaptive whitened 0.291966455, fixed depth 1 whitened 0.295281195, and fixed depth 2 whitened 0.288464804.

The raw comparator selected depths 1 and 2 with upper-depth weight 0.25607431; the whitened comparator selected depths 1 and 2 with weight 0.25607431. Both searched every feasible fixed-depth pair from depths 1–4.

## Compute, timing, and integrity

Adaptive and comparator totals were exactly 307585049440 integer counted FLOPs each. Adaptive used 4320 base predictions, 5375 refiner calls, and 4320 gate evaluations. The adaptive breakdown was 304055389440 base-predictor FLOPs, 259200 action-normalization FLOPs, 3496072000 refiner FLOPs, 16031520 gate-feature FLOPs, and 17297280 gate-score FLOPs (33328800 gate FLOPs total); 21600 comparison/min operations are reported separately. Counterfactual depth-3/4 exits were evaluated only to search the required comparator; the adaptive policy never used those depths and no stage-2 or stage-3 gate score was computed.

Synchronized median latency on mps was 0.194041s for base prediction over all transitions, 0.0169062s from cached base to binary adaptive outputs, 0.0016025s for fixed depth 1, and 0.00278346s for fixed depth 2. These timings are separate from counted FLOPs and support no speedup claim.

All 240 fixed fresh episodes were retained with no exclusions or replacements. Frozen hashes, causal boundaries, finiteness, exact selected predictions, exact integer compute, chronology, focused tests, bootstrap values, and every headline value were independently checked from the lossless arrays. Independent recomputation passed: True.
