# PushT adaptive-computation replication pilot

Final scientific label: `pusht_replication_pilot_not_supported`.

## Paper-level result

1. Heterogeneous extra-refinement value existed under the fixed fit-only rule: True.
2. The PushT-specific causal gate selected `stage_dual_r10_q0.75` and its evaluation call histogram at depths 1–4 was [1089, 42, 46, 263].
3. The learned causal allocation did not satisfy every predeclared pilot-positive condition; this process-valid negative is terminal.
4. Raw episode-averaged benefit was 0.00390979699 (exploratory 95% interval 0.00272872528, 0.00508864426); fit-derived-whitened benefit was -0.000309900129 (-0.00125899386, 0.000629201146). Positive values favor adaptive allocation.
5. Strongest limitations: This is one environment, one WeakPolicy DGP, one refiner, one internally selected gate family, and a bounded pilot. The analytic comparator is an expectation-level transition-independent mixture; exploratory intervals do not account for candidate-selection multiplicity. The target latents inherit the cached pretrained encoder's representation, and prediction utility was not tested in planning or control.
6. A separate fresh PushT confirmation is not warranted from this bounded result without new scientific rationale.

## Fit-only refinement signal

| Depth | Raw MSE | Fit-whitened MSE | Raw marginal gain | Fraction raw benefiting |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.122760675 | 0.261367866 | 0.0146127194 | 0.831019 |
| 2 | 0.114452381 | 0.249440313 | 0.00830829445 | 0.626852 |
| 3 | 0.110624659 | 0.244465888 | 0.00382772168 | 0.468056 |
| 4 | 0.110916017 | 0.245890225 | -0.000291358041 | 0.348148 |

## Untouched evaluation comparisons

| Comparison (positive favors adaptive) | Mean episode benefit | Exploratory 95% low | Exploratory 95% high |
| --- | ---: | ---: | ---: |
| fit_whitened_vs_fixed_depth_1 | 0.00414655782 | 0.00294197921 | 0.00528841683 |
| fit_whitened_vs_fixed_depth_4 | 0.00156333954 | -0.000221841098 | 0.00341116805 |
| fit_whitened_vs_primary_analytic | -0.000309900129 | -0.00125899386 | 0.000629201146 |
| fit_whitened_vs_within_episode_permutation | 0.00256173713 | 0.00182237643 | 0.00336832294 |
| raw_vs_fixed_depth_1 | 0.00663129351 | 0.0050587178 | 0.00818297809 |
| raw_vs_fixed_depth_4 | 0.00526328725 | 0.0032214695 | 0.00737390289 |
| raw_vs_primary_analytic | 0.00390979699 | 0.00272872528 | 0.00508864426 |
| raw_vs_within_episode_permutation | 0.00541152297 | 0.00425098652 | 0.00667399903 |

Adaptive and the primary analytic comparator each used exactly 102905055196 counted FLOPs. This includes 16201500 gate FLOPs and does not equate call-count equality with FLOP equality. Adaptive made 1440 base-predictor calls, 2363 refiner calls, and 2100 reached gate evaluations; 10500 comparison/min operations are reported separately. Synchronized latency is in `LATENCY.json` and is not used to claim speedup.

Stagewise combined score/gain Spearman values were [0.429762516764331, 0.37671439671439666, 0.3908497077617027]. All intervals are exploratory because this is a bounded pilot with internal candidate selection.

## Integrity

Independent recomputation passed: True. Fit, selection, and evaluation roles were seed-disjoint, contained the fixed episode counts, and had no replacements. The cached base remained frozen; causal features excluded target/future/contact/reward/success/simulator state/geometry; all outputs were finite; evaluation was generated only after the gate freeze. No HDF5/H5 corpus was opened.
