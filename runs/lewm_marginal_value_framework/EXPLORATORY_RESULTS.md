# Existing-artifact exploratory results

All results in this document reuse already-consumed pilot or confirmation
cohorts. They are exploratory and do not modify any accepted terminal label.
The fixed analysis definitions are in `ANALYSIS_PROTOCOL.md`; lossless derived
arrays and machine-readable tables are under `numeric/`.

## Independent artifact audit

No environment, encoder, predictor, refiner, or gate was executed. The analysis
recomputed losses in float64 from saved target and dense-exit arrays, reconstructed
gate heads from saved causal features or inputs and frozen weights, reconstructed
depth choices, and rebuilt every counted-compute total.

| Study | Rows / episodes | Recomputed call histogram | Headline loss and compute audit |
|---|---:|---|---|
| Cube v5 confirmation | 60,800 / 1,600 (38 each) | `[50,973, 7,267, 1,773, 787]` | Sparse raw `0.003433540759280532`, sparse whitened `0.09876419993642473`, and total `4,332,936,120,435` FLOPs exactly reproduce the accepted report. |
| PushT four-depth pilot | 1,440 / 80 (18 each) | `[1,089, 42, 46, 263]` | Raw `0.10740739382814737`, whitened `0.252459692826501`, and total `102,905,055,196` FLOPs reproduce the accepted report (floating difference below `1e-16`). |
| PushT binary confirmation | 4,320 / 240 (18 each) | `[3,265, 1,055]` | Raw `0.11865693567454487`, whitened `0.29196645548960315`, and total `307,585,049,440` FLOPs reproduce the accepted report. |

Cube’s independently reconstructed float32 affine scores differ from stored
backend scores by at most `1.2041e-5`; PushT pilot and binary maxima are
`8.80e-7` and `1.03e-6`. All independently reconstructed choices are exact.
Cube dense-selected and actual sparse predictions differ only at the accepted
numerical contract (`2.05e-11` mean absolute raw-loss difference and `4.54e-9`
for whitened loss).

The fail-closed verifier reports 23/23 checks passed in
`numeric/validation_report.json`.

## Can the saved scores support a compute-price sweep?

- **PushT binary:** all 4,320 stage-1 scores are saved. They support an exact
  finite-cohort score-ranked top-k sweep without using confirmation outcomes to
  fit or rank anything. They do **not** support a calibrated (lambda) sweep in
  loss-per-FLOP units because the score is the minimum of two differently
  standardized heads.
- **Cube:** stage-1 scores are complete, but stage-2 and stage-3 scores exist
  only for the 9,827 and 2,560 rows reached by the frozen sparse policy. A more-
  permissive sweep would require counterfactual scores that were never saved and
  would violate the no-regeneration rule. The stored arrays support only a
  conservative top-k thinning of frozen continuations, from a gate-only depth-1
  endpoint through the accepted policy.
- **PushT four-depth pilot:** all dense counterfactual scores are saved, so a
  threshold sweep is mechanically possible. Because the cohort is consumed and
  the frozen four-depth operating point was formally negative, such a sweep
  would be especially vulnerable to retrospective selection. It was not used to
  manufacture a favorable multistage frontier.

## Quality versus counted compute

![Exploratory quality-versus-compute frontiers](/Users/rishisim/Documents/research/lewm-p2-contact/runs/lewm_marginal_value_framework/figures/quality_compute_frontiers.png)

### Cube conservative thinning

At retention zero, the policy pays one gate per transition but performs only
fixed depth 1. It is consequently worse than a gate-free transition-independent
mixture on both endpoints. At every nonzero retention setting (`0.125` through
`1.0`), the causal score allocation has lower raw and whitened loss than the
strongest transition-independent lower envelope at the same counted total.
This covers equivalent fixed-depth means from `1.0518895` through `1.2529541`,
not merely the accepted endpoint.

At retention one, the audit reproduces the frozen allocation and obtains
dense-selected exploratory benefits of `6.769003433e-6` raw and
`0.000608622055` whitened. The tiny difference from the accepted sparse-path
benefits is exactly the saved dense/sparse numerical tolerance; the confirmation
claim remains the accepted sparse value.

This is encouraging range evidence, but the intermediate policies are
consumed-cohort diagnostics and the stagewise retention fraction is not one
shared compute price.

### PushT binary exact top-k

Both endpoints favor score-ranked allocation at every grid point from 10% to
90% optional depth-2 allocation (equivalent fixed-depth means `1.1118613` to
`1.9118613`). With no optional refinement the gate-only policy loses to the
lower envelope. At the largest schedulable count below gate-free fixed depth 2
(`4,268/4,320`), raw retains a very small advantage (`2.55e-5`) while whitened
is slightly negative (`-1.50e-5`). Thus the two-endpoint advantage occupies a
broad interior budget range but not both endpoints.

The accepted `1,055/4,320` optional-call operating point exactly reproduces
the confirmed raw benefit `0.002733782041` and whitened benefit
`0.001569236827`. This curve is the clearest empirical support for a budget-
allocation interpretation, but every intermediate point is exploratory on the
same consumed confirmation cohort.

## Oracle headroom and allocation efficiency

For binary PushT, the outcome oracle receives the identical optional-call count
and selects the transitions with the largest realized next-step gain separately
for each endpoint. Allocation uplift is measured relative to transition-
independent random assignment at that same call histogram. At the accepted
budget, the learned ranking captures:

- `61.99%` of raw oracle allocation uplift; and
- `52.99%` of whitened oracle allocation uplift.

The learned router is therefore materially informative but far from saturated.
The missing `38–47%` is diagnostic headroom, not a promise that an executable
causal model can attain the outcome oracle.

For multistage studies, `numeric/oracle_headroom.csv` reports only stage-local
headroom among rows reached by the frozen policy. Cube captures `7.97–12.06%`
of raw and `13.02–28.76%` of whitened oracle uplift across the three reached
stages. The PushT pilot captures `31.84–58.85%` raw and `25.97–50.05%`
whitened. These are not global prefix-constrained oracle optima.

## Calibration and rank association

![Allocation histograms and marginal-gain calibration](/Users/rishisim/Documents/research/lewm-p2-contact/runs/lewm_marginal_value_framework/figures/allocation_and_calibration.png)

The heads generally rank useful refinements better than they calibrate their
scale:

- Cube endpoint-head calibration slopes are `0.207–0.263` for raw and
  `0.311–0.600` for whitened. The gate score’s endpoint-specific Spearman
  association is weak for raw (`0.080–0.111`) and stronger for whitened
  (`0.133–0.296`). This does not contradict the accepted combined standardized-
  gain rank diagnostic, which is a different target.
- PushT binary calibration slopes are `0.571` raw and `0.390` whitened. The
  frozen minimum score has Spearman `0.4593` with raw gain and `0.3958` with
  whitened gain, reproducing the accepted diagnostics to numerical tolerance.
- In the negative PushT pilot, stage-3 mean whitened realized gain among reached
  rows is negative despite a positive mean predicted gain. Positive rank
  association therefore does not imply a successful multistage endpoint.

Because the deployed score is a minimum in standardized units, none of these
calibration results turns it into one scalar expected loss reduction per FLOP.

## Gate overhead sensitivity

![Gate overhead sensitivity](/Users/rishisim/Documents/research/lewm-p2-contact/runs/lewm_marginal_value_framework/figures/gate_overhead_sensitivity.png)

Repricing the frozen allocation while holding choices and losses fixed shows:

- Cube favors the causal policy on both endpoints at the accepted ledger and at
  `2x` gate cost; whitened benefit is negative by `5x`.
- PushT binary favors the causal policy on both endpoints through `10x` gate
  cost; whitened benefit is negative by `25x`.
- The PushT four-depth pilot remains mixed even with zero gate cost: raw favors
  routing, whitened does not.

These are accounting sensitivities, not latency or energy measurements. They
show why overhead belongs in the system budget even when it is sunk at an
already-reached binary decision.

## Finite integer scheduling

The lower-envelope comparator is expectation-level. At each accepted operating
point, deterministic floor/ceiling depth counts differ from the adaptive target
by less than one refinement-call cost:

- Cube: `-161,395` / `+103,565` FLOPs;
- PushT four-depth pilot: `-591,132` / `+59,300` FLOPs; and
- PushT binary: `-156,768` / `+493,664` FLOPs.

Outcome-independent seeded assignments show ordinary finite-assignment
variation around the analytic mixture. The saved schedules and per-transition
losses are in `numeric/finite_schedule_arrays.npz`; no one seeded draw replaces
the stronger analytic expectation comparator.

## Chronology retained

The four-depth PushT pilot remains formally negative: raw benefit over its
matched envelope is `0.003909796985`, but whitened benefit is
`-0.000309900129`. The later binary confirmation remains a separate, locked,
fresh result. The exploratory binary frontier does not retroactively validate
the four-depth pilot or turn any consumed point into confirmation.

## Evidence meaning for the paper

The saved data support a useful exploratory statement: **causal score rankings
beat the transition-independent lower envelope over a nontrivial interior
budget range in both Cube conservative thinning and PushT binary top-k.** They
do not support the stronger claim that the deployed gates implement an optimal
compute-price rule. The frontier can strengthen explanation and visualization,
but it cannot change the central confirmed claim without a new study—and the
paper-value rule separately determines that no such study is warranted.
