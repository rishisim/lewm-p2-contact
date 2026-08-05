# Existing-artifact exploratory analysis protocol

Status: **frozen before execution** on 2026-07-18. All analyses below consume
already-opened confirmation or pilot cohorts and are exploratory. They cannot
create or upgrade a confirmation claim.

## Sources and non-regeneration rule

- Cube: the accepted v5-v004 lossless execution array, compiled gate, gate-fit
  object, whitening object, and counted-operation ledger.
- PushT four-depth chronology: the accepted replication-pilot evaluation array,
  evaluation inputs, frozen gate, and fit whitening/action statistics.
- PushT binary: the accepted 240-episode confirmation lossless array together
  with the same frozen pilot gate and fit statistics.
- No simulator, encoder, predictor, refiner, or gate will be executed. Source
  artifacts are read-only.

## Fixed exploratory grids

1. **Cube conservative frontier.** Use retention fractions
   `0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1`. At every reached stage,
   retain the highest stored gate scores among rows that the frozen policy
   originally continued, with the retained count equal to the stated fraction
   rounded to the nearest integer (half up). This never asks for an unstored
   score. The endpoint `1` must reproduce the frozen depth choices. This is a
   finite-cohort top-k *thinning diagnostic*, not a compute-price sweep and not
   a newly deployable policy.
2. **PushT binary frontier.** Rank every transition by its stored causal stage-1
   score. Evaluate optional-depth-2 counts corresponding to fractions
   `0, 0.1, ..., 0.9`, the frozen count, and the largest integer count whose
   counted cost (including one gate evaluation per row) does not exceed the
   gate-free fixed-depth-2 cost. No outcome is used in the ranking or grid.
3. **Gate overhead.** Reprice the frozen policies at multipliers
   `0, 0.5, 1, 2, 5, 10, 25, 50` of the accepted gate FLOP ledger; losses and
   allocations remain fixed.
4. **Calibration.** Use deterministic equal-count deciles of each endpoint head's
   predicted marginal gain. Report bin means, least-squares slope/intercept, and
   Spearman association against the realized next-refinement gain.
5. **Oracle headroom.** For PushT binary, at every fixed optional-call count,
   compare the learned score ranking with the outcome-oracle top-k ranking,
   separately for raw and fit-whitened loss. For multistage Cube and the PushT
   pilot, report stage-local top-k headroom only among rows reached by the frozen
   policy. Do not call this a globally optimal prefix-constrained oracle.
6. **Finite scheduling.** For each frozen operating point and endpoint, take the
   selected analytic lower-envelope depth pair; use the floor and ceiling of its
   fractional upper-depth count and an outcome-independent seeded permutation.
   Report realized loss and counted-cost slack. These schedules diagnose the
   analytic-expectation versus executable-integer distinction.

## Chart contract

- Decision question: does the learned causal allocation dominate the strongest
  transition-independent fixed-depth envelope over a nontrivial counted-compute
  interval, and how much outcome-oracle allocation value remains?
- Primary chart: four small multiples (Cube/PushT binary by raw/whitened loss),
  line and marker encodings, x-axis equal to extra counted MFLOPs per transition
  above gate-free fixed depth 1, and the accepted operating point marked with a
  star. The Cube line is explicitly labeled as conservative thinning.
- Diagnostics: allocation histograms, predicted-versus-observed marginal-gain
  deciles, and gate-overhead sensitivity. No dual axes or truncated bar axes.

## Interpretation limits

- Consumed-cohort curves estimate feasibility and descriptive range; they are
  look-after-outcome exploratory analyses even though the ranking and grids are
  outcome-independent.
- The two co-primary losses remain separate. The frozen gates use the minimum of
  two standardized predicted-gain heads; that score is not a calibrated scalar
  expected loss reduction per FLOP.
- A negative or mixed result is retained as evidence, not treated as an
  implementation defect.
