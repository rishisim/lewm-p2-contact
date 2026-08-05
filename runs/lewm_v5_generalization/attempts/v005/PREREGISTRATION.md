# LeWM V5 zero-shot one-factor generalization preregistration

Status: normative v005 text, frozen before every new regime smoke and target
episode.

This study consumes the process-valid V5 terminal package at
`runs/lewm_v5_readiness_program/v5_package_versions/v004`. It never reruns,
edits, overwrites, or regenerates V5. Candidate
`stage_dual_r0.01_q0.85`, thresholds
`0.19465729885363534`, `0.1568665868361991`,
`0.15153321811129283`, compiled heads, PlanOracle-native whitening, base
model, refiner, solver, causal features, gradient boundaries, and compute
prices are byte-frozen.

The bounded matrix contains exactly three one-factor shifts:

1. MarkovOracle replaces PlanOracle; action noise remains 0.1 and every other
   DGP setting remains fixed.
2. PlanOracle action noise changes only from 0.1 to 0.2.
3. PlanOracle random-action probability changes only from 0 to 0.1.

Each regime has six fresh excluded smoke episodes followed by exactly 3,000
fresh target episodes (114,000 modeled rows). There is no sequential
expansion. Episode IDs and environment, policy, oracle-NumPy, action-space,
bootstrap, comparator, histogram, and qualification seeds are preassigned,
mutually disjoint, and checked against prior recorded runs and V5 V001–V004.
Two hundred replacement tuples per regime are frozen. Replacement is allowed
only for a mechanical exception before artifact completion. Target, loss,
contact, motion, phase, reward, and success cannot affect replacement,
exclusion, retention, stopping, or terminal mapping.

Power uses only the consumed V5 episode-level co-primary contrasts. The family
is three regimes by two endpoints. Bonferroni one-sided alpha is 0.05/6. The
fixed rule selects the smallest 500-episode grid point whose union-bound
family-power lower bound is at least 0.95 when only 35% of the V5 effect
remains. That point is 3,000 episodes per regime, with calculated lower bound
0.953835915981. Sensitivities are immutable; power cannot guarantee
support under shifted DGPs.

Within each regime and endpoint, the primary comparator is the strongest
transition-independent analytic mixture over fixed depths 1–4 at exactly the
adaptive total counted FLOPs, including reached feature and dual-head gate
overhead. Controls are seeded weakly-more-compute allocation, fixed depth 1,
and within-episode exact-call-histogram randomization. Raw MSE and the unchanged
V5 PlanOracle-native-whitened endpoint are retained; the latter is called
fixed-whitened under shifts to emphasize that no whitening is recalibrated.

All 9,000 raw target episodes and all execution traces must be complete and
hash-sealed before target arrays are first opened. The episode is the
bootstrap unit. Exactly 20,000 fixed-seed replicates produce two-sided 95%
individual intervals and Bonferroni one-sided lower bounds at quantile 0.05/6.
Every regime is reported separately.

A regime-endpoint claim is supported iff its simultaneous lower bound is
strictly positive. All six supported maps to
`zero_shot_generalization_supported`; one through five maps to
`zero_shot_generalization_partial`; zero maps to
`zero_shot_generalization_failed`. Any required integrity failure has
precedence and maps to `generalization_execution_invalid`. Auxiliary controls,
rank signs, heterogeneity, latency, energy, and post-hoc fields cannot alter
the decision.

Per regime, reporting includes stagewise score/next-stage-gain rank signs and
reached counts; regime heterogeneity; call-depth, gate-score, causal-feature,
and solver-gain shifts; exact compute-price implications; routing calibration;
base and refiner calls; gate evaluations; feature, head, and total FLOPs;
non-FLOP operations; synchronized latency; and energy availability. Contact,
motion, phase, reward, and success may be opened only after the immutable
independent terminal decision and only for interpretation.

An independent implementation rehashes inputs and sealed sources and
recomputes calls, losses, exact compute, comparators, all 20,000 replicates,
simultaneous bounds, diagnostics, chronology, and mapping. A process-valid
partial or failed result is terminal and cannot be repaired or retried.
Version-forward repair is permitted only after a zero-target-outcome
procedural invalidity with exact source/AST/object/hash evidence and unchanged
science.
