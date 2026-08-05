# Manuscript handoff

## Bottom line

Keep the paper’s current bounded empirical identity. The marginal-value lens
improves the explanation of *why* transition-specific allocation can help and
gives an exact interpretation for binary equal-cost top-k allocation. It does
not support retitling or recasting LeWM as a new optimal budget-allocation
framework, and it does not justify another cohort.

The strongest justified framing is:

> LeWM tests whether a non-anticipatory transition score can reassign a fully
> counted refinement budget more effectively than the strongest transition-
> independent fixed-depth envelope. The answer is positive at locked operating
> points in fresh Cube and PushT confirmations. Binary PushT additionally admits
> an exact score-ranked top-k budget interpretation; multistage Cube remains an
> empirically validated heuristic rather than a proven myopic optimum.

## Exact suggested contribution bullets

If revising the introduction bullets, use language at this strength:

1. **Matched-budget causal allocation.** “We provide two fresh-cohort tests in
   which a non-anticipatory transition score reallocates a fully counted
   refinement budget and improves both raw and whitened next-latent prediction
   over the strongest transition-independent fixed-depth envelope: frozen Cube
   PlanOracle and an environment-specific binary PushT confirmation.”
2. **Budget/accounting clarity.** “We explicitly charge routing overhead and
   separate an expectation-level lower-envelope comparator from exact finite-
   cohort allocation and executable integer schedules, isolating where compute
   is spent from how much counted arithmetic is spent.”
3. **Boundaries and chronology.** “We retain the formally negative four-depth
   PushT pilot, partial Cube shift results, exploratory-only compute frontiers,
   unsupported planning bridge, and negative latency evidence, delimiting the
   claim to predictive fidelity at matched counted compute.”

Do not add a contribution bullet claiming a new marginal-value theorem,
compute-price router, adaptive world model, or general quality/compute frontier.

## Compact formal paragraph suitable for methods or discussion

> With one optional equal-cost refinement and a fixed cohort budget of $K$
> calls, selecting the $K$ largest conditional expected loss reductions is
> optimal by an exchange argument; a threshold is the corresponding Lagrangian
> form. This gives the binary PushT score a top-k *ranking interpretation* if the
> score is monotone in conditional gain. It does not make the deployed score a
> calibrated compute price: the gate takes the minimum of two differently
> standardized gain heads. For multistage Cube, an optimal stopping rule also
> contains the option value of later refinements, so a myopic immediate-gain
> threshold is not generally optimal.

This paragraph is a clarification, not a claim of theoretical novelty.

## Figure and table recommendation

The existing fresh-confirmation effect figure should remain primary. If space
permits, add the generated frontier as an explicitly exploratory secondary or
appendix figure:

- source: `figures/quality_compute_frontiers.pdf`;
- four panels: Cube conservative thinning and PushT binary exact score top-k,
  each with raw and whitened loss;
- required caption qualifier: “All intermediate budgets reuse consumed outcome
  cohorts and are exploratory; the star marks the only accepted operating point
  in each study. Cube’s curve only thins originally stored continuations and is
  not a full compute-price sweep.”

The key visual takeaway is honest and useful: both settings have an interior
range above the transition-independent lower envelope, while gate-only and
near-full-depth extremes show why routing overhead and endpoint choice matter.
Do not present the connecting lines as simultaneous confirmation.

If adding one compact table instead, use the three-row distinction from
`FRAMEWORK.md`: expectation budget, exact finite-cohort budget, and executable
scheduling. That table is less likely than the exploratory curves to be
misread as a new confirmation result.

## Exploratory statements that may be added

These statements are supported only with an explicit consumed-cohort label:

- “In a frozen-score exploratory sweep, both losses favored causal allocation
  at eight nonzero Cube thinning settings and across 10–90% optional allocation
  in binary PushT.”
- “At the confirmed PushT budget, the causal score captured 62.0% of raw and
  53.0% of whitened outcome-oracle allocation uplift over random assignment.”
- “The score ranks gains more reliably than it calibrates their scale; endpoint-
  head calibration slopes are generally below one.”

No confirmatory manuscript claim changes. The accepted Cube and PushT effect
sizes, lower bounds, and terminal labels remain exactly as before.

## Required caveats

- The existing score is not one scalar expected loss reduction divided by cost.
- The binary top-k proposition assumes equal optional costs, additive loss, and
  scores available before allocation; unequal costs give knapsack.
- Multistage myopic allocation ignores future option value unless additional
  conditions hold.
- Cube cannot be swept more permissively from saved scores because later-stage
  scores were generated only for frozen-policy reached rows.
- Every new curve and calibration result consumes previously opened outcomes and
  is exploratory.
- The outcome oracle is noncausal diagnostic headroom, not an attainable method.
- Gate-overhead sensitivity is counted-FLOP accounting, not latency or energy.
- PushT remains environment-specific and all WeakPolicy episodes had zero task
  success.
- No planning, control, universal robustness, direct Cube-to-PushT transfer, or
  wall-clock speedup claim follows.

## Literature wording

Use “complementary to” or “a matched-compute empirical test” relative to ACT,
PonderNet, budgeted early exit, Mixture-of-Depths, LoopWM, and adaptive
imagination. Make no “first” claim. `LITERATURE_POSITIONING.md` gives primary
links and the exact novelty boundary.

## Fresh-study decision

No fresh cohort was run. The frozen paper-value rule failed because the proposed
formulation is established in broad form and is not genuinely equivalent to the
deployed multistage gate. This is a terminal, successful stop decision.
