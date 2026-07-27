# Task D paired closed-loop PushT result

## Decision

Do **not** proceed to a two-axis Task E dispatcher on this evidence. The full
presealed 24-start × 2-candidate-seed × 12-cell evaluation completed with 576
finite records and no replacements, but the apparent per-start package
heterogeneity was not reproducible across candidate seeds. Refinement was
harmful on average, and cross-seed constrained joint-oracle gains were small,
direction-dependent, and failed the preregistered joint-versus-one-axis
margins.

The appropriate next step is to improve the refinement/value target before
dispatcher training. A population-only controller is not supported as a
scientific follow-on either: its mean extreme-axis effect was uncertain and
the joint oracle did not consistently exceed the population-only opportunity.

## Execution integrity

The Task C audit sealed its protocol, report, configuration, result, solver,
runner, base checkpoint, and refiner checkpoint hashes before Task D outcomes.
The Task D cohort reconstruction excluded the union of Task A source episodes,
Task B expert fit/selection/evaluation episodes, and Task C's row-zero source
episode. Pilot and sealed cohorts used unique, disjoint source episodes. Exact
rows, goals, states, exclusions, rules, seeds, and hashes are in
`task_d_cohort.json`.

A two-start, one-seed, complete-grid pilot found and repaired one Task D-only
list-to-array adapter incompatibility before sealed execution. No cell or
analysis choice changed. The repaired pilot completed all 24 records with
finite outcomes and exact ledgers. The outcome-blind midpoint gate then found
288/288 tranche-1 records valid, zero failures, and 1.4 MiB of compact data, so
the frozen rule required tranche 2. The final sealed ledger contains all
576/576 unique `(start,seed,cell)` records, zero mechanical/nonfinite failures,
and exact per-call Task C identities.

The environment/policy semantics yielded one or two planner calls per 50-step
episode because each planned horizon expands to 25 raw actions. The
preregistered ten-call budget charge remains a conservative fixed allowance;
observed work is also reported and was never used to make a package eligible.
All packages preserved Task A's unclipped fitted CEM mean.

## Complete quality/work/latency frontier

Return is higher-better and cumulative cost lower-better. Each mean pools 24
start clusters and two repeated candidate seeds. Planner and episode medians
are synchronized milliseconds; episode latency is outcome-dependent.

| package | success | normalized return | cumulative cost | counted FLOPs/call | planner ms | episode ms |
|---|---:|---:|---:|---:|---:|---:|
| d0-p64 | .854 | .531 | 4,805 | 496,478,859,240 | 469 | 515 |
| d0-p128 | .896 | .565 | 4,409 | 857,093,356,520 | 661 | 709 |
| d0-p300 | .896 | **.569** | **4,280** | 1,826,244,819,960 | 1,299 | 1,345 |
| d1-p64 | .812 | .519 | 4,956 | 500,625,240,040 | 683 | 728 |
| d1-p128 | .875 | .550 | 4,527 | 865,386,118,120 | 868 | 912 |
| d1-p300 | .812 | .548 | 4,595 | 1,845,680,979,960 | 1,471 | 1,522 |
| d2-p64 | .854 | .543 | 4,670 | 504,771,620,840 | 698 | 741 |
| d2-p128 | .875 | .530 | 4,684 | 873,678,879,720 | 891 | 933 |
| d2-p300 | .792 | .538 | 4,695 | 1,865,117,139,960 | 1,517 | 1,562 |
| d4-p64 | .792 | .514 | 5,103 | 513,064,382,440 | 733 | 776 |
| d4-p128 | .812 | .532 | 4,758 | 890,264,402,920 | 935 | 980 |
| d4-p300 | .771 | .524 | 4,805 | 1,903,989,459,960 | 1,560 | 1,608 |

These operation counts are Task C's incomplete counted-work proxy, not complete
or measured FLOPs. Material exclusions include activations, normalization,
attention softmax/scaling/masking, other base elementwise operations, RNG,
top-k comparisons, gather/indexing, movement/allocation, Python/controller
work, synchronization, and kernel overhead. Exact call/row ledgers, counted
work, constant-work Task C latency, and outcome-dependent episode latency are
kept distinct.

## Paired effects and interaction

The average depth-4 minus depth-0 normalized-return effect was **-0.0318**
(start-bootstrap 95% CI -0.0493 to -0.0162). The average population-300 minus
population-64 effect was 0.0179 (95% CI -0.0333 to 0.0602). The extreme
depth-by-population difference-in-differences was -0.0283 (95% CI -0.0789 to
0.0236). Thus deeper refinement did not contribute downstream value beyond
population; it reduced return on average.

All 30 preregistered within-population depth and within-depth population
contrasts, paired permutation results, success risk differences, and
deterministic resampling hashes are preserved in `task_d_results.json`.
Negative cells were not filtered. With only 24 start clusters, task-success and
interaction precision remains limited; continuous return is the more sensitive
co-primary endpoint here.

## Heterogeneity and candidate-seed stability

Per-seed winners were diverse (entropy 2.117 nats across ten observed winning
packages), but that diversity was sampling-sensitive:

- exact package-winner agreement across seeds: **0/24**;
- mean within-start Kendall rank correlation: **0.077**;
- mean pairwise rank-sign agreement: **0.539**;
- preregistered reproducible-preference fraction: **0.104**;
- between-start centered-package variance: 0.00539;
- within-start/across-seed variance: 0.00841;
- median selected-action-block cross-seed L2 difference: 6.84.

Within-start sampling variance exceeded between-start package heterogeneity.
Accordingly, noisy per-start winners are not stable labels.

## Budget-constrained opportunity

At the largest counted-work and latency anchors, the A→B cross-seed joint
oracle exceeded its train-selected fixed package by 0.039 return, but the
reverse B→A direction was **-0.018**. This misses the preregistered 0.05
joint-versus-fixed/mixed margin even at the point-estimate level. At smaller
budgets, joint-minus-fixed ranged from -0.016 to 0.018. Start-bootstrap
comparison intervals include zero (the work-budget lower bounds reach -0.070),
so uncertainty does not rescue an opportunity claim.

The joint oracle also failed to consistently exceed both one-axis oracles. At
the largest anchor, joint minus population-only was -0.021 in one direction
and 0.016 in the other; joint minus depth-only was 0.010 and 0.022. Smaller
budgets were likewise inconsistent, and several joint allocations
underperformed executable mixtures on held candidate randomness.

For every work budget and separately every latency budget, the machine-readable
record includes the held-seed fixed package, executable transition-independent
mixture and slack, joint multiple-choice allocation and independently
recomputed slack, depth-only and population-only allocations, non-executable
reference scope, and 9,999 histogram-preserving assignment controls. The
in-sample oracle is not used for the decision.

## Task E mapping and limitations

The preregistered conjunction fails on reproducibility, joint baseline margin,
joint-versus-one-axis margin, and refinement value. Task E should stop in its
two-axis dispatcher form. The evidence points to refinement/value-target work
before any new dispatcher experiment, rather than extracting labels from these
noisy winners.

This is a 24-start, two-seed PushT result. It cannot establish small binary
interactions, exhaust CEM sampling variability, convert incomplete counted
work into complete FLOPs, or claim adaptive-control benefit. Latency budgets
use Task C's synchronized constant-work medians; observed episode latency is
outcome-dependent and is not causal evidence for package quality.
