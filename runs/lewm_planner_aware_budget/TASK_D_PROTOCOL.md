# Task D: paired closed-loop PushT fixed-grid protocol

Status: frozen before pilot or sealed Task D outcomes.

## Scientific question and fixed packages

This experiment asks whether closed-loop PushT outcomes vary reproducibly across
fixed `(refinement depth, CEM population)` packages enough to justify a
two-axis dispatcher. It does not train a dispatcher. The complete grid is depth
`{0,1,2,4}` by population/elite `{64/8,128/16,300/38}`, with 20 CEM
iterations, horizon 5, receding horizon 5, action block 5, Task A's base/cost
model and controller/reset semantics, Task B's refiner, and Task C's
innovation-prefix solver.

Task C's fitted CEM mean is not clipped to the declared `[-1,1]` action space.
That qualified behavior is preserved uniformly and is not repaired here.
Common candidate randomness is owned by
`(candidate_seed,start_dataset_row,replan_index)`: depths share innovations and
populations consume prefixes of a maximum-population stream. Only iteration
zero transformed candidates must be prefixes; later fitted distributions can
diverge.

## Prospective feasibility and staged execution

Task A's held-out cumulative-cost SD was 2,994.064. With 24 start clusters, a
two-sided 5% paired comparison has approximate 80% minimum detectable
differences of 1,780 cost units at paired correlation 0.5 and 2,518 at
correlation 0. Task success near 0.75 has roughly 0.17 half-width at this size,
so binary interaction inference is explicitly precision-limited.

Task C's 12 synchronized median planner-call latencies sum to 11.8175 seconds.
Ten calls per non-terminated 50-step episode imply about 94.5 minutes of
planner time for 24 starts by two candidate seeds, before environment and I/O.
The bounded full design is therefore feasible. Two disjoint pilot episodes run
all 12 cells with one pilot seed and may be inspected only for mechanics,
schema, restart integrity, and runtime. They cannot tune cells or analysis.

The sealed cohort has 24 unique source episodes and two candidate seeds. Tranche
1 contains the first 12 presealed starts and tranche 2 the remaining 12. After
tranche 1, execution continues automatically unless projected total wall time
exceeds six hours, compact records project above 2 GiB, the device/dependency
stack is unavailable, or repeated mechanical/nonfinite failure exceeds 5%
after one code repair and deterministic restart. No endpoint or package
outcome enters this gate. Partial failures are retained without replacement.

## Cohort, order, and restart contract

The cohort is sampled before outcomes from `pusht_expert_train.lance`, one row
per source episode, with a 25-step goal. Every source episode used by Task A's
smoke/tuning/heldout reconstruction, Task B's expert
fit/selection/evaluation reconstruction, or Task C row-zero smoke is excluded.
Pilot and sealed episodes are disjoint. The machine-readable ledger records
rows, episode identifiers, start/goal steps and states, dataset identity,
reconstruction rules, hashes, and exclusions. There is no outcome replacement.

Each `(phase,start,candidate_seed)` block uses a deterministic permutation of
the 12 cells derived from order seed 2026072611. Common reset state, goal, and
environment seed are reused across its cells. Compact records are written
atomically, keyed by `(phase,row,seed,cell)`, and an existing valid key is
skipped. Resume must reproduce the same schedule and record content except
measured latency. Raw trajectories, videos, per-call tensors, and logs remain
under ignored `work/task_d/`.

## Endpoints, failures, and inferential unit

The start episode is the inferential unit; candidate seeds are repeated
measurements, not independent starts. Co-primary endpoints are:

1. Task success: Task A PushT termination within at most 50 environment steps.
2. Normalized return:
   `1 - cumulative_cost / (initial_distance * 50)`, higher is better, where
   cumulative cost sums full-state Euclidean cost over actually executed steps.

Cumulative task cost (lower is better) is reported alongside normalized return.
Final task cost and episode length are secondary. Early termination is part of
the outcome; no post-termination outcome imputation is added. Planner-call and
end-to-end episode latency are synchronized and reported separately, with
early-termination latency labeled outcome-dependent.

Failures/nonfinite values are recorded without replacement and count as
failure for success. Continuous analyses report missingness plus complete-case
and conservative worst-observed sensitivity; no failed start is silently
dropped from the failure-rate endpoint.

## Frozen inference and multiplicity

All 12 marginal cell estimates and paired contrasts are retained, including
negative effects. Continuous inference uses start-cluster bootstrap and paired
start-level permutation; binary inference uses paired risk differences with
the same whole-start resampling. Bootstrap and permutation each use 9,999
draws, seeds 2026072691 and 2026072692. Two global 11-df package tests are Holm
adjusted across co-primary endpoints. The 30 simple contrasts (depths within
population and populations within depth) are Holm adjusted within endpoint.
Depth, population, and interaction omnibus terms form a separate Holm family
within endpoint. Candidate-seed fixed effects are included; start clusters
contain both seeds and all cells.

## Heterogeneity and out-of-seed stability

Candidate seeds 2026072601 and 2026072602 have symmetric train/evaluate roles.
Package rankings selected on one seed are evaluated on the other, then roles
reverse and estimates average. The declared normalized-return tolerance is
0.03. Reports include tolerance-aware winner entropy, pairwise rank/sign
stability, selected-action block L2/max-absolute stability, between-start and
within-start/across-seed variance, and the fraction whose training-seed
preference has held-seed regret at most 0.03 and beats the held-seed strongest
fixed package by at least 0.03. An in-sample winner is never treated as a
stable label.

## Work, latency, and constrained opportunity

Exact calls and executed rows use Task C identities. `counted_flops` is only
Task C's incomplete dense/conv/attention-matmul/refiner/CEM arithmetic proxy;
activation, normalization, softmax, RNG, top-k, indexing/movement, Python,
controller, synchronization and kernel overhead remain excluded. No
exact-complete-FLOP or matched-FLOP claim is authorized.

Counted-work budgets per allowed planner call are 513,064,382,440;
890,264,402,920; and 1,865,117,139,960. Separate synchronized-latency budgets
are 0.690, 0.900, and 1.530 seconds per allowed call. Eligibility/allocation
charges ten calls per start so early termination cannot buy package access;
observed executed work and latency are also reported.

For each budget and cross-seed direction, analyses compare the strongest
train-seed-selected feasible fixed package; an executable,
transition-independent mixture with aggregate slack; a non-executable analytic
convex-envelope reference; and a per-start multiple-choice hard-budget joint
oracle selected on the train seed and evaluated on the held seed. Joint
allocation is independently ledger-verified and compared with brute force in
bounded tests. Depth-only (one global population) and population-only (one
global depth) constrained oracles use the identical budget. Histogram controls
permute the joint oracle's exact package counts over starts using seed
2026072693. In-sample oracles are optimistic diagnostics only.

## Frozen Task E decision

Two-axis Task E is supported only if all hold with start-bootstrap uncertainty:

- at least two materially different-work frontier packages are within 0.03
  held-seed mean return of the best fixed package;
- the lower 95% bound on reproducible preference fraction exceeds 0.25;
- the held-seed joint oracle exceeds both the fixed and executable mixture by
  at least 0.05 return, with lower bound above zero;
- it exceeds each one-axis oracle by at least 0.02, with lower bound above zero;
- positive-depth refinement adds at least 0.02 over its depth-zero population
  counterpart, with lower bound above zero;
- effects survive fixed-call work sensitivity and are not solely candidate
  sampling, latency, or early termination;
- success risk-difference lower bound is no worse than -0.05 and worst-quartile
  return degradation no worse than -0.05.

Otherwise: prefer a population-only controller if only that axis passes, a
depth-only controller if only depth passes, stop Task E if neither axis or
reproducibility passes, and improve refinement/value targets first if
refinement is harmful or unstable.
