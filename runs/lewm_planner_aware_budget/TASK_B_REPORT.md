# Task B final report

## Decision

Task B is complete for integration validity. The planner-independent PushT
refiner passed the frozen B3 offline gate and fixed depths 0, 1, 2, and 4 run
inside the installed CEM solver. This is not a new control-effect claim.

## Data and training

The bounded corpus contains disjoint WeakPolicy, expert, and executed
off-policy episodes. Fit/selection/evaluation contain 8,736/3,289/5,455
examples. Every source has explicit lengths 1, 2, and 3 and every
source×split×length cell exceeds 100 examples. Task A qualification episodes
were excluded. Unexecuted CEM candidates were never used as targets.

The canonical action is the PushT environment coordinate itself: five
chronological 2-D actions form one 10-D block. Real-transition training rows
are finite and within the declared `[-1,1]` environment range; out-of-range
expert rows were excluded. CEM candidates remain identity-coordinate
extrapolations and are not clipped inside the refiner.

The first aggregate-row pilot failed on off-policy raw MSE. Equal
source×length weighting fixed that imbalance without changing data or
architecture. A second pilot caught depth-4 regressions versus depth 2; frozen
whitened supervision and the preregistered monotonic penalty corrected it.

## Offline validity

Depth 4 is finite and better than base in raw and frozen-whitened latent MSE in
all nine held-out source×length cells. Eight cells exceed 1% improvement in
both metrics (the frozen requirement was five). Every supported depth
transition remains within the 1% regression limit.

All 18 episode-bootstrap upper 95% bounds for depth-4 relative raw/whitened
regression are below 1%; the least favorable is expert length 1 raw at
`-0.73%`. The strongest effects occur off-policy: raw improvements are
11.4–13.9% and whitened improvements are 28.5–30.5%.

The compact checkpoint is 1,301,521 bytes with SHA-256
`b55521a9870e9e29be6e77d055756fa5c82fecb13d6596b13c242481f76f6a4d`.
Dense references cover every source, prefix length, and supported positive
depth. Batch-512 synchronized MPS median latency was 1.60/2.06/2.29/2.63 ms at
depths 0/1/2/4.

## CEM integration

The project-owned adapter leaves the installed package unchanged. Depth zero
delegates exactly to the native Task A `get_cost`. Positive depths use explicit
left padding and masks at rollout prefix lengths `[1,2,3,3,3]`, preserve
candidate/batch shapes, run frozen under inference mode, and record base/stage
calls and rows.

The installed CEM solver smoke used one common start, 32 candidates, three
iterations, horizon five, and common seed `26072610`. Including one fixed
candidate probe, each depth performed 20 base calls and 640 base rows. Positive
stage call/row counts were exactly `[20]*depth` and `[640]*depth`. Refinement at
all positive depths changed finite candidate costs and rankings. Selected
actions did not change in this deliberately small smoke, so no planning
improvement is claimed.

The FLOP ledger remains intentionally incomplete: refiner dense-linear FLOPs
can be derived from the frozen architecture, while image/goal encoding, base
prediction, CEM sampling/top-k/statistics, tensor movement, and policy overhead
are not presented as a complete planner FLOP total.
