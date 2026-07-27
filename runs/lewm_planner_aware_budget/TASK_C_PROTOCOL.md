# Task C: fixed PushT CEM population-axis protocol

Status: frozen before opening or generating Task C grid outcomes.

## Fixed grid and validation

The complete grid is refinement depth `{0,1,2,4}` × population
`{64,128,300}`. Population labels are `p064`, `p128`, and `p300`; cell labels
are `d{depth}_p{population:03d}`. Values must be plain positive integers
(booleans and floats are rejected) and must be members of the frozen sets.

The elite rule preserves Task A's fraction `1/8`. It is computed by integer
round-half-up, `floor(population / 8 + 1/2)`, and is not independently
tunable. The resulting elite counts are `{64:8,128:16,300:38}`. Configuration
fails closed if the supplied population, label, or elite count disagrees, if
`elite >= population`, or if solver `num_samples/topk` disagree with the
validated boundary.

Everything else is frozen: 20 CEM iterations, horizon 5, receding horizon 5,
action block 5, batch size 1, Task A cost model/checkpoint/reset/controller and
Task B refiner checkpoint/action bounds. Population 128/depth 0 is the native
Task A regression anchor.

## Candidate randomization

Seed ownership is `(candidate_seed, start_id, replan_index)`. A project-owned
solver creates a fresh generator for each planning call. For each CEM
iteration it draws a maximum-population standard-normal innovation tensor of
shape `[batch,300,5,10]`; a population consumes its leading prefix. Candidate
zero is then replaced with the fitted mean, exactly as in Task A.

Thus iteration-zero transformed candidates are exact prefixes because their
initial distributions are equal. At later iterations only the standard-normal
innovations are promised to be prefixes; fitted distributions can diverge, so
transformed candidates need not be equal. No global Torch RNG state is read.
Distinct declared seeds must produce distinct streams.

## Measurements and order

The fixed start is Task A-compatible dataset row 0 with goal row 25. Grid calls
use a deterministic interleaved order: repetition-major, with a seeded
permutation (`26072631`) of all 12 cells in every repetition. Each cell has two
untimed warmups followed by seven timed planner calls. The solver/model are
constructed before warmup. Every call uses replan index equal to its
within-cell call counter so seed/counter ownership is explicit; raw records
are written only below the ignored `work/task_c/` root.

The planner-call timing boundary begins immediately before `solve` and ends
immediately after it. MPS is synchronized immediately before and after.
Checkpoint/model/dataset loading, input construction, and result
serialization are excluded. Median is NumPy's ordinary median; p95 is the
linear empirical quantile (`numpy.quantile(..., method="linear")`). Warmups
are excluded. Constant-work planner calls are never mixed with terminated
episode latency. A bounded one-replan smoke per cell records the real planner
path separately; no control-quality inference is authorized.

Frozen measurement identity:

- Apple M5 MacBook Pro (`Mac17,2`), 10 CPU cores, 32 GiB unified memory;
- macOS 26.5.2 (25F84), arm64;
- Python 3.10 environment used by Tasks A/B;
- PyTorch 2.12.0, NumPy 2.2.6, MPS available;
- base checkpoint SHA-256
  `0be0611227823a00b2f9acac299375a013bb23cd2a33b8ffa94a3bdb5e19b737`;
- refiner checkpoint SHA-256
  `b55521a9870e9e29be6e77d055756fa5c82fecb13d6596b13c242481f76f6a4d`.

## Executed-work and counted-FLOP ledger

For one batch-1 planning call with population `P`, iterations `I=20`, horizon
`H=5`, and selected depth `D`:

- candidate sequences evaluated = `P I`;
- predicted transition rows = base rows = `P I H`;
- base calls = `I H`;
- each active refiner stage has `I H` calls and `P I H` rows;
- goal encoder calls = `I`, rows = `I`;
- image encoder calls = `I`, rows = `I`;
- terminal-cost calls = `I`, rows = `P I`;
- sampling, update, and top-k invocations are each `I`;
- sampled innovation scalars = `300 I H 10` (maximum stream, independent of P);
- transformed candidate scalars = `P I H 10`;
- top-k values/indices = `elite I`; elite candidate scalars = `elite I H 10`.

Observed ledgers must exactly match these identities. A multi-replan smoke is
the sum of its actual completed planning calls; there is no assumed fixed
episode length.

Counted FLOPs use operation-level multiply/add conventions declared in the
aggregate. Dense linear operations count a multiply and add separately;
elementwise arithmetic is counted only where its exact executed tensor size is
known. The ledger separately covers base prediction, each refiner stage,
encoding, terminal squared-error cost, and CEM scale/shift and mean/variance
updates when their checkpoint/shape formulas are verified. Top-k comparisons,
random-number generation, indexing/gather, clipping (none), tensor movement,
allocation, Python/controller work, MPS synchronization, and device-kernel
implementation details are latency-only exclusions. Unless all material
operations become technically countable, results are named **counted FLOPs**,
never complete measured FLOPs, and include coverage/exclusions plus the
narrowest profiler/instrumentation unblocker. FLOPs and latency remain separate
axes.

## Completion and failure

Completion requires all 12 cells to execute finite real CEM/refiner calls,
exact prefix/randomness and work-ledger tests, finite actions/costs, the
population-128/depth-0 native-equivalence anchor, positive-depth cost/ranking
effects and exact stage counts at every population, seven valid synchronized
latencies per cell, and a componentized honest counted-FLOP result.

Failure is any identity/configuration/randomness violation, nonfinite result,
checkpoint mismatch, unsupported depth/population, anchor mismatch beyond the
declared numerical tolerance, missing cell, or malformed latency schema.
After Task C grid outcomes are opened, no population, elite, seed, order,
warmup, repetition, estimator, timing-boundary, or ledger rule may change.
Repairs to mechanics require an explicit protocol amendment made before
rerunning affected outcomes.
