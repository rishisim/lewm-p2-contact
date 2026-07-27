# Task C population-axis decision record

## Decision

The population axis is mechanically complete and auditable for the fixed
depth `{0,1,2,4}` × population `{64,128,300}` grid. All 12 cells executed the
real MPS CEM/refiner path with the frozen 20 iterations, horizon/receding
horizon/action block 5, batch size 1, base/refiner checkpoints, and Task A cost
path. This is a compute-mechanics result only; it makes no control-effect or
success claim and introduces no dispatcher.

Population/elite pairs are frozen to `64/8`, `128/16`, and `300/38` by the
preregistered 1/8 round-half-up rule. Invalid, ambiguous, unsupported, and
population/top-k-inconsistent inputs fail closed.

## Randomness and regression anchors

Every planning call owns its seed through
`(candidate_seed,start_id,replan_index)`. Each iteration draws `[1,300,5,10]`
standard-normal innovations and each population consumes a prefix. Unit tests
establish repeatability, global-RNG and run-order independence, distinct-seed
independence, and exact 64⊂128⊂300 prefixes for all 20 iterations and multiple
replans. Iteration-zero transformed candidates are exact prefixes; later
transformed candidates may diverge because population-specific fitted
distributions diverge.

For the same population-128 candidate tensor, the depth-zero boundary and
native Task A `get_cost` are bitwise equal (`max |Δcost| = 0`). Every positive
depth changes finite costs and candidate rankings at all three populations.
Exact stage-call/row identities pass for every cell.

## Exact work per planning call

With population `P`, elite count `K`, iterations `I=20`, and horizon `H=5`,
each call observes:

- `P I` candidate sequences and `P I H` predicted/base rows;
- `I H = 100` base calls;
- 100 calls and `P I H` rows for each selected refiner stage;
- 20 image encodes and 20 goal encodes, one row each;
- 20 terminal-cost calls and `P I` terminal-cost rows;
- 20 sampling, update, and top-k invocations;
- one `[1,300,5,10]` innovation tensor per iteration, then
  `[1,P,5,10]` transformed candidates and `[1,K,5,10]` elites.

The one-replan bounded smoke for each cell has finite actions, costs, and
metrics and exactly matches these identities. The qualified Task A CEM does
not clip its fitted mean to the declared `[-1,1]` action space: all 12 smoke
means contain values outside that range (recorded min/max are in the aggregate).
Task C preserves this behavior rather than changing planner semantics. The
controller/environment handling remains the Task A path; Task D must not infer
that the planner itself enforces bounds.

## Counted FLOPs and synchronized latency

Counted FLOPs cover hash-pinned executed dense/conv operations (one
multiply-accumulate is two FLOPs) and explicit attention `QKᵀ`/`AV` matmuls,
refiner dense/residual arithmetic, terminal
squared error, candidate scale/shift, and elite mean/std arithmetic. They are
integer operation counts, not complete measured FLOPs. Activation,
normalization, attention softmax/scaling/masking, other base-model elementwise
operations, RNG, top-k comparisons, gather/indexing, allocation/movement,
Python/controller work, synchronization, and device-kernel overhead remain
explicit latency-only exclusions. The
smallest unblocker for complete matched-FLOP evaluation is operator-complete
MPS profiler counters for SDPA, normalization, activation, RNG, and top-k.

Latency is seven synchronized constant-work planner calls per cell after two
excluded warmups, in the preregistered deterministic interleaved order. Values
below are milliseconds; p95 uses NumPy's linear empirical quantile.

| depth | population | elites | counted FLOPs | median ms | p95 ms |
|---:|---:|---:|---:|---:|---:|
| 0 | 64 | 8 | 496,478,859,240 | 462.282 | 487.425 |
| 0 | 128 | 16 | 857,093,356,520 | 677.391 | 680.331 |
| 0 | 300 | 38 | 1,826,244,819,960 | 1,317.202 | 1,339.706 |
| 1 | 64 | 8 | 500,625,240,040 | 671.821 | 678.844 |
| 1 | 128 | 16 | 865,386,118,120 | 869.893 | 873.697 |
| 1 | 300 | 38 | 1,845,680,979,960 | 1,512.445 | 1,530.157 |
| 2 | 64 | 8 | 504,771,620,840 | 689.467 | 697.927 |
| 2 | 128 | 16 | 873,678,879,720 | 887.919 | 892.273 |
| 2 | 300 | 38 | 1,865,117,139,960 | 1,518.910 | 1,540.346 |
| 4 | 64 | 8 | 513,064,382,440 | 722.477 | 738.185 |
| 4 | 128 | 16 | 890,264,402,920 | 921.636 | 930.043 |
| 4 | 300 | 38 | 1,903,989,459,960 | 1,566.088 | 1,603.061 |

## Task D readiness

The second axis is ready for Task D configuration, deterministic candidate
construction, exact executed-work budgets, and latency-indexed comparisons.
It is **not yet ready for a claim of complete matched FLOPs**: the named
operator exclusions are material. Task D may use the present quantity only as
`counted_flops` with its coverage metadata, or first add the narrow
operator-complete MPS instrumentation above. Any bounded-action change would
also be a separately preregistered planner change, not a Task C repair.

Machine-readable details, component breakdowns, run order, smoke ledgers,
anchor, effect checks, coverage, and exclusions are in `task_c_results.json`.
Raw timings remain ignored under `work/task_c/`.
