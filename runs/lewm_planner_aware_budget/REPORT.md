# Fixed PushT CEM qualification

## Decision

`fixed_cem_qualified`. A fixed, non-adaptive CEM planner using population 128,
20 CEM iterations, 16 elites, horizon 5, and receding horizon 5 passed the
prospectively frozen competency screen on 20 held-out PushT starts: 15/20
successes (75%; Wilson 95% interval 53.1%–88.8%). This establishes a competent
local control baseline. It does not establish any adaptive-computation benefit.

## Locked path

- Checkpoint: local `pusht/lewm_object.ckpt`, SHA-256
  `0be0611227823a00b2f9acac299375a013bb23cd2a33b8ffa94a3bdb5e19b737`.
  Its 303 state-dict tensors are exactly equal to all 303 tensors in the local
  released `quentinll/lewm-pusht` weights (weights SHA-256
  `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`).
- Dataset: local `pusht_expert_train.lance`, Lance version 2191, 2,336,736
  rows. No checkpoint or dataset was downloaded.
- Reset: the tracked evaluator selects a dataset row, calls installed PushT
  `_set_state(state)`, and targets `goal_state` 25 rows later through
  `_set_goal_state`.
- Success: installed PushT termination, requiring joint agent/block position
  error below 20 and wrapped block-angle error below `pi/9`, within 50
  environment steps.
- Planner: installed `stable_worldmodel.solver.CEMSolver`, closed-loop
  `WorldModelPolicy`, five 2-D raw actions per model action block, MPS device,
  fixed candidate seed `26072610`. Evaluation used `stable-worldmodel 0.1.0`,
  `stable-pretraining 0.1.7`, PyTorch 2.12.0, and Python 3.10.20.

`PROTOCOL.md` and `config.json` were frozen before new outcomes. The two smoke,
six tuning, and twenty held-out rows use distinct seeds and disjoint row IDs;
all 20 held-out starts also came from distinct source expert episodes. No
outcome-based replacement occurred.

The only prior closed-loop PushT artifact found locally was the untracked
historical evaluator log under the Stable WorldModel cache: the released
300-population/30-iteration config recorded 14/15 successes (and an earlier
5/5 run), but it had no frozen start ledger or dense per-start metrics. It was
treated as capability evidence only, not merged into this qualification.

## Bounded tuning

The cheap 64-population/3-iteration smoke was mechanically valid on 2/2 starts
and had 0/2 successes. It served only to validate loading, reset, action shape,
finite metrics, and latency accounting.

| population | iterations | elites | tuning success | mean task cost | median latency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 10 | 8 | 3/6 | 6515.7 | 469 ms |
| 128 | 10 | 16 | 5/6 | 4752.8 | 682 ms |
| **128** | **20** | **16** | **5/6** | **4303.6** | **1323 ms** |
| 300 | 15 | 30 | 3/6 | 6510.7 | 3575 ms |
| 300 | 30 | 30 | 4/6 | 5926.6 | 3086 ms |

The selected arm tied for best tuning success and won the frozen second
tie-break (lower mean cumulative task cost). `fixed_cem.json` was then written
before the held-out run.

## Held-out performance

| metric | result |
| --- | ---: |
| Success | 15/20 (75%) |
| Cumulative task cost, mean / median | 4712.3 / 3399.2 |
| Normalized return, mean / median | 0.559 / 0.628 |
| End-to-end latency, median / p95 | 698 ms / 1400 ms |
| Task-cost standard deviation | 2994.1 |
| Distinct task costs at `1e-6` | 20/20 |

Latency is synchronized per-start wall time from reset through termination or
the 50-step cap, including preprocessing, CEM, and environment execution but
excluding one-time dataset/checkpoint loading. Successful episodes often stop
early (median 24 steps); every failure consumed all 50 steps, so latency is an
episode-outcome quantity rather than a constant per-decision planner cost.

All five failures timed out. Their mean cumulative task cost was 9391.6 and
mean normalized return was 0.275, compared with 3152.6 and 0.654 for
successful starts. Four failures reduced final distance substantially but did
not satisfy both termination tolerances; one ended farther from the goal than
it began. This is evidence of residual hard-start/control failure, not evidence
for any particular physical mechanism.

## Adequacy and later labels

The held-out result passes the preregistered minimum of 50% success and at
least three successes, with finite mechanically valid metrics. Both success
and failure occur, and continuous task cost has nonzero spread with 20 distinct
values. Outcome variation is therefore sufficient to construct later paired
planning-regret labels, subject to a separate preregistered common-random-number
collection.

The cohort is a deterministic sample of valid rows from the released expert
dataset, not a fresh environment-start distribution and not proven disjoint
from original world-model training. MPS execution is seeded but not claimed
bitwise deterministic. The six-start tuning screen is deliberately small; the
held-out interval communicates the resulting uncertainty. Raw trajectories,
logs, and videos remain ignored under `work/`.

Compact evidence is in `results.json`; the runnable fixed configuration is
`fixed_cem.json`.
