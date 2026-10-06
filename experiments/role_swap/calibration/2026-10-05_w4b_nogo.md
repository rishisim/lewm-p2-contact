# Role-swap Step-4 evaluation protocol

Status: **frozen before evaluation** (2026-10-05). **Main run: NO-GO.**
All results below are independent pre-evaluation calibration. No main evaluation has run.
The best tested configuration and conditional planning N are frozen for review;
they do not establish an adequate reference or authorize the main run.

## Fidelity gate

The controlled no-peg loop and the verified smoke-eval both succeed on 43/50
expert start/goal pairs (seed 42, goal offset 25, CEM 300 x 30, top-30,
50 env steps). Removing the dataset's fixed green T decoration gives 36/50.
The correction restores that same fixed decoration in current and goal frames;
it does not render a task-dependent target marker. Mean rendered/dataset pixel
MAE is 0.411 for starts and
0.385 for goals on the 0-255 scale.

No-peg mode creates no peg body or shape and renders no peg; state/goal state
remain nine dimensional with peg xy=(-1000,-1000). Proprio uses indices
[0,1,5,6]. The shared controlled loop executes the full budget using wheel
WorldModelPolicy and history length 1, replanning every 25 steps. The fidelity
criterion is upstream ever-success, including agent and T position error;
Step-4 scoring below excludes agent position and uses final success.

Evidence: `$LEWM_WORK_ROOT/runs/w4b-fidelity/rendered.json` and
`without-marker/rendered.json`; verified smoke result:
`runs/smoke-eval/20261005T231544642249Z_c3dac797/metrics.json` under the work root.
The stored smoke result is reused, rather than executing another smoke run.

## Fixed semantics, observations, and costs

- Each model action is 5 consecutive 2-D physical env actions (10-D);
  horizon=5 blocks and receding horizon=5 blocks (25 env steps).
- LeWM CEM: population 300, iterations 30, top-30; identical for all arms.
- Wheel transforms: ToImage, float32 /255, ImageNet normalization, resize 224.
  Goal frames render every object's goal state plus the upstream fixed green
  decoration at (256,256,pi/4). The existing W2 peg remains radius 15,
  mass 1, RGB(230,126,34).
- The calibration copies checkpoint normalization once and freezes it at
  `runs/w4b-calibration/normalization.json` (method `torch_sample_std`;
  SHA256 `839239731fb7c7734fa66057cce9408af8efa043d9045fee5c1c980f058c41f3`). The concurrent normalization
  work is not edited. Future arms and reference eligibility must use this
  same normalization artifact; resume/feasibility checks reject mismatches.
- Reference cost = final T position error + 100 x wrapped T angle error,
  plus final peg target error for move_peg, or 2 x maximum peg displacement
  for preservation conditions, including displacement before replanning.
  LeWM uses the upstream predicted-final-embedding MSE goal cost.
- Strict thresholds (unchanged): T position error <20 px and wrapped angle
  error <pi/9; peg preservation max displacement <10 px; peg target error
  <15 px. move_peg also requires T success. Report ever-success and
  disturbance-allowed T success as companions, not substitutes for final success.

## Seeds and separation

- Pilot: master seed 1,000,000,000; reserved master range
  [1,000,000,000,1,000,999,999], 10 bases, preserved at
  `$LEWM_WORK_ROOT/runs/w4-pilot` (moved out of the repository).
- Calibration: master seed 2,000,000,000; reserved master range
  [2,000,000,000,2,000,999,999], grid/feasibility planner seed F_cal=22042, followed by independent
  validation planner seed E_cal=22043 for both reference and pretrained.
  There are 20 accepted
  bases per T-displacement filter from the same independent candidate stream;
  19 bases are identical across filters (21 distinct scenes total).
  Actual scene seeds are SeedSequence([master,attempt]) uint32 values,
  not constrained to the numerical master ranges. Recorded pilot and
  calibration scene seeds have been checked to be disjoint.
- Main scene master seed: 300,000,000, reserved range [300,000,000,300,999,999].
  Do not use pilot/calibration bases for feasibility or evaluation.
- Main feasibility planner seed F=101; reference evaluation seed E=202.
  F alone selects eligibility, and E reports reference results on that set.
  Neither has been run on main evaluation scenes.
- Learned evaluation seed namespaces: 42,43,44 for the three repeats,
  with the same namespace across arms; per-episode solver seed is the
  low 32 bits of SHA256(namespace:base_id:condition). Training seeds are
  checkpoint provenance and are separate from these planning seeds.

## Calibration grid and selection

The complete grid is budget {50,100}, reference CEM {100x10 top-10,
300x30 top-30}, minimum 25-step weak-rollout T displacement {40,60} px,
and matched translation magnitude {[40,80],[60,100]} px. Reference uses
8 process workers. Thresholds and costs above remain fixed. Targets are
80% for off_path and move_T_matched; 60% for on_path and move_peg.

For identical inputs/settings, reuse exact seed-matched episodes across the
acceptance filters. The 50-step budget is the exact prefix of the 100-step
run; the regression test compares actions, states, and score. off_path/on_path
are also shared across translation bands, since their construction is unchanged.
No reused observation is counted twice within a 20-base configuration.

Prefer passing budget-50 configurations, then the fewest changes from
minimum 60, range [60,100], reference 100x10. If none pass, minimize the sum
of shortfalls from the four targets, then apply that preference. Calibration
reference common-feasible counts are descriptive and are not main seed-F
eligibility. Detailed episodes, populations, per-episode times, and copied
inputs stay under `$LEWM_WORK_ROOT/runs/w4b-calibration`.

## Frozen selected configuration and readiness

Budget **100 env steps**; reference **300 x 30, top-30, 8 CPU workers**;
LeWM **300 x 30, top-30, MPS**; minimum T displacement **60 px** from the
25-step weak rollout; matched displacement magnitude **[60,100] px**.
All action semantics, thresholds, costs, normalization, and rendering above
are unchanged. Among tied minimum-deficit configurations, this retains the
original displacement settings. No grid cell meets all four targets.

The reference validation still falls below every target. On the calibration
common-feasible D set, reference E succeeds on 9/9 off_path and 7/9 on_path:
its -22.2 point difference fails the frozen reference-difference bar (<5 points).
The G common set is empty. These are planner/measurement readiness failures;
they are not evidence that the objectives are physically impossible or that
LeWM loses task-dependent information. Do not start the main run with this
configuration merely by increasing N.

## Complete reference calibration grid

Each cell uses 20 bases. Rates are final joint success, in percent, under
F_cal=22042. CEM 100x10 uses top-10; 300x30 uses top-30.

| Min T px | Translation px | Budget | Reference CEM | off_path | on_path | move_peg | move_T_matched |
|---:|:---:|---:|:---:|---:|---:|---:|---:|
| 60 | 60-100 | 50 | 100x10 | 50 | 20 | 5 | 15 |
| 60 | 60-100 | 100 | 100x10 | 50 | 25 | 10 | 15 |
| 60 | 40-80 | 50 | 100x10 | 50 | 20 | 10 | 25 |
| 60 | 40-80 | 100 | 100x10 | 50 | 25 | 10 | 25 |
| 60 | 60-100 | 50 | 300x30 | 70 | 40 | 5 | 25 |
| 60 | 60-100 | 100 | 300x30 | 70 | 45 | 5 | 25 |
| 60 | 40-80 | 50 | 300x30 | 70 | 40 | 5 | 25 |
| 60 | 40-80 | 100 | 300x30 | 70 | 45 | 5 | 25 |
| 40 | 60-100 | 50 | 100x10 | 50 | 20 | 5 | 20 |
| 40 | 60-100 | 100 | 100x10 | 50 | 25 | 5 | 20 |
| 40 | 40-80 | 50 | 100x10 | 50 | 20 | 5 | 30 |
| 40 | 40-80 | 100 | 100x10 | 50 | 25 | 5 | 30 |
| 40 | 60-100 | 50 | 300x30 | 70 | 40 | 0 | 30 |
| 40 | 60-100 | 100 | 300x30 | 70 | 45 | 0 | 30 |
| 40 | 40-80 | 50 | 300x30 | 70 | 40 | 0 | 30 |
| 40 | 40-80 | 100 | 300x30 | 70 | 45 | 0 | 30 |

## Selected configuration: independent validation E_cal=22043

All 20 calibration bases are evaluated unconditionally. The reference is
rerun with E_cal after selection, rather than reusing grid success numbers.
Pretrained uses the same validation namespace. These are calibration results,
not main evaluation results. Wilson intervals and every episode's times are
in `evidence.json`; complete trajectories/actions/populations are retained.

| Condition | Reference successes | Pretrained successes | Reference mean s/episode | Pretrained mean s/episode |
|:---|---:|---:|---:|---:|
| off_path | 15/20 (75%) | 7/20 (35%) | 11.40 | 8.96 |
| on_path | 7/20 (35%) | 2/20 (10%) | 12.09 | 8.96 |
| move_peg | 2/20 (10%) | 0/20 (0%) | 10.69 | 8.95 |
| move_T_matched | 5/20 (25%) | 2/20 (10%) | 10.99 | 8.95 |

F_cal selects **9/20 D bases (11 excluded)** and **0/20 G bases (20 excluded)**.
On D's nine bases, reference E is off_path 100%, on_path 77.8%; pretrained
is off_path 44.4%, on_path 11.1% (paired difference -33.3 points,
cluster-bootstrap 95% CI [-66.7,0.0]). There is no estimable common-set G
contrast. Eligibility is paired by base, never by individual condition.

Across all 80 validation episodes, reference mean/median/p90/max is
11.29/10.96/12.70/18.03 s; pretrained is 8.95/8.85/9.26/9.83 s.
Reference grid timing spans the IPC fix: an unused rendered snapshot in
Condition was still serialized despite stripping the separate snapshot's
image. Removing it reduces the condition payload from 503,547 to 312 bytes.
Physical costs/rollouts pass parity tests. Time projections below use the
post-fix independent validation, not the slower early grid episodes.

## Inspected failures and scope limits

Four median-cost failed reference trajectories were rendered and visually
inspected at steps 0,25,50,100 and at the goal. Replaying all 100 recorded
float32 actions from the saved start snapshot reproduces every state exactly
(maximum replay error 0.0 for all four):

| Condition / base suffix | Observed failure |
|:---|:---|
| off_path / 1223 | T remains unmanipulated; final error 152.6 px and 31.0 deg; peg stationary. |
| on_path / 1198 | T makes partial progress, then stalls; final position error 119.5 px, angle error 0.009 deg; peg displacement only 0.16 px. This base passed F_cal but fails E_cal. |
| move_peg / 771 | Peg stationary; target error 82.3 px; closest sampled agent-peg distance 295.3 px. T remains fixed. |
| move_T_matched / 1419 | T stationary; target error 81.1 px; no peg disturbance. |

These examples indicate navigation/contact-search failure and local search
stalls, rather than a trajectory replay or observation-loop discrepancy.
Agent position is absent from the reference role cost, so pre-contact
navigation can have a flat objective. CEM's fixed normalized exploration and
25-step planning horizon do not reliably find the contacts from these random
starts. More iterations/budget and shorter translation bands did not remove
the floor. This diagnosis is supported by inspected trajectories, not a proof
that a stronger planner cannot solve the scenes.

The existing collection path (`src/lewm_research/data/collect.py:19`) uses
with_target=False, whereas the fidelity-corrected evaluation includes the
upstream fixed decoration. That train/evaluation visual shift remains;
fine-tuned arms need the PLAN's separate no-peg forgetting check. Training,
normalization, and collected data were not changed by W4b.

Deferred material changes: contact-directed reference initialization or
navigation shaping (`probe/reference.py`), and/or a start distribution placing
the agent closer to both manipulated objects (`probe/conditions.py`). Aligning
future collection rendering would affect `data/collect.py`. These change
scientific assumptions and require a new disjoint calibration and freeze;
none is silently included in these results.

## Frozen conditional N and wall-time planning

**Planning N=2,000 candidate bases per evaluation repeat** (8,000 condition
episodes per arm/repeat). This is a conditional D-only budget estimate;
**main run remains NO-GO, and G has no defensible powered N from these data**.
No sample-size increase repairs the observed reference gap or an empty G set.

For paired binary outcomes, q=P(discordance), variance=q-delta^2. Use
n_eff=ceil((z_0.975+z_0.80)^2*(q_upper-delta^2)/delta^2), where q_upper is
its Wilson 95% upper bound. On D's common set, pretrained discordance is
3/9=0.3333 and q_upper=0.6458. This yields **218 effective common bases**
for a 15-point difference and **500** for a 10-point difference. The latter
uses the same discordance as a proxy for future ft_mixed/ft_block comparisons;
that cross-arm discordance has not been measured.

Observed D eligibility is 9/20=0.45 (Wilson lower bound 0.2582). Requiring
500 common bases yields 1,112 candidates at the observed fraction or 1,937
at the lower bound; round the latter to 2,000. For the 15-point comparison
alone, the corresponding candidate counts are 485 and 845. This calculation
powers CI exclusion of zero at an assumed effect, not the full compound bar's
effect-size/localization requirements. It does not turn pretrained pilot
behavior into a guaranteed fine-tuned-arm effect, and it does not credit the
three repeats as independent new bases.

At N=2,000, using measured episode means and serial episodes:

- 3 learned arms x 3 repeats x 4 conditions x N x 8.953 s = **179.1 hours
  (7.46 days)** of evaluation.
- Separate reference F and E x 4 conditions x N x 11.293 s = **50.2 hours
  (2.09 days)** with 8 candidate workers.
- Combined = **229.3 hours (9.55 days)**. At N=100, the learned-arm portion
  alone would be about 9.0 hours, but the common sets would be too small for
  the conservative minimum-effect planning calculation.

These projections assume stable per-episode latency. They exclude training,
scene generation, readout, A/B/C, and other preprocessing. The 3x3 multiplier
is an evaluation workload estimate; it does not change the training workstream.
The main protocol reports each checkpoint/repeat and clusters by base when
pooling repeats. Success/readout/ABC evidence is still conditional on the
trained checkpoints as specified in PLAN.md.

## Evidence and reproducibility

All full products are under `$LEWM_WORK_ROOT/runs/`, never the repository.
The work-root cache and governed USB are unchanged. Canonical commands:
`uv run lewm probe-fidelity`, `uv run lewm probe-calibrate --workers 8`.
Calibration resumes only matching configs; feasibility additionally requires
matching bases, budget, translation band, rendering, normalization, and a
separate planning seed. No main evaluation was performed, and no commit was
made, as requested.

Retained evidence: `w4b-calibration/{grid.json,chosen.json,evidence.json,
inspection.json,inspected-trajectories.png,protocol_freeze.json}`; per-episode
records are in `chosen-reference/episodes.jsonl` and
`chosen-lewm/episodes.jsonl`. `chosen-feasibility` is the exact grid seed-F
budget prefix with a source manifest. Images/populations for reused filter
cells remain at the recorded source runs; they are not main A/B/C banks.



## Frozen decision bar (copied from PLAN.md)

For `ft_block`:
1. Gap exists: D or G shows LeWM success lower by >= 15 points in the less-familiar
   condition, paired cluster-bootstrap 95% CI excluding 0, on the common feasible set,
   while the reference planner's difference under evaluation seed E is < 5 points.
2. Localization (labels are descriptive, not proofs of mechanism):
   - linear-readout deficit: peg readout error >= 2x T readout error AND >= 1.5x the
     pixel-ridge peg error (weak linear accessibility, not proof of information loss);
   - observed-state cost mismatch: Spearman(B,C) in the hard condition < its control by
     >= 0.2, with peg readout < 1.25x pixel-ridge error;
   - prediction-associated degradation: Spearman(B,C) >= 0.5 in BOTH conditions (adequate
     absolute agreement), Spearman(A,B) in the hard condition lower than control by >= 0.2,
     and Spearman(A,C) lower by >= 0.2.
3. Coverage: `ft_mixed` counts as a sufficient remedy only if (a) success in the hard
   condition improves by >= 10 points over `ft_block` (paired CI excluding 0), (b) the
   control condition regresses by < 5 points, and (c) the gap shrinks by >= 50%.
   Otherwise report only "reduced disparity" (or none).
