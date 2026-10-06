# Role-swap Step-4 protocol — W4c

Status: **frozen diagnostic configuration; main run NO-GO**.
No main evaluation has run. This protocol replaces W4b's task construction;
its unchanged historical record is `calibration/2026-10-05_w4b_nogo.md`.

## Task construction

Sample a non-overlapping agent/T/clutter-peg scene, with unchanged physics.
Run 300 physical steps of BlockWeakPolicy from that scene. D uses the **first**
25-step window whose start agent is within 40 px of the T center, T translation
is >=40 px, and peg position is stationary throughout (absolute tolerance
1e-6 px). Retain the exact snapshot at t, including all body velocities.
Goal agent/T states are the real t+25 states; peg remains fixed.

`off_path` retains the scene's peg and requires its center >=80 px from the
T center start-goal segment. `on_path` moves only the peg to the exact segment
midpoint, with zero peg velocity. It must overlap neither agent nor T at both
endpoints and stay within the peg wall margin. Discard incompatible scenes;
no midpoint jitter. Inserting this obstacle changes the task, so on_path is
**not** guaranteed reachable by the original witness.

For G, restore the same original scene and run PegWeakPolicy with a 30 px
box for 300 steps. Select the first 25-step window with agent within 40 px
of peg, peg translation 40-100 px, and T pose stationary throughout (1e-6
absolute tolerance). `move_peg` uses the real future peg and agent, T fixed.
`move_T_matched` uses a separate block-rollout window in the same scene,
agent within 40 px of T, T translation in the peg displacement's 20 px bin
([40,60), [60,80), [80,100]), wrapped rotation <pi/9, peg stationary.
Both original G goals and off_path have recorded rollout action witnesses.
Witnesses are provenance only and never initialize the reference or LeWM.
D/G share scene IDs but may have different exact starts and future agent goals.
Before acceptance, replay each original off_path/move_peg/move_T_matched
action witness from a fresh simulator space and the saved start snapshot;
require final joint success against the unchanged t+25 goal. This rejects
contact-cache-sensitive windows rather than silently calling unreachable
restored starts reachable. Body snapshots do not capture Chipmunk cached
contact impulses. Store each replay score and final-state discrepancy.
Accept 20 scenes valid for both families; pairing and bootstrap remain by scene.

Legacy BaseScene files remain readable for historical seed checks, but cannot
be evaluated under W4c. Replay W4b using its original Git revision. There is
one canonical task-construction implementation.

## Semantics, rendering, normalization, and scoring

Each model action is five physical 2-D actions (10-D). Horizon and receding
horizon are five blocks =25 env steps. Execute the full frozen 50 or 100-step
budget, with history length 1 and wheel WorldModelPolicy/CEM; never stop early.
LeWM CEM is 300x30, top-30, identical across all checkpoints, on MPS here.
Transforms: ToImage, float32 /255, ImageNet normalization, resize 224.
W2 physics are unchanged: peg radius 15 px, mass 1, RGB(230,126,34),
block-like friction/damping. Initial clutter peg centers are at least 180 px
from T and 100 px from agent. No collection or training changes were made.

The inherited W4b no-peg fidelity gate is 43/50 with fixed upstream decoration
and 36/50 without it; W4c does not rerun that passing gate. Full evidence is
in the archived W4b record.

**Rendering follows each checkpoint's training data.** Pretrained `lewm-pusht`
uses the upstream fixed green T decoration at (256,256,pi/4) in current and
goal images. Fine-tuned arms collected with `with_target=False` use no green
decoration in either image. `probe-eval --with-target/--no-with-target` explicitly
overrides the per-arm default; every config records `with_target` and
`observation_rendering`. Reference pixels do not affect physical costs. Its
calibration uses fixed decoration. Feasibility is physical and can therefore
be shared across differently rendered learned arms. The no-peg forgetting
check must likewise use each learned arm's training rendering.

Normalization is copied once from pretrained to
`$LEWM_WORK_ROOT/runs/w4c-calibration/normalization.json` and reused by all cells
and validation. Resume rejects changed config, bases, checkpoint or normalization;
reference E must match F's CEM and approach weight. Main arms use this same
frozen normalization. Its canonical JSON SHA256 is
`839239731fb7c7734fa66057cce9408af8efa043d9045fee5c1c980f058c41f3`,
verified identical to W4b. W4b normalization and concurrent training are not edited.

Unchanged physical role cost: final T position error (px) +100 x wrapped angle
error (rad) + final peg target error for move_peg, otherwise 2 x maximum peg
displacement, including prior executed steps. Reference optionally adds
**w x final agent-to-manipulated-object center distance (px)** to each candidate's
cost; manipulate peg for move_peg, T otherwise. This shaping is reference-only,
absent from success scoring, LeWM cost, and A/B/C physical role cost.
LeWM minimizes upstream predicted-final-embedding MSE to the goal embedding.

Final joint success: T error <20 px and wrapped angle error <pi/9; preservation
max peg displacement <10 px; move_peg final peg target error <15 px and T
success. Agent goal position is rendered but excluded from success. Report
trajectory ever-success and disturbance-allowed T success as companions.

## Seeds and calibration selection

Pilot master reserved [1,000,000,000,1,000,999,999]; W4b calibration master
reserved [2,000,000,000,2,000,999,999]. W4c reserves the new disjoint master
range **[2,100,000,000,2,100,999,999]**, using master 2,100,000,000.
Actual scene seeds are SeedSequence([master,attempt]) uint32 values; assert
they are disjoint from recorded pilot/W4b scene seeds in construction.json.
Grid/feasibility F_cal=23042; independent validation E_cal=23043 for reference
and pretrained. Calibration is not main evaluation or main feasibility.

Main master/range remain 300,000,000 / [300,000,000,300,999,999]; F=101 alone
selects paired eligibility; E=202 independently evaluates reference. One learned
planning namespace 42 across seven checkpoints: pretrained + ft_block and
ft_mixed each at three training seeds. Episode seeds are low 32 bits of
SHA256(namespace:base_id:condition), identical across arms. Training seeds are
checkpoint provenance and are not independent new evaluation scenes.

Grid: w={0,0.1,0.3} x CEM={100x10 top-10,300x30 top-30}, eight CPU candidate
workers, 20 bases per family. Exhaust the six-cell grid at budget 50. If none
meets readiness, extend the best 50-step cell (minimum target deficit, then
D difference, then CEM/w) to budget 100. Select among the resulting seven
tested configurations and validate independently. This bounded budget
expansion fits the 100-minute session limit; it is not an exhaustive 100-step
weight/CEM grid and NO-GO concerns only the tested configurations. Targets:
off_path and move_T_matched >=80%; on_path and move_peg >=60%.
Require grid unconditional absolute reference D difference <5 points as a
conservative plausibility screen. Prefer a passing budget-50 cell, then smaller
CEM and smaller w. If none passes, minimize summed target deficit, then D
difference, then budget/CEM/w; this selects a diagnostic config, not a GO.
Validate selection under E_cal and run pretrained on all 80 validation episodes.
Reference 100-step grid runs replay their exact 50-step physical action prefixes
to reconstruct live contact state, advance the wheel CEM RNG by the first two
searches, and execute the remaining two calls. A regression test checks equality
of all actions, states and scores against a fresh 100-step run; runtime replay
checks reject state mismatches. Independent reference E runs are fresh.
Pretrained validation may be precomputed at budget 100 on MPS concurrently
with reference F calibration. Selection uses only F reference outcomes; retain
the exact budget prefix after selection, with hashes and source provenance.
GO additionally requires validation targets, nonempty F common sets for D/G,
and reference E absolute D difference <5 points on D's common feasible set.

## Calibration evidence and frozen settings

The diagnostic configuration is frozen before main evaluation. **Main run: NO-GO**. No main evaluation has run.
Budget **50 env steps**; reference **300x30, top-30, w=0.1, 8 CPU workers**; LeWM **300x30, top-30, MPS**.

Complete F_cal=23042 grid, final joint success percentages (20 bases each):

| Budget | CEM | w | off_path | on_path | move_peg | move_T_matched |
|---:|:---|---:|---:|---:|---:|---:|
| 50 | 100x10 | 0 | 85 | 0 | 80 | 95 |
| 50 | 100x10 | 0.1 | 85 | 0 | 80 | 95 |
| 50 | 100x10 | 0.3 | 80 | 0 | 85 | 95 |
| 50 | 300x30 | 0 | 95 | 5 | 90 | 95 |
| 50 | 300x30 | 0.1 | 90 | 5 | 100 | 95 |
| 50 | 300x30 | 0.3 | 90 | 0 | 100 | 95 |
| 100 | 300x30 | 0.1 | 90 | 5 | 80 | 95 |

Independent E_cal=23043 validation, all bases unconditionally:

| Condition | Reference | Pretrained | Reference mean s | Pretrained mean s |
|:---|---:|---:|---:|---:|
| off_path | 18/20 (90%) | 5/20 (25%) | 7.63 | 5.41 |
| on_path | 0/20 (0%) | 0/20 (0%) | 7.71 | 5.50 |
| move_peg | 20/20 (100%) | 1/20 (5%) | 5.78 | 5.49 |
| move_T_matched | 19/20 (95%) | 10/20 (50%) | 7.49 | 5.47 |

Reference validation unconditional D difference is **-90.0 points**. Frozen readiness checks are unmet; no main evaluation.
The empty D common set leaves its reference E difference unestimable. These readiness failures do not establish LeWM information loss.

F-selected common sets (calibration eligibility only):

- D: **0/20 included, 20 excluded**.
- G: **19/20 included, 1 excluded**.
  reference: move_peg 100.0%, move_T_matched 100.0%; hard-control difference 0.0 points, bootstrap 95% CI [0.0, 0.0].
  lewm-pusht: move_peg 5.3%, move_T_matched 52.6%; hard-control difference -47.4 points, bootstrap 95% CI [-73.7, -21.1].

25-step displacement distributions (px):

| Condition | min | median | mean | max |
|:---|---:|---:|---:|---:|
| off_path | 82.02 | 194.28 | 191.28 | 356.34 |
| on_path | 82.02 | 194.28 | 191.28 | 356.34 |
| move_peg | 50.80 | 77.68 | 76.24 | 99.07 |
| move_T_matched | 40.05 | 70.69 | 71.95 | 98.82 |

G bin counts ([40,60), [60,80), [80,100]): **{'move_peg': [6, 5, 9], 'move_T_matched': [6, 5, 9]}**.
Scene seeds are disjoint from 31 unique recorded pilot/W4b seeds.
All 60 original replay witnesses solve; all 20 midpoint endpoint checks pass.
The superseded initial attempt rejected three contact-cache-sensitive futures and contributes no observations;
its diagnosis is retained at `runs/w4c-snapshot-audit/superseded.json`.

D: N unestimable (zero common-feasible calibration bases).
G: discordance 0.5789, Wilson q upper 0.7686; effective bases 261 (15 points), 596 (10 points); eligibility 0.9500, Wilson lower 0.7639; candidate bases 628 (observed), 781 (lower bound).

Joint planning N: **unestimable**. Conditional N from estimable contrasts: **800**; neither repairs reference readiness.

Pretrained latency was measured concurrently with reference calibration and tests; projections assume those measured latencies persist.

Measured independent episode means: reference **7.152 s**, pretrained **5.465 s** (LeWM planning **5.247 s**).
At N=800 (conditional), 7 checkpoints x1 repeat x4 conditions:

- Mac learned: **34.01 h**; reference F/E: **12.72 h**; total **46.72 h**.
- One CUDA GPU, LeWM planning assumed 5x faster, same CPU reference: learned **7.89 h**; total **20.60 h**.

Projections exclude training, generation, readout and A/B/C. CUDA is an assumption, not a benchmark.
All full evidence is under `$LEWM_WORK_ROOT/runs/w4c-calibration`: `grid.json`, `chosen.json`,
`construction.json`, `construction_audit.json`, `evidence.json`, `chosen-reference`, `chosen-lewm`,
`pretrained-budget100` and `protocol_freeze.json`. Reused learned images/populations are at the source in `reuse.json`.
Wilson intervals, all episode times and full power inputs are in evidence.json. Tests: `uv run pytest -q` passes (see `runs/w4c-tests/pytest.log`). No commit was made.

## Sample size and timing method

Use W4b's paired-binary method separately on each F-selected common set.
q is pretrained hard/control discordance; use its Wilson 95% upper bound
(and at least the assumed effect). For delta in {0.15,0.10}, effective bases
=ceil((z_0.975+z_0.80)^2*(q_upper-delta^2)/delta^2). The 10-point requirement
is a proxy for future ft_mixed/ft_block discordance, not measured cross-arm
power. Divide the larger effective count by the Wilson lower bound on F
eligibility. Take the larger candidate count across D/G and round up to the
next hundred; an empty common set leaves joint N unestimable. A remaining contrast may
provide an explicitly conditional N and workload projection, never main-run
authorization. Increasing N
cannot fix reference readiness. This powers CI exclusion of zero, not the
full compound decision bar or guaranteed fine-tuned effects.

Evaluation workload is **7 checkpoints x 1 repeat x 4 conditions x N**, plus
separate reference F/E x4xN. Use independent validation episode means at the
chosen reference cost and worker count. CUDA projection, if Mac exceeds
12 h, divides measured LeWM **planning** time by five, retaining non-planning
and reference CPU time; this is an assumption, not a measured GPU benchmark.
Exclude training, generation, readout and A/B/C from the evaluation projection.

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
