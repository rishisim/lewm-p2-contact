# Role-swap Step-4 protocol — W4d

Status: **GO for the primary contrasts below**. No main evaluation has run.
Primary contrasts: **D' = near_path vs off_path; G = move_peg vs move_T_matched**.
W4d is one bounded redesign of W4c D. G and the physical planner/scoring are unchanged.
The prior W4c midpoint `on_path` is retained as an explicitly requested diagnostic,
not a primary condition: its independent reference success was 0/20.

## Task construction

Sample the same non-overlapping agent/T/clutter-peg scene with unchanged physics.
Run 300 physical steps of BlockWeakPolicy. D uses the first 25-step window
whose start agent is within 40 px of T, T translation is >=40 px, and peg
position is stationary throughout (absolute tolerance 1e-6 px). Retain the
exact start snapshot including body velocities; goals use the real t+25
agent/T states. `off_path` retains the original peg, >=80 px from the T
center start-goal segment. `on_path` places the peg at its exact midpoint.
`near_path` uses the same D start and goal, replacing only the peg with
a perpendicular offset L from the segment midpoint. Its goal peg is fixed
there; peg linear/angular velocities are zero. A per-base seeded side
(NumPy default_rng(scene_seed), choice([-1,1])) is preferred independently
of L and outcomes; try the opposite side only if geometry is invalid.
Require no peg overlap with agent/T at both endpoints and center within
[45,467]^2. Do not jitter, alter goals, or replace calibration bases.
Record bases where neither side fits; exclude them from D eligibility.

For G, restore the original scene and run PegWeakPolicy (30 px box) for
300 steps. Select the first 25-step window with agent within 40 px of peg,
peg translation 40–100 px and T pose stationary throughout (1e-6 tolerance).
`move_peg` uses its real future agent/peg, with T fixed. `move_T_matched`
uses a separate block window in the same scene, agent within 40 px of T,
T translation in the same 20 px peg displacement bin ([40,60), [60,80),
[80,100]), wrapped rotation <pi/9 and stationary peg.
D/G share scene IDs but may have different exact starts and agent goals.
Replay original off_path/move_peg/move_T_matched witnesses from fresh
simulator spaces before scene acceptance; require unchanged final joint
success. This rejects Chipmunk contact-cache-sensitive restored futures.
Witnesses never initialize either planner. For near_path the original D
witness is instead replayed with the relocated peg to measure **naive
disturbance**: maximum peg displacement >10 px during all 25 steps.
This establishes whether the obstacle affects that witness, not whether
near_path is solvable; reference F/E establish planner feasibility.

New main scenes use `generate_bases(N, 300000000, near_path_distance=L)`
when D is retained; geometry-invalid candidates are rejected. Calibration
uses `with_near_path` on the exact 20 stored W4c bases. Legacy BaseScene
files remain readable but require their original revision for evaluation.

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
`$LEWM_WORK_ROOT/runs/w4d-calibration/normalization.json` (exact W4c copy) and reused by all cells
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

## Seeds and bounded calibration selection

Reuse the exact W4c calibration master 2,100,000,000 and its 20 accepted
scene IDs (previous pilot/W4b disjointness remains verified). F_cal=23042
alone selects L; E_cal=23043 independently validates reference and pretrained.
This is calibration, not main evaluation or main feasibility. Main master
300,000,000 / [300,000,000,300,999,999], F=101 and E=202 remain reserved.
Main learned planning namespace 42 is shared across the **five trained
checkpoints**: lewm-pusht, ft_block_s0, ft_block_s1, ft_mixed_s0, ft_mixed_s1.
One evaluation repeat each. Episode seeds are low 32 bits of
SHA256(namespace:base_id:condition), identical across arms.

Test L={40,55,70,85} px only. Reference is frozen at 50 env steps,
300x30 CEM, top-30, approach w=0.1 and eight CPU candidate workers.
LeWM is 50 steps, 300x30, top-30 on MPS with upstream fixed decoration.
Original W4c control episodes are reused after exact base, config,
normalization and checkpoint-hash checks; source condition/population
artifacts remain at their original run roots. Only near_path is new.
A candidate requires near_path F success >=60% of the original 20 bases
(geometry exclusions count against coverage) and nonzero naive disturbance.
Among candidates meeting the reference F common-set |near-off| <5-point
bar, maximize naive disturbance; tie by smaller L. Report the unconditional
difference as a companion, not an additional selection screen.
If none meets the common-set bar, minimize its difference, then maximize naive
disturbance, then smaller L. A bounded fallback is diagnostic and cannot
relax the independent E <5-point frozen bar on the F common feasible set.
F common-set success differences are zero by construction, not validation.
Do not select a second L after inspecting E. Retain D only if chosen L
passes independent E targets (off >=80%, near >=60%), nonempty F common
eligibility and reference E absolute difference <5 points. Otherwise drop D
and use G only. G requires move_peg >=60%, matched >=80%, nonempty common
eligibility and reference E absolute difference <5 points. GO applies only
to the contrasts retained and does not authorize a main run automatically.

## Calibration evidence and frozen settings

| L (px) | Geometry valid | Reference F success | Success / all 20 | Naive disturbance | F abs near-off (points) |
|---:|---:|---:|---:|---:|---:|
| 40 | 17/20 | 9/17 (52.9%) | 45.0% | 9/17 (52.9%) | 35.3 |
| 55 | 20/20 | 15/20 (75.0%) | 75.0% | 5/20 (25.0%) | 15.0 |
| 70 | 19/20 | 16/19 (84.2%) | 80.0% | 1/19 (5.3%) | 5.3 |
| 85 | 20/20 | 14/20 (70.0%) | 70.0% | 5/20 (25.0%) | 20.0 |

Chosen **L=55 px**; naive disturbance **5/20 (25.0%)**. F unconditional abs difference **15.0 points**.

Independent E=23043 validation (all available bases; unchanged controls reused):

| Condition | Reference | Pretrained | Reference mean s | Pretrained mean s |
|:---|---:|---:|---:|---:|
| off_path | 18/20 (90.0%) | 5/20 (25.0%) | 7.63 | 5.41 |
| near_path | 15/20 (75.0%) | 2/20 (10.0%) | 8.69 | 4.44 |
| move_peg | 20/20 (100.0%) | 1/20 (5.0%) | 5.78 | 5.49 |
| move_T_matched | 19/20 (95.0%) | 10/20 (50.0%) | 7.49 | 5.47 |

F-selected common feasible sets (calibration eligibility only):

- D': **15/20 included, 5 excluded**; independent readiness passes.
  Naive disturbance on this common set: **1/15 (6.7%)**; eligibility removes some witness-disturbing bases, limiting D interaction coverage.
  reference: off_path 100.0%, near_path 100.0%; hard-control difference **0.0 points**, paired bootstrap 95% CI [0.0, 0.0].
  pretrained: off_path 33.3%, near_path 13.3%; hard-control difference **-20.0 points**, paired bootstrap 95% CI [-46.7, 6.7].
- G: **19/20 included, 1 excluded**; independent readiness passes.
  reference: move_peg 100.0%, move_T_matched 100.0%; hard-control difference **0.0 points**, paired bootstrap 95% CI [0.0, 0.0].
  pretrained: move_peg 5.3%, move_T_matched 52.6%; hard-control difference **-47.4 points**, paired bootstrap 95% CI [-73.7, -21.1].

**GO** for D' = near_path vs off_path; G = move_peg vs move_T_matched. No main evaluation has run.

## Sample size and measured Mac timing

Use the inherited paired-binary approximation separately on each retained
F-selected common set. q is pretrained hard/control discordance; use its
Wilson 95% upper bound (at least the effect). Effective bases for delta in
{0.15,0.10} = ceil((z_0.975+z_0.80)^2*(q_upper-delta^2)/delta^2). Divide
the larger count by the Wilson lower bound on F eligibility. Take the
maximum over retained contrasts and round up to the next hundred. The
10-point count is a proxy, not measured cross-arm power. This powers CI
exclusion of zero, not the full compound bar or guaranteed fine-tuned effects.

D_prime: discordance 0.3333, Wilson q upper 0.5829; effective bases 196 (15 points), 450 (10 points); eligibility 0.7500, Wilson lower 0.5313; candidate bases 600 (observed), 847 (lower bound).
G: discordance 0.5789, Wilson q upper 0.7686; effective bases 261 (15 points), 596 (10 points); eligibility 0.9500, Wilson lower 0.7639; candidate bases 628 (observed), 781 (lower bound).

Planning **N=900 candidate bases**; five checkpoints x1 repeat x4 primary conditions.
Measured independent episode means: reference **7.396 s**, pretrained **5.201 s** (LeWM planning **5.012 s**).
Mac learned: **18000 episodes, 26.00 h**; reference F/E: **7200 episodes, 14.79 h**; total **40.80 h**.
Measured Mac episode means; pretrained is a latency proxy for all five checkpoints. Serial episodes, eight reference CPU candidate workers. Excludes training/generation/readout/ABC. Controls reused from W4c.
Calibration base generation and contact-cache witness filtering are additional unprojected work.

Full evidence: `$LEWM_WORK_ROOT/runs/w4d-calibration` (`grid.json`,
`chosen.json`, `L*/construction.json`, `L*/reference-F`, `near-reference-E`,
`near-pretrained-E`, merged `reference-F/reference-E/pretrained-E`,
`reuse.json`, `evidence.json`, `protocol_freeze.json`, `driver.log`).
Reproduce the bounded workflow with `python -m lewm_research.probe.near_path`
using the project environment; resume rejects changed episode configs.
Tests and validation are recorded under `$LEWM_WORK_ROOT/runs/w4d-tests`.
No commit was made.

## Frozen decision bar (copied from PLAN.md)

For `ft_block`:
1. Gap exists: D' or G shows LeWM success lower by >= 15 points in the less-familiar
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
