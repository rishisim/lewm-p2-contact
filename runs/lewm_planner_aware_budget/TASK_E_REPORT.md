# Task E PushT planner-alignment diagnosis

## Decision

Stop and refactor the refinement axis; do **not** train Task F yet, do not train
a selective refiner/dispatcher, and do not repeat the fixed grid. Keep the
verified depth-0/population-300 package as the production reference.

The smallest planner-aligned repair is to retire the current latent-MSE-only
refiner from planning and make any future refiner earn re-entry on common
depth-0 final-CEM banks using a simulator-ranking target. This evaluation
establishes that such a target addresses a real mismatch, but it does not yet
meet the frozen Task F gate requiring a stable sign across candidate-bank
seeds or a reproducible causal regime. A bounded ranking-target feasibility
study, not full training, is therefore the next admissible step.

## Integrity and scope

The protocol and 2-pilot/8-sealed-start manifest were committed before
candidate generation or simulator outcomes. All starts use unique source
episodes disjoint from Tasks A-D and Task B expert fit/selection/evaluation.
The complete sealed test contains 16 banks (8 starts × 2 independent seeds),
300 candidates per bank, and 4,800/4,800 finite simulator labels. A further
eight mechanically predeclared proposal-prefix banks contain 512/512 valid
labels.

Every primary bank is exactly the final population-300 tensor from depth-0
CEM iteration 20. Candidate tensors were generated once and reused unchanged
for depths 0/1/2/4 and every schedule ablation. All 4,800 model-visible action
sequences exactly equal the actions received by the unwrapped environment;
the observed bank range was not clipped (`-5.09` to `4.58` globally). The
qualified Task C mean/controller semantics are preserved.

All 192 deterministic replay checks (12 per sealed bank) reproduced executed
actions, states, and costs bitwise. The mechanical pilot clarified that
PushT's inherited `_set_state` callable advances one physics tick. Task E
therefore verifies exact repeated reset→callable reconstruction rather than
incorrectly comparing the post-call state with the pre-call dataset vector.
No installed package, checkpoint, dataset, prior run, or original checkout was
modified. Raw tensors, trajectories, states, and scores remain under ignored
`work/task_e/`.

## Fixed-bank ranking result

Lower regret is better; higher other metrics are better. Means pool 16 banks
but uncertainty for primary contrasts resamples the eight source-start
clusters with both seeds retained.

| depth | top-choice regret | normalized regret | concordance | Kendall tau-b | top-30 recall | choices changed | improve / harm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,363.6 | .333 | .5255 | .0510 | .0563 | — | — |
| 1 | 2,363.6 | .333 | .5278 | .0556 | .0542 | 0/16 | 0 / 0 |
| 2 | 2,475.2 | .353 | .5301 | .0603 | .0563 | 6/16 | 1 / 5 |
| 4 | 2,591.0 | .369 | .5344 | .0687 | .0625 | 11/16 | 3 / 8 |

Depth 4 minus base improved mean concordance by only 0.0089 (95% cluster
bootstrap CI -0.0077 to 0.0259) while **increasing** top-choice regret by
227.4 (benefit-oriented contrast -227.4, 95% CI -620.3 to 123.2). Depth 2
increased regret by 111.6 (benefit CI -422.1 to 201.6). Depth 1 slightly
reordered candidates but never changed the selected candidate. No primary
contrast survived the frozen uncertainty/multiplicity interpretation.

The critical failure is tail selection, not just globally reversed rankings:
deeper refinement marginally increases average concordance while moving the
argmin toward worse simulator actions. Base itself has substantial headroom:
its chosen candidate loses 33.3% of the within-bank simulator-cost range on
average, despite an oracle candidate already being present in the same bank.
Population sampling therefore cannot by itself fix this observed scoring
error.

## Mechanism localization

**Representation/metric misalignment is supported.** Task B's locked
evaluation established depth-4 latent-MSE improvement in all nine
source×history cells: raw improvements range from 0.9% to 13.8%, and whitened
improvements from 2.4% to 30.5%. On actual final-CEM candidates here, the same
checkpoint increases selected-action regret and harms 8 of its 11 changed
choices. A latent Euclidean target is therefore not aligned with the
planner's simulator-referenced ordering at the decision tail.

**Compounding/tail drift is suggestive but not a deployable regime.** Applying
depth 4 only at rollout transition 4 reduced mean regret by 70.8, but the
bank-bootstrap interval touches zero (0.0 to 212.4). Applying it at transition
5 instead increased regret by 75.8. Repeating depth 4 over transitions 1-4 or
1-5 increased regret by 227.4. Concordance rises monotonically across the
prefix while selected regret worsens, directly separating a broad rank metric
from the operational argmin. The isolated transition-4 point effect is too
sparse and seed-sensitive to authorize a schedule/controller.

**Overshoot is supported at selection, not as a stable beneficial shallow
exit.** Depth 1 never changes a choice. Depth 2 changes six and harms five;
depth 4 changes eleven and harms eight. Refiner update norms rise mechanically
with depth (roughly 0.06–0.08 at depth 1, 0.12–0.17 at depth 2, and 0.23–0.31
at depth 4 in the pilot), but no positive-depth exit yields reproducible
top-choice value.

**Contact selectivity is insufficient.** A simulator-only median proximity
split shows slightly larger concordance changes in near-interaction candidates
(depth-4 +0.0124, bank-bootstrap CI +0.0008 to +0.0248) than free-motion
candidates (+0.0098, CI -0.0009 to +0.0208). This does not translate to
top-choice improvement, uses a simulator-opened label unavailable to the
model, and is not a stable inference-time causal regime.

**Action magnitude and candidate distribution do not rescue refinement.**
Depth-4 concordance change is -0.0016 in the high-action-magnitude half and
+0.0052 in the low half, both uncertain. On the mechanically sealed
iteration-zero proposal-prefix banks, refinement also slightly reduces
concordance (`.5578` at base versus `.5569` at depth 4) and increases
top-choice regret (9,671 to 9,804). The failure is therefore not confined to
the final-CEM bank, though absolute proposal-bank regret is much larger.

## Candidate-seed stability and opportunity

The depth-2 benefit-oriented top-choice effect changes from -269.8 cost units
on seed 2026072701 to +46.7 on seed 2026072702; depth 4 changes from -491.3 to
+36.6.
Only 1/8 starts has a same-nonzero depth-2 sign across seeds. Depth 4 has 2/8
same-sign and 2/8 opposite-sign starts. Descriptively, within-start/across-bank
variance is 164,075 versus 251,862 between-start variance for depth 2, and
433,315 versus 342,991 for depth 4. Bank randomness remains material before
any depth-specific CEM distribution update, consistent with Task D's larger
closed-loop instability rather than explained solely by those updates.

The post-outcome base-versus-depth-4 oracle lowers mean regret by 210.6 by
selecting depth 4 in only 3/16 banks. Its histogram-randomization p-value is
0.0037, confirming optimistic opportunity in those already-labelled banks,
not predictability. Depth 2 is selected in 1/16 banks (randomization p=.126);
depth 1 is never selected. There are too few positive, seed-stable labels for
the preregistered inference-feature probe, so no probe was fit and no
dispatcher claim is made.

## Frozen next-step mapping

The Task F conjunction fails its stable-sign/reproducible-regime gate:

- verified latent-versus-ranking gap: **pass**;
- nontrivial depth-0 ranking headroom: **pass**;
- positive depth materially changes ranking: **pass** for depths 2/4;
- stable sign across bank seeds or reproducible causal regime: **fail**;
- effect not explained solely by population sampling: **pass**;
- bounded data/compute plan: **pass**.

Selective one-axis refinement also fails because no inference-available regime
has stable ≥0.01 normalized benefit or a qualifying cross-validated probe.
Population-only continuation is not selected: Task D did not support that axis
and Task E shows that the best available candidate is commonly already in the
bank but misranked.

If a later bounded feasibility task establishes seed-stable learnability, the
smallest Task F recipe remains the frozen one: common final depth-0 CEM banks;
simulator cumulative-cost pairwise ranking as the sole new target; five-step
rollout loss; a hinge constraint preventing base top-choice-regret
degradation; and an optional residual gate that may output exactly zero.
Splits must hold out whole source episodes and banks. Fixed-bank regret and
concordance with stable seed sign, followed by a small closed-loop safety
confirmation, are required before any grid. No planner-aware model was trained
in Task E.

## Limitations

This is an eight-start, two-bank-seed development diagnosis, not a new
adaptive-control claim. Binary success is not meaningful for all 25-step
open-loop candidates. Proximity is an analysis-only proxy rather than a model
input. Operation counts remain Task C's incomplete counted-work proxy; no
exact-complete-FLOP claim is made.
