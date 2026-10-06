# Size control: is the peg dropped because it is small?

Follow-up to the Step-4 main run (RESULTS.md). Question: is the peg-readout/planning
deficit explained by the peg being small (radius 15 px of 512 → ~6.6 px in the 224-px
input: diameter ~13 px, about one 14-px ViT patch and often split across patches),
rather than by its changed role (P1)?

## Phase A design (written 2026-10-06, before any size-control numbers were computed)

- Env: `PushTPeg(peg_radius=...)`, default 15 (unchanged). Nothing else in the env changes.
- Pool: the same held-out validation episodes as the main readout (intersection of the
  recorded val splits per training dataset), 4000 frames per dataset sampled with seed 42.
  Agent and T poses come from those dataset states. Main-run condition endpoints are not
  used. The peg position of each frame is replaced by one draw (seed 42) that is uniform
  over [30+45, 512-30-45]^2 and does not overlap the agent or T at **radius 45**. So all
  three radii share identical agent, T and peg positions, and only the peg radius changes.
- Radii: 15, 30, 45. Every frame is re-rendered at every radius (no dataset pixels are
  reused), so radius 15 is the in-experiment control. Rendering follows each checkpoint's
  frozen rule: lewm-pusht with the fixed green-T decoration, fine-tuned checkpoints without.
- Readout: unchanged `nested_readout` (ridge, 5 outer/3 inner episode-grouped folds, seed 42,
  2000 cluster bootstrap samples) on CLS and projected embeddings, plus mean-predictor and
  32x32 pixel-ridge baselines. Checkpoints: lewm-pusht, ft_block_s0/s1, ft_mixed_s0/s1.
- Reported: peg error, T error, pixel-baseline peg error, mean-baseline peg error, and
  peg/T and peg/pixel ratios with episode-cluster bootstrap 95% CIs.

### Pre-stated reading (applied mechanically to radius 45, fine-tuned checkpoints)

Cells = 4 fine-tuned checkpoints × {cls, projected} = 8 cells, point estimates. Labels are
assigned in this order, so they are mutually exclusive:

1. **size-unexplained** if peg error > 0.8 × mean-predictor peg error (radius 45);
2. else **size-explained** if peg error ≤ 2 × T error AND peg error ≤ 1.5 × pixel-ridge peg
   error (both radius 45) AND peg error drops from radius 15 to 45 with a paired
   episode-bootstrap 95% CI of the reduction excluding 0;
3. else unlabeled.
- Overall reading:
  - all 8 cells size-explained → **"size substantially explains the readout deficit"**;
  - all 8 cells size-unexplained → **"size alone does not explain it"**;
  - anything else → **"partial/ambiguous"**.

lewm-pusht is reported alongside but does not enter the reading. Folds and bootstrap draws
are identical across radii (same episode groups and seeds), so radius comparisons are paired.
Sanity check: the radius-15 re-rendered pool must reproduce the deficit (peg > 0.8 × mean);
if it does not, the result concerns this pool only and is reported as such. Point estimates
whose CI straddles a threshold are flagged. Scope of a negative result: it means test-time
enlargement does not rescue frozen-encoder readout; it cannot exclude a training-time size
effect (that is Phase B) and says nothing directly about planning.

Design reviewed by gpt-6-astra (read-only) before running; adopted: ordered/mutually exclusive
labels, required paired 15→45 reduction, radius-15 sanity check, corrected patch rationale.

Phase B (big-peg fine-tuning + G contrast at N=100) runs only if the reading is not
"size substantially explains", and only after the user approves the GPU spend.

## Phase A results

_Pending._
