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

## Phase A results (run 2026-10-06, ~4 min compute; `lewm probe-size-control`)

Pool: 2800 frames from 28 held-out episodes (14 per training dataset × 100 frames; the val-split
intersection caps it below 4000 per dataset). Peg errors in px, [episode-bootstrap 95% CI],
paired across radii. Full table incl. T, ratios and flags: `size_control_phaseA_full.md`.
Run dir: `~/lewm-work/runs/size-control/phaseA/`.

| Checkpoint | Kind | Peg r15 | Peg r30 | Peg r45 | T r45 | Peg/mean r45 | Peg/pixel r45 | 15→45 reduction |
|---|---|---|---|---|---|---|---|---|
| ft_block_s0 | cls | 139.1 [136.7, 141.5] | 138.6 [136.3, 141.1] | 137.6 [135.1, 140.2] | 19.4 [16.2, 23.3] | 0.98 [0.97, 0.99] | 9.47 [6.47, 14.05] | 1.5 [0.5, 2.3] |
| ft_block_s0 | projected | 141.8 [139.1, 144.6] | 142.2 [139.5, 145.3] | 142.8 [139.7, 146.3] | 24.9 [20.7, 30.1] | 1.02 [1.00, 1.03] | 9.82 [6.69, 14.59] | -1.0 [-2.6, 0.3] |
| ft_block_s1 | cls | 139.1 [136.7, 141.5] | 138.5 [136.1, 140.9] | 137.3 [134.9, 139.8] | 20.9 [17.4, 25.1] | 0.98 [0.97, 0.99] | 9.45 [6.46, 14.03] | 1.8 [0.9, 2.6] |
| ft_block_s1 | projected | 141.6 [139.0, 144.4] | 142.1 [139.4, 145.0] | 142.7 [139.7, 146.0] | 24.1 [20.4, 28.6] | 1.01 [1.00, 1.03] | 9.82 [6.69, 14.57] | -1.1 [-2.5, 0.1] |
| ft_mixed_s0 | cls | 139.3 [136.7, 141.8] | 139.4 [136.8, 142.1] | 139.3 [136.6, 142.1] | 20.4 [18.0, 23.4] | 0.99 [0.98, 1.00] | 9.58 [6.55, 14.22] | -0.1 [-0.6, 0.4] |
| ft_mixed_s0 | projected | 140.9 [138.3, 143.6] | 141.2 [138.5, 143.9] | 141.5 [138.7, 144.4] | 26.4 [21.7, 31.8] | 1.01 [0.99, 1.01] | 9.74 [6.66, 14.45] | -0.6 [-1.2, -0.0] |
| ft_mixed_s1 | cls | 139.4 [136.9, 142.0] | 139.6 [137.0, 142.3] | 139.4 [136.7, 142.2] | 20.8 [17.6, 24.5] | 0.99 [0.98, 1.00] | 9.59 [6.56, 14.23] | 0.0 [-0.5, 0.5] |
| ft_mixed_s1 | projected | 141.2 [138.5, 143.9] | 141.5 [138.7, 144.3] | 141.7 [138.9, 144.6] | 27.3 [22.9, 32.3] | 1.01 [1.00, 1.02] | 9.75 [6.68, 14.46] | -0.5 [-1.1, -0.0] |
| lewm-pusht | cls | 138.5 [135.2, 141.6] | 121.3 [109.8, 134.5] | 92.1 [77.5, 109.8] | 36.2 [29.9, 43.8] | 0.65 [0.55, 0.78] | 6.27 [3.97, 10.04] | 46.4 [30.2, 59.5] |
| lewm-pusht | projected | 141.1 [137.8, 144.3] | 141.4 [137.0, 145.8] | 139.5 [133.1, 146.4] | 44.6 [34.0, 57.8] | 0.99 [0.96, 1.03] | 9.50 [6.27, 14.32] | 1.6 [-3.6, 5.6] |

| Baseline (peg error, px) | r15 | r30 | r45 |
|---|---|---|---|
| Pixel ridge 32x32, no decoration (fine-tuned) | 116.6 [77.5, 169.8] | 30.1 [22.8, 38.2] | 14.5 [9.7, 21.4] |
| Pixel ridge 32x32, green-T decoration (pretrained) | 115.2 [76.9, 167.0] | 30.6 [23.0, 38.9] | 14.7 [9.7, 22.0] |
| Mean predictor | 140.7 [138.0, 143.3] | 140.7 [138.0, 143.3] | 140.7 [138.0, 143.3] |

### Pre-stated reading, applied mechanically

All 8 fine-tuned cells are **size-unexplained** (peg error at r45 = 0.98–1.02 × mean predictor,
threshold 0.8). Radius-15 sanity check passed in all 8 (deficit reproduced). Flags: the 15→45
reduction CI straddles 0 in 4 cells; no cell is near the 0.8 × mean threshold.

**Overall reading: "size alone does not explain it."** Per the design, this means test-time
enlargement does not rescue frozen-encoder readout; it does not exclude a training-time size
effect and says nothing directly about planning.

### Interpretation

At radius 45 the peg is easy to read linearly from raw 32×32 pixels (14.5 px), yet every
fine-tuned checkpoint's CLS and projected embedding stays at mean-predictor level (~140 px) while
reading the T to 19–27 px, a peg/pixel ratio of ~9.5. Enlarging the peg to T scale changes
nothing for the fine-tuned encoders (largest reduction 1.8 px). The pretrained checkpoint, which never saw a
peg in training, does partly pick up a large one in CLS (138 → 92 px), so the fine-tuned encoders look
specifically insensitive to the orange disc rather than unable to see objects of that size.
Caveats: this pool differs from the main readout pool (no condition endpoints; narrower peg
support), so absolute numbers are not comparable (mean baseline 141 vs 186 px; r15 pixel ridge
117 px here vs 19 px there). Within this pool, r15 is therefore uninformative about a deficit
relative to pixels; the informative contrast is r45. The CIs come from only 28 episode clusters.
Big pegs are out of distribution for these frozen encoders, which Phase B addresses.

## Phase B design (written 2026-10-06, before any Phase B number was computed)

User approved the GPU spend (cap $3) on 2026-10-06.

- Data: `lewm collect --peg-radius 45`, block and mixed policies, 2000 × 100 steps, clutter,
  seed 1 (`peg45_blockpolicy.h5`, `peg45_mixedpolicy.h5`). One deviation from "existing collector",
  found in code review before collection: the mixed policy's peg-episode target box was a
  hardcoded ±30 px (= agent+peg contact distance at radius 15), which sits inside a radius-45
  peg and makes the agent shove the peg near-continuously. It now scales as radius + 15 (30 at
  radius 15, byte-identical old datasets; 60 at 45). Peg-contact frames: mixed 39.5% (37.2% at
  r15); block 5.6% (2.9% at r15; incidental contact rises with peg size).
- Fine-tuning: `ft45_block_s0`, `ft45_mixed_s0` from lewm-pusht, Step-3 recipe (2000 steps,
  batch 128, bf16, seed 0), one Vast GPU.
- Evaluation: G only (move_peg vs move_T_matched), N=100 bases built at radius 45 with new
  reserved construction master **310000000** (main used 300000000; dry-run 290000000).
  Frozen settings: budget 50, LeWM CEM 300×30 top-30 seed 42 (no T decoration for fine-tuned
  arms), reference CEM 300×30 top-30, approach weight 0.1, F=101, E=202, frozen W4d
  normalization, success tolerances unchanged. Primary set: common F-feasible bases.
- Readout: the Phase A pool and procedure at radii 15 and 45 for ft45 checkpoints.

### Pre-stated Phase B reading (per arm; original gap = matching s0 checkpoint, common_feasible)

Original G gaps: ft_block_s0 0.688, ft_mixed_s0 0.666.

- **size explains the planning failure** if big-peg gap < 0.15 AND the gap's 95% CI upper bound
  is below the original gap;
- **relevance (P1) supported** if big-peg gap ≥ 0.15 AND the protocol's linear-readout deficit
  persists at radius 45 (peg error ≥ 2 × T error AND ≥ 1.5 × pixel-ridge error, cls and projected);
- otherwise **ambiguous**.

Overall: both arms give the same label → that label; otherwise "mixed/ambiguous". Reference-E
gap is reported as the task-difficulty control (if it is itself ≥ 0.15, that is flagged).

## Phase B results

_Pending._

## Out-of-scope ideas

- Why the main-pool pixel ridge reads a radius-15 peg to 19 px but this pool's only to 117 px
  (condition-endpoint frames vs dataset frames; peg-position distribution). Worth one look before
  quoting the main-run peg/pixel ratio as evidence on its own.
- The pretrained encoder reads the big peg better than fine-tuned ones: a direct test of whether
  fine-tuning on peg data actively suppresses the peg (e.g. readout across fine-tuning steps
  using the 250-step checkpoints, if retained).
