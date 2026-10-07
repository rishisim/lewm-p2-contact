# Pin check 2: training duration vs objective-level blindness

## Protocol note (dated 2026-10-06, 19:55 CDT; written before any pin2 training or evaluation)

**Question.** Does the G gap (LeWM cannot plan move_peg but can plan move_T_matched) come from
pretraining lock-in / insufficient training, or from objective-level blindness? Every earlier
fine-tune was 2,000 steps from the T-focused pretrained `lewm-pusht`.

**Arms** (dataset `peg_mixedpolicy.h5`, existing recipe: batch 128, bf16, upstream AdamW lr 5e-5 /
wd 1e-3, LinearWarmupCosine schedule over the run's own step budget, grad clip 1.0, seed 0):
- `pin2_ft_mixed_long`: fine-tune from `lewm-pusht`, 10,000 steps.
- `pin2_scratch_mixed`: random init, same architecture/config as `lewm-pusht`, 20,000 steps
  (new `--from-scratch` flag in `lewm finetune`; no other training-semantics change).
Both runs share one rented GPU. Validation interval is 1,000 steps (instead of 250) to save
wall time. This affects only logging, not training. Checkpoints are written every 250 steps.

**Evaluation (exploratory reuse, declared here).** This reuses parts of the main run instead of
using a fresh construction:
- Bases: the FIRST 100 bases of `~/lewm-work/runs/main/construction/bases.json`, in file order.
- G contrast only: move_peg vs move_T_matched.
- Frozen protocol LeWM settings: budget 50, CEM population 300 x 30 iterations, top-30,
  seed namespace 42 (episode seed = sha256(42:base:condition), so episodes match the main run),
  frozen normalization `w4d-calibration`, fine-tuned-arm rendering (`--no-with-target`), CUDA.
- Reused, not rerun: main-run reference-F (feasibility) and reference-E results, and the main
  run's `ft_mixed_s0` episodes on the same 100 bases as the matched 2k-step control.
- Common feasible set: the main report's definition (reference-F feasible), intersected across
  all compared arms.
- Linear peg/T readout: the existing `probe-readout` procedure on the same frame pool as the main
  run (seed 42, default options). I verify pool identity against main `readout/pool.npz`.
- Because of the reuse, n=100, and one seed per arm, this is exploratory. It is not a
  confirmatory result.

**Pre-stated reading (applied mechanically).**
Definitions: G gap = success(move_T_matched) - success(move_peg) in percentage points on the
common feasible set, with a paired 95% bootstrap CI (report conventions, 2000 samples, seed 42).
Readout numbers use the **projected** embedding (the planner's latent) as primary. cls is reported
as secondary and does not change the reading. "peg readout" = held-out linear peg-position error
(px), "mean-predictor error" = the readout's mean baseline, "T readout error" = held-out linear
T-position error, all from the same readout run.
1. Competence gate (scratch only): move_T_matched success >= 40% on the common feasible set,
   else "scratch not competent - uninformative" and scratch is excluded from steps 2-4.
   `pin2_ft_mixed_long` is always informative.
2. **objective-level blindness**: every informative arm has G gap >= 15 pts with 95% CI
   excluding 0, AND peg readout > 0.8x mean-predictor error.
3. **lock-in / insufficient training**: any informative arm has G gap < 15 pts AND peg readout
   <= 2x T readout error.
4. otherwise **partial**.

**Fallbacks if the 3-hour budget (19:48-22:48 CDT) or the $5 cap is threatened** (in order, each
recorded): (1) cut scratch to 12,000 steps (its LR schedule is then set to 12,000); (2) evaluate
on the first 60 bases; (3) drop the scratch arm and report the long arm only.

### Amendment 1 (2026-10-06, 20:25 CDT; before any pin2 evaluation or readout)

- Instance A (Vast 54563887, H100 NVL, $2.756/h) ran at 0.32 s/step for one run alone and
  0.70 s/step per run with both concurrent, so sharing the GPU gave no gain. That is 2.3x slower
  than the earlier H100 SXM. Under the $5 cap, fallback (1) was applied first (scratch restarted
  with a 12,000-step budget). The projection still did not fit, so fallback (3) was applied at
  20:06 (scratch dropped; the long arm continued alone).
- 20:08: the user raised the cap to $10 and asked for a better GPU. Fallbacks (1) and (3) were
  reversed. Scratch was restarted at the **original 20,000 steps** on instance B (Vast 54565474,
  H100 SXM, $2.353/h). B's driver (CUDA 12.5) cannot run the locked torch 2.14.1+cu130, so B uses
  **torch 2.14.1+cu126** (same version, different CUDA build). The 12k/A-scratch partial runs
  (<=300 steps) were abandoned and are unused.
- Long-arm training is unchanged: 10,000 steps on A from step 0, alone after step ~300. For
  ~4 minutes it shared A with the abandoned scratch run, which affects speed only.
- Evaluation episodes for each arm run as two parallel probe-eval processes, one per G condition.
  Episode seeds depend only on (42, base, condition), so this matches a single combined run.

## Results (2026-10-06, 21:35 CDT)

### Training curves (val_pred_loss on held-out episodes, seed-0 split)

| Step | pin2_ft_mixed_long (from lewm-pusht) | pin2_scratch_mixed (random init) | ft_mixed_s0 (main, 2k) |
| ---: | ---: | ---: | ---: |
| 1,000 | 0.0302 | 0.2220 | 0.0288 |
| 2,000 | 0.0276 | 0.1395 | 0.0258 |
| 3,000 | 0.0262 | 0.0879 | |
| 5,000 | 0.0240 | **0.0738** (minimum) | |
| 7,000 | 0.0220 | 0.1276 | |
| 10,000 | **0.0218** (final) | 0.1632 | |
| 15,000 | | 0.1712 | |
| 20,000 | | **0.1909** (final) | |

- Long arm: val_pred_loss improves steadily, then plateaus from ~7k steps (0.0258 at 2k -> 0.0218
  at 10k). Train pred loss is 0.013 (mean of last 100 steps). Wall time 4,414 s at 0.34 s/step on
  the H100 NVL.
- Scratch arm: train pred loss falls to 0.011, but val_pred_loss bottoms at 5k and then rises
  2.6x. **It overfits the ~900-episode dataset.** Wall time 3,521 s at 0.12 s/step on the
  H100 SXM. Val SIGReg is ~43 for all arms (diagnostic only; ordered windows inflate it).
- Loss values are not comparable across differently initialized encoders: each arm predicts in
  its own latent space.

### G contrast: first 100 main-run bases, reference-F common feasible set (n = 85, identical for all arms)

| Arm | Steps | move_peg % (95% CI) | move_T_matched % (95% CI) | G gap pp (95% CI) |
| --- | ---: | --- | --- | --- |
| ft_mixed_s0 (main run, reused) | 2,000 | 2.4 [0.0, 5.9] | 67.1 [56.5, 77.6] | 64.7 [52.9, 76.5] |
| **pin2_ft_mixed_long** | 10,000 | 0.0 [0.0, 0.0] | 72.9 [63.5, 82.4] | **72.9 [63.5, 82.4]** |
| **pin2_scratch_mixed** | 20,000 | 1.2 [0.0, 3.5] | **8.2 [2.4, 14.1]** | 7.1 [1.2, 12.9] |
| reference-E (main run, reused) | - | 100.0 [100.0, 100.0] | 98.8 [96.5, 100.0] | -1.2 [-3.5, 0.0] |

Paired bootstrap CIs (report `_contrast`, 2000 samples, seed 42). Every arm has all 85 feasible
bases paired, with no missing episodes. Source: `~/lewm-work/runs/pin-training/analysis.json`
(script `experiments/role_swap/pin_training_analysis.py`).

### Linear readout (main-run frame pool; pool identity c623fbfd... matches main)

Held-out ridge error in px (95% CI). Mean-predictor baseline: peg 186.08, T 151.56.

| Checkpoint | Feature | Peg error | T error | Peg / mean | Peg / T |
| --- | --- | --- | --- | ---: | ---: |
| ft_mixed_s0 (main) | projected | 164.13 [153.44, 174.97] | 17.45 [16.36, 18.70] | 0.882 | 9.40 |
| **pin2_ft_mixed_long** | **projected** | 163.32 [152.49, 174.46] | 15.95 [14.77, 17.54] | **0.878** | **10.24** |
| pin2_ft_mixed_long | cls | 161.02 [149.62, 172.55] | 13.76 [12.96, 14.69] | 0.865 | 11.70 |
| pin2_scratch_mixed | projected | 172.34 [161.30, 184.30] | 51.18 [48.10, 54.50] | 0.926 | 3.37 |
| pin2_scratch_mixed | cls | 164.10 [152.70, 176.50] | 46.00 [43.50, 48.40] | 0.882 | 3.57 |

Driver: `experiments/role_swap/pin_training_readout.py`. It adds the pin2 arms (same dataset and
seed-0 split as ft_mixed_s0) to the split list. The pool is unchanged, and the script reproduces
the main ft_mixed_s0 numbers exactly.

### Pre-stated reading, applied mechanically

1. Competence gate (scratch): move_T_matched = 8.2% < 40%, so **"scratch not competent -
   uninformative"**. Scratch is excluded from steps 2-4.
2. Informative arms: `pin2_ft_mixed_long` only. Its G gap is 72.9 pts >= 15 with CI [63.5, 82.4]
   excluding 0, and its projected peg readout 163.3 px > 0.8 x 186.1 = 148.9 px (cls: 161.0 also
   passes). **Both clauses hold for every informative arm.**
3. Lock-in clause: no informative arm has G gap < 15, so it does not apply.

**Reading: objective-level blindness.** It rests on one informative arm, the 5x-longer
fine-tune. The from-scratch test of pretraining lock-in was uninformative.

What this means: 5x more fine-tuning on mixed-policy data (peg pushed in ~37% of frames) does not
close the gap or improve peg encoding. move_peg stays at 0/85 while move_T_matched rises slightly
(67 -> 73%). Peg readout stays at mean-predictor level (0.88x) while T readout improves
(17.5 -> 15.9 px). Validation loss plateaued, so more steps of the same objective are unlikely to
help. The lock-in alternative is **not excluded**: the scratch model never became a competent
planner, even for T, so it cannot say whether a non-T-pretrained encoder would represent the peg.
Its readout shows no peg encoding either (0.93x mean), but that comes from an incompetent,
overfit model. Exploratory: n = 85 paired bases, one seed per arm, reused bases and controls.

### Fallbacks used

- (1) cut scratch to 12k: applied at 20:02 on instance A, then **reversed** at 20:08 when the cap
  was raised to $10 (Amendment 1). The final scratch arm ran the full 20,000 steps.
- (2) 60 bases: **not used**. Both arms were evaluated on all 100 bases.
- (3) drop scratch: applied at 20:06, then **reversed** at 20:08 (same reason).

### Cost / time

- Instance A (54563887, H100 NVL, 64 vCPU): 19:51-21:33 CDT. Ledger: GPU $4.515 + storage $0.038
  + network $0.030 = **$4.58**. Ran the long-arm training and eval.
- Instance B (54565474, H100 SXM, 32 vCPU): 20:08-21:22 CDT. Ledger: GPU $3.060 + storage $0.026 =
  **$3.09** (network charges not yet posted at write-up). Ran scratch training and eval.
- **Total ~$7.67**, under the user-raised $10 cap (originally $5). Both instances were destroyed
  and verified with `vastai show instances` -> `[]`. Independent local watchdogs targeted only
  54563887 and 54565474 and were cancelled after destruction. The temporary SSH key was deleted.
- Wall time: 19:48-~21:40 CDT (about 1 h 52 min of the 3 h limit).
- Eval speed differed by host: ~10 s/episode on A vs ~1.5-2 s/episode on B, which is CPU-bound.
  B used torch 2.14.1+cu126 (old driver). A and the main run used cu130/MPS respectively.
- Artifacts (outside git): checkpoints `~/lewm-work/stable-worldmodel/checkpoints/pin2_{ft_mixed_long,scratch_mixed}/`
  (weights sha256 b8c1f255... / cf7ecb31..., verified against remote), episodes, logs and readouts
  under `~/lewm-work/runs/pin-training/`. Remote training-run records are in `A/finetune`, `B/finetune`.
  The abandoned <=300-step scratch attempt on A is in `remote_sync/pin2_scratch_mixed_ABANDONED_on_A` (unused).

### Out-of-scope ideas (not run)

- The scratch arm overfits: val loss is minimal at 5k steps. A from-scratch check of lock-in would
  need more data (e.g. adding pusht_expert_train) or early stopping at the val minimum. Its
  competence gate should be checked before any G reading.
- Evaluate the long arm's intermediate checkpoints (2k/5k/7k) to see whether move_T gains
  plateau with val loss.
- Probe whether peg information exists before the projector (patch tokens rather than CLS)
  in the long arm. That would separate "encoder discards the peg" from "CLS summary discards it".
