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

## Results

_Pending._
