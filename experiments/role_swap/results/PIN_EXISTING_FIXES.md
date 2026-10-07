# Pin check 3: does an existing fix make LeWM keep and plan with the peg?

Branch `research/pin-existing-fixes` (from `research/role-swap-probe` @ 2ef3ea1). Started 2026-10-06 19:48 CDT; hard limit 3 h.

## Protocol note (2026-10-06, written before any pin3 result exists)

**Status: exploratory.** This check reuses the main run's bases, reference runs and ft_mixed_s0 episodes. It is not part of the frozen PROTOCOL.md decision chain.

**Question.** Does an existing fix already make LeWM keep the peg in its latent and plan with it?

**Arms.** Both start from `lewm-pusht` and train on `peg_mixedpolicy.h5` for 2,000 steps with batch 128, bf16, seed 0, AdamW lr 5e-5 / wd 1e-3 and the unchanged LeJEPA loss (pred + 0.09·SIGReg). This is identical to `ft_mixed_s0` except for one added term:
- `pin3_idm`: inverse-dynamics auxiliary loss from arXiv 2610.03137 (LeWM+I), weights α_inv = α_pid = 1 (the paper's values). See "Matched from 2610.03137" below.
- `pin3_pegsup` (privileged upper bound): a linear head `Linear(192→4)` on the projected embedding `emb` of all 4 window frames. It predicts (peg x, peg y, T x, T y) from `state[..., [7,8,2,3]]`, normalized as (coord − 256)/256. Loss is MSE averaged over elements, weight 1.0 (pre-declared). The head is training-only and discarded: `weights.pt` holds the plain LeWM.

Auxiliary heads are initialized from a dedicated generator, so the global RNG stream (data order, SIGReg projections) matches `ft_mixed_s0`. With zero weights, the code path is bit-identical to the current loss (tested).

**Evaluation (reuse).**
- **Bases:** the first 100 bases of `~/lewm-work/runs/main/construction/bases.json`, in file order.
- **Contrast:** G only (hard = `move_peg`, control = `move_T_matched`).
- **Planner:** frozen LeWM settings: budget 50, CEM 300×30 top-30, seed namespace 42, fine-tuned rendering without the green target decoration (`--no-with-target`), W4d normalization.
- **Episode seeds** are `episode_seed(42, base_id, condition)`, so they match the main `ft_mixed_s0` episodes exactly (asserted in the analysis).
- **Device:** LeWM evaluation runs on a rented CUDA GPU. The main run used MPS, so planner numerics can differ slightly; the `ft_mixed_s0` control is reused from the MPS episodes.
- **Reused, not rerun:** reference-F feasibility and reference-E gap come from the main run.
- **Readout:** the existing nested grouped ridge readout of peg/T xy, on the main run's persisted pool (identical frames, renders, folds and bootstrap).

**Primary quantities (pre-declared).**
- *G gap* = control − hard success on the **common_feasible** set: the first 100 bases that reference-F found feasible, using the main report's rule. The unconditional set is also reported.
- *Readout criterion* uses the **projected** embedding (the space the planning cost uses), ridge error in px. The cls readout is reported but not used for the reading.

**Pre-stated reading (applied mechanically).** "Meets criteria" means G gap point estimate < 15 pts AND projected peg readout error ≤ 2 × projected T readout error.
1. **"existing fix suffices"**: pin3_idm meets criteria.
2. **"privileged supervision fixes, IDM does not"**: pin3_pegsup meets criteria and pin3_idm does not. The gap stays open for a label-free method.
3. **"neither fixes planning"**: both G gaps ≥ 15 pts. Then note whether readout improved: an arm "improved readout" if it passes the ≤ 2× readout criterion; "partly improved" if its projected peg error falls ≥ 25% below ft_mixed_s0's (164.1 px). If readout passes but planning fails, the failure is in the latent-distance cost, not in missing information.
4. **"partial"**: anything else (e.g. a gap < 15 pts without the readout criterion).

**Fallbacks if the 3-hour budget is threatened (apply in order, record which):** (1) evaluate on the first 60 bases; (2) drop pin3_pegsup and report pin3_idm only.

**Not done:** other arms, sweeps, seeds, conditions, D′/near_path, A/B/C, DINO-WM, new methods.

## Matched from 2610.03137

(filled in below the line after training; nothing above this line is changed after results)

---
