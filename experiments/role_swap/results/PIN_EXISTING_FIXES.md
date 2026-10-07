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

Read from the arXiv abstract and HTML method section (time-boxed, about 10 min).

| Item | Paper (LeWM+I) | pin3_idm | Match |
|---|---|---|---|
| Encoder-IDM term | ‖h(z_t, z_{t+1}) − a_t‖², every consecutive pair in the window (3 pairs at context 3) | same: 3 pairs of the 4-frame window | ✓ |
| Predictor-IDM term | ‖h(sg(z_t), ẑ_{t+1}) − a_t‖², last context position only; updates predictor and head | same: h(sg(z_2), pred[:, −1]) vs a_2 | ✓ |
| Shared head | 3-layer MLP 384→256→256→10 (≈167K params), discarded at test time | same dims; `weights.pt` = plain LeWM | ✓ |
| Head activation | not stated in what I read | GELU (pre-declared) | assumption |
| Latent z | encoder latent, D = 192, same space the predictor predicts | projected `emb` (D = 192), the predictor's space | ✓ |
| Loss reduction | squared norm (sum over action dims) | sum over the 10 action dims, mean over batch/pairs | ✓ (the text also says "MSE"; read as squared norm per Eq. 2–3) |
| Actions | frameskip-packed (5 env steps), 10-D | same packed actions, z-scored by the lewm-pusht normalization (the model's own input); paper normalization not stated | assumption |
| Weights | (λ_sigreg, α_inv, α_pid) = (0.09, 1, 1) | (0.09, 1, 1) | ✓ |
| Optimizer | AdamW lr 5e-5, wd 1e-3, batch 128, bf16 | same; head in the same optimizer and grad-clip | ✓ |
| Training | from scratch, 30 epochs on SLIM | **fine-tune** of lewm-pusht, 2,000 steps on peg_mixedpolicy | **deviation** (fixed by this check's design; the paper reports no fine-tuning) |

Consequence of the matched weighting: at step ~20 the encoder-IDM term was ≈38 against pred_loss ≈0.08, so the IDM terms dominate the gradient early (with grad-norm clipping at 1.0, as in the existing recipe). This is what the paper's weights imply under z-scored actions. It is recorded, not tuned.

## Training

Both arms trained concurrently on one A100-SXM4-80GB (Vast instance 54564256) from 01:04:51Z to 01:25:30Z, ≈20.5 min for 2,000 steps each. Recipe as declared. Final logged values (step 2000):

| Run | train_pred_loss | val_pred_loss | aux terms (val) |
|---|---:|---:|---|
| ft_mixed_s0 (reference, finetune_runs.md) | 0.0225 | 0.0258 | — |
| pin3_idm | 0.6046 | 0.5860 | enc-IDM 31.35, pred-IDM 3.78 |
| pin3_pegsup | 0.0227 | 0.0264 | pegsup 0.0107 (normalized-coordinate MSE) |

The IDM arm's world-model prediction loss ended **~23× worse** than ft_mixed_s0's. It spiked to 2.1 at step 250 and recovered only partly. The encoder-IDM term barely moved (≈38 → 31). The chance level is high because the peg datasets' raw actions have std ≈0.43 per axis, while the model z-scores with lewm-pusht stats (std ≈0.21). Normalized actions therefore have ≈4.3 variance per dim, ≈43 summed over 10 dims. The predictor-IDM term fell to ≈3–4, but the predictor receives the action as input; the paper's own ablation shows this term alone does nothing. The privileged head trained cleanly without hurting prediction.

## Planning: G contrast, first 100 main-run bases

Success at budget 50. Gap = control (move_T_matched) − hard (move_peg). 95% paired cluster-bootstrap CIs (seed 42, 2,000 samples). Computed with the main report's internals; reference-E and feasibility come from the main run. Episode seeds were asserted identical to the main ft_mixed_s0 episodes.

| Arm | Set | Bases | Hard success | Control success | Gap control−hard (95% CI) | Reference-E gap | Gap − ft_mixed_s0 gap (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ft_mixed_s0 (main episodes, MPS) | common_feasible | 85 | 0.024 [0.000, 0.059] | 0.671 [0.565, 0.776] | **0.647** [0.529, 0.765] | −0.012 [−0.035, 0.000] | — |
| pin3_idm | common_feasible | 85 | 0.012 [0.000, 0.035] | 0.235 [0.141, 0.329] | **0.224** [0.129, 0.318] | −0.012 [−0.035, 0.000] | −0.424 [−0.565, −0.271] |
| pin3_pegsup | common_feasible | 85 | 0.012 [0.000, 0.035] | 0.529 [0.424, 0.635] | **0.518** [0.400, 0.624] | −0.012 [−0.035, 0.000] | −0.129 [−0.271, 0.000] |
| ft_mixed_s0 (main episodes, MPS) | unconditional | 100 | 0.020 [0.000, 0.050] | 0.600 [0.510, 0.700] | 0.580 [0.480, 0.690] | −0.130 [−0.200, −0.070] | — |
| pin3_idm | unconditional | 100 | 0.010 [0.000, 0.030] | 0.200 [0.120, 0.280] | 0.190 [0.110, 0.270] | −0.130 [−0.200, −0.070] | −0.390 [−0.520, −0.270] |
| pin3_pegsup | unconditional | 100 | 0.010 [0.000, 0.030] | 0.470 [0.370, 0.570] | 0.460 [0.360, 0.560] | −0.130 [−0.200, −0.070] | −0.120 [−0.240, −0.010] |

**Hard (move_peg) success stays at the floor in every arm (1–2%).** pin3_idm's smaller gap comes entirely from its control collapsing (0.67 → 0.24), consistent with the degraded predictor. It does not come from planning the peg move.

## Readout: linear peg/T position, main run's persisted pool

Nested grouped ridge, identical pool/renders/folds as the main readout. The pool's mixed-policy episodes were asserted absent from the pin3 arms' training episodes. Error in px (95% CI).

| Arm | Readout | Peg error | T error | Peg/T ratio |
| --- | --- | --- | --- | --- |
| ft_mixed_s0 (main) | projected | 164.1 [153.4, 175.0] | 17.5 [16.4, 18.7] | 9.40 [8.45, 10.33] |
| pin3_idm | projected | 107.2 [101.2, 113.7] | 33.2 [31.0, 35.2] | 3.23 [2.98, 3.53] |
| pin3_pegsup | projected | 19.2 [17.6, 21.1] | 14.4 [13.4, 15.4] | 1.34 [1.20, 1.48] |
| ft_mixed_s0 (main) | cls | 161.9 [150.3, 173.5] | 14.4 [13.7, 15.3] | 11.23 |
| pin3_idm | cls | 87.6 [82.3, 93.4] | 30.9 [29.0, 32.9] | 2.83 |
| pin3_pegsup | cls | 17.1 [16.0, 18.4] | 15.0 [14.0, 16.0] | 1.15 |
| mean predictor | — | 186.1 [178.3, 193.8] | 151.6 [141.5, 161.1] | — |

## Pre-stated reading, applied

Criteria: G gap (common_feasible point estimate) < 15 pts AND projected peg error ≤ 2× projected T error.

| Arm | G gap | < 15 pts? | Peg/T (projected) | ≤ 2×? | Meets criteria |
|---|---:|---|---:|---|---|
| pin3_idm | 22.4 pts | no | 3.23 | no | **no** |
| pin3_pegsup | 51.8 pts | no | 1.34 | yes | **no** |

Both G gaps are ≥ 15 pts, so the reading is **3: "neither fixes planning"**. Readout clause:
- **pin3_pegsup: improved readout** (passes ≤ 2×; peg error 164 → 19 px). The peg position is now linearly present in the projected embedding the planner uses, yet move_peg success stays at 1%. As pre-stated, the failure is then in the latent-distance planning cost, not missing information.
- **pin3_idm: partly improved** (projected peg error 164 → 107 px, −35% ≥ 25%; ratio 3.2 > 2). Its planning is worse overall: control success fell to 0.24.

Not "existing fix suffices": the paper's IDM loss, matched to its form and weights, does not make LeWM plan the peg move here. Not "privileged supervision fixes": even privileged position labels fix the information but not the planning.

**Caveats.**
- Exploratory reuse of the first 100 bases (85 common-feasible), so CIs are wide.
- The pin3 arms were evaluated on CUDA; the ft_mixed_s0 control episodes were evaluated on MPS. Planner numerics differ slightly, but hard success is at the floor either way.
- IDM was matched as a 2,000-step fine-tune, not the paper's from-scratch training. Its action-normalization scale is an assumption that makes the IDM term dominate (see Training). A weaker IDM weight or dataset-z-scored actions might not damage the predictor; that was not tested (no sweeps by design).
- One seed per arm.

## Fallbacks, time, cost

- **Fallbacks:** none used. Full 100 bases, both arms.
- **Time:** started 19:48 CDT; GPU work 19:55–20:37 CDT; results ≈20:45 CDT; write-up done within the 3 h limit.
- **Cost:** Vast instance 54564256 (A100-SXM4-80GB, $1.022/h incl. 60 GB disk). Ledger: GPU $0.771 + storage $0.017 + download $0.020 + upload $0.014 = **$0.82** (cap $3). Destroyed 01:37:19Z and verified absent from `vastai show instances`. The watchdog (destroying only 54564256) was cancelled after the manual destroy, and the temporary SSH key was removed.
- **Artifacts:**
  - checkpoints `~/lewm-work/stable-worldmodel/checkpoints/{pin3_idm,pin3_pegsup}/` (SHA-256 verified against the instance; trainer_state contains the discarded aux heads);
  - eval runs `~/lewm-work/runs/pin-fixes/eval/{pin3_idm,pin3_pegsup}/`;
  - readout `~/lewm-work/runs/pin-fixes/readout/`;
  - analysis `~/lewm-work/runs/pin-fixes/analysis/analysis_n100.{json,md}`;
  - instance logs `~/lewm-work/runs/pin-fixes/logs_vast_54564256/`.
- **Code:** `lewm finetune --idm-weight / --pegsup-weight` (zero-weight path byte-identical, tested), `lewm probe-pool-readout`, `experiments/role_swap/pin_fixes_analysis.py`. Tests: `tests/test_finetune.py`, `tests/test_pin_fixes.py` (34 passed).
- **Process note:** a parallel session switched the shared checkout's branch mid-run. This check moved to its own worktree (`../lewm-p2-contact-pin-existing`), and its pre-registration commit 9889639 was restored to this branch unchanged.

## Out-of-scope ideas (not done)

- The pegsup result separates information from planning. The latent-distance cost (MSE to the goal embedding) may underweight a small object even when the object is linearly decodable. A cost-side test (e.g. reweighting embedding directions, or goal distance in a peg-aware subspace) would test P1 directly.
- IDM with dataset-z-scored actions or a smaller α to keep the predictor intact (a weight/normalization sweep, excluded here).
- IDM trained from scratch on the peg data, as in the paper.
- Whether pegsup's planning gap closes at a larger planning budget or with a CEM population change. If not, that points further at the cost landscape.
