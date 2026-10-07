# Pin check 1: nonlinear (MLP / kNN) readout of peg position

Question: is the peg absent from the LeWM embedding, or present but not linearly decodable?

## Pre-registration (written 2026-10-06, before any MLP/kNN numbers were computed)

**Data (no new rendering/encoding).** Reuse the cached size-control readout pools, which have the same frames, groups,
targets and seed (42) as the existing ridge readout:
- `~/lewm-work/runs/size-control/phaseA`: lewm-pusht, ft_block_s0/s1, ft_mixed_s0/s1 at radius 15 and 45.
- `~/lewm-work/runs/size-control/phaseB/readout`: ft45_block_s0, ft45_mixed_s0 at radius 15 and 45.
Embeddings: cached `cls` and `projected` (192-d). Targets: peg xy and T xy (state columns 7,8 and 2,3).

**Readouts** (all with the same episode-grouped outer CV as `nested_readout`, i.e. `grouped_folds(groups, 5, seed=42)`;
errors are out-of-fold Euclidean pixel errors; CIs are the existing episode-cluster bootstrap `cluster_mean`, 2000 samples):
- *linear*: the existing nested ridge results, read from cache (not recomputed).
- *MLP*: 2 hidden layers of 256, ReLU; Adam (lr 1e-3, weight decay 1e-4), batch 256, at most 200 epochs;
  train-fold-only standardization of inputs and targets; early stopping (patience 20, best weights restored) on one inner
  grouped validation split (20% of training-fold episodes, never shared with the inner-train episodes). One network per
  target pair (peg, T). Torch seed = 42 + fold.
- *kNN*: k = 10, uniform weights, Euclidean distance on train-fold-standardized features.
- *pixel-MLP*: the same MLP on the cached 32x32 RGB downsampled pixels of the same pool (fairness baseline);
  pixel kNN reported as well.
- *mean*: train-fold mean predictor.

**Reading** (applied mechanically to point estimates, separately per checkpoint x radius x embedding):
- **peg present, nonlinearly decodable**: MLP peg error ≤ 2 x MLP T error AND ≤ 1.5 x pixel-MLP peg error.
- **peg effectively absent**: MLP peg error AND kNN peg error both > 0.8 x mean-predictor peg error.
- otherwise **partial**.
(If both the first and second conditions held, which is not expected, "present" is reported and the conflict flagged.)
`projected` is the embedding the planner's goal cost uses, so it is the primary label; `cls` is reported alongside.

**Implication.**
- present → the planning failure lies in how the latent-distance goal cost weighs the peg (cost geometry), not in information loss.
- absent → the encoder drops the information (P1 read as information loss).
- partial → neither clean statement is licensed; report the partial labels as they are.

## Results

(to be filled after the run)
