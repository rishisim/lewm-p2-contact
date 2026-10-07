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

## Results (run 2026-10-06; artifacts `~/lewm-work/runs/pin-mlp/`, summary.json / summary.md)

Pool: 2,800 held-out dataset frames, 28 episodes. The peg moves within episodes (median within-episode spread ≈ 138 px),
so there are many distinct peg positions. Cached ridge fold assignments were checked to equal `grouped_folds` for every unit.
Entries are Euclidean pixel error [episode-cluster bootstrap 95% CI]. The Linear column is the cached nested ridge. Labels are
the pre-registered rule applied mechanically to point estimates.
Implementation deviation (stricter than pre-registered): MLP input/target scalers are fitted on the inner-train episodes
only, not on the whole outer-train fold. kNN uses outer-train standardization as pre-registered.

| Checkpoint | Radius | Embedding | Linear peg | Linear T | MLP peg | MLP T | kNN peg | kNN T | Pixel-MLP peg | Pixel-MLP T | Mean peg | Mean T | MLP-peg/MLP-T | MLP-peg/pixel-MLP-peg | MLP-peg/mean-peg | kNN-peg/mean-peg | Label | Conflict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ft_block_s0 | 15 | cls | 139.09 [136.67, 141.50] | 18.84 [15.67, 22.51] | 140.56 [138.11, 142.99] | 32.79 [23.76, 42.87] | 145.31 [142.37, 148.42] | 48.24 [37.56, 59.82] | 175.82 [121.76, 246.28] | 67.90 [50.45, 92.04] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 4.287 | 0.799 | 0.999 | 1.033 | absent | False |
| ft_block_s0 | 15 | projected | 141.78 [139.13, 144.58] | 24.10 [19.76, 29.26] | 142.40 [139.76, 145.09] | 65.58 [46.38, 85.96] | 145.57 [142.56, 148.73] | 52.36 [38.17, 70.08] | 175.82 [121.76, 246.28] | 67.90 [50.45, 92.04] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 2.171 | 0.810 | 1.012 | 1.035 | absent | False |
| ft_block_s0 | 45 | cls | 137.63 [135.15, 140.23] | 19.39 [16.17, 23.31] | 141.02 [137.92, 144.39] | 31.83 [24.47, 40.05] | 144.55 [141.42, 148.13] | 47.23 [37.16, 58.01] | 61.08 [40.02, 88.52] | 62.58 [49.97, 77.39] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 4.431 | 2.309 | 1.002 | 1.027 | absent | False |
| ft_block_s0 | 45 | projected | 142.83 [139.72, 146.27] | 24.91 [20.67, 30.06] | 141.74 [139.11, 144.45] | 60.83 [43.08, 80.32] | 145.15 [142.07, 148.53] | 50.51 [37.96, 66.02] | 61.08 [40.02, 88.52] | 62.58 [49.97, 77.39] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 2.330 | 2.321 | 1.007 | 1.032 | absent | False |
| ft_block_s1 | 15 | cls | 139.11 [136.70, 141.54] | 20.07 [16.48, 24.26] | 140.48 [138.07, 142.87] | 36.87 [27.11, 48.04] | 145.38 [142.45, 148.51] | 48.31 [37.46, 60.04] | 175.82 [121.76, 246.28] | 67.90 [50.45, 92.04] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 3.810 | 0.799 | 0.998 | 1.033 | absent | False |
| ft_block_s1 | 15 | projected | 141.60 [138.95, 144.40] | 24.23 [20.06, 29.13] | 142.10 [139.45, 144.86] | 61.26 [43.29, 80.52] | 145.56 [142.67, 148.71] | 53.53 [39.29, 71.68] | 175.82 [121.76, 246.28] | 67.90 [50.45, 92.04] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 2.320 | 0.808 | 1.010 | 1.034 | absent | False |
| ft_block_s1 | 45 | cls | 137.32 [134.89, 139.85] | 20.92 [17.36, 25.12] | 140.85 [137.85, 144.02] | 37.07 [28.67, 46.79] | 145.10 [141.72, 148.80] | 46.92 [36.94, 57.63] | 61.08 [40.02, 88.52] | 62.58 [49.97, 77.39] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 3.800 | 2.306 | 1.001 | 1.031 | absent | False |
| ft_block_s1 | 45 | projected | 142.74 [139.75, 146.04] | 24.07 [20.42, 28.60] | 141.71 [139.05, 144.46] | 61.82 [43.28, 81.98] | 145.51 [142.27, 149.16] | 51.91 [38.57, 68.43] | 61.08 [40.02, 88.52] | 62.58 [49.97, 77.39] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 2.292 | 2.320 | 1.007 | 1.034 | absent | False |
| ft_mixed_s0 | 15 | cls | 139.25 [136.69, 141.83] | 20.97 [17.77, 24.80] | 140.37 [137.93, 142.87] | 34.10 [25.57, 43.43] | 144.29 [142.14, 146.49] | 44.20 [33.09, 57.38] | 175.82 [121.76, 246.28] | 67.90 [50.45, 92.04] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 4.117 | 0.798 | 0.998 | 1.025 | absent | False |
| ft_mixed_s0 | 15 | projected | 140.94 [138.29, 143.59] | 26.50 [21.24, 32.79] | 141.00 [138.41, 143.66] | 52.03 [37.09, 68.96] | 144.57 [142.44, 146.73] | 52.54 [37.35, 71.75] | 175.82 [121.76, 246.28] | 67.90 [50.45, 92.04] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 2.710 | 0.802 | 1.002 | 1.027 | absent | False |
| ft_mixed_s0 | 45 | cls | 139.31 [136.59, 142.09] | 20.35 [17.96, 23.35] | 140.45 [137.95, 142.88] | 32.88 [24.70, 42.06] | 144.63 [142.14, 147.31] | 43.59 [32.87, 55.99] | 61.08 [40.02, 88.52] | 62.58 [49.97, 77.39] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 4.271 | 2.299 | 0.998 | 1.028 | absent | False |
| ft_mixed_s0 | 45 | projected | 141.52 [138.71, 144.40] | 26.40 [21.71, 31.77] | 141.06 [138.43, 143.68] | 51.72 [37.23, 68.29] | 145.02 [142.66, 147.46] | 52.42 [37.56, 71.23] | 61.08 [40.02, 88.52] | 62.58 [49.97, 77.39] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 2.728 | 2.310 | 1.002 | 1.031 | absent | False |
| ft_mixed_s1 | 15 | cls | 139.44 [136.86, 142.04] | 20.80 [17.14, 25.17] | 140.43 [137.95, 142.92] | 32.01 [24.43, 40.49] | 144.31 [142.07, 146.50] | 43.97 [33.08, 56.66] | 175.82 [121.76, 246.28] | 67.90 [50.45, 92.04] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 4.387 | 0.799 | 0.998 | 1.026 | absent | False |
| ft_mixed_s1 | 15 | projected | 141.20 [138.49, 143.90] | 28.40 [23.45, 34.01] | 141.18 [138.58, 143.80] | 51.18 [36.14, 67.88] | 144.75 [142.49, 147.10] | 49.95 [36.27, 67.15] | 175.82 [121.76, 246.28] | 67.90 [50.45, 92.04] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 2.758 | 0.803 | 1.003 | 1.029 | absent | False |
| ft_mixed_s1 | 45 | cls | 139.41 [136.70, 142.22] | 20.83 [17.65, 24.45] | 140.56 [138.04, 143.10] | 33.82 [26.04, 42.55] | 144.73 [142.06, 147.54] | 43.59 [33.19, 55.33] | 61.08 [40.02, 88.52] | 62.58 [49.97, 77.39] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 4.156 | 2.301 | 0.999 | 1.029 | absent | False |
| ft_mixed_s1 | 45 | projected | 141.74 [138.90, 144.62] | 27.31 [22.88, 32.25] | 141.75 [139.00, 144.56] | 52.00 [37.14, 68.87] | 144.86 [142.39, 147.33] | 49.78 [36.45, 66.92] | 61.08 [40.02, 88.52] | 62.58 [49.97, 77.39] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 2.726 | 2.321 | 1.007 | 1.030 | absent | False |
| lewm-pusht | 15 | cls | 138.47 [135.25, 141.65] | 38.11 [30.45, 47.59] | 140.33 [137.41, 143.17] | 29.06 [23.04, 35.85] | 144.05 [141.70, 146.55] | 43.05 [33.46, 52.62] | 191.25 [132.98, 266.13] | 63.27 [48.77, 81.12] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 4.830 | 0.734 | 0.997 | 1.024 | absent | False |
| lewm-pusht | 15 | projected | 141.13 [137.77, 144.30] | 51.47 [41.37, 62.51] | 140.49 [137.78, 143.10] | 38.90 [29.56, 49.85] | 144.67 [142.23, 147.23] | 46.70 [36.85, 56.98] | 191.25 [132.98, 266.13] | 63.27 [48.77, 81.12] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 3.612 | 0.735 | 0.998 | 1.028 | absent | False |
| lewm-pusht | 45 | cls | 92.08 [77.48, 109.78] | 36.18 [29.86, 43.82] | 127.65 [111.15, 144.87] | 31.63 [25.91, 38.08] | 131.44 [126.76, 135.57] | 42.11 [33.27, 51.23] | 68.27 [43.99, 99.48] | 66.80 [54.20, 81.51] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 4.035 | 1.870 | 0.907 | 0.934 | absent | False |
| lewm-pusht | 45 | projected | 139.51 [133.11, 146.44] | 44.61 [33.96, 57.79] | 140.60 [136.08, 145.15] | 40.24 [30.93, 51.59] | 138.56 [134.02, 142.95] | 43.89 [34.67, 54.13] | 68.27 [43.99, 99.48] | 66.80 [54.20, 81.51] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 3.494 | 2.060 | 0.999 | 0.985 | absent | False |
| ft45_block_s0 | 15 | cls | 138.55 [136.28, 140.87] | 19.69 [16.10, 24.00] | 140.11 [137.78, 142.44] | 36.17 [26.62, 46.68] | 144.64 [142.22, 147.15] | 47.39 [36.86, 58.71] | 175.82 [121.76, 246.28] | 67.90 [50.45, 92.04] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 3.873 | 0.797 | 0.996 | 1.028 | absent | False |
| ft45_block_s0 | 15 | projected | 140.92 [138.53, 143.41] | 25.34 [20.68, 30.86] | 142.33 [139.75, 144.99] | 65.78 [47.00, 85.94] | 145.13 [142.52, 147.97] | 52.18 [37.78, 70.53] | 175.82 [121.76, 246.28] | 67.90 [50.45, 92.04] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 2.164 | 0.810 | 1.012 | 1.031 | absent | False |
| ft45_block_s0 | 45 | cls | 136.06 [134.12, 138.02] | 20.18 [16.74, 24.61] | 138.75 [136.58, 141.11] | 35.96 [26.92, 45.76] | 143.64 [141.18, 146.27] | 47.78 [37.34, 59.07] | 61.08 [40.02, 88.52] | 62.58 [49.97, 77.39] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 3.858 | 2.272 | 0.986 | 1.021 | absent | False |
| ft45_block_s0 | 45 | projected | 139.71 [137.60, 141.93] | 27.06 [22.57, 32.40] | 141.06 [138.74, 143.45] | 62.81 [44.78, 81.72] | 144.38 [141.90, 147.08] | 52.04 [38.27, 69.12] | 61.08 [40.02, 88.52] | 62.58 [49.97, 77.39] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 2.246 | 2.310 | 1.002 | 1.026 | absent | False |
| ft45_mixed_s0 | 15 | cls | 139.48 [136.96, 142.09] | 22.35 [18.24, 27.51] | 140.34 [137.89, 142.87] | 35.43 [27.11, 44.59] | 144.80 [142.55, 147.07] | 44.61 [33.28, 58.37] | 175.82 [121.76, 246.28] | 67.90 [50.45, 92.04] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 3.961 | 0.798 | 0.997 | 1.029 | absent | False |
| ft45_mixed_s0 | 15 | projected | 141.42 [138.73, 144.18] | 26.76 [21.91, 32.21] | 141.15 [138.48, 143.89] | 50.63 [34.87, 67.89] | 144.37 [142.15, 146.70] | 54.46 [37.72, 75.04] | 175.82 [121.76, 246.28] | 67.90 [50.45, 92.04] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 2.788 | 0.803 | 1.003 | 1.026 | absent | False |
| ft45_mixed_s0 | 45 | cls | 137.60 [135.12, 140.16] | 22.24 [18.48, 26.94] | 139.58 [137.13, 142.07] | 34.07 [26.34, 42.63] | 143.81 [141.67, 146.09] | 44.50 [33.54, 57.75] | 61.08 [40.02, 88.52] | 62.58 [49.97, 77.39] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 4.097 | 2.285 | 0.992 | 1.022 | absent | False |
| ft45_mixed_s0 | 45 | projected | 140.77 [138.11, 143.41] | 25.90 [21.38, 30.96] | 140.95 [138.33, 143.59] | 52.12 [36.61, 69.85] | 144.46 [142.14, 146.84] | 53.58 [37.64, 73.00] | 61.08 [40.02, 88.52] | 62.58 [49.97, 77.39] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] | 2.704 | 2.308 | 1.002 | 1.027 | absent | False |

| Pixel cache | Radius | With target | Linear peg | Linear T | MLP peg | MLP T | kNN peg | kNN T | Mean peg | Mean T |
|---|---|---|---|---|---|---|---|---|---|---|
| pixel_r15_0_6b7a1a8f539b9f16 | 15 | False | 116.59 [77.50, 169.76] | 50.24 [42.63, 58.46] | 175.82 [121.76, 246.28] | 67.90 [50.45, 92.04] | 137.47 [135.00, 140.12] | 50.98 [37.88, 64.66] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] |
| pixel_r15_1_233b91f3f935268b | 15 | True | 115.16 [76.94, 166.99] | 50.21 [42.59, 58.49] | 191.25 [132.98, 266.13] | 63.27 [48.77, 81.12] | 137.32 [134.66, 140.04] | 50.86 [37.89, 64.64] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] |
| pixel_r45_0_443d3af5dc418c6e | 45 | False | 14.54 [9.72, 21.40] | 50.55 [43.14, 58.23] | 61.08 [40.02, 88.52] | 62.58 [49.97, 77.39] | 43.84 [34.92, 53.30] | 83.59 [71.41, 96.92] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] |
| pixel_r45_1_d2fee0ffe7a335b1 | 45 | True | 14.69 [9.70, 21.96] | 50.60 [43.03, 58.44] | 68.27 [43.99, 99.48] | 66.80 [54.20, 81.51] | 41.44 [32.45, 51.02] | 85.42 [73.36, 98.25] | 140.71 [138.03, 143.33] | 147.67 [124.93, 169.68] |

### Reading applied

**All 28 rows (7 checkpoints x radius {15, 45} x {cls, projected}) are labelled "peg effectively absent"; no conflicts.**
On the primary `projected` embedding the MLP peg error is 0.998–1.012x the mean predictor and kNN is 0.985–1.035x, for every
checkpoint at both radii, including ft45_block_s0/ft45_mixed_s0, which were fine-tuned on radius-45 data.
Pre-registered implication: **information is dropped by the encoder (P1 read as information loss)**, not a cost-geometry
weighting of information that is present.

### Interpretation and caveats

The MLP is not a strong readout on this small pool. Its T error (29–66 px) is worse than ridge's (19–51 px). Pixel-MLP is
far worse than pixel ridge (r45: about 61–68 px vs 14.5 px; r15: worse than the mean predictor). So the "absent" label does
not rest on the MLP alone. What carries it is the model-free kNN check. kNN does recover T from the same embeddings (43–54 px
vs 148 px mean) and recovers the r45 peg from 32x32 pixels (41–44 px, ≈0.3x mean). Yet on every embedding its peg error sits
at the mean-predictor level. So three readout families (ridge, MLP, kNN) find no peg information in the 192-d embeddings, while
two of the same families find it in coarse pixels at r45.

One row is mechanically "absent" but borderline: lewm-pusht `cls` at r45. There the cached linear ridge already gets 92 px
(0.65x mean), while MLP (0.91x) and kNN (0.93x) do worse than linear. Its `projected` embedding, which the planner uses, is
at the mean level for all three. At r15 the pixel baselines themselves barely see the peg (pixel ridge 116 px), so the
r15 rows say little about the encoder relative to the input. The r45 rows carry the evidence.

"Absent" is relative to the readouts tried here (a small MLP, k=10 kNN, and ridge, with n = 2,800). A larger or better-tuned
decoder, or one trained on far more frames, could still find a weak, highly entangled peg signal. The result does not show
that the encoder discards all peg information. It shows the peg is not recoverable at a usable level, for any of these
readouts, from the embedding the planning cost uses.

### Out-of-scope ideas (not run)

- Repeat on the main N=400 readout pool (`~/lewm-work/runs/main/readout`, 6,000 frames / 428 groups incl. condition
  endpoints) for a larger-n MLP.
- Tune the MLP (smaller width, stronger weight decay or dropout, input PCA) until it at least matches ridge on T and pixel
  ridge on the r45 peg. That would make it a fair nonlinear probe and tighten the "absent" bound.
- Patch-token (pre-CLS) readout: test whether the peg is in the ViT patch tokens but lost at CLS pooling/projection.
  That separates "encoder never sees it" from "pooling drops it".
- Planning-side check: the goal-cost sensitivity to peg displacement (latent distance vs peg offset curve) as a direct
  measure of the cost geometry.
