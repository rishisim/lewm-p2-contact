# Paper status

## Verdict

**Begin the full manuscript now.** The evidence is sufficient for a bounded paper about **next-latent predictive quality at matched counted FLOPs**. It is not sufficient for a paper whose central claim is downstream control, zero-shot cross-environment transfer, universal robustness, or practical speed.

## Strongest defensible claim

Learned causal allocation can improve next-latent predictive fidelity at matched counted compute: a frozen four-depth policy passed a fresh 1,600-episode Cube PlanOracle confirmation, and an environment-specific, pilot-derived binary depth-1/2 PushT policy passed a separately locked fresh 240-episode confirmation against the strongest transition-independent analytic fixed-depth mixture.

## What is established

- **Cube confirmation:** On 1,600 fresh `swm/OGBCube-v0` PlanOracle episodes (60,800 transitions), the frozen V5 allocator reduced episode-averaged raw MSE by `6.769003361969305e-6` (simultaneous one-sided lower bound `5.423955197986978e-6`) and PlanOracle-native-whitened MSE by `6.086220971507569e-4` (lower bound `5.135570380185851e-4`) relative to the strongest transition-independent analytic exact-total-compute mixture. All three stagewise score/gain Spearman signs were positive. Source: `runs/lewm_v5_readiness_program/v5_package_versions/v004/decision.json`; arrays: `metrics/v5_confirmation_episode_metrics.npz` and `metrics/bootstrap_replicates.npz` within that package.
- **Fresh PushT binary confirmation:** After a consumed PushT pilot suggested simplifying the policy to mandatory depth 1 plus one selectively allocated depth-2 step, that rule was locked and evaluated on 240 fresh `swm/PushT-v1` WeakPolicy episodes (4,320 transitions). Raw benefit was `0.002733782040519924` (simultaneous lower bound `0.002432267824803084`); pilot-frozen fit-whitened benefit was `0.0015692368266779146` (lower bound `0.001268753083313291`). Both primary arms used exactly `307,585,049,440` counted FLOPs. Stage-1 score/combined-gain Spearman was `0.45773974181990207`. Source: `runs/lewm_pusht_binary_confirmation/DECISION.json`; arrays: `EVALUATION_ARRAYS.npz`.
- **Mechanism, not merely depth:** Cube also beat a within-episode call-histogram comparator and a weakly-more-compute seeded comparator; PushT binary beat within-episode call randomization. These controls support transition-specific allocation rather than only a favorable global depth histogram.
- **Process separation:** Discovery/pilot cohorts were consumed before their respective fresh confirmations, frozen objects were hash-checked, episode was the inferential unit, and the terminal labels were independently reproduced from lossless arrays.

## What is partial

- **Cube zero-shot DGP robustness is partial, not universal.** The fully frozen Cube V5 policy and whitening were carried without recalibration into three one-factor shifts, 3,000 fresh episodes each. Both endpoints passed for increased PlanOracle action noise; the frozen-whitened endpoint passed under MarkovOracle and 10% random-action contamination, but the raw endpoint did not. The terminal label is `zero_shot_generalization_partial`. Source: `runs/lewm_v5_generalization/attempts/v005/decision.json`.
- **Five-step composition is exploratory.** On 100 already-consumed Cube confirmation episodes and 3,400 overlapping starts, the K=5 adaptive-minus-matched effect favored adaptive routing in raw (`-6.597279931162478e-5`) and whitened (`-0.0011870892398969934`) terminal MSE. Overlap and consumed data preclude a new confirmation claim. Source: `runs/lewm_frozen_gate_planning_bridge/RESULTS.json`.
- **PushT is method replication after PushT-specific fitting, not zero-shot Cube-to-PushT transfer.** The PushT refiner, whitening, and gate were fitted/selected on PushT roles before the pilot; only the simplified binary policy was frozen into the fresh confirmation.

## What failed

- **The original four-depth PushT pilot was formally negative.** On its 80-episode evaluation role, raw benefit over the primary exact-compute analytic comparator was positive (`0.003909796985389183`, exploratory 95% interval `[0.0027287252766780828, 0.0050886442555326565]`), but the required fit-whitened benefit was `-0.00030990012922212844` (`[-0.001258993862851237, 0.0006292011461374453]`). The immutable label is `pusht_replication_pilot_not_supported`. Source: `runs/lewm_pusht_replication_pilot/PILOT_DECISION.json`.
- **The prediction-to-planning bridge is unsupported.** In the 20-start Cube candidate decomposition, both off-policy prediction intervals crossed zero, zero starts met the informativeness rule, and actual-terminal latent goal cost did not usefully rank physical error (raw mean Spearman `0.023720525194694596` on nine rank-defined starts). No planning or control claim follows. Source: `runs/lewm_planning_bridge_decomposition/RESULTS.json`.
- **No wall-clock speedup was observed.** Counted FLOP parity is the efficiency axis. Sparse routing overhead made measured adaptive latency slower than fixed depth 1 in the reported Cube and PushT timings.

## What is unavailable

- Any closed-loop task-success or control improvement attributable to allocation.
- Zero-shot transfer of the Cube gate/refiner to PushT.
- Evidence under a successful or competent PushT control policy: the 240 pilot-role episodes and the 240 binary-confirmation episodes each had `success_fraction = 0.0` under `WeakPolicy(dist_constraint=100)`.
- End-to-end latency or energy advantage; pixel encoding was excluded from the latency microbenchmarks and energy was not measured.
- Universal environment or policy robustness.
- Novelty priority. LoopWM is known relevant adaptive-computation work in text environments; a focused related-work pass remains a writing task.

## Paper boundary

Lead with **predictive fidelity at matched counted compute**. Treat Cube as the clean frozen-policy confirmation, the shift study as an explicit external-validity map, the negative four-depth PushT pilot plus fresh binary confirmation as honest cross-domain method replication, and the multi-step/planning work as the boundary showing why prediction does not yet imply control. Do not use “speedup,” “universal,” “zero-shot to PushT,” “planning improvement,” “control improvement,” or “first adaptive world model.”

## Reproducibility status

`evidence_check.py` independently reloads the terminal NPZ arrays, recomputes all headline means and bootstrap bounds, reconstructs the strongest analytic mixtures and compute totals, checks chronology, and hashes the source artifacts. It currently returns `"all_checks_passed": true`. The binary study also records a verifier-only post-run correction from bitwise comparison to tight float64 tolerance across `einsum`/`matmul` paths; no scientific array or analysis was rerun.

The tracked repository state is unchanged at `86e4bb150de0d1546a3cf57f3ef301294eb04368` on `main`, equal to `origin/main`, with no tracked or staged diff. The accepted untracked experiment roots remain unmodified and are pinned by the hashes in `EVIDENCE_INDEX.json`; this synthesis adds only `runs/lewm_paper_evidence_synthesis/`.
