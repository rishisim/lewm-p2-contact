# Results narrative

The primary estimand throughout is the episode-averaged reduction in next-latent mean-squared error relative to the strongest transition-independent analytic mixture of fixed refinement depths available at the adaptive policy's counted compute. Positive comparator-minus-adaptive effects favor learned allocation. All base predictors, refiners, routing policies, thresholds, normalizers, and whitening transforms were frozen before the corresponding evaluation cohort was opened. Prediction, compute, latency, and downstream decision utility are reported as separate outcomes.

## Frozen adaptive allocation passed the fresh Cube confirmation

The accepted four-depth Cube policy was first tested on 1,600 fresh `swm/OGBCube-v0` PlanOracle episodes, yielding 60,800 evaluated transitions. The adaptive path used `4,332,936,120,435` fully counted FLOPs, including base prediction, mandatory and optional refiner calls, causal feature construction, and reached gate heads. Its observed depth histogram was `[50,973, 7,267, 1,773, 787]`, corresponding to 1.21668 actual refiner calls per transition. Charging the gate overhead to the comparator gave an analytic-equivalent mean depth of 1.25295.

Both preregistered co-primary endpoints favored adaptive allocation. Raw episode-averaged MSE was `0.003433540759` for adaptive routing and `0.003440309763` for the analytic comparator, a benefit of `6.769003362e-6`; the simultaneous one-sided lower bound was `5.423955198e-6`. PlanOracle-native-whitened MSE was `0.09876419994` for adaptive routing and `0.09937282203` for the comparator, a benefit of `0.0006086220972` with lower bound `0.0005135570380`. The effects correspond to `0.1968%` and `0.6125%` of comparator MSE, respectively. Thus, the result is precise but modest in absolute and relative magnitude.

The routing diagnostics were consistent with input-dependent allocation. Score/gain Spearman correlations were positive at all reached stages: `0.2626` over 60,800 stage-1 rows, `0.1712` over 9,827 stage-2 rows, and `0.1412` over 2,560 stage-3 rows. Adaptive routing also beat a within-episode call-histogram randomization and a seeded fixed-depth mixture with `103,565` more counted FLOPs. Together, these controls make a favorable global depth histogram an insufficient explanation for the confirmed effect.

*Evidence:* `runs/lewm_v5_readiness_program/v5_package_versions/v004/decision.json`; `metrics/v5_confirmation_episode_metrics.npz`, `metrics/bootstrap_replicates.npz`, `metrics/compute_ledger_realized.json`, and `metrics/stagewise_ranking.json` within that package.

## Zero-shot robustness of the frozen Cube policy was partial

We next carried the complete V5 Cube mechanism—including the PlanOracle-native whitening matrix—without recalibration into three predeclared one-factor distribution shifts. Each regime contained 3,000 fresh episodes and 114,000 transitions; the six regime-by-endpoint claims shared a Bonferroni familywise one-sided alpha of `0.05`.

Increasing PlanOracle action noise from `0.1` to `0.2` preserved both endpoints. Raw benefit was `6.690627725e-6` with simultaneous lower bound `5.233178062e-6`, and fixed-whitened benefit was `0.0006314822713` with lower bound `0.0005518158912`. Under the structural shift from PlanOracle to MarkovOracle, fixed-whitened benefit remained positive (`0.0001773618066`, lower bound `0.00008606212246`), but the raw point estimate was negative (`-2.513778663e-6`, lower bound `-5.677096742e-6`). With 10% random-action contamination, fixed-whitened benefit was positive (`0.0004263060037`, lower bound `0.0003605363394`), whereas raw benefit was effectively null (`9.599565688e-8`, lower bound `-3.128983968e-6`). Four of six simultaneous lower bounds were therefore positive, yielding the terminal label `zero_shot_generalization_partial`.

This experiment establishes an external-validity envelope within the Cube environment, not universal robustness. In particular, robustness depended on the coordinate weighting: the frozen-whitened endpoint transferred more consistently than raw MSE, even though the whitening matrix itself was not recalibrated.

*Evidence:* `runs/lewm_v5_generalization/attempts/v005/DGP_MATRIX.json`, `decision.json`, and `metrics/*_episode_metrics.npz`.

## The original four-depth PushT pilot was formally negative

PushT tested method replication rather than zero-shot transfer. A PushT-specific recurrent refiner was fitted on 120 fresh episodes, and one of six causal routing candidates was selected on a separate 40-episode role. The frozen candidate was then opened once on an 80-episode evaluation role (1,440 transitions) generated with `WeakPolicy(dist_constraint=100)`. The adaptive and primary analytic-comparator arms each used exactly `102,905,055,196` counted FLOPs. The adaptive depth histogram was `[1,089, 42, 46, 263]`.

The pilot produced strong but discordant diagnostics. Raw MSE was `0.1074073938` for adaptive routing versus `0.1113171908` for the analytic comparator, a benefit of `0.003909796985` with exploratory 95% interval `[0.002728725277, 0.005088644256]`. Stagewise score/combined-gain Spearman correlations were `0.4298`, `0.3767`, and `0.3908`. However, the fit-whitened co-primary endpoint narrowly favored the comparator: adaptive MSE was `0.2524596928` versus `0.2521497927`, giving benefit `-0.0003099001292` with interval `[-0.001258993863, 0.0006292011461]`.

The pilot rule required positive point benefits on both raw and fit-whitened endpoints, exact compute equality, causal validity, finite outputs, and nonnegative stagewise rank signs. Because the fit-whitened point estimate failed, the immutable terminal label was `pusht_replication_pilot_not_supported`. The positive raw result and routing correlations are exploratory evidence inside a formally negative study; they do not convert the four-depth pilot into a success.

*Evidence:* `runs/lewm_pusht_replication_pilot/CONFIG.json`, `PILOT_DECISION.json`, and `EVALUATION_ARRAYS.npz`.

## A pilot-derived binary PushT hypothesis passed fresh confirmation

Inspection of the consumed pilot isolated a simpler hypothesis: depth 1 is mandatory, and only the first gate decides whether to allocate one depth-2 step. On the already-consumed pilot evaluation arrays, this binary rule produced positive raw and fit-whitened point benefits (`0.002435240881` and `0.001233341903`) with a call histogram of `[1,089, 351]`. These values were used only to generate the hypothesis. Stages 2 and 3, depths 3 and 4, and their scores were excluded from the confirmation policy.

The binary protocol was locked at `2026-07-18T23:30:23.485504Z`, before generation of a disjoint contiguous 240-episode cohort. All 240 episodes were retained, each contributed 18 transitions, and no evaluation or replacement rule was changed. The policy made 3,265 depth-1 and 1,055 depth-2 decisions across 4,320 transitions. Its adaptive and strongest analytic pairwise-comparator arms each used exactly `307,585,049,440` counted FLOPs; the comparator searched every feasible fixed-depth pair among depths 1–4 and selected the depth-1/2 pair for both endpoints.

Both fresh co-primary endpoints passed. Raw MSE was `0.1186569357` for adaptive routing and `0.1213907177` for the comparator, a benefit of `0.002733782041` with simultaneous lower bound `0.002432267825`. Pilot-frozen fit-whitened MSE was `0.2919664555` for adaptive routing and `0.2935356923` for the comparator, a benefit of `0.001569236827` with lower bound `0.001268753083`. The relative reductions were `2.2521%` and `0.5346%` of comparator MSE. Episode-level signs were positive/zero/negative for 216/0/24 raw episodes and 166/0/74 fit-whitened episodes. The stage-1 score correlated with the combined standardized gain at Spearman `0.457739742` (raw `0.459258874`; fit-whitened `0.395781309`). Independent recomputation from the lossless arrays reproduced the calls, comparators, effects, bootstrap bounds, sign counts, ranks, and exact compute.

The result is a fresh confirmation of a hypothesis derived from a consumed pilot. It is neither an untouched first-shot PushT result nor zero-shot transfer of the Cube mechanism: the refiner, whitening transform, and routing head came from the PushT-specific fit/selection program.

*Evidence:* `runs/lewm_pusht_binary_confirmation/CONFIG.json`, `DECISION.json`, `EVALUATION_ARRAYS.npz`, and `INDEPENDENT_CHECK.json`.

### Clearly labeled post-hoc metric-definition sensitivity

The formal PushT endpoint uses the pilot-fit whitening transform frozen with a covariance eigenvalue floor ratio of `1e-3`; its confirmed benefit remains `0.001569236827`. As a non-confirmatory sensitivity analysis, we re-estimated the whitening covariance on the saved 240-episode confirmation targets while holding predictions, calls, and exact-compute comparator construction fixed. Under this outcome-cohort-defined metric, the whitened benefit was `0.001336506988` at floor ratio `1e-3`, weakened to `0.0008799668240` at `3e-4`, and became negative (`-0.0001742822988`) at `1e-4`. This post-hoc analysis cannot replace the frozen primary metric, because it uses confirmation outcomes to define the transform. It does show that the whitened magnitude and even sign are not invariant to plausible metric regularization, limiting metric-general claims and increasing the importance of the independently positive raw endpoint.

*Sensitivity source and method:* `runs/lewm_pusht_binary_confirmation/EVALUATION_ARRAYS.npz`; recomputation in `runs/lewm_paper_evidence_synthesis/evidence_check.py`.

## Multi-step prediction remained directionally favorable, but planning did not

On 100 already-consumed Cube confirmation episodes, the frozen allocator was composed autoregressively for five steps from 3,400 overlapping starts. At every horizon, the matched comparator preserved the exact adaptive depth histogram, refiner-call count, reached-gate count, and counted compute while permuting depth labels globally across transitions. At K=5, adaptive terminal raw MSE was `0.02967818` versus `0.02974415` for matched allocation, an adaptive-minus-matched difference of `-6.597279931e-5`. Whitened MSE was `0.155283` versus `0.156470`, a difference of `-0.001187089240`. These values establish only a small exploratory composition advantage: the starts overlap, the source episodes were already consumed, and autoregressive histories diverge after the first step.

The subsequent candidate decomposition did not support a downstream bridge. Across 20 independent starts and 64 fixed candidates per start, adaptive-minus-matched off-policy terminal prediction effects were `-2.735299392e-6` raw (95% start-cluster interval `[-6.392055823e-5, 5.519269647e-5]`) and `-0.0001401812773` whitened (`[-0.0004938837164, 0.0001551248760]`). Both intervals crossed zero. More importantly, zero of 20 starts met the predeclared physical-outcome informativeness rule. Among the nine starts with even a defined physical rank, actual-terminal raw latent goal cost had mean Spearman `0.0237205` with physical block error, and all-start top-5 overlap was `0.040` versus chance `0.078125`. Adaptive and matched raw selections differed on only one start. The decomposition therefore supersedes the nominally promising label of the earlier tie-heavy ranking pilot for paper-level interpretation: prediction fidelity did not yield supported planning or control utility.

*Evidence:* `runs/lewm_frozen_gate_planning_bridge/RESULTS.json` and `phase_b_metrics.npz`; `runs/lewm_planning_bridge_decomposition/RESULTS.json` and `decomposition_metrics.npz`.

## Counted-FLOP efficiency did not imply wall-clock acceleration

Latency was measured after the scientific decisions and was excluded from every terminal mapping. On MPS, the Cube adaptive sparse path was slower than fixed depth 1 at batch sizes 1, 32, 256, and 1,024; at batch 1, medians were `3.8366 ms` and `3.4151 ms`, respectively. In the PushT binary confirmation, the cached-base adaptive selection path took `16.9063 ms`, compared with `1.6025 ms` for fixed depth 1 and `2.7835 ms` for fixed depth 2; the common base-predictor measurement was `194.0410 ms` over all 4,320 transitions. Pixel encoding was excluded, and energy was not measured. The efficiency claim is therefore restricted to predictive quality at matched counted FLOPs, with no wall-clock or energy advantage claimed.
