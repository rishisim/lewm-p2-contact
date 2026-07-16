# LeWM contact-free adaptive-computation V5 preregistration — package v004

## Frozen claim and candidate

This package tests one claim only: under the fixed PlanOracle DGP, the frozen causal contact-free stage-specific adaptive policy has lower episode-averaged raw latent MSE and lower episode-averaged PlanOracle-native-whitened latent MSE than the strongest transition-independent analytic mixture with exactly the same total counted compute. The policy is `stage_dual_r0.01_q0.85`, with stage thresholds `0.19465729885363534`, `0.1568665868361991`, and `0.15153321811129283`. Base model, four-exit refiner, gate coefficients, thresholds, and whitening are immutable and hash-sealed.

This is an adaptive world-model computation claim, not a claim to be the first adaptive world model; LoopWM is relevant related work.

## Cohort and DGP

V5 uses exactly 1,600 fresh PlanOracle episodes and no sequential expansion. `DGP.json` is normative: stable-worldmodel 0.1.0, OGBench 1.2.1, MuJoCo 3.10.0, Gymnasium 1.3.0, `swm/OGBCube-v0`, one agent, 224×224 pixels, 200 environment steps, frameskip 5, history 3, modeled steps 3 through 40, `action_noise=.1`, `p_random_action=0`, `noise_smoothing=.5`, and `min_norm=.4`, with the deterministic reset amendment. `cohort_seed_ledger.json` fixes all confirmation and replacement identifiers before excluded package smoke. The 12 package-smoke episodes are disjoint and permanently excluded.

Only `pixels` and `action` may be loaded by model or gate code. Contact, reward, success, privileged simulator state, targets, and future information cannot enter fitting, whitening, features, routing, stopping, exclusion, or decision statistics. Contact is post-hoc interpretation only. V3 test targets, the combined V3 cache, and the released HDF5 corpus are forbidden; V4 remains consumed diagnostic evidence.

## Frozen execution and compute

The reported adaptive prediction is produced by actual sparse execution: later adapters run only for active rows. Dense all-exit execution is a separately accounted comparator and numerical shadow. Calls, stopping depths, histograms, same-path gate scores/features, and manual sparse versus `StagewiseResidualCascade.forward_selected` must be exact. Cross-batch sparse/dense selected latents use the sealed float32/MPS contract `rtol=2e-6`, `atol=2e-7`, and maximum absolute error `4.76837158203125e-7`; tolerance never applies to calls.

Every reached gate decision counts 3,801 feature FLOPs and 4,184 dual-affine-head FLOPs, 7,985 total, plus five separately reported non-FLOP comparison/min operations. Feature construction is counted at every reached decision. Two independent derivations must agree. Base calls, refiner calls, FLOPs, non-FLOP operations, and synchronized latency are reported separately.

## Comparators and statistics

The adaptive result is compared with: the strongest transition-independent analytic exact-total-compute mixture; a fixed-seed integer mixture with equal or weakly greater total compute; fixed depth 1; and a within-episode random reassignment that preserves each episode’s exact call histogram. Both raw and native-whitened losses use the frozen outcome mapping. No comparator may condition on transition features.

Inference uses exactly 20,000 deterministic paired episode bootstrap replicates. All six individual 95% lower bounds named in `outcome_mapping.json` must be strictly positive. Bonferroni simultaneous 95% one-sided lower bounds for the two co-primary effects, raw and native-whitened versus analytic exact-total-compute mixtures, must both be strictly positive. Gate-score/next-stage-gain Spearman rank correlation must be finite and strictly positive at each stage. Calls, hashes, path sets, chronology, seeds, isolation, finite arrays, gradients, frozen modules, numerical equivalence, and exact compute must all pass.

Any integrity failure maps to `v5_execution_invalid`; otherwise any statistical or stagewise failure maps to `v5_confirmation_failed`; only an all-criteria pass maps to `v5_confirmation_passed`. A valid failure is not retried.

## Power justification and uncertainty

`power_analysis.json` uses only the accepted 300-episode discovery cohort. It defines each planning effect as 80% of the smaller of the independently reproduced simultaneous discovery lower bound and the minimum contiguous 50-episode block mean. With Bonferroni alpha .025 per co-primary endpoint, `n=1600` is the first 50-episode grid size achieving a union-bound joint-power lower bound of at least .95 at design effect 1.5 and at least .90 at stress design effect 2.0. This calculation cannot guarantee a V5 pass; future DGP stability, paired variance, heterogeneity, and design-effect adequacy remain irreducible uncertainty.

## Package rule

All normative code, frozen objects, ledgers, mappings, source hashes, and this preregistration are sealed before package smoke. A smoke failure invalidates v004; it may not be patched in place. Package construction and smoke generate zero V5 confirmation outcomes.

Package v004 is the prospective, pre-outcome operational successor to v001, v002, and v003. Package v001 produced zero raw confirmation episodes and zero modeled rows because generation was invoked with the evaluation-only interpreter; all v001 identifiers remain consumed. Package v002 also produced zero raw episodes and zero modeled rows: its preseal qualification found that the inherited FLOP derivation attempted to recreate an already carried immutable operation graph. Package v003 repaired that output conflict but, still before sealing or rollout, its fresh synthetic qualification seed exposed a brittle fixed `1e-15` agreement check between two algebraically equivalent mixture implementations; their maximum difference was `1.7763568394002505e-15`, with exact depths and weights. All v002 and v003 identifiers are consumed. Package v004 retains the v003 graph verification and replaces only that synthetic diagnostic threshold with a float64 machine-precision-scaled ceiling; neither inference implementation changes. The scientific design above is unchanged. `runtime_contract.json` assigns DGP construction and rollout generation exclusively to the generation interpreter, while sparse execution, dense shadow, analysis, latency, and independent verification run exclusively under the evaluation interpreter. `launcher.py` is the sole fail-closed dispatcher. Generation validates its interpreter, exact package versions, imports, seal, and a global PlanOracle constructor before creating a role raw directory or entering the primary/replacement loop. A wrong interpreter therefore produces one package-level preflight error, consumes zero identifiers, and creates no per-seed failure ledger.
