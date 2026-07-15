# Preregistration: LeWM adaptive-computation V4 confirmation

Status: frozen before any V4 smoke or confirmation episode generation. This is confirmation of the already-frozen stagewise solver and shared full-feature linear gate, not discovery, tuning, or recalibration.

## Frozen scientific object

- Stagewise solver SHA-256: `e63277943a356f3e28b4c4d1a1eb56acc8771fc07eccccdca67f878fed5ba782`.
- Linear student SHA-256: `fb59da4fd2a3895679a2cbba006c2a1f4c6619968b594f9ab1117531560f6302`.
- Frozen tournament SHA-256: `5f474b17a37cc05c39b66495aad0baceb7c7ec4913eb3d6b2966eae230e52b66`.
- Family `full_linear`, operating point `b1.25`, compute price `0.00014899746351320145`.
- Gate: 1,050 parameters and 8,196 incremental FLOPs per evaluated decision.
- Student normalization/depth encoding and discovery whitening are loaded from the hashed checkpoints and never refit.

No V4 result may train, recalibrate, select, or alter a solver, student, normalizer, whitening transform, compute price, feature set, depth encoding, stopping rule, baseline seed, bin, or sample size. The policy's realized V4 call rate is accepted; it is not forced to 1.25.

## Authorization audit and sample size

The prior calibration result is historical authorization only: raw adaptive MSE 0.0031314349443701435, equal-call matched MSE 0.0031499073418161604, benefit 1.847239744601695e-5 with clustered 95% CI [8.881163526542912e-6, 2.9195376017610194e-5]. Whitening was positive but inconclusive; the policy was FLOP-nondominated and slower on the recorded MPS sparse path. These facts are audited, not retested as fresh confirmation.

N=300 is fixed. The calibration episode-cluster SE is 5.18805512991e-06. Detecting half the calibration effect (9.23619872301e-06) gives projected SE 2.84161482423e-06, noncentrality 3.250335, and two-sided alpha=.05 normal-approximation power 0.901539. N will not change after outcomes.

Exactly 12 fresh pipeline-smoke episodes and exactly 300 confirmation episodes are preregistered. Smoke is physically separate, role-labelled, immutable, and forever excluded. Only environment exceptions or malformed generation may consume the next frozen replacement seed; every failure is retained. There are no performance exclusions and no sequential expansion.

## Primary comparator and decision

Raw latent MSE is primary. All intervals use exactly 10,000 hash-seeded episode-cluster bootstrap replicates. The primary transition-independent comparator is the strongest convex depth mixture at the adaptive policy's exact fully counted total FLOPs. The adaptive gate's feature construction, normalization, head, and control-flow FLOPs are converted to fractional expected later-adapter calls and granted to the baseline. Its analytic expected loss is exactly FLOP matched. Its integer realization uses MILP-optimal depth counts at the ceiling of that call budget, so it receives weakly more compute; the residual FLOPs are reported. Assignment is hash seeded and independent of transitions. Both raw CI lower bounds must exceed zero. Equal solver-call mixtures remain secondary allocation-quality decompositions.

A raw pass additionally requires CI lower >0 versus fixed d1 and exact histogram shuffle; score permutation may not significantly beat either of its own exact-total baselines; adaptive must be nondominated on block-call and fully counted FLOP frontiers; and every freshness, receipt, hash, exact-call, finite-value, causality, sparse-equivalence, and no-gradient audit must pass. The complete mechanical verdict mapping is in `decision_rule.json`.

Frozen discovery whitening is applied without refitting. Full pass requires whitened analytic exact-total benefit CI lower >0. Positive whitening with a crossing interval yields raw-only confirmation. A negative whitening point estimate yields failure and flags a likely motion/variance shortcut.

## Freshness, one-shot access, and isolation

All mechanically recorded prior identifiers were enumerated before selecting new disjoint seed blocks. Every accessible prior raw Cube episode outside the permanently untouched V3 test set is hashed for exact duplicate comparison. V3 test targets are never opened. The released checkpoint lacks its original pretraining episode manifest; therefore mechanical pretraining nonmembership cannot be claimed despite newly generated documented seeds and exact-duplicate checks.

An exclusive confirmation receipt is created after all 300 raw trajectories exist and before any evaluator opens confirmatory observations as target/next latents. The complete frozen policy is evaluated once. No partial confirmation metric is printed or inspected during evaluation, and no choice changes or rescue episodes are allowed.

## Secondary physical analysis

Contact, impact, other free motion, and static labels are prohibited from solver and gate inputs and attached only after the primary verdict. Frozen discovery-fit fine and coarse bins match contact to noncontact on trajectory phase, target block-motion magnitude, and five-action magnitude. Unmatched and coarsened-exact overlap-weighted contrasts, three fixed seed blocks, and offsets -2 through +3 around impact onset are reported. “Contact-aware” is allowed only when the fine-bin matched call and prediction-benefit contrasts are positive with clustered CI lower >0. A null physical contrast limits that claim without changing the primary verdict.

## Compute and deployment

FLOPs are recomputed from the common base, V1 call, later adapters, feature construction, normalization, linear head, and control flow. MPS benchmarks at batches 1, 32, 256, and 1024 include base prediction, solver, scoring, routing, indexing, synchronization, and compare fixed d1–d4, exact-total and equal-call mixtures, dense adaptive, frozen reference sparse, and optimized sparse. Optimized execution must select exactly the reference depths and produce numerically equivalent outputs. Deployment status compares optimized to reference sparse under the frozen thresholds in `config.json`; latency never substitutes for the FLOP/statistical endpoint. Energy is reported only if reliable.

## Interpretation

A pass supports adaptive computation for this frozen visual latent physical world model at one preregistered operating point. It is not a claim of the first adaptive world model; LoopWM remains relevant adaptive-depth world-model work in text environments. Overall compute-responsive allocation and the stronger contact-aware claim remain separate.
