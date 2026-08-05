# Planning bridge decomposition

## Paper-level answer

1. **Does the K=5 adaptive advantage transfer to these off-policy candidates at exactly matched compute?** DIRECTIONALLY YES, BUT WEAKLY: adaptive-minus-matched terminal prediction MSE was -2.7352994e-06 raw and -0.00014018128 whitened, satisfying the sealed point-estimate rule. Both paired 95% start-cluster intervals crossed zero: [-6.3920558e-05, +5.5192696e-05] and [-0.00049388372, +0.00015512488].
2. **Can the frozen goal-latent cost rank real block outcomes with the actual future observation?** NOT SUPPORTED IN THIS POOL: no start met the predeclared informativeness rule. Among the 9 subthreshold starts where a physical rank was at least defined, the actual-terminal raw goal cost had mean Spearman 0.023721 and all-start stable top-5 overlap 0.040 versus chance 0.078125; the whitened analogues were 0.034639 and 0.070.
3. **Is any planning/control claim now justified?** UNSUPPORTED. goal-cost alignment and candidate informativeness remain unsupported, and adaptive changes no informative-start selection; the nominal prior ranking label does not justify planning or control
4. **Strongest caveats.** This is a descriptive decomposition of a consumed 20-start pilot; starts are the independent clusters; the candidate pool is PlanOracle-centered and tie-heavy; physical error is block-position error only; stable tie breaking cannot make uninformative starts evidential; intervals are descriptive; and no closed-loop control was run.
5. **Next experiment.** None is proposed: the fixed decision rule permits a follow-on only if downstream planning evidence is supported.

## Exact reproduction

All configured frozen-object and prior-artifact SHA-256 hashes verified. The 20 start identifiers and seed tuples were exact; warm-start pixels, goal pixels, past actions, reconstructed 64-candidate action arrays, and state hashes were bitwise exact. Replayed terminal block positions/errors differed from the stored arrays by at most 0 m / 0 m. Candidate 0 restored and repeated exactly for every start.

All original model calls, matched permutations, and raw selected indices reproduced exactly. The maximum absolute difference in the original lossless predicted raw goal-cost array was 0, inside the predeclared V5 numerical contract.

Terminal observations were rendered and encoded one 64-image start batch at a time. Only compact float32 actual terminal latents and pixel hashes were retained; no terminal-pixel archive remains.

## Candidate-rollout prediction fidelity

| condition | mean raw terminal latent MSE | mean V5-whitened terminal latent MSE | refiner calls | reached gate evaluations | counted FLOPs |
|:--|--:|--:|--:|--:|--:|
| adaptive | 0.07559018 | 0.25194918 | 7481 | 7353 | 456014729065 |
| matched | 0.07559292 | 0.25208937 | 7481 | 7353 | 456014729065 |
| fixed_d1 | 0.07549305 | 0.25379349 | 6400 | 0 | 455669593600 |
| fixed_d4 | 0.07549362 | 0.24775958 | 25600 | 0 | 460756825600 |

Paired effects use each start's mean over its fixed 64 candidates, so the independent sample size is 20.

| metric | mean adaptive-minus-matched | median | 95% bootstrap interval for mean | 95% interval for median | starts favoring adaptive |
|:--|--:|--:|:--|:--|--:|
| raw MSE | -2.7352994e-06 | +3.0274934e-06 | [-6.3920558e-05, +5.5192696e-05] | [-6.5105651e-06, +1.185312e-05] | 7/20 |
| whitened MSE | -0.00014018128 | +0 | [-0.00049388372, +0.00015512488] | [-7.6765104e-05, +3.8299833e-05] | 9/20 |

Adaptive and matched have identical depth histograms, refiner calls, reached-gate evaluations, and counted FLOPs separately at all 20x5 start/rollout-step cells. The lossless per-cell arrays are in `decomposition_metrics.npz`.

## Latent-goal-cost alignment ceiling

| actual-terminal cost | starts with defined physical rank | mean Spearman on defined starts | all-start mean top-5 overlap | chance | all-start mean selected physical regret (m) |
|:--|--:|--:|--:|--:|--:|
| raw | 9 | 0.023721 | 0.040 | 0.078125 | 0.00016634758 |
| whitened | 9 | 0.034639 | 0.070 | 0.078125 | 0.00016634758 |

No start met the fixed >=5-group informativeness rule, so these nine defined correlations are descriptive diagnostics, not a clean representation verdict. Still, using the actual future observation removes rollout error and did not reveal a useful physical ranking signal.

By contrast, model-predicted costs tracked their corresponding actual-terminal latent costs well, even though neither tracked physical error usefully:

| condition | raw: model vs actual-cost Spearman | raw top-5 overlap | whitened Spearman | whitened top-5 overlap |
|:--|--:|--:|--:|--:|
| adaptive | 0.708988 | 0.570 | 0.735591 | 0.530 |
| matched | 0.709277 | 0.560 | 0.736738 | 0.520 |
| fixed_d1 | 0.708514 | 0.560 | 0.734123 | 0.540 |
| fixed_d4 | 0.715721 | 0.560 | 0.735460 | 0.500 |

## Candidate-set informativeness

The fixed rule marked 0/20 starts informative; 4 were exactly constant and 11 had physical range at most 0.1 mm. Overall informativeness is **unsupported** under the sealed 15/20 rule. The 10 constant/tie-heavy starts are reported but are not evidence for or against planning quality.

| start | range (m) | meaningful groups | tied fraction | top-5 boundary tie size | informative | raw adaptive-match regret contribution to mean (m) | raw selections differ |
|:--|--:|--:|--:|--:|:--:|--:|:--:|
| pilot-cube-000 | 2.6020852e-18 | 1 | 1.000 | 64 | no | +0 | no |
| pilot-cube-001 | 0.037547835 | 2 | 0.984 | 63 | no | +0 | no |
| pilot-cube-002 | 0.045996838 | 2 | 0.984 | 63 | no | +0 | no |
| pilot-cube-003 | 0.0026054593 | 4 | 0.969 | 60 | no | +0 | no |
| pilot-cube-004 | 5.0306981e-17 | 1 | 1.000 | 64 | no | +0 | no |
| pilot-cube-005 | 0 | 1 | 1.000 | 64 | no | +0 | no |
| pilot-cube-006 | 0.00045890788 | 3 | 0.969 | 62 | no | +0 | no |
| pilot-cube-007 | 0.0024661902 | 4 | 0.953 | 61 | no | +0 | no |
| pilot-cube-008 | 0.011849253 | 3 | 0.969 | 62 | no | +0 | no |
| pilot-cube-009 | 0.00094703524 | 2 | 0.984 | 63 | no | -4.7351762e-05 | yes |
| pilot-cube-010 | 4.6403853e-17 | 1 | 1.000 | 64 | no | +0 | no |
| pilot-cube-011 | 0 | 1 | 1.000 | 64 | no | +0 | no |
| pilot-cube-012 | 0.0025979705 | 2 | 0.984 | 63 | no | +0 | no |
| pilot-cube-013 | 0 | 1 | 1.000 | 64 | no | +0 | no |
| pilot-cube-014 | 8.6820464e-05 | 1 | 1.000 | 64 | no | +0 | no |
| pilot-cube-015 | 0.0057669967 | 2 | 0.984 | 63 | no | +0 | no |
| pilot-cube-016 | 0 | 1 | 1.000 | 64 | no | +0 | no |
| pilot-cube-017 | 1.3010426e-18 | 1 | 1.000 | 64 | no | +0 | no |
| pilot-cube-018 | 2.6020852e-18 | 1 | 1.000 | 64 | no | +0 | no |
| pilot-cube-019 | 3.469447e-18 | 1 | 1.000 | 64 | no | +0 | no |

Adaptive and matched raw selections differed on 1/20 starts and 0 informative starts. Their all-start mean physical-regret difference was -4.7351762e-05 m. The per-start contributions above sum to that mean difference.

## Link-by-link decision table

| link | decision | reason |
|:--|:--|:--|
| candidate rollout transfer | supported | adaptive start-cluster mean terminal MSE is lower than matched in both spaces under the sealed directional rule, but both descriptive 95% intervals cross zero |
| goal cost alignment | unsupported | no useful alignment is supported: 0/20 starts are informative, while the 9 subthreshold rank-defined starts have raw mean Spearman 0.023721 and all-start top-5 overlap 0.040 versus 0.078125 chance |
| candidate informativeness | unsupported | 0/20 starts meet the fixed >=5 meaningful groups and >0.1 mm range rule; 15/20 were required |
| downstream planning evidence | unsupported | goal-cost alignment and candidate informativeness remain unsupported, and adaptive changes no informative-start selection; the nominal prior ranking label does not justify planning or control |

## Interpretation

The paper's chain is heterogeneous value of refinement -> learned equal-compute allocation -> multi-step prediction fidelity -> decision utility. V5 v004 established the first two links and the prior bridge gave a nominal positive five-step composition result. This decomposition diagnoses only the missing bridge from prediction fidelity to decision utility; it neither fits a new objective nor supplies downstream-control evidence.

All intervals and ranks are descriptive because the pilot was already consumed and contains only 20 independent start clusters.
