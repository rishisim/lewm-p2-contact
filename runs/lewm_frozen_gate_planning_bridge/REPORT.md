# Frozen-gate composition and planning bridge

## Paper-level answer

1. **Question.** Does the completely frozen V5 policy retain its accepted one-step advantage through five-step autoregressive latent composition, and only then does it improve selection among identical open-loop Cube candidates at exactly matched computation?
2. **Evidence obtained.** K=1 compatibility passed on 8 consumed V5 episodes. Phase B evaluated 3,400 overlapping valid starts from 100 episodes selected before multi-step metrics, with only the initial three encoded frames teacher-forced.
3. **Did the benefit survive K=5?** Yes under this exploratory consumed-data criterion: the K=5 raw point estimate favored adaptive routing and the whitened endpoint remained inside the fixed 5% margin.
4. **Did candidate selection improve?** The fixed pilot rule technically passed: adaptive had slightly higher mean rank correlation and slightly lower real selected-candidate regret, with equal top-5 recovery. This does not establish a practically meaningful planning advantage because correlations were near zero and candidate outcomes were tie-heavy. Adaptive/matched mean Spearman: `0.01836663904091594` / `0.016644616900534617`; mean real regret: `0.000746566` / `0.000793917` m; mean top-5 overlap: `0.050` / `0.050`.
5. **Strongest caveats.** This is exploratory evidence; composition starts overlap within consumed episodes; the ranking bridge has only 20 pilot starts and PlanOracle-centered candidates; 10 of 20 starts had real candidate-error range at most 1e-12 m and only 16 had defined Spearman correlation; no new confirmation cohort was launched; autoregressive condition histories diverge after K=1; no closed-loop or wall-clock claim follows.
6. **Next experiment.** No larger or closed-loop control task is warranted yet. The precise next experiment is an independently seeded, preregistered small ranking pilot whose condition-independent start/candidate procedure targets contact-active cases and must satisfy a predeclared real-outcome-spread check before model scores are opened, while retaining the same frozen objects and exact compute matching.

**Terminal label:** `planner_ranking_promising`

## Frozen decision criterion

The K=5 raw mean adaptive-minus-matched difference was `-6.59727993e-05` (negative favors adaptive). The whitened difference was `-0.00118708924`. The whitened no-degradation limit was fixed before Phase B at adaptive mean no greater than 1.05 times matched mean. Failed Boolean criterion components: `none`.

## Composition table

| K | condition | terminal raw MSE | terminal whitened MSE | cumulative raw | cumulative whitened |
|---:|:--|--:|--:|--:|--:|
| 1 | adaptive | 0.00357140 | 0.101631 | 0.00357140 | 0.101631 |
| 1 | matched | 0.00358548 | 0.102803 | 0.00358548 | 0.102803 |
| 1 | fixed_d1 | 0.00359081 | 0.103544 | 0.00359081 | 0.103544 |
| 1 | fixed_d4 | 0.00352153 | 0.096985 | 0.00352153 | 0.096985 |
| 1 | base | 0.00470556 | 0.283961 | 0.00470556 | 0.283961 |
| 2 | adaptive | 0.00748569 | 0.111667 | 0.01105708 | 0.213298 |
| 2 | matched | 0.00750358 | 0.112891 | 0.01108906 | 0.215694 |
| 2 | fixed_d1 | 0.00754142 | 0.113697 | 0.01113222 | 0.217241 |
| 2 | fixed_d4 | 0.00732960 | 0.106659 | 0.01085113 | 0.203643 |
| 2 | base | 0.00982149 | 0.301729 | 0.01452705 | 0.585690 |
| 3 | adaptive | 0.01335377 | 0.123419 | 0.02441086 | 0.336717 |
| 3 | matched | 0.01341960 | 0.124561 | 0.02450866 | 0.340255 |
| 3 | fixed_d1 | 0.01345538 | 0.125422 | 0.02458760 | 0.342663 |
| 3 | fixed_d4 | 0.01299913 | 0.117820 | 0.02385026 | 0.321464 |
| 3 | base | 0.01672722 | 0.316538 | 0.03125427 | 0.902227 |
| 4 | adaptive | 0.02106641 | 0.138461 | 0.04547727 | 0.475178 |
| 4 | matched | 0.02115272 | 0.139693 | 0.04566138 | 0.479947 |
| 4 | fixed_d1 | 0.02120672 | 0.140591 | 0.04579432 | 0.483254 |
| 4 | fixed_d4 | 0.02025229 | 0.131825 | 0.04410255 | 0.453289 |
| 4 | base | 0.02527864 | 0.333800 | 0.05653291 | 1.236027 |
| 5 | adaptive | 0.02967818 | 0.155283 | 0.07515545 | 0.630461 |
| 5 | matched | 0.02974415 | 0.156470 | 0.07540553 | 0.636417 |
| 5 | fixed_d1 | 0.02990402 | 0.157644 | 0.07569834 | 0.640898 |
| 5 | fixed_d4 | 0.02823635 | 0.147236 | 0.07233891 | 0.600525 |
| 5 | base | 0.03447296 | 0.352275 | 0.09100587 | 1.588302 |

## Exact allocation matching

At each K separately, matched depth labels were a fixed-seed permutation of adaptive labels over all 3,400 transitions. Every per-step depth histogram, refiner-call total, and reached-gate-evaluation total matched exactly. Matched gate evaluations were executed and discarded; they could not affect the forced transition-independent depth labels.

## Stagewise and stability evidence

All stored loss and prediction-norm arrays were finite: `True`. The manual sparse adaptive output satisfied the accepted dense-selected numerical contract: `True`. Per-horizon/per-stage Spearman score-versus-realized-next-refinement-gain summaries are stored in `RESULTS.json`; lossless per-case scores and gains are in `phase_b_metrics.npz`.

Timing was collected only as synchronized, single batched-pass diagnostics. It is confounded by diagnostic shadows and is not evidence of wall-clock acceleration.

## Candidate-ranking pilot

The pilot used 20 fresh, explicitly consumed simulator start/goal seeds and 64 identical five-block candidates per start. Candidate 0 followed frozen PlanOracle actions; the other 63 used the fixed condition-independent perturbation rule. Each candidate was executed from an exactly restored simulator state. Privileged simulator state was used only for exact restore, goal rendering, and real-outcome measurement, never as a model or gate input.

| condition | mean Spearman | mean real regret (m) | mean top-5 overlap | selected in real top-5 |
|:--|--:|--:|--:|--:|
| adaptive | 0.0184 | 0.000747 | 0.050 | 0.050 |
| matched | 0.0166 | 0.000794 | 0.050 | 0.050 |
| fixed_d1 | 0.0169 | 0.000747 | 0.050 | 0.050 |
| fixed_d4 | 0.0176 | 0.000794 | 0.060 | 0.050 |

The formal pilot label should be read cautiously: adaptive and matched selected different candidates on only 1 of 20 starts. Their mean Spearman difference was `+0.00172202` and their mean-regret difference was `-4.73518e-05` m. Ten starts had candidate-error range at most 1e-12 m; the median range was `4.34102e-05` m. Mean top-5 overlap was 0.050 for both, below the random-set expectation of `0.078125`.

| condition | candidate transitions | refiner calls | reached gate evaluations | total counted FLOPs |
|:--|--:|--:|--:|--:|
| adaptive | 6400 | 7481 | 7353 | 456014729065 |
| matched | 6400 | 7481 | 7353 | 456014729065 |
| fixed_d1 | 6400 | 6400 | 0 | 455669593600 |
| fixed_d4 | 6400 | 25600 | 0 | 460756825600 |

Pilot terminal label: `planner_ranking_promising`. Every adaptive/matched depth histogram, refiner-call total, and reached-gate-evaluation total matched separately for every start and rollout step. Per-candidate costs and real outcomes are stored losslessly in `phase_c_metrics.npz` and `phase_c_simulator.npz`.

## Scope and provenance

The frozen base model, stagewise refiner, compiled gate, thresholds, action normalization, and V5 whitening were loaded read-only and hash-checked against the accepted v004 package. Only `pixels` and `action` were materialized from raw consumed Phase-B episodes. In Phase C, privileged simulator state was used only to restore identical starts, render the goal, and measure real outcomes; it was never supplied to a model or gate. V3 targets/caches and the released HDF5 corpus were not opened. Exact frozen-input hashes, versions, selected identifiers, seeds, and configuration are in `STUDY_RECORD.json`, `CONFIG.json`, and `PHASE_C_PILOT_CONFIG.json`; final study-artifact hashes are in `ARTIFACT_HASHES.json`.

## Interpretation for the paper

This result is not a new confirmation claim. It answers the missing composition link only for the frozen mechanism and bounded consumed Cube data. It does not claim universal generalization, downstream closed-loop improvement, wall-clock acceleration, or novelty priority.
