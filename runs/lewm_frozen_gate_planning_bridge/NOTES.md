# Chronological scientific notes

This log records bounded scientific decisions and elapsed-time checkpoints.

- 2026-07-18 10:27:43 MST (elapsed 0.000 h): Fixed the consumed Phase-B cohort (episodes 0000-0099), five-step rollout rule, matched permutation seed 9401723, and 5% relative whitened non-degradation margin before opening any multi-step comparison.

- 2026-07-18 10:28:16 MST (elapsed 0.009 h): Phase A passed on 8 consumed episodes: wrapper inputs/base path reproduced, depth decisions were exact, selected outputs satisfied the accepted numerical contract, and frozen module/gradient audits were unchanged.

- 2026-07-18 10:29:07 MST (elapsed 0.023 h): Final pre-Phase-B preflight passed: adaptive sparse matched the accepted sparse path, forced matched execution was bitwise equal to the frozen selected-depth API, and histogram/gate-evaluation totals were exact.

- 2026-07-18 10:29:21 MST (elapsed 0.027 h): Completed Phase-B rollout horizon K=1; exact matched depth histogram and reached-gate count checks passed.

- 2026-07-18 10:29:22 MST (elapsed 0.028 h): Completed Phase-B rollout horizon K=2; exact matched depth histogram and reached-gate count checks passed.

- 2026-07-18 10:29:23 MST (elapsed 0.028 h): Completed Phase-B rollout horizon K=3; exact matched depth histogram and reached-gate count checks passed.

- 2026-07-18 10:29:24 MST (elapsed 0.028 h): Completed Phase-B rollout horizon K=4; exact matched depth histogram and reached-gate count checks passed.

- 2026-07-18 10:29:25 MST (elapsed 0.028 h): Completed Phase-B rollout horizon K=5; exact matched depth histogram and reached-gate count checks passed.

- 2026-07-18 10:29:25 MST (elapsed 0.028 h): Phase B met the frozen exploratory go criterion. Candidate ranking may be tested only as a small pilot using common candidates.

- 2026-07-18 10:38:58 MST (elapsed 0.188 h): Before any pilot simulator outcome or model cost, fixed 20 pilot start/goal seed tuples, 64 candidates per start, the PlanOracle-centered condition-independent proposal rule, top-k=5, matched seed, and planner-ranking decision rule.

- 2026-07-18 10:39:35 MST (elapsed 0.198 h): Generated and executed all 1,280 fixed pilot candidates across the 20 now-consumed start/goal identifiers. Full MuJoCo state restoration repeated candidate 0 exactly for every start; no performance-based exclusion occurred.

- 2026-07-18 10:39:56 MST (elapsed 0.204 h): Phase C completed with terminal label planner_ranking_promising; frozen planner criterion components were {'adaptive_mean_spearman_strictly_greater_than_matched': True, 'adaptive_mean_real_regret_strictly_less_than_matched': True, 'adaptive_mean_top5_overlap_at_least_matched': True, 'finite_and_exact_compute': True}. No follow-on task was launched.

- 2026-07-18 10:45:49 MST (elapsed 0.302 h): Final interpretation checkpoint: retained the prespecified planner_ranking_promising label because all fixed pilot rule components passed, but added the post-result boundary that near-zero rank correlations, 10/20 effectively constant candidate sets, and only 1/20 adaptive/matched selection disagreements make the planning evidence weak and do not justify a larger or closed-loop control task.

- 2026-07-18 10:49:52 MST (elapsed 0.369 h): Final verification passed: 8/8 focused invariant tests, lossless arrays recomputed the reported Phase-A/B/C summaries, all adaptive/matched histogram/call/gate checks were exact, simulator restoration replay was exact, all arrays were finite, and all 9 frozen-input hashes were reverified.
