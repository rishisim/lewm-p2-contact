# Proposal 2 Final Synthesis

## Summary Table

| Test | Verdict | Implication |
| --- | --- | --- |
| A: Horizon control | A3 | Contact effect is conditional on trajectory phase; prior run found the effect survived only in Q3 under stricter within-quartile and bootstrap checks. |
| B: Multi-step rollout | B2 | Multi-step rollout did not show contact-specific divergence; control windows diverged harder later under the paired-window constraints. |
| C: Random data | C1 | Prior weak/random fallback run found expert trajectory bias was masking the signal: onset MSE at t=0 was 0.2512 vs 0.1483 at t=-5, and horizon control survived. |
| D: Alternative events | D2 | Alternative event definitions on the expert trajectories did not produce a cleaner localized onset spike; B and D ratios were elevated but sparse/noisy, not diagnostic. |

## Overall Conclusion

OVERALL MIXED

One diagnostic shows a signal but the others don't. The idea is not dead but needs a different foundation. Specific recommendation: the only clear positive result is C1, which implies the contact-onset claim should be reframed as a trajectory-diversity/data-regime claim, not as an expert-trajectory geometric-contact claim. Discuss with Dr. Ding before further work.

Note: C1 is included from the prior completed run's user-visible result message; its artifact worktree was no longer present for fresh disk verification in this checkout.
