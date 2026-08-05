# LeWM V5 zero-shot generalization report

Terminal result: `zero_shot_generalization_partial`.

The fixed V5 policy was evaluated once on exactly 3,000 fresh episodes in each
of three preregistered one-factor DGP shifts. All 9,000 cohorts and 342,000
modeled rows were hash-sealed before analysis. The independent verifier
reproduced the terminal evidence and passed.

| Regime | Regime map | Raw effect and simultaneous lower bound | Fixed-whitened effect and simultaneous lower bound | Depth 1–4 histogram |
|---|---|---:|---:|---|
| markov_oracle | mixed | -2.5137787e-06 [-5.6770967e-06, +∞) | 0.00017736181 [8.6062122e-05, +∞) | [103059, 6748, 3507, 686] |
| plan_action_noise_0p2 | supported | 6.6906277e-06 [5.2331781e-06, +∞) | 0.00063148227 [0.00055181589, +∞) | [96236, 12960, 3261, 1543] |
| plan_random_action_0p1 | mixed | 9.5995657e-08 [-3.128984e-06, +∞) | 0.000426306 [0.00036053634, +∞) | [96637, 12788, 3166, 1409] |

The simultaneous family is the fixed set of six regime-by-endpoint claims,
using 20,000 episode bootstrap replicates and Bonferroni one-sided alpha
0.05/6. Positive values favor the frozen adaptive policy over the strongest
transition-independent analytic mixture at exactly matched total counted
compute, including gate feature and dual-head overhead.

Overall, 4 of six co-primary
claims have strictly positive simultaneous lower bounds. The preregistered
mapping therefore yields `zero_shot_generalization_partial`. Auxiliary seeded,
fixed-depth-1, and within-episode-histogram controls, stagewise rank signs,
heterogeneity, latency, energy availability, and post-hoc interpretation did
not affect that mapping.

Latency is reported separately from FLOPs at
`metrics/latency_and_resources.json`. Energy was not measured because this
runtime has no reliable resettable per-path energy interface. Contact, motion,
phase, reward, and success were opened only after the immutable decision and
are reported only in `metrics/posthoc_interpretation.json`.

The confirmed V5 claim remains restricted to the fixed Cube PlanOracle DGP:
lower episode-averaged raw and PlanOracle-native-whitened latent MSE than the
strongest transition-independent analytic allocation at exactly matched
counted compute. This generalization study does not establish universal
generalization, downstream control improvement, contact-aware routing,
wall-clock acceleration, or “first adaptive world model.” LoopWM remains
relevant related work.
