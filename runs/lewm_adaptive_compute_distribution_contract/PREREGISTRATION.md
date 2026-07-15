# Preregistration: Cube policy distribution contract

Status: frozen before every new smoke or main rollout. This is a bounded
discovery/diagnostic study, not confirmation. No V5 episodes may be generated.

## Controlling generator contract

OGBench v1.2.1 at `1d4140997f60c52c6fb0702ec100dc988b18c548` maps `dataset_type=play` to the
non-Markovian `CubePlanOracle` and `dataset_type=noisy` to `CubeMarkovOracle`.
The paired local generator uses the exact same `swm/OGBCube-v0` constructor,
200-action horizon, capture path, and preprocessing for both policies. The
first 30 main environment seeds are identical between policies. Wrapper RNG
and NumPy-global oracle RNG streams are separate and recorded.

## Fixed design

- 12 PlanOracle and 12 MarkovOracle smoke episodes, permanently excluded.
- 90 fresh PlanOracle discovery-judge episodes.
- 30 paired MarkovOracle diagnostic-control episodes on the first 30 Plan seeds.
- No sequential expansion and zero V5 confirmation episodes.

## Isolation

Only the physically isolated 420-episode V3 discovery cache and already-consumed
90-episode calibration cache can supply offline targets. The combined V3 cache
is hash-checked opaquely and never loaded. Raw HDF5 reads are limited by those
two cache allowlists and are used only for raw-input and post-hoc composition
descriptors. Every derived cache proves empty intersection with the pinned V3
test episode set. Contact, impact, state, motion, phase, task, and success are
never solver/gate inputs or primary decision thresholds.

## Frozen reference and decision

The core metrics are: raw_action_abs_ge_0_99_rate, raw_action_rms, raw_action_coord4_mean, normalized_action_abs_gt3_rate, history_temporal_change_norm, target_norm, d0_raw_mse. A 10,000-replicate,
episode-resampled 90-episode reference distribution supplies marginal 95%
bands and a joint 95% max-T region. It is frozen before new target generation.

`distribution_contract_passed` requires every item in the predeclared JSON
rule, including core-region membership, material paired closeness to Plan over
Markov, restored positive d1-to-d2 gain, all frozen-gate discovery criteria,
and all validity audits. If reconstruction fails, only a smallest bounded
version/action-semantics or replay check is allowed; the gate cannot be tuned.
If distribution matches but the gate fails, V5 remains unauthorized and the
next study is domain-robust gate discovery.

Before any fresh rollout, the offline-only recomputation found that discovery
and calibration had positive but materially different d1-to-d2 means
(`1.65336e-4` and `4.08993e-5`). The required restoration rule is therefore the
directional rule stated in the task (`PlanOracle mean > 0`); the two split means
and pooled episode-bootstrap band are frozen and reported descriptively, not
used as an added equivalence threshold. The seven-metric joint distribution
region remains unchanged and includes policy/action and base-model d0 metrics,
not d1-to-d2 gain.

## Claim limits

This study cannot support a confirmatory, contact-aware, or “first adaptive
world model” claim. V4 remains a valid negative result for its actual
MarkovOracle distribution and is diagnostic only. LoopWM is relevant related
work in text environments.
