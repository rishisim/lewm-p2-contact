# LeWM domain-robust causal contact-free gate preregistration — v001

Status: normative design frozen before every fresh fit, selection, smoke, and
confirmation episode. At this writing all four fresh-outcome counts are zero.

## Claim and immutable scientific boundary

This study tests whether one newly fit causal contact-free routing gate lowers
episode-averaged raw latent MSE and the unchanged V5 PlanOracle-native-whitened
latent MSE relative to the strongest transition-independent analytic allocation
at exactly matched total counted compute in each of four predeclared DGPs.
Every DGP is reported separately and the eight DGP-by-endpoint claims form one
simultaneous family.

The base world model, four-exit stagewise refiner, solver causality, sparse
execution semantics, target construction, action normalization, gradient
boundaries, and V5 fixed-whitening matrix are frozen. The only scientific
intervention is the gate/routing policy. V5 v004 and v005 are consumed terminal
evidence. v005 outcome summaries informed diagnosis only; no V5 or v005 episode
may enter fitting, normalization, whitening, candidate selection, smoke, or
confirmation.

The claim is restricted to causal contact-free latent-prediction adaptive
compute at exact counted FLOPs. It cannot establish downstream control benefit,
contact-aware routing, wall-clock acceleration, universal robustness, or
priority.

## Bounded four-DGP envelope

The fixed order and settings in `DGP_MATRIX.json` are:

1. `native_plan`: PlanOracle, action noise 0.1, random-action probability 0.
2. `markov_oracle`: MarkovOracle with all other native settings fixed.
3. `plan_action_noise_0p2`: PlanOracle with only action noise changed to 0.2.
4. `plan_random_action_0p1`: PlanOracle with only random-action probability
   changed to 0.1.

No additional DGP may be added after this preregistration. Environment,
packages, reset mechanics, oracle construction, frame history, and modeled
steps remain the V5/v005 contract.

## Strict role isolation

For each DGP, exactly 300 fresh fit episodes are used only for gate fitting,
fit-only feature transforms, gain standardization, and an auxiliary robust
whitening matrix. Exactly 500 disjoint fresh selection episodes are used once
to select among already locked candidates; no selected head is refit. Exactly
six further episodes are permanently excluded mechanical smoke. Up to 4,500
prospectively ordered confirmation slots per DGP are assigned now; the fixed
common prefix length is chosen once by the sealed power rule after selection
and before smoke or confirmation.

Every primary and replacement episode ID and environment, policy,
oracle-NumPy, and action-space seed is assigned prospectively. Every fit,
normalization, whitening, selection, qualification, comparator, histogram,
bootstrap, smoke, and latency RNG identifier is also assigned. All strings and
numeric identifiers must be unique across new roles/DGPs and disjoint from all
permitted recorded prior runs, V5 package versions v001--v004, and v005. The
freshness verifier may inspect only metadata/ledgers; it may not open V3 target
arrays, NumPy-load the combined V3 cache, or read the released HDF5 corpus.

Two hundred replacement tuples per DGP and role are preassigned. Replacement
is allowed only after a rollout exception determined without target, loss,
contact, privileged state, motion, phase, reward, or success. A global registry
prevents tuple reuse across every role and DGP. Persistence failure is not a
rollout failure: verified complete orphan artifacts are adopted, and partial or
hash-inconsistent artifacts cause a procedural stop rather than replacement.
There is no outcome-conditioned exclusion and no sequential expansion.

## Causal candidate family

The feature graph is the unchanged counted 1,046-dimensional V5 graph:
detached latent history, normalized action history, current prediction, last
update, and eleven causal norm/cosine/change summaries. It has no target,
future, contact, state, reward, success, episode, or batch-reduction input.
Only raw episode arrays `pixels` and `action` may be materialized before the
post-terminal interpretation boundary.

`candidate_grid.json` fixes 24 stage-specific ridge candidates:

- architecture in `{balanced_pooled_dual, domain_envelope_eight}`;
- ridge penalty in `{0.01, 1, 100}`;
- sequential fit-score quantile in `{0.55, 0.65, 0.75, 0.85}`.

Each stage has endpoint-specific predicted-gain heads. Fit-role gain targets
are standardized separately by DGP, stage, and endpoint. The pooled architecture
fits one equally DGP-balanced raw head and one equally balanced fixed-whitened
head per stage. The envelope architecture fits raw/fixed heads separately in
each DGP; at runtime a single DGP-agnostic policy uses the minimum of all eight
compiled affine scores. DGP identity is never a gate input. Per-DGP feature
transforms for envelope heads are algebraically compiled into raw-feature
weights and biases.

Thresholds are sequential fit-only quantiles: a stage threshold is computed
only among fit rows reached under earlier fit thresholds. All 24 complete
objects and fit-only transforms are hash-locked before any selection array is
opened. Selection never recalibrates a transform, threshold, or head.

The V5 fixed-whitening endpoint remains co-primary and unchanged. A pooled
fit-only robust-whitening matrix may be reported as auxiliary only. It does not
enter the candidate ranking, simultaneous family, or terminal mapping.

## Prospective selection rule

Eligibility is mechanical, not effect-sign based. In every DGP a candidate
must have finite inputs and outputs, exact sequential call reconstruction,
valid exact-compute comparators, physical mean calls in `[1.05, 2.5]`, at least
5% of rows reaching stage 2, at least 1% reaching stage 3, exact within-episode
histograms, and a seeded comparator with weakly more counted compute. No
contact, privileged, target, reward, success, motion, or phase field can enter
routing or eligibility.

For candidate `c`, DGP `d`, and endpoint `e`, define

`Z(c,d,e) = selection_mean_exact_compute_contrast / max(fit_episode_SD(c,d,e), 1e-12)`.

Eligible candidates are ranked lexicographically by:

1. maximize the minimum `Z` over all eight DGP-by-endpoint cells;
2. maximize the minimum selection Spearman score/next-stage-combined-gain
   coefficient over all twelve DGP-by-stage cells;
3. minimize worst-DGP counted FLOPs per row;
4. prefer fewer heads;
5. prefer smaller ridge penalty;
6. prefer lower fit-score quantile;
7. lexical candidate ID.

Point estimates determine ranking; no selection confidence screen or positive
effect eligibility filter is allowed. If no mechanically eligible candidate
exists, that is a process-valid preconfirmation scientific failure. The full
selection ledger remains evidence and no smoke or confirmation is generated.

## Exact computation price

The frozen model prices are 70,529,190 base FLOPs per row, 669,184 mandatory
depth-1 refiner FLOPs, and 264,960 FLOPs for each additional refiner call. The
causal feature graph costs 3,801 FLOPs per reached decision. Each 1,046-wide
affine head costs 2,092 FLOPs.

Thus the pooled dual gate costs 7,985 FLOPs and five separately reported
non-FLOP comparison/min operations per reached decision. The eight-head
envelope gate costs 20,537 FLOPs and eleven non-FLOP operations. Executable
graph and independent symbolic derivations must agree before any rollout.
Measured synchronized latency is reported separately and never substituted for
FLOPs. Energy is measured only if a reliable resettable per-path interface is
available; unavailability is reported.

For every DGP and endpoint, the primary comparator exhaustively enumerates the
strongest transition-independent analytic allocation over fixed depths 1--4 at
the exact adaptive total FLOPs, including selected-gate feature/head overhead.
Controls are a fixed-seed integer allocation using equal or weakly more counted
compute, fixed depth 1, and within-episode randomization preserving each exact
call histogram. No comparator conditions on transition features.

## Prospective power and fixed confirmation size

`power_rule.json` is binding. After the selected gate is frozen, fit and
selection episode moments for the eight exact-compute co-primary contrasts are
used once. For each cell, the planning effect is 35% of the smaller positive
fit and selection mean. The planning SD is the maximum of the fit and selection
one-sided 95% upper SD limits and the consumed V5 endpoint SD. One-sided
per-claim alpha is `0.05/8`.

For each common grid size `{500,1000,...,4500}`, marginal normal-approximation
power is computed and the family-power lower bound is the union bound
`max(0, 1 - sum(1-power_i))`. The smallest grid point with family-power lower
bound at least 0.95 is frozen as `N*`. All DGPs use exactly the same `N*` and
the corresponding prefix of already assigned slots. There is no expansion.
If any planning mean is nonpositive or no grid point qualifies, the bounded
program maps to a process-valid scientific failure before confirmation; power
criteria are never weakened and a larger cohort is not improvised.

The inherited V5 35%-retention calculation extended to an eight-claim family
places 3,500 episodes per DGP above 0.95, but that is a nonbinding feasibility
yardstick, not a guarantee for the selected gate. The 4,500 maximum is sealed
from local storage/runtime evidence before fit outcomes.

## Confirmation, inference, and reporting

After the selected gate, all transforms, thresholds, feature definitions,
compute derivations, confirmation size/IDs, comparators, analysis source,
standalone verifier source, inference, and mapping are sealed, exactly six
excluded smoke episodes per DGP must pass mechanical qualification. Smoke
losses cannot be analyzed. Then exactly `N*` fresh confirmation episodes per
DGP are generated and executed once. Every raw and execution file is complete,
manifested, rehashed, and input-sealed before the first confirmation target
array is opened for analysis.

The episode is the resampling unit. Exactly 20,000 fixed-seed paired bootstrap
replicates are stored. Individual intervals are two-sided 95% percentile
intervals. Simultaneous one-sided lower bounds use Bonferroni quantile
`0.05/8 = 0.00625`; NumPy linear quantile interpolation is fixed. A claim is
supported iff its simultaneous lower bound is strictly positive. Every DGP is
reported separately, with descriptive heterogeneity.

Per DGP, nonterminal diagnostics report stagewise raw and fixed-whitened
score/next-stage-gain ranks and reached counts, routing calibration, causal
feature and solver-gain shifts, depth distribution, base/refiner calls, gate
evaluations, feature/head/total FLOPs, non-FLOP operations, latency, and energy
availability. Rank signs, controls, heterogeneity, latency, energy, and the
auxiliary whitening endpoint cannot alter the terminal label.

## Independent verification and terminal mapping

Before confirmation, a separately written standalone verifier is source-sealed.
It imports only the standard library and NumPy, emits JSON to stdout, and does
not import or mutate production state. From declarative contracts and sealed
artifacts it independently rehashes all inputs, validates ledger chronology and
role isolation, recomputes fit transforms/candidates/selection/no-refit,
reconstructs causal features from primitives, recomputes scores, calls, losses,
compute, comparators, every bootstrap replicate, simultaneous bounds, and the
mapping. Producer and verifier bootstrap arrays must match elementwise and by
hash. A non-scientific exclusive-output wrapper captures the audit.

Integrity failure has precedence and maps to
`domain_robust_gate_execution_invalid`. Otherwise all eight supported maps to
`domain_robust_gate_confirmed`; one through seven maps to
`domain_robust_gate_partial`; zero maps to `domain_robust_gate_failed`.
Preregistered no-candidate or power-infeasibility outcomes also map to the
failed label and must be independently verified.

Once any confirmation outcome is opened, a process-valid confirmed, partial,
or failed result is terminal and cannot be repaired or retried. Version-forward
is permitted only after a procedural invalidity with zero confirmation
artifacts/outcomes and explicit source, normalized-AST, scientific-object, and
hash equivalence; every invalid attempt remains immutable. Post-terminal
contact or other forbidden interpretation fields may be opened only if they
were outcome-independently recorded. Their absence is reported rather than
repaired.
