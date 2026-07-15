# LeWM Cube policy distribution-contract study

## Outcome

Mechanical decision: **`generator_reconstruction_failed`**.

Fresh PlanOracle did not satisfy the frozen offline distribution contract. The study therefore does not isolate V4's policy choice as the complete cause; only the smallest bounded version/action-semantics or replay check is justified.

A future, entirely fresh V5 confirmation is **not authorized**.
No V5 episode was launched here. This result is discovery evidence, not confirmation.

## Generator provenance and design

The controlling upstream source is OGBench v1.2.1, tag commit
`1d4140997f60c52c6fb0702ec100dc988b18c548`. Its official
`cube-single-play-v0` recipe uses `dataset_type=play`, and
`generate_manipspace.py` maps play to the non-Markovian `CubePlanOracle`;
the noisy recipe maps to `CubeMarkovOracle`. Installed source bytes, package
versions, the 95 GB offline HDF5 SHA-256, constructors, preprocessing, frozen
model objects, seeds, bootstrap estimands, metrics, and decision rules were
sealed before any new rollout.

The bounded counts were executed exactly: 12 physically separate smoke episodes
per policy (permanently excluded), 90 PlanOracle discovery episodes, and 30
MarkovOracle controls. The excluded smoke exposed an installed Cube reset bug:
`seed` was accepted but not forwarded, and the variation space was reseeded from
entropy. That failed smoke audit is preserved. Protocol Amendment 01 was frozen
before any main rollout; it explicitly seeded the normal Cube variation and
physical-state RNGs. All 30 main controls were then paired on identical
environment seeds and exact initial states. Wrapper and NumPy-global oracle RNG
streams remained separate. There was no sequential expansion and no
confirmatory episode.

## Pre-main protocol amendment

The local incompatibility and its correction were discovered using only the
excluded smoke data. The original failed pairing audit, installed source hashes,
corrected initial-state-only probe, corrected RNG values for every main episode,
and amendment seal are retained in `audit/`. Sample sizes, preprocessing,
reference bands, metrics, gate, and decision rules were unchanged.

## Distribution readout

Fresh PlanOracle's joint max-T statistic was
`3.85832` against the frozen 95% threshold
`2.64656`. Its paired standardized distance to
the offline center was `0.661692`, versus
`136.209` for MarkovOracle (ratio
`0.00485793`; Plan was closer on
7/7 core coordinates).

| Dataset | Episodes | |a|≥.99 | Action RMS | a[4] mean | |z|>3 | d0 MSE | d1→d2 gain | Contact | Calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Offline discovery | 420 | 0.0724071 | 0.858462 | 0.159004 | 0.00405238 | 0.00422941 | 0.000165336 | 0.486717 | 1.28239 |
| Offline calibration | 90 | 0.0725 | 0.857495 | 0.160415 | 0.00325556 | 0.00425077 | 4.08993e-05 | 0.488596 | 1.2538 |
| Fresh PlanOracle | 90 | 0.0674444 | 0.846155 | 0.157906 | 0.00386667 | 0.00454739 | 2.98976e-05 | 0.487719 | 1.25146 |
| Paired MarkovOracle | 30 | 0.3072 | 1.6046 | -0.504798 | 0.198933 | 0.0848554 | -0.000135029 | 0.220175 | 1.40263 |
| Consumed V4 MarkovOracle | 300 | 0.316697 | 1.61619 | -0.545498 | 0.2019 | 0.0770857 | -8.38167e-05 | 0.198246 | 1.38254 |

Raw pixel/observation summaries, latent histories/targets, raw and normalized
actions, d0 and fixed d1–d4 raw/whitened losses, updates, gains, every gate
feature block and normalized-z summary, scores, rankings, price exceedance,
calls, and post-hoc physical composition are retained in machine-readable
artifacts. Contact, impact, state, motion, phase, task, and success did not enter
training, inference, policy selection, or the primary contract thresholds.

## Frozen gate on 90 PlanOracle episodes

The unchanged gate used mean calls `1.25146` and raw MSE
`0.00333303`. Episode-bootstrap benefits (baseline minus
adaptive) were:

- exact-total-FLOP analytic: `8.06208e-06`,
  95% CI [`3.80774e-06`, `1.2448e-05`];
- conservative exact-total-FLOP seeded: `7.79205e-06`,
  95% CI [`2.50187e-06`, `1.31674e-05`];
- fixed d1: `1.67313e-05`,
  95% CI [`1.1234e-05`, `2.23016e-05`];
- exact call-histogram randomization: `1.08786e-05`,
  95% CI [`4.81321e-06`, `1.69799e-05`].

The exact-total comparator counted the 8,196-FLOP gate overhead at every
evaluated decision; fixed base/V1 and later-adapter costs use the unchanged
historical convention. Analytic exact matching, conservative integer matching,
exact histogram preservation, whitening, and call/frontier audits are recorded.
The separate synchronized MPS latency audit passed (28 timed path/batch rows); latency did not enter the statistical or FLOP verdict.

## Validity and claim boundary

Frozen-object byte hashes, no-gradient state, causal feature signatures, d0/d1
identity, dense/reference/optimized sparse equivalence, row counts, paired
initial states, raw-hash freshness, seed nonmembership, exact calls/FLOPs,
bootstrap intervals, and decision mapping were audited. Every new cache proves
empty V3-test intersection; the combined V3 cache was never opened with NumPy,
and V3 test targets remain untouched.

V4 is retained as a valid negative result for its actual MarkovOracle
distribution and was used only as a labeled diagnostic comparator. It defined
no band, threshold, model, normalization, stopping rule, or selection. This
study makes no confirmatory or contact-aware claim and no “first adaptive world
model” claim. LoopWM remains relevant adaptive-computation related work in text
environments.

## Strongest caveats and next step

- The released LeWM checkpoint lacks a complete original pretraining episode
  manifest; exact nonduplication was checked against every accessible recorded
  episode outside the opaque V3 test set, but full pretraining nonmembership
  cannot be mechanically proven.
- The public upstream command uses a 1,001-step collection horizon, while the
  released local LeWM artifact has 201 states/200 actions; this reconstruction
  therefore pins the local 200-action contract and tests it empirically.
- This is one environment, one frozen solver/gate, and a 90-episode discovery
  judge. Actual latency is hardware-specific and separate from FLOPs.


The authorized post-decision diagnostic replayed exactly one existing PlanOracle trajectory with zero new seeds or policy trajectories. Exact required transition replay was successful; the upstream 1,001-step versus released-local 200-action horizon mismatch remains. No model or gate was tuned.

The preregistered smallest bounded check is complete and passed exact replay; it did not resolve the offline-generator mismatch.

Next highest-information step: obtain or reconstruct the exact released HDF5 generator provenance (environment implementation, reset semantics, horizon, command/config, and seed handling), then preregister a new bounded generator-reconstruction study. Do not tune the gate or launch V5.
