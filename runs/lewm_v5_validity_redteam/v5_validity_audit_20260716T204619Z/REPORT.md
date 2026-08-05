# V5 adversarial validity audit

## Terminal outcome: `v5_validity_supported_with_caveats`

**Bottom line:** the frozen V5 confirmation appears genuine. I found no
demonstrated target leak, cohort reuse, outcome-conditioned recovery,
implementation error, numerical error, or comparator construction error that
can explain or reverse the positive result. The exact primary statistics,
calls, outputs, bootstrap intervals, ranks, and terminal mapping reproduce
independently over all 1,600 episodes and 60,800 transitions.

This is support with substantial caveats, not a clean bill of health. The
package fails several defense-in-depth fault tests, its unqualified “exact
FLOPs” wording is too broad, its historical transitive runtime is incompletely
attested, and the original base-training raw corpus is unavailable. None is
shown in the preserved canonical evidence, and none reverses the comparator
result.

## The precise claim supported

For the frozen v004 PlanOracle DGP and its prospectively generated 1,600-episode
confirmation cohort, the frozen causal policy allocated solver depth better
than the strongest transition-independent comparator at the same **declared
latent-prediction operation budget**. This statement is limited to this
data-generating distribution and accounting convention.

It does **not** establish broad generalization, a practically large gain,
lower latency, lower energy, or a literal end-to-end arithmetic FLOP total.

## Independently reproduced result

| Quantity | Clean-room result | Discrepancy |
|---|---:|---:|
| Episodes / transitions | 1,600 / 60,800 | exact |
| Adaptive raw MSE | 0.0034335407592805318 | within declared tolerance |
| Raw analytic-comparator MSE | 0.0034403097626425011 | exact |
| Raw benefit | 6.7690033619693052e-06 | 0 |
| Raw 95% interval | [5.4239551979869781e-06, 8.1629610044480406e-06] | 0 |
| Raw relative reduction | 0.196756% | sensitivity only |
| Adaptive native-whitened MSE | 0.098764199936424726 | within declared tolerance |
| Native-whitened comparator MSE | 0.099372822033575489 | exact |
| Native-whitened benefit | 0.00060862209715075702 | 1.08e-19 |
| Native-whitened 95% interval | [0.00051355703801858512, 0.00070842720416841825] | 0 |
| Native-whitened relative reduction | 0.612463% | sensitivity only |
| Call histogram d1/d2/d3/d4 | [50973, 7267, 1773, 787] | exact |
| Reached gate stages | [60800, 9827, 2560] | exact |
| Declared adaptive total | 4,332,936,120,435 | exact under ledger |
| Seeded comparator total | 4,332,936,224,000 | +103,565 |
| Recomputed terminal | `v5_confirmation_passed` | exact |

Every leave-one-episode-out effect is positive. All 16 contiguous 100-episode
raw block means are positive. Alternative accumulation orders and standard
quantile conventions preserve the sign and decision.

## Tests performed

The sealed matrix contained 49 tests. **44 passed and 5
failed; none is partial, unresolved, or not run.** Of 43 decision-critical
tests, 39 passed and 4 failed. Failed
IDs are: FRZ-03, CMP-01, FLT-02, FLT-03, NEG-03.

| Domain | Pass | Fail | Failed IDs |
|---|---:|---:|---|
| `causal_input_target_leakage` | 4 | 0 | — |
| `chain_of_custody` | 5 | 0 | — |
| `clean_room_recomputation` | 7 | 0 | — |
| `compute_comparator_fairness` | 3 | 1 | `CMP-01` |
| `confirmation_isolation` | 5 | 0 | — |
| `exclusions_recovery` | 3 | 0 | — |
| `fail_closed_fault_injection` | 3 | 2 | `FLT-02`, `FLT-03` |
| `frozen_implementation` | 3 | 1 | `FRZ-03` |
| `inference_sparse_execution` | 4 | 0 | — |
| `negative_controls` | 4 | 1 | `NEG-03` |
| `statistical_numerical_sensitivity` | 3 | 0 | — |

### Chain of custody and isolation

- The v001-v004 chronology is coherent. v001 generated zero outcomes after 101
  wrong-runtime slot-0 failures; v002 and v003 stopped before sealing/raw
  outcomes; only v004 completed. Scientific bytes/functions are unchanged
  across versions.
- v004 contains exactly 1,600 unique raw IDs/paths/hashes and 60,800 ordered
  rows, with zero replacements and no confirmation failure ledger.
- The scan compared all 1,600 V5 archives against 2,472 accessible prior raw
  archives, plus 31,629 structured prior IDs and 28,427 prior seeds. Exact,
  low-32-bit, SeedSequence, state, action, observation, pixel, and calibrated
  near-duplicate tests found no collision.
- The unavailable original base-training raw corpus is an explicit limitation.
  V5 nevertheless postdates training and uses fresh, unique seed tuples.
- Initial/final v001-v004 aggregate SHA-256 is
  `ef8fdadfd5ea2221a833e1582b85ca1ea441cbbd90dd29911acc3d433c8a4619`; the full readiness
  program aggregate is
  `ec48c0670f8167f761debff613929578b83b5489b88ba129468fe3462827ebaa`.
  No path, content, or stat record changed during the audit.

### Causal input and leakage

Static dataflow and runtime access tracing agree: model/gate code materializes
only `pixels` and `action`. Gate features use past/current latent history,
action history, the current prediction, and the most recent solver update.
Targets, future observations/latents, loss/gain, contact, reward, done,
success, oracle decisions, future summaries, and V5 normalization canaries do
not enter a decision. Varying those copied canaries leaves allowlisted inputs
and semantic outputs exact.

### Frozen execution and complete replay

All frozen model/gate/whitening/source/input hashes currently match. Module
state is frozen and gradient-free before/after. The full MPS packaged replay
reproduces targets, all dense exits, selected sparse outputs, calls, scores,
and features for all 60,800 rows.

The independent explicit sqrt/sum feature transcription differed by at most
`3.814697265625e-6` from float32 reduction order, passed the preregistered audit
`atol/rtol`, and preserved every call/output exactly. Only 88/1,600 episodes
met the package's tighter sealed score-allclose under that deliberately
different algebraic order. The required second, vector-norm-order
transcription was bit exact for features, scores, calls, and outputs. The
initial failure was an audit-harness acceptance bug (absolute-only comparison
without the preregistered relative term), retained in
`REPLAY_RESULTS_INITIAL_FAILURE.json`; no tolerance was relaxed.

### Compute and comparator fairness

The declared integer ledger is internally exact:

- 60,800 base and mandatory d1 calls;
- 13,174 later-adapter calls;
- 73,187 gate evaluations × 7,985 = 584,398,195 gate FLOPs;
- declared total 4,332,936,120,435;
- seeded comparator 4,332,936,224,000, or 103,565 more.

The analytic comparator is global and transition-independent. It uses a d1/d2
mixture with d2 weight `0.25295409752763076`; using aggregate target loss to
select the strongest global pair makes it a stronger benchmark but supplies
no row-level routing information.

The total is not a literal end-to-end count. The ledger inherits dense-linear
constants that omit visible biases, GELUs, alpha multiplication, and residual
adds; it also excludes the executed pixel encoder. Adding the directly visible
704 or 832 operations per later adapter changes the raw effect to
`6.773222841315813e-6` or `6.77398761790643e-6`, respectively—slightly
stronger—and the seeded comparator remains weakly more compute. The justified
wording is “exact under the frozen latent-prediction ledger.”

### Negative controls

- Within-episode/reachability score permutations: 2,000 repetitions,
  raw/white plus-one p = 0.00049975.
- Random and constant/mixed matched-compute policies: 2,000 each,
  raw/white p = 0.00049975.
- Score-to-gain breaks and perturbed reversed order: 2,000 each,
  raw/white p = 0.00049975; deterministic reversal is worse.
- Unused fields: exact semantic invariance.
- **Retained failure:** 1,000 whole-episode target reassignments give raw
  p = 0.164835 and white p = 0.000999. A second 1,600×1,600 pair-matrix
  implementation agrees. This null changes the physical prediction task and
  has 15.1× the SD of matched allocation nulls. A separately labeled post-hoc
  loss-profile block permutation gives raw/white p = 0.00049975; it diagnoses
  but does not erase the preregistered failure.

### Fault injection

Thirteen copy-only corruptions were exercised. Nine behaved as required and
four exposed fail-open behavior (`FLT-02D, FLT-02E, FLT-03A, FLT-03B`):

- one-byte execution/gate changes, missing/extra files, manifest hash drift,
  thresholds, compute price, wrong runtime, and post-manifest canaries reject;
- a duplicate manifest identifier is accepted by the existing-manifest
  preflight (the smoke final verifier later rejects it);
- a consistently rebound episode-block order is semantically accepted;
- a `../` raw path with a declared matching hash escapes the root;
- a byte-identical external symlink passes the final hash/path verifier.

Canonical evidence contains none of these states: paths are relative and
confined, files are regular, IDs/paths/hashes are unique, rows are ordered,
and both complete snapshots match.

## Findings

### Blockers

None.

### Major

- **V5-AUD-MAJ-001 — Artifact validators have pre-seal path and manifest fail-open surfaces.** These are meaningful supply-chain/hardening weaknesses and could permit substitution or duplication before the confirmation input seal. They did not compromise the preserved result: canonical paths are confined and regular, IDs/paths/hashes are unique, row order is exact, and initial/final snapshots are unchanged.
- **V5-AUD-MAJ-002 — The unqualified exact-FLOP claim is not literal end-to-end arithmetic.** The claim must be scoped to the declared convention. Comparator fairness survives: common omissions cancel, adding 704 or 832 visible operations per later adapter slightly increases the raw advantage, and every sensitivity keeps the seeded comparator weakly more compute.
- **V5-AUD-MAJ-003 — Historical transitive runtime environment is incompletely attested.** An exact historical environment cannot be reconstructed solely from the seal. Current dual-runtime probes and a bitwise-exact 60,800-row MPS replay make a decision-changing historical substitution unsupported, but cannot retroactively attest it.
- **V5-AUD-MAJ-004 — Complete original base-model training raw corpus is unavailable.** All accessible evidence is clean—2,472 prior raw archives, 31,629 IDs, 28,427 seeds, transformed seeds, initial states, actions, observations and pixels—and V5 was generated later from fresh unique seeds. This leaves a bounded provenance uncertainty, not evidence of direct cohort leakage.
- **V5-AUD-MAJ-005 — The sealed audit plan has an internally over-strict terminal mapping.** No failed test was relabeled. Because the user requires exactly one outcome, the governing top-level definitions are applied: no Blocker, exact primary reproduction, and bounded Major caveats map most truthfully to v5_validity_supported_with_caveats.

### Minor

- **V5-AUD-MIN-001 — Predeclared whole-episode target permutation fails in raw space.** Whole-target reassignment changes the physical prediction task and has 15.1× the null SD of matched allocation controls. All sharper reachability/histogram/score-gain controls pass; a separately labeled post-hoc loss-profile permutation also passes. The preregistered failure remains visible.
- **V5-AUD-MIN-002 — Historical execution device/command is not explicit in the execution manifest.** Reproducibility/attestation gap; no observed numerical discrepancy.
- **V5-AUD-MIN-003 — No post-decision complete package path-set manifest exists.** The audit's initial/final full snapshots close the preserved-evidence gap.
- **V5-AUD-MIN-004 — Raw loader safely ignores rather than schema-rejects unknown fields.** No target leak; strict unknown-field rejection would improve defense in depth.
- **V5-AUD-MIN-005 — Chronology relies on mutable local timestamps and has a narrow TOCTOU window.** Strict ordering, pre/post hashes, exact replay and unchanged final snapshots provide strong but not notarized custody.
- **V5-AUD-MIN-006 — Initial replay audit acceptance logic produced a false failure.** Synthetic validation identified the audit-harness defect. No tolerance was relaxed: order-matched features/scores are exact; explicit algebra passes audit tolerance; every call/output is exact. Only 88/1,600 episodes satisfy the package's stricter score allclose under that deliberately different order.

## Preregistered terminal-mapping defect

The sealed audit plan made every one of the 49 tests mandatory for either
supported outcome. It also limited invalidation to a validated Blocker and
inconclusive to missing/corrupt essential evidence or persistent critical
disagreement capable of hiding a Blocker. The five resolved failures are real
and remain failures, but no Blocker exists and the primary evidence is complete
and mutually agreeing. The sealed rules therefore leave an unintended empty
case: no label fits.

I have not converted failures into passes. Because the user requires exactly
one of four outcomes, I apply the governing user-level definitions and choose
`v5_validity_supported_with_caveats` as the least misleading exact label: the confirmation is
independently supported, while important bounded Major caveats remain. Calling
it invalidated would falsely imply a demonstrated invalidating path; calling it
inconclusive would falsely imply the primary result or essential evidence is
unresolved.

## Generalization, practical size, and latency

- **Frozen-confirmation validity:** supported with the caveats above.
- **Generalization:** not established. No output from the separate
  generalization task was used.
- **Practical effect:** raw MSE reduction is about 0.197% and
  native-whitened reduction about 0.612% versus the analytic
  comparator—statistically clear but small.
- **Latency:** no win is established. Synchronized MPS adaptive medians are
  3.837 ms at
  batch 1 versus 3.415
  ms for fixed d1, and
  49.693 ms
  at batch 1,024 versus
  48.115 ms for fixed
  d1. Pixel encoding and simulator rollout are excluded from these timings.
- **Energy:** not measured.

## Artifact locations

- Audit worktree: `/Users/rishisim/Documents/research/.worktrees/lewm-p2-contact/v5-validity-audit-20260716T204619Z`
- Audit run: `/Users/rishisim/Documents/research/.worktrees/lewm-p2-contact/v5-validity-audit-20260716T204619Z/runs/lewm_v5_validity_redteam/v5_validity_audit_20260716T204619Z`
- Canonical read-only evidence: `/Users/rishisim/Documents/research/lewm-p2-contact/runs/lewm_v5_readiness_program/v5_package_versions/v004`
- Test ledger: `/Users/rishisim/Documents/research/.worktrees/lewm-p2-contact/v5-validity-audit-20260716T204619Z/runs/lewm_v5_validity_redteam/v5_validity_audit_20260716T204619Z/TEST_RESULTS.json`
- Findings: `/Users/rishisim/Documents/research/.worktrees/lewm-p2-contact/v5-validity-audit-20260716T204619Z/runs/lewm_v5_validity_redteam/v5_validity_audit_20260716T204619Z/FINDINGS.json`
- Final source reconciliation: `/Users/rishisim/Documents/research/.worktrees/lewm-p2-contact/v5-validity-audit-20260716T204619Z/runs/lewm_v5_validity_redteam/v5_validity_audit_20260716T204619Z/SOURCE_FINAL_HASHES.json`
- This report: `/Users/rishisim/Documents/research/.worktrees/lewm-p2-contact/v5-validity-audit-20260716T204619Z/runs/lewm_v5_validity_redteam/v5_validity_audit_20260716T204619Z/REPORT.md`

No commit, push, pull request, retraining, replacement outcome, V3 target
inspection, or canonical evidence mutation was performed.
