# V5 Adversarial Validity Audit Plan

## Status and scope

This document is preregistered before inspection of detailed V5 per-episode
results. The only numerical outcomes known at sealing time are the headline
claims supplied in the audit brief. The pre-audit chain-of-custody snapshot is
`SOURCE_SNAPSHOT.json`.

The audit asks whether the already completed, frozen V5 confirmation is a
leak-free, correctly computed confirmatory result. It does not optimize V5,
retrain any component, generate replacement outcomes, inspect V3 test targets,
or assess broad out-of-distribution generalization.

All mutations and fault injections will operate on independent regular-file
copies inside this audit directory. Canonical evidence under
`/Users/rishisim/Documents/research/lewm-p2-contact/runs/lewm_v5_readiness_program/v5_package_versions`
is read-only and will be hashed again at audit completion.

## Evidence-handling protocol

1. Record `lstat` metadata and SHA-256 for every directory, regular file, and
   symlink in v001–v004 before detailed inspection.
2. Treat all package code, manifests, decisions, and prior independent audit
   outputs as claims under test rather than authorities.
3. Do not import `analysis.py`, `independent_verify.py`, or their helper
   functions in the clean-room recomputation.
4. Track every predeclared test by the IDs in `TEST_MATRIX.json`. Any later
   diagnostic is labeled `post_hoc`.
5. Validate audit harnesses on synthetic fixtures before interpreting a
   disagreement as a V5 defect.
6. When a check unexpectedly fails, reproduce with a technically independent
   method, minimize the discrepancy, and classify it as evidence, environment,
   or audit-harness behavior.
7. Rehash the complete v001–v004 evidence path set at the end and compare both
   content hashes and path sets to the initial snapshot.

## Predeclared numerical tolerances

These tolerances apply to agreement checks, not to changing the preregistered
V5 decision:

- Identifiers, seeds, row order, path sets, hashes, categorical outcomes,
  Boolean calls, reached-stage counts, call histograms, episode counts,
  transition counts, and integer FLOPs: exact equality.
- Float64 per-transition scalar errors independently recomputed from the same
  stored tensors: `atol=1e-12`, `rtol=1e-10`.
- Float32 model outputs in dense-versus-sparse or replay comparisons:
  `atol=2e-6`, `rtol=2e-5`; resulting gate calls must still be exact except
  where a score is within `2e-6` of a threshold, which triggers an explicit
  boundary analysis rather than an automatic pass.
- Raw aggregate effects and raw confidence bounds: `atol=5e-12`,
  `rtol=1e-9`.
- Native-whitened aggregate effects and confidence bounds: `atol=5e-10`,
  `rtol=1e-8`.
- Bootstrap quantiles produced from identical resample indices:
  `atol=5e-12`, `rtol=1e-9` for raw and `atol=5e-10`, `rtol=1e-8` for
  whitened quantities. Alternative quantile definitions are reported as
  sensitivity analyses and are not expected to be byte-identical.
- Stagewise ranks and signs: exact rank ordering after deterministic
  average-rank tie handling; reported rank statistics `atol=1e-10`.
- Serialized files are compared by semantic content unless a manifest
  explicitly requires byte identity.

Any discrepancy exceeding tolerance is a failure requiring diagnosis; it is
not automatically a V5 finding until the clean-room harness has passed a
synthetic fixture and a second method reproduces the discrepancy.

## Predeclared audit domains and falsification behavior

### 1. Chain of custody and chronology

Reconstruct v001–v004 and recovery events from filesystem metadata, ledgers,
state files, manifests, seals, and source differences. Establish whether model,
gate, normalization, compute-price, stopping-rule, analysis, and
outcome-mapping changes preceded the first V5 outcome. Search for launches,
partial outputs, abandoned versions, retries, replacements, missing episodes,
post-outcome version selection, timestamp contradictions, path-set gaps,
TOCTOU windows, and executable dependencies outside the seals.

Falsification: a performance-relevant post-outcome change, outcome-conditioned
selection or replacement, unexplained earlier outcome, irreconcilable
chronology, or unsealed executable capable of changing the decision fails the
domain.

### 2. Confirmation isolation and cohort leakage

Inventory accessible earlier training, discovery, calibration, qualification,
and normalization cohorts, excluding V3 test targets. Compare exact IDs,
seeds, transformed seeds, file hashes, initial-state fingerprints, observation
sketches, action sketches, and trajectory fingerprints where the evidence
supports them. Use nearest-neighbor or quantized-hash tests for near
duplicates, with thresholds calibrated on synthetic perturbations and
within-cohort neighbors.

Falsification: confirmed V5 membership or a causally equivalent duplicate in
an earlier model-selection/fitting cohort; or inability to account for the V5
cohort's derivation when prior-cohort contamination cannot otherwise be
excluded.

### 3. Causal inputs and target leakage

Statically enumerate every value entering a gate decision and trace its source,
transformation, shape, and decision-time availability. Instrument copied
loaders and gate code to log data access. Inject sentinel privileged fields
covering future observations, targets, losses, oracle gain, contact, reward,
done, success, and future summaries. Perturb each sentinel independently and
compare gate inputs/scores/calls.

Falsification: any privileged or future-derived value changes a gate feature,
score, or decision; confirmation-derived normalization is used; or runtime
access cannot be reconciled with the causal contract.

### 4. Frozen implementation and executed-code integrity

Verify hashes and loadability of models, solver exits, compiled gate, gate fit,
whitening, normalization, thresholds, runtime contract, source files,
interpreter, device, and dependency inventory. Search dynamic imports,
environment-dependent paths, stale bytecode, fallback branches, mutable global
state, nondeterministic kernels, gradients, and unrecorded dependencies.
Determine the strongest evidence for what actually executed.

Falsification: an unsealed or mismatched performance-relevant component,
environment-controlled implementation substitution, or missing executed-code
evidence capable of changing calls or metrics.

### 5. Clean-room numerical recomputation

Implement analysis independently with only standard numerical libraries and
the frozen artifacts. Validate on synthetic fixtures with hand-computed
answers. Recompute per-transition/per-episode errors, adaptive decisions from
stored scores, call histograms, reached stages, fixed exits,
transition-independent analytic and seeded comparators, raw/whitened effects,
gate-inclusive exact compute, comparator compute dominance, bootstrap
replicates and intervals, simultaneous lower bounds, stagewise ranks, and
terminal mapping.

Falsification: a verified discrepancy beyond the predeclared tolerance,
incorrect outcome mapping, comparator compute shortfall, or inability to
reproduce a decision-critical number from preserved evidence.

### 6. Inference and sparse-execution correctness

Audit row selection/restoration and `forward_selected`; compare dense and
sparse evaluation, batch sizes/orders, dtypes/devices, stage reachability,
NaN/unreached rules, and threshold boundaries. Replay frozen inference from
raw inputs over all 60,800 transitions when feasible. If a complete replay
fails, exhaust both (a) the packaged runtime path and (b) an independently
batched direct-module path, then quantify every untested row/stage.

Falsification: wrong row restoration, order dependence, unreachable-stage
contamination, non-equivalent sparse execution, boundary instability changing
calls, or material unreplayed surface after two strategies.

### 7. Compute and comparator fairness

Re-derive FLOPs from executed tensor shapes and operation ledgers, including
feature extraction and gate scoring. Check exact integer arithmetic, stage
cost mapping, analytic mixture construction, seeded comparator construction,
transition-independent-information constraints, histogram/call-count
comparators, and rounding bounds. Keep latency out of the FLOP claim.

Falsification: omitted adaptive overhead, seeded comparator using less compute,
privileged comparator information, wrong cost mapping, or rounding capable of
creating/reversing the observed benefit.

### 8. Exclusions, recovery, and outcome conditioning

Account for every preregistered episode and attempted generation via raw and
execution manifests, logs, ledgers, state files, and recovery artifacts.
Verify the 1,600-episode cohort, 60,800 transitions, zero replacements, path
sets, unique IDs, and rule-based restarts/exclusions.

Falsification: missing/extra/duplicate/replaced episodes, performance-dependent
recovery, unexplained partial outcomes, or manifest/path-set disagreement.

### 9. Fail-closed fault injection

On independent copies, inject: one-byte outcome corruption; model/gate hash
mismatch; missing raw file; extra raw file; duplicate identifier; manifest
mismatch; reordered rows; traversal path; symlink substitution; changed
threshold; changed compute price; wrong runtime/interpreter condition; and an
unexpected privileged field. Run each applicable seal, loader, preflight, and
final verifier.

Falsification: an applicable verifier reports success or proceeds to a
confirmatory decision despite the corruption. A control is `not_applicable`
only when the package has no claimed mechanism for that fault; lack of a
needed mechanism is separately assessed as a finding.

### 10. Negative controls and falsification tests

Use deterministic master seeds and 2,000 repetitions per random control family
when computationally practical:

- within valid episode/stage strata, permute scores or decisions while
  preserving reachability;
- random matched-total-call policies;
- constant/mixed policies matched to total compute;
- episode or valid-transition-block target-alignment permutation;
- break score-to-marginal-gain association while preserving compute and stage
  availability;
- reverse or perturb score ordering where the ordering has a causal meaning.

Add unused fields and verify exact invariance. Report randomization tail
probabilities with the plus-one correction `(extreme+1)/(R+1)`, effect
distributions, and whether controls preserve their stated constraints.
Controls diagnose causal allocation; they do not redefine confirmation.

Predeclared seeds:

- score/decision permutation: `510001`
- random matched-call policy: `510002`
- constant/mixed matched-compute policy: `510003`
- target/block permutation: `510004`
- broken score-gain association: `510005`
- ordering reversal/perturbation: `510006`
- synthetic harness fixtures: `730001`, `730002`, `730003`
- sensitivity bootstrap cross-check: `880001`

If 2,000 repetitions cannot be completed, at least 1,000 are mandatory. Fewer
than 1,000 makes the corresponding control unresolved unless its state space
is exhaustively enumerated.

Falsification: the real allocation is not distinguishable from controls that
destroy its causal association (plus-one one-sided `p >= 0.05`) or a
supposedly unused field changes results. A failed negative control is
diagnostic and becomes a severity finding only after assessing whether its
null is scientifically appropriate and its constraints were truly matched.

### 11. Statistical and numerical sensitivity

Recompute with alternative aggregation orders, pairwise/Kahan/`math.fsum`
accumulation, float64, independent bootstrap index generation, documented
quantile conventions, leave-one-episode-out influence, seed blocks, and
reasonable non-normative sensitivities. Preserve the preregistered decision.

Falsification: sign or terminal decision changes under a numerically
equivalent implementation; a small number of episodes dominate enough to make
the scientific interpretation fragile; or undocumented conventions are
decision-critical.

## Finding severity

- **Blocker:** demonstrated or credible leak, bug, process violation, or
  accounting error capable of creating or reversing the confirmatory claim.
- **Major:** materially weakens the result or leaves an important failure
  surface unresolved, but is not presently shown capable of creating or
  reversing the claim.
- **Minor:** disclosure, hardening, or reproducibility issue that does not
  threaten the decision.

Severity is based on causal capability and evidence, not merely on which test
found the issue.

## Terminal decision mapping

Exactly one terminal outcome will be emitted:

- `v5_validity_invalidated`: one or more validated Blockers, including a
  demonstrated leak, outcome-conditioned cohort/process, verified
  decision-reversing numerical or compute error, or fail-open defect that
  actually compromises preserved evidence.
- `v5_validity_audit_inconclusive`: no validated invalidation, but essential
  evidence is unavailable/corrupt; source integrity changed without
  reconciliation; isolation cannot be maintained; critical independent
  implementations still disagree after synthetic validation and serious
  diagnosis; or a mandatory decision-critical test remains unresolved such
  that a Blocker cannot be excluded.
- `v5_validity_supported_with_caveats`: no Blocker; all mandatory
  decision-critical checks pass; the exact claimed decision is independently
  reproduced; but at least one Major issue or bounded important uncertainty
  remains, or Minor issues materially constrain reproducibility/hardening.
- `v5_validity_supported`: no Blocker or Major; all mandatory checks for
  support pass; clean-room metrics and exact compute agree; full or
  substantively complete replay agrees; controls behave diagnostically; fault
  tests fail closed; source hashes remain stable. Non-material Minor
  disclosures are permitted but must not create an important unresolved
  surface.

Invalidation takes precedence over inconclusive; inconclusive takes precedence
over supported outcomes whenever the missing surface could conceal a Blocker.

## Mandatory checks by candidate outcome

For either supported outcome, all tests marked `mandatory_for_support` in
`TEST_MATRIX.json` must be `pass`, except explicitly bounded non-critical
replay coverage may be `partial` only for
`v5_validity_supported_with_caveats` after both replay strategies are
exhausted. All decision-critical recomputation, cohort accounting, causal
input, comparator fairness, and final source-integrity checks must be full
passes.

For invalidation, the triggering Blocker must pass synthetic harness
validation, reproduce with a second independent method where technically
possible, and have a documented causal path to creating or reversing the
claim.

For inconclusive, the report must identify the exact missing evidence or
persistent disagreement, alternatives attempted, and why the residual surface
could conceal a Blocker.

## Reporting

`REPORT.md` will separate:

1. validity of this frozen V5 confirmation;
2. generalization beyond its data-generating distribution;
3. practical effect size; and
4. actual latency.

Only item 1 is the confirmatory validity judgment of this audit.
