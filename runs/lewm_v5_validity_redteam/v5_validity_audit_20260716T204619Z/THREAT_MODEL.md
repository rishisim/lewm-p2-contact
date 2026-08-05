# V5 Validity Threat Model

## Protected claim

The protected asset is the integrity of the frozen V5 confirmatory statement:
the adaptive policy produced the preregistered terminal pass on exactly 1,600
confirmation episodes and 60,800 evaluated transitions, using only causal
inputs, with exact gate-inclusive compute and fair transition-independent
comparators.

The audit does not protect a claim of broad generalization, practical
importance, or latency improvement.

## Trust boundaries

The following are untrusted until independently checked:

- v001–v004 package source, manifests, seals, reports, and state files;
- `analysis.py` and `independent_verify.py`;
- serialized raw and execution arrays;
- filenames, filesystem metadata, and path lists;
- model, solver, gate, whitening, normalization, and threshold artifacts;
- package-provided FLOP ledgers and comparator construction;
- recovery logs and zero-replacement assertions;
- runtime/environment attestations.

The OS kernel, SHA-256 implementation, local read-only access to the canonical
evidence, Git object database for committed source, and two independent
numerical implementations are the minimal operational trust base. SHA-256
collision attacks and malicious kernel/hardware behavior are outside scope.

## Threat actors and failure modes

No malicious actor is assumed. The threat model covers accidental leakage,
researcher degrees of freedom, outcome-conditioned process choices, stale or
wrong code, incomplete seals, serialization/path bugs, numerical mistakes,
and deliberate-corruption resilience. A hypothetical insider with canonical
write access is modeled only through detectable content/path/stat changes,
manifest inconsistencies, and chronology gaps.

## Attack surfaces

### Cohort construction

- exact reuse of V5 IDs or seeds in training/discovery/calibration;
- transformed-seed collisions or deterministic cohort overlap;
- duplicated/near-duplicated initial states or trajectories;
- hidden qualification on V5-derived summaries;
- missing, replaced, retried, or performance-filtered episodes.

### Gate causality

- future observations/latents, targets, loss, marginal gain, contact, reward,
  done, success, oracle decision, or future episode summaries;
- V5-derived normalization/whitening;
- indirect leakage through cached arrays, global state, filename/order,
  episode length, or error-dependent missingness;
- permissive loaders that forward unexpected fields.

### Executed implementation

- package/source hash mismatch;
- dynamic import or environment-variable substitution;
- stale bytecode, fallback code, editable install, or external helper;
- dtype/device/batch/order/nondeterministic branch;
- sparse row-selection/restoration bug;
- threshold boundary mismatch.

### Analysis and statistics

- wrong target/prediction alignment;
- wrong aggregation unit or weights;
- incorrect whitening;
- unfair comparator mixture or information;
- seeded comparator compute shortfall;
- omitted gate overhead;
- integer/float rounding;
- bootstrap seed/index/quantile error;
- wrong multiplicity correction, ranks, or outcome mapping;
- cherry-picked version or post-outcome analysis change.

### Artifact integrity

- incomplete path manifests;
- TOCTOU between preflight and execution;
- symlink/path traversal;
- extra/missing files;
- duplicate identifiers or reordered rows;
- seals that hash declarations rather than executed bytes;
- verifiers that fail open after corruption.

### Recovery and chronology

- abandoned unfavorable partial launch;
- favorable retry or replacement;
- changes after first outcome visibility;
- inconsistent timestamps or backfilled manifests;
- recovery branch conditioned on performance.

## Adversary goals

1. Create an apparent adaptive benefit without causal allocation skill.
2. Make an unfair lower-compute comparator look compute matched.
3. Hide contamination or privileged features.
4. Change code/data after preregistration while preserving superficial seals.
5. drop, replace, reorder, or duplicate outcomes to improve the result.
6. Cause an incorrect bootstrap, confidence bound, or terminal mapping.
7. Make verifiers accept corrupted or substituted artifacts.

## Security properties under test

- **Cohort independence:** V5 is absent from all earlier fitting and selection.
- **Causal availability:** every gate input exists at its decision time.
- **Freeze integrity:** performance-relevant bytes and parameters were fixed
  before outcome visibility and match the executed path.
- **Cohort completeness:** every preregistered episode is present once, with no
  replacement or performance-conditioned exclusion.
- **Numerical correctness:** independent implementations reproduce all
  decision-critical metrics within preregistered tolerance.
- **Compute fairness:** adaptive compute includes overhead and each comparator
  satisfies its information and compute constraints.
- **Execution correctness:** sparse adaptive inference is equivalent to its
  dense reference on reached rows and preserves identity/order.
- **Fail-closed integrity:** corruptions are rejected before a successful
  confirmatory decision.
- **Robust interpretation:** destroyed allocation information does not mimic
  the real effect, and the result is not a numerical or single-episode
  artifact.

## Evidence-to-threat mapping

- Git/source diffs, manifests, ledgers, state, and timestamps address freeze,
  chronology, recovery, and version selection.
- Cross-cohort hashes/fingerprints address contamination.
- Static dataflow plus runtime instrumentation and canaries address causal
  availability.
- Clean-room recomputation plus synthetic fixtures address analysis defects.
- Full replay, dense/sparse comparisons, and boundary tests address inference.
- Operation-graph reconstruction and integer arithmetic address compute.
- Copy-only mutation campaigns address fail-closed behavior.
- Randomization controls and sensitivity analyses address causal allocation
  plausibility and fragility.
- Initial/final complete snapshots address canonical source integrity.

## Residual risks that cannot be silently waived

- An executed environment not reconstructable from preserved evidence.
- Earlier inaccessible cohorts needed to rule out overlap.
- A full raw replay that remains impossible after two technically distinct
  strategies.
- Timestamp provenance limited to mutable local filesystem metadata.
- External or deleted partial launches not represented in any preserved log.

If any residual risk could credibly conceal a Blocker, the terminal outcome is
inconclusive, not supported with an omitted caveat.
