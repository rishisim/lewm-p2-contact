# Validation report

## Overall assessment: Ready to share with caveats

The literature boundary, formal result, artifact recomputation, and exploratory
frontiers are adequate for the paper-value decision. The numerical artifacts
are ready to share as exploratory supporting material. They are not suitable
for upgrading any confirmation claim.

## Methodology review

- The analysis answers the stated question: whether the formulation changes the
  paper’s contribution, not whether additional settings can be accumulated.
- Fit, selection, pilot, and confirmation populations remain distinct in the
  narrative. Every new outcome analysis is labeled consumed-cohort exploratory.
- The strongest transition-independent comparator is recomputed separately for
  raw and whitened loss at each counted total.
- Gate feature/score FLOPs are included at every reached decision. The formal
  treatment separately explains when gate cost is sunk locally.
- The Cube frontier is restricted to policies supported by stored sparse scores;
  no unstored counterfactual score is imputed.
- Oracle headroom is endpoint-specific and explicitly noncausal. Multistage
  headroom is stage-local, not mislabeled as a global prefix optimum.

## Calculation spot checks

- Cube accepted losses, call histogram, and `4,332,936,120,435` FLOPs: **verified
  exactly from accepted arrays**.
- PushT four-depth pilot losses, histogram, and `102,905,055,196` FLOPs:
  **verified**; the negative whitened result is preserved.
- PushT binary losses, histogram, and `307,585,049,440` FLOPs: **verified**.
- Frozen score/depth choices: **independently reconstructed exactly** for all
  three studies. Backend affine-score differences remain below `1.21e-5`.
- Fixed grids and frozen operating points: **verified**.
- Episode-array shapes: **verified** (`9 x 1,600` Cube and `12 x 240` PushT
  binary frontier points by episodes).
- Source hashes: **verified against accepted manifests**.
- Machine verifier: `verify_outputs.py` returns `all_checks_passed` for 23 checks.

## Visualization review

The three PNG/PDF figure pairs were rendered and visually inspected. Axes,
units, endpoint labels, accepted operating-point stars, zero references, and
exploratory qualifiers are present. Bar charts start at zero; line charts use
focused loss scales appropriate for small quality differences. No dual axes or
3D encodings are used.

## Remaining caveats

1. The frontier points share consumed cohorts and are statistically dependent;
   no simultaneous interval family was predeclared for them.
2. Cube dense-selected curves differ from the accepted sparse execution only by
   the pre-existing numerical tolerance; the confirmation number must continue
   to use the sparse path.
3. Calibration and oracle summaries use realized outcomes and cannot define or
   tune a deployable gate on these cohorts.
4. One seeded integer schedule is not the transition-independent expectation
   envelope; it is a finite-scheduling sensitivity only.
5. The focused literature review is sufficient to reject broad novelty but is
   not an exhaustive systematic search capable of proving absence of every
   narrower precedent.

## Reproducible paths

- analysis: `analyze_existing.py`
- fail-closed verifier: `verify_outputs.py`
- independent audit: `numeric/independent_audit.json`
- validation results: `numeric/validation_report.json`
- lossless episode frontiers: `numeric/cube_frontier_episode_arrays.npz` and
  `numeric/pusht_binary_frontier_episode_arrays.npz`
- finite schedules and losses: `numeric/finite_schedule_arrays.npz`
