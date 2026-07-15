# LeWM source/lock adjudication

This sealed diagnostic tests whether one immutable public stable-worldmodel
source/lock pair is coherent and relevant enough to justify another Cube
mechanics replay. The outcome is
`coherent_generator_candidate_not_identified`: neither of the two
preregistered candidates passes every static eligibility rule.

Start with `REPORT.md`. `candidate_inventory.json` contains immutable identity,
hash, lock-history, file, collector, and normalized dependency evidence;
`source_lock_contract_diffs.json` isolates contract drift. `decision.json` and
`metrics.json` record the fail-closed stop before environment or HDF5 execution.
`independent_verify.py` recomputes the result directly from git objects without
importing `audit_candidates.py`. `verify_manifest_readonly.py` verifies the
final artifact set and reports unrecorded eligible files without writing.
