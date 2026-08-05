# Curated branch validation

Date: 2026-08-05

- All retained JSON records parse successfully.
- All retained Python sources compile successfully under the project Python
  3.10 runtime.
- The focused canonical v008 suites for analysis, fit selection, generation,
  seed/power rules, and terminal workflow pass: 153 tests passed.

The complete v008 historical-lineage suite is not expected to pass from this
curated branch alone. It deliberately opens the full v001--v007 attempt trees,
the frozen cohort seed ledger, and other excluded run artifacts by path. Those
artifacts remain in the local external run archive and are hash-bound by the
retained state, ledger, seals, and invalidity records. The branch is therefore
a compact paused-program record, not a self-contained authorization to resume
the experiment.
