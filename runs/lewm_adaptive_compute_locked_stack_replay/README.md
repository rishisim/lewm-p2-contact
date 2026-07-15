# LeWM locked-stack replay

This run attempted the single preregistered locked dependency candidate and
stopped before scientific execution because the immutable lock and immutable
source disagree on the root-project version. The controlling outcome is
`locked_stack_unavailable`.

The sealed scope remains eight discovery episodes, 64 transitions, and 24
frames, but zero HDF5 rows were opened and zero probes were executed. See
`REPORT.md`, `environment_provenance.json`, `decision.json`, and
`independent_verification.json`.

The prior generator-forensics run is unchanged, including its known stale
manifest entry for the verifier log. This directory contains the new detached
clone and run-local virtual environment; the final manifest represents those
large trees by immutable source commit/lock plus an environment binding and
requirements manifest, with explicit exclusions documented in the report.
