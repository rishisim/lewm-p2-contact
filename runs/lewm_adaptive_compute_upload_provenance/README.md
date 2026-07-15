# LeWM upload-adjacent provenance diagnostic

This sealed, read-only run reconstructs the complete immutable history of the
public `quentinll/lewm-cube` dataset repository, distinguishes Git/LFS/Xet/HTTP
hash namespaces, inspects one bounded archive prefix, and adjudicates whether
upload metadata binds the released Cube archive to a coherent generator.

The outcome is `upload_metadata_contains_no_generator_binding`. See
`REPORT.md`, `decision.json`, `evidence_ledger.json`, and
`independent_verification.json`. No HDF5 content, Cube mechanics, V3 target,
model, gate, trajectory, or V5 work occurred.
