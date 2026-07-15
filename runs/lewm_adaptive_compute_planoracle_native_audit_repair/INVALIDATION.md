# Audit-repair invalidation

Terminal outcome: `planoracle_audit_repair_invalid`

The pre-data seal verified, and all 12 excluded smoke rollouts were generated from the sealed seed ledger with the deterministic reset amendment. During the first sealed smoke execution, the mandatory bitwise comparison between the actual sparse selected output and the dense-shadow output selected by the same calls failed. The runner raised `RuntimeError: sparse/dense selected output mismatch` before writing a smoke execution artifact.

Under the sealed outcome mapping, any sparse/dense failure is an invalid run. Because correcting the sparse implementation would require changing normative runner code after the pre-data seal, this run was not patched or continued. Zero prospective audit-repair episodes were generated, no prospective loss or gate outcome was inspected, no bootstrap or latency analysis was performed, and V5 was not launched.

The numerical evidence from the prior discovery run is unaffected, but this attempted audit repair does not provide a procedurally valid prospective verdict.
