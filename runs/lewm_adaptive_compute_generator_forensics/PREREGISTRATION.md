# Generator-mechanics reconstruction preregistration

Status: frozen before any selected raw transition outcome was loaded and before
the installed environment was constructed or executed. `protocol.json` is the
machine-controlling specification.

Exactly eight episodes are selected from the mechanically isolated V3 discovery
allowlist by the salted SHA-256 rule in the protocol. The main audit uses exactly
eight fixed one-step indices and three fixed frame indices per episode, with no
sequential expansion. Calibration is not used. V3 test targets and the combined
V3 cache remain unopened.

The audit first resolves HDF5 alignment solely internally and seals it before any
environment comparison. It then tests, in order: initial reset/IK and task state;
exact qpos/qvel restoration; privileged fields and the 28-D observation; RGB
rendering under only the frozen orientation/channel conventions; normalized 5-D
action to 7-D control; action-path one-step dynamics; and, where technically
possible, direct-control dynamics. Tolerances and the earliest-failure decision
tree are fixed in `protocol.json` and may not be relaxed after results are seen.

This is mechanics-only diagnostics. It records no policy trajectory or model/gate
evaluation, performs no solver or gate tuning, and cannot authorize V5 or establish
a distribution-contract pass. Contact/state labels remain post-hoc mechanics
evidence only. FLOPs and latency are outside scope.
