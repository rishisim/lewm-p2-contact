# LeWM source/lock adjudication preregistration

Status: sealed before any missing-history fetch, environment construction, or
scientific HDF5-row access. `protocol.json` is the machine-controlling protocol;
the SHA-256 sidecars bind both files.

## Scope and candidate discovery

This diagnostic asks whether exactly one immutable public stable-worldmodel
source commit has a coherent same-commit lock and a materially relevant Cube
collector. Candidate discovery is deliberately narrow and uses only existing
provenance. Candidate 1 is the immutable `0.0.4` tag target
`089efbb87746e153e9ee051bf228b9c12aa3889a`. Candidate 2 is the already-evidenced
history-boundary commit `a265229cb29688715651cbd831a3b4c10b8f98b4`, whose
message is `fix cube script` and whose collector/config blobs are the strongest
known direct match. No branch head, arbitrary version sweep, or third candidate
is permitted. Missing objects may be fetched by exact object/tag reference from
the public repository after sealing.

For each candidate, the audit reads only immutable git objects. It records tag
and commit identity/date, project version, normalized base and `env` extra
requirements, editable-root version and normalized root metadata in that same
commit's `uv.lock`, lock hash and last-change commit, required collector/wrapper
file existence and hashes, collector facts, and temporal plausibility. Source
and lock are never edited, regenerated, combined across commits, or repaired.

## Eligibility and binding

A candidate passes only if all of these are true: (1) source project version
equals the editable-root lock version; (2) normalized source base requirements
equal normalized locked-root base requirements, and normalized source `env`
requirements equal the root metadata used for the `env` extra; (3) the Cube
collector, controlling config, Cube wrapper, expert policy, and directly used
wrapper files exist; (4) collector facts materially match single Cube,
ExpertPolicy, 10,000 trajectories, 200 actions, seed 3072,
`terminate_at_goal=false`, and 224x224 multiview capture; (5) the commit predates
a plausible generation/upload bound (the preserved HDF5 mtime is supportive,
not conclusive, while the immutable March upload is an outer bound); and (6)
CPython 3.10.20 can install the candidate's own unmodified lock frozen on this
arm64 macOS host without substitution. Version or timestamp agreement alone is
insufficient.

Frozen install is attempted only after static rules (1)-(5) leave exactly one
candidate. Zero or multiple static passes maps to
`coherent_generator_candidate_not_identified` and forbids HDF5/environment
execution. One static pass whose frozen environment cannot be bound maps to
`coherent_candidate_environment_unavailable`. A candidate environment must be a
new run-local detached clone and run-local virtual environment, and imports must
prove no resolution from the current project environment.

## Conditional mechanics replay

Scientific replay is allowed only after exactly one candidate passes all
eligibility and binding checks. It reuses only discovery episodes 3316, 5355,
6405, 2420, 6763, 8240, 9767, and 8510; transition rows 0, 1, 2, 5, 10, 25, 50,
100; and frame rows 0, 50, 200 (64 transitions, 24 frames). V3 test targets and
the combined V3 cache remain unopened. No new policy trajectory is generated.

Reset reports reconstructible cube/goal fields only. Observations are split
into stored/raw, kinematic-derived, contact, full, and contact-excluded groups.
The replay compares 5-D action to 7-D control, action-path one-step dynamics,
and stored-control one-step dynamics. EGL is tested first; GLFW/macOS pixels are
descriptive only and excluded from mechanics and provenance decisions.

Every transition uses three deterministic restore modes: fresh/cold `MjData`,
cold with `qacc_warmstart` zeroed, and reused data. Exact time, qpos, qvel,
target mocap, and appropriate action/control are injected. The audit inventories
act, userdata, plugin state, applied forces, mocap, ctrl, warmstart, and other
`mjSTATE`-relevant fields against HDF5 coverage. Hidden state is never optimized.
Material differences among legitimate restore modes with omitted required state
support the separate `transition_nonidentifiable_from_partial_state` finding.

Metrics, tolerances, current-runtime reference errors, ratio definitions, and
decision mapping are frozen in `protocol.json`. A mechanics match never proves
the exact released generator. No rollout, solver/gate inference or tuning,
FLOP/latency claim, contact-aware performance claim, V5, or “first adaptive
world model” claim is permitted; LoopWM remains related work.

