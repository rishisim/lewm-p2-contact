# LeWM locked-stack mechanics replay preregistration

Status: sealed before construction of the run-local source clone or Python
environment and before any scientific environment execution. `protocol.json`
is the machine-controlling specification. Its SHA-256 sidecar is the seal.

This bounded diagnostic replays only the already-consumed Cube generator probes
for discovery episodes 3316, 5355, 6405, 2420, 6763, 8240, 9767, and 8510.
It uses exactly transition rows 0, 1, 2, 5, 10, 25, 50, and 100 and frame rows
0, 50, and 200: 64 transitions and 24 frames. V3 test targets and the combined
V3 cache remain unopened. No episode, row, or version expansion is permitted.

The sole candidate is stable-worldmodel commit
`a265229cb29688715651cbd831a3b4c10b8f98b4` with immutable `uv.lock` SHA-256
`0d1659e36b8cb4b054b95ae69cfabb174cc15193dbba126e0c8cb5da5751724d`.
The required interpreter is CPython 3.10.20. Installation must use
`/opt/homebrew/bin/uv sync --frozen`, a run-local clone, and a run-local
`UV_PROJECT_ENVIRONMENT`, with only locked environment/format extras and no
development group. Any platform incompatibility ends the experiment as
`locked_stack_unavailable`; no substitution or unlocked resolution is allowed.

Reset is descriptive only: released data omit the reset seed and end-effector
start variation, so only reconstructible cube/goal fields are reported. Pixel
comparison is causal only if the collector's EGL backend is genuinely available;
otherwise macOS/GLFW rendering is descriptive and excluded from the generator
decision. Observation evidence is split into stored/raw, derived kinematic, and
contact fields.

Every frozen transition is replayed with three deterministic restore modes:
fresh/cold data reset followed by exact time, qpos, qvel, target mocap and control
injection; the same with `qacc_warmstart` explicitly zeroed; and prior reused-data
behavior. No hidden simulator state is optimized. Model dimensions and HDF5
coverage for act, userdata, plugin state, applied forces, mocap, ctrl, warmstart,
and other MuJoCo state are inventoried. Material restore-mode variation is tested
against the fixed transition tolerance and supports a separate
`transition_nonidentifiable_from_partial_state` finding only when legitimate
modes differ materially and required state is absent.

Primary thresholds, material-improvement rules, comparisons, and the decision
tree are frozen in `protocol.json`. A mechanics match never proves the exact
released generator. This task creates zero policy trajectories, zero model/gate
rows, and zero V5 episodes; it performs no tuning or performance claim.
