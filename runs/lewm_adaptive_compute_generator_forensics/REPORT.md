# LeWM Cube generator-mechanics reconstruction audit

## Outcome

The layer-localized decision is **`reset_only_mismatch`** and the provenance
decision is **`generator_provenance_unresolved`**. The installed
`swm/OGBCube-v0` path is not the exact generator of the sampled released HDF5
rows. The one directly evidenced historical source candidate also failed the
same frozen checks, so no exact historical generator was recovered.

This result does not revise the prior distribution-contract verdict and is not a
mechanics match, a distribution-contract pass, or authorization for V5.

## Frozen scope and alignment

The protocol was sealed before selected transition outcomes were read or the
environment was executed. A salted SHA-256 rule selected exactly 8 episodes from
the isolated 420-episode V3 discovery allowlist: 3316, 5355, 6405, 2420, 6763,
8240, 9767, and 8510 (hash-score order). Their intersection with V3 test is
empty. Calibration and the combined V3 cache were not loaded.

Exactly 8 transition indices per episode (64 total) and 3 frame indices per
episode (24 total) were used, with no expansion. Internal HDF5 alignment was
sealed first: for every selected episode and every row 1–200, `prev_qpos` and
`prev_qvel` exactly equal the prior row's `qpos` and `qvel`. Thus action `t`
maps state row `t` to state row `t+1`, and its post-step control is stored on
row `t+1`. Row 200 is the truncation sentinel.

## Layer results

| Layer | Frozen result | Max abs error | Mean abs error |
|---|---:|---:|---:|
| Injected qpos | pass, bit-exact | 0 | 0 |
| Injected qvel | pass, bit-exact | 0 | 0 |
| Reset/IK qpos | fail | 3.59151 | 0.0769616 |
| 28-D observation | fail | 1.0 | 0.00794454 |
| RGB pixels, identity best | fail | 122 | 0.284632 |
| 5-D action → 7-D control | fail | 0.00277103 | 0.000388066 |
| Action-path qpos transition | fail | 0.00145644 | 0.0000647881 |
| Action-path qvel transition | fail | 0.283596 | 0.00195933 |
| Stored-control qpos transition | fail | 0.0000341251 | 0.000000200272 |
| Stored-control qvel transition | fail | 0.00251568 | 0.0000245643 |

The stored start and goal cube positions reset exactly, but the robot IK state
does not because the released reset seed/semantics are absent. State injection
itself is sound: raw qpos/qvel, arm joint fields, block position/quaternion,
gripper opening, and gripper velocity reconstruct exactly. Derived end-effector
pose differs (position max 0.00070342; yaw max 0.00185591), and recomputed
contact differs by as much as 1.0. These are evidence of generator/model-stack
incompatibility, not permission to replace stored mechanics fields or use contact
as a gate input.

Every best pixel convention was identity; no flip or BGR transform helped. Mean
pixel mismatch rate was 12.5318%, mean PSNR 43.5862 dB (minimum 37.4053 dB), and
mean absolute channel errors were R 0.232718, G 0.353869, B 0.267310. Bit equality
was the frozen pass rule. SSIM is explicitly unavailable because `skimage` is not
installed in the sealed runtime; no package was added after results.

Direct stored-control replay is roughly two orders of magnitude closer in qpos
mean error than the action path, which localizes an additional action/IK-control
mapping mismatch. It still fails the predeclared 1e-9 transition tolerance, so
the underlying physics/render/model stack is not exact either.

## Provenance and historical candidate

The [Hugging Face dataset repository](https://huggingface.co/datasets/quentinll/lewm-cube)
head is `02a19a67...`; its tiny card and file metadata contain no generator
command, package lock, seed, or source commit. The public
[stable-worldmodel collector at `a265229`](https://github.com/galilai-group/stable-worldmodel/blob/a265229cb29688715651cbd831a3b4c10b8f98b4/scripts/data/collect_cube.py)
and [Hydra config](https://github.com/galilai-group/stable-worldmodel/blob/a265229cb29688715651cbd831a3b4c10b8f98b4/scripts/data/config/ogb.yaml)
are the strongest direct recipe evidence found: single Cube, ExpertPolicy,
10,000 trajectories, 200 steps, 224×224 rendering, seed 3072, and no termination
at goal. Attribution of the released archive to this recipe remains inference.

That single candidate was frozen before execution, checked out under the run
directory, and tested on the same 64 transitions and 24 frames with installed
OGBench v1.2.1. It produced the same summaries and failures as the installed
path. The weaker stable-worldmodel 0.0.4 timestamp-only candidate was not run
because archive mtime is not direct generator evidence. No broad version sweep
occurred. The official [OGBench v1.2.1 recipe](https://github.com/seohongpark/ogbench/blob/1d4140997f60c52c6fb0702ec100dc988b18c548/data_gen_scripts/generate_manipspace.py)
is not the released recipe: its public Cube command uses 1,000 episodes and a
1,001-step horizon.

## Verification and claim boundary

Focused tests passed. Independent recomputation imported none of the analysis
implementations, recomputed the salted sample, empty test intersection, HDF5
alignment, counts, every stored numeric aggregation, pass flags, and decision
mapping. It read only the eight frozen HDF5 episode slices and did not execute
the environment.

No V5 confirmation was launched or authorized. No solver, gate, or model tuning
occurred. Zero policy trajectories and zero model/gate evaluation rows were
recorded. V3 test targets remained unopened, and the combined V3 cache was never
loaded with NumPy. FLOPs, latency, and contact-aware performance claims are
outside this mechanics task. This work makes no “first adaptive world model”
claim; LoopWM remains relevant related work.

## Smallest next experiment

First seek immutable upload-side generation metadata, a package lock/container,
or an exact MuJoCo/dm-control/render stack tied to the archive. Only if that
evidence identifies a specific dependency stack should one additional frozen
candidate replay these same transitions. Do not tune the gate or launch V5.
