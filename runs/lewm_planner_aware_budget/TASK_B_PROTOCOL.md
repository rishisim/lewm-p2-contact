# PushT planner-independent refiner protocol

Status: B1 frozen; B2 readiness gate failed before training.

## B1 contract

The stable interface is `MaskedStagewiseRefiner.forward(history, actions, mask,
base, depth)`. Shapes are `[N,3,192]`, `[N,3,10]`, `[N,3]`, and `[N,192]`.
History is chronological and left padded. The only legal masks are `[0,0,1]`,
`[0,1,1]`, and `[1,1,1]`; padded values are multiplied by the mask inside the
module. Repetition is never padding. Depths are `0,1,2,4`; depth zero returns
the input base tensor itself and executes no residual stage.

Each 10-vector is five chronological physical PushT actions. PushT declares
each raw action in `[-1,1]^2`; the canonical transform and inverse are identity.
There are no source-fit statistics. Range and finite checks fail closed.

Every positive stage is the same residual MLP conditioned on masked latent
history, masked canonical actions, the explicit mask, current prediction, and a
stage embedding. Its final layer starts at zero. The output is a next-latent
prediction in the frozen 192-dimensional Task A space. Inputs contain no target,
future observation, reward, simulator geometry/state, contact label, goal, CEM
cost/rank, or selected-action outcome. Inference requires a hash-checked
checkpoint, `eval()`, frozen parameters, and inference mode.

The ledger counts base calls/rows and each residual stage's calls/rows. Dense
linear FLOPs may be added after an export is selected, but are not a complete
planner FLOP ledger. Candidate sampling, top-k, goal/image encoding, tensor
movement, and policy overhead remain separately identified.

## Frozen data and decision protocol

- Seed: model initialization/training `26072621`; deterministic pilot sampling
  `26072622`; loader ordering `26072623`; latency input ordering `26072624`.
- Split unit: complete episode and source. No episode may cross fit, selection,
  or evaluation. Task A smoke/tuning/held-out start rows and their source
  episodes are excluded from fit and selection.
- Roles: WeakPolicy-like; expert/competent-policy; mixed/off-policy. Each role
  must appear in held-out evaluation and each must contain lengths 1, 2, and 3.
- Readiness pilot: at least 20 complete episodes per fit source and 10 per
  selection/evaluation source, at least 100 transitions per role/length cell,
  nonzero contact/recovery examples where the source exposes them, exact target
  alignment, finite values, provenance hashes, and canonical action range.
- Model selection: lowest selection frozen-whitened MSE at depth 4 among the
  single preregistered architecture's training epochs, subject to no raw-MSE
  regression above 1% at depths 1 or 2 in any role/length cell. No architecture
  search is permitted.
- Primary offline pass: in every planner-relevant role/length cell, each
  supported positive depth is finite and depth 4 is noninferior to base in both
  raw and fit-frozen-whitened latent MSE (upper paired episode-bootstrap 95%
  bound on relative regression at most 1%). In addition, depth 4 must improve
  both metrics by at least 1% in at least half of the nine cells. Any deeper
  exit may regress by at most 1% relative to the preceding supported exit.
- Diagnostics: raw and frozen-whitened MSE, per-episode paired differences,
  update norms, improvement/headroom calibration, old refiner only on valid
  length-3 WeakPolicy-like rows, actual median/p95 per-depth latency, and exact
  call/row ledgers.
- Export: compact state dict plus architecture/constants/manifest, SHA-256,
  dense reference inputs/outputs covering every supported length/depth, and
  ledger references. No positive depth is available before this export passes.

Only after the offline gate passes may a project-owned CEM adapter be added.
It must apply one fixed depth to all candidates, iterations, rollout steps, and
replans, and must preserve the qualified Task A depth-zero path.

## B2 readiness decision

The local audit found:

1. `pusht_expert_train.lance` (2,336,736 rows, Lance version 2191), which can
   provide competent real transitions and prefix lengths after exclusion of
   Task A cohorts.
2. The prior WeakPolicy pilot: 120 fit, 40 selection, and 80 evaluation
   episodes, with 2,160/720/1,440 real-transition rows. Every row has exactly
   three latent/action histories; there is no mask or prefix-length field.
3. No existing executed competent-CEM or mixed/off-policy trajectory corpus
   with real next observations/latents. Task A retains outcome summaries and
   start ledgers, not candidate-execution transition targets.

Thus existing records cannot satisfy the frozen roles and length cells.
Unexecuted CEM candidates are not targets, and truncating a length-3 record can
change the frozen base prediction and is not a valid aligned length-1/2 example.
Training is blocked before B3.

The smallest expansion choice is a bounded, ignored collection of real PushT
transitions: execute sampled canonical action blocks from frozen mixtures of
expert-policy, Task A CEM-selected, WeakPolicy, and perturbed/off-policy actions;
retain full observations/actions/source/episode/contact provenance; derive
masked lengths 1–3 with the frozen base checkpoint; and first collect only the
readiness minimum above. This is new environment collection and requires an
explicit go-ahead under the task's expansion gate.

## Authorized continuation outcome

Authorization was received after the readiness decision. The bounded
collection, single frozen architecture, offline evaluation, export, and
fixed-depth CEM integration were then completed without changing this contract.
See `TASK_B_REPORT.md`, `task_b_results.json`,
`pusht_refiner_manifest.json`, and `task_b_cem_smoke.json`.
