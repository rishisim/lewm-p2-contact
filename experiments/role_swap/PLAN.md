# Role-swap probe: implementation plan (v3, final)

Status: final after review rounds 1-3 (Codex gpt-6-astra, medium). Owner: Rishi Simhadri. Date: 2026-10-05.

## Goal and claim boundary

Test whether a compact JEPA world model (LeWM, one 192-d vector per frame) loses
information about an object whose relevance depends on the later task, and
localize failures to representation, cost, or predictor.

Claim boundary (narrowed per review): results speak to *generalization across
matched manipulation objectives in one simulator, under specified interaction
coverage*, for the trained checkpoints. They do not by themselves establish a
general P1 law. Multiple training seeds are run if the pilot budget allows;
otherwise conclusions are per-checkpoint.

Scope: steps 1-4. DINO-WM comparison is later.

## Pinned stack (verified)

- `stable-worldmodel==0.1.1` PyPI wheel (2026-06-06), extras `[train,env,format]`.
  Its `solver.CEMSolver(model=...)`, `policy.WorldModelPolicy`,
  `wm.utils.load_pretrained` match upstream le-wm `8edfeb3`. Do NOT use the Git
  HEAD (planning API moved to `planning.solver.CEMSolver(cost=...)`).
- `lucas-maes/le-wm` @ `8edfeb3` as a git submodule at `third_party/le-wm`
  (pristine; never edited). The old customized `le-wm/` snapshot is removed.
- Python 3.10 via `uv`; `uv.lock` committed. Torch MPS build.
- Pretrained `quentinll/lewm-pusht` (weights.pt + config.json); dataset
  `pusht_expert_train.h5.zst` (13.1 GB). Record sha256 of both.

## Repository layout

```
pyproject.toml, uv.lock
lewm_storage.py, storage.json, storage_manifest.json   # unchanged (tests preserved)
third_party/le-wm/                 # submodule @ 8edfeb3
src/lewm_research/
  paths.py        # work root resolution (LEWM_WORK_ROOT, default ~/lewm-work)
  device.py       # cuda/mps/cpu
  runs.py         # run dir creation: config, git sha, versions, seed, timing, metrics.json
  lewm.py         # load checkpoint + its normalization; encode (cls & proj); rollout; cost
  envs/pusht_peg.py   # PushTPeg + snapshot/restore + goal rendering; registers swm/PushTPeg-v1
  policies/weak.py    # BlockWeak, PegWeak, Mixed
  data/collect.py     # fixed-length collection (no termination) -> HDF5 (swm format)
  data/verify.py      # from old le-wm/verify_dataset.py
  train/finetune.py   # pretrained-init fine-tuning loop (upstream losses)
  probe/conditions.py # base scenes + matched conditions + success/disturbance scoring
  probe/reference.py  # simulator-dynamics reference planner
  probe/rollout_eval.py # controlled planning loop (fixed budget, role-aware scoring)
  probe/readout.py, probe/abc.py, probe/stats.py
  cli.py          # typer; one subcommand per stage
configs/role_swap/*.yaml
experiments/role_swap/{PLAN.md, PROTOCOL.md, results/}
tests/
```
`.gitignore`: anchor `data/` to repo root (`/data/`) so `src/lewm_research/data/` is tracked.

## Storage

- Governed USB unchanged (Cube/Reacher/TwoRoom provenance). It has 34 GB free,
  too little for PushT, so new regenerable artifacts go to `LEWM_WORK_ROOT`
  (internal SSD, default `~/lewm-work`): `stable-worldmodel/` (STABLEWM_HOME for
  this experiment: downloads, datasets, checkpoints) and `runs/`.
- Budget (to verify in pilot): PushT zst 13 GB + decompressed (measure) +
  PushT-Peg datasets (~2 x 15 GB at 224 px, compressed HDF5 if supported) +
  checkpoints (<1 GB). Delete the .zst after verified decompression only if
  space requires. Committed results: compact JSON/CSV/MD/PNG only.

## Action semantics (fixed everywhere)

Model action = block of 5 consecutive 2-D env actions (10-D). Plan horizon 5
blocks = 25 env steps; receding horizon 5 blocks = execute all 25, then replan.
Eval budget 50 env steps = 2 planning calls. Reference planner, LeWM planner,
A/B/C banks and training windows (frameskip 5) all use these semantics.
Normalization: all arms use the pretrained checkpoint's normalization (StandardScaler
fit on `pusht_expert_train` action/proprio), persisted as an artifact next to each
checkpoint. Candidate banks are stored in physical action coordinates and
transformed per model.

## Step 1: setup + smoke test

1. Remove `le-wm/`; add submodule; `pyproject.toml` with pinned deps; `uv sync`; run existing storage tests.
2. Verify downloads (sha256), decompress PushT to `$LEWM_WORK_ROOT/stable-worldmodel/datasets/`.
3. Tiny gates first (minutes each): import/load pretrained, one MPS planning call,
   one forward+backward step, save/reload parity, 2-episode collection.
4. `cli smoke-eval`: upstream eval logic (dataset start/goal sampling, CEM 300x30
   top-30, horizon 5, budget 50, 50 episodes, seed 42) with device resolution in
   our wrapper; no upstream file edits. Report success and timings.
5. Pilot benchmarks (each reported separately): (a) simulator env-steps/s and
   snapshot/restore cost; (b) LeWM planning call latency on MPS; (c) training
   step time (forward+backward, batch sizes 32/64/128, fp32 on MPS) and peak memory;
   (d) data collection frames/s and bytes/frame.

## Step 2: PushT-Peg environment and data

### Environment `swm/PushTPeg-v1` (subclass of 0.1.1 `PushT`)
- Peg: dynamic disc radius 15 px (window 512), mass 1, block-like friction/damping,
  color RGB(65,105,225). Created in `_setup()` override.
- Observation `state` = original 7 dims + peg xy (9 dims); update
  `observation_space`. `proprio` = agent xy + agent velocity computed by index in
  `reset`, `step` and `_get_info` (goal proprio) overrides.
- Override `_get_obs()` (9 values) and replace `_set_state()` with an exact setter
  that assigns every body's position/angle/velocity/angular velocity from a full
  snapshot and does NOT step physics (the 0.1.1 setter drops peg, mis-assigns
  velocity and steps physics). `reset()` override builds consistent 9-value
  defaults and goal fields (`goal_state`, `goal_pose`, `_goal`).
- `get_snapshot()/restore_snapshot()`: positions, angles, velocities, angular
  velocities of agent, block, peg (and goal fields). No physics step on restore.
  Test: identical 50-step futures after restore.
- Goal rendering: `render_state(state)` draws a full frame from a state vector
  without touching the live space (temporary body poses or a separate space).
  Goal observation = rendered goal state.
- `terminate_on_success` option (default False for collection/eval); our probe
  scores success itself.
- Reset option `peg_xy` or `peg_placement` sampler (non-overlapping; walls margin 30 px).

### Policies
`BlockWeakPolicy` (upstream WeakPolicy logic, ±100 px around block), `PegWeakPolicy`,
`MixedPolicy` (per-episode p=0.5). Env-id check accepts the PushT family.

### Training datasets (fixed length, no early termination)
Both arms share the SAME peg placement distribution (uniform non-overlapping)
and episode count/length; only the policy differs:
- `peg_blockpolicy` (primary): BlockWeakPolicy. Peg is incidental.
- `peg_mixedpolicy` (control): MixedPolicy. Peg is frequently manipulated.
Record per dataset: fraction of episodes with peg displacement > 5 px, peg
contact frames. Episode count from pilot (target 1,000 x 100 steps); episode-level
train/val split (90/10) before windowing.

## Step 3: fine-tuning

- Own loop in `train/finetune.py`, using 0.1.1 LeWM classes initialized from
  `quentinll/lewm-pusht` weights; loss = upstream next-embedding MSE + SIGReg
  (weight 0.09, knots 17, num_proj 1024), AdamW lr 5e-5 wd 1e-3, grad clip 1.0,
  history 3, frameskip 5, fp32 on MPS.
- Tests: load parity (fine-tune code with 0 steps reproduces pretrained
  embeddings/predictions exactly); save/reload parity.
- Arms: `pretrained`, `ft_block`, `ft_mixed`; identical step budget set from pilot
  (cap ~6 h per arm); seeds: 1 per arm baseline, a 2nd seed per arm if budget allows.
- Sanity: val loss on held-out peg episodes; 20-episode no-peg PushT eval to
  detect forgetting.

## Step 4: evaluation

### Base scenes and matched conditions (`probe/conditions.py`)
W4c supersedes W4b's far-start construction (historical record:
`calibration/2026-10-05_w4b_nogo.md`). Sample a random non-overlapping clutter
scene with unchanged physics, then run 300-step weak-policy traces:
- D: first 25-step BlockWeakPolicy window starting within 40 px of the T,
  T displacement >=40 px, peg stationary throughout (tolerance 1e-6 px).
  `off_path` retains the clutter peg, >=80 px from the T center corridor;
  `on_path` moves it to the exact midpoint. Reject the scene if the midpoint
  overlaps agent/T at either endpoint. Retain the exact start snapshot,
  including velocities, and the real agent/T future goal.
- G: first peg-centered (30 px box) window starting within 40 px of the peg,
  peg displacement 40-100 px, T stationary. Pair with a separate block-centered
  window from the same scene starting within 40 px of T, T displacement in the
  same 20 px magnitude bin ([40,60), [60,80), [80,100]), wrapped rotation <pi/9,
  peg stationary. Both goals include the real agent future. Each family shares
  a scene ID; D and G need not share start poses or a translation vector.
Store rollout times and action witnesses as construction provenance. Reject
a scene unless all three original task witnesses achieve final joint success
after a fresh snapshot restore (cached contact impulses are not serialized). Original
futures are reachable by construction; inserting the on-path obstacle changes
reachability and requires reference calibration. These witnesses never seed CEM.
Per-arm rendering follows training data: pretrained uses the fixed upstream
T decoration; fine-tuned arms use `with_target=False`. Both current and goal
frames obey this rule, recorded in each evaluation config.
Scoring (computed on every trajectory; agent position excluded):
- T success: final T position error < 20 px and angle error < pi/9.
- Peg preserved: max peg displacement over the trajectory < 10 px.
- Peg target: final peg error < 15 px.
- Also record ever-achieved success.
Contrasts are descriptive task-generalization comparisons, not causal separations
(geometry and manipulation difficulty differ across conditions). Paired by base,
cluster bootstrap by base:
- Disturbance contrast D: P(T success & peg preserved | on_path) vs same for off_path.
- Disturbance-allowed companion: P(T success | on_path) vs P(T success | off_path),
  reported alongside D to show how much of D is preservation vs obstruction.
- Target contrast G: P(success | move_peg) vs P(success | move_T_matched).

### Reference planner (`probe/reference.py`)
Simulator-dynamics CEM with identical action semantics; fixed physical-unit cost:
T pos error (px) + 100 x angle error (rad) + peg term (preserve: 2 x MAX peg
displacement over the rollout, px; target: final peg error px). Trajectory-aware by
construction (simulator rollouts expose the whole trajectory). W4c optionally adds
reference-only endpoint approach shaping: w x agent-to-manipulated-object distance
(px), calibrated over w in {0,0.1,0.3}; scoring and A/B/C physical role cost stay
unchanged. Population/iterations chosen from the pilot so one episode
<= ~20 s (e.g. 100 x 10), parallelized across processes. It is a reference, not an
optimum (no oracle normalization is used). Seeds: feasibility seed F selects the common
feasible set (bases where the reference solves all conditions of a contrast); a separate
evaluation seed E produces the reported reference numbers. Contrasts are reported on
the common set AND unconditionally, with exclusion counts. Elites/populations come from
the wheel CEM's callbacks (`solver/cem.py` candidates/costs), not a custom CEM.

### Controlled planning eval (`probe/rollout_eval.py`)
Own loop (not `World.evaluate`, whose success = env termination): reset to the
base condition via snapshot, goal image via `render_state`, LeWM planning with
0.1.1 `CEMSolver` + `WorldModelPolicy` semantics (300x30 top-30 unless pilot forces
less, identically for all arms), run full 50-step budget, score as above.
N is frozen in PROTOCOL.md from an independent pilot (pilot bases disjoint from
evaluation bases; default 100 evaluation bases = 400 condition episodes per arm).

### Readout (`probe/readout.py`)
One common, episode-disjoint evaluation pool (frames from held-out episodes of both
datasets + condition episodes). For CLS and projected embeddings separately:
ridge with grouped inner CV (groups = episodes), targets peg xy and T xy.
Baselines: mean predictor; ridge on 32x32 downsampled pixels. Report held-out
mean pixel error (not cross-object R^2 comparison).

### A/B/C (`probe/abc.py`)
For 50 feasible bases x {off_path, on_path, move_peg, move_T_matched}: one shared
bank of K=64 physical-coordinate action sequences per condition, used for every arm:
16 deduplicated reference-planner final-population members, 16 Gaussian perturbations
of them (sigma in {0.1, 0.3, 0.6} of the normalized action range, round-robin), 32 uniform random. Keep banks with >= 1 successful and >= 1
failed candidate and C range > 20 px; report exclusions.
- A: model-predicted final embedding vs goal embedding (upstream MSE cost).
- B: real final observation (real history) encoded, same cost.
- C: ground-truth role cost on the real executed trajectory, same definition as the
  reference cost (preserve conditions use max peg displacement, so away-and-back is
  penalized). An endpoint-only variant C_end is reported as a diagnostic.
Metrics: Spearman(A,C), Spearman(B,C), Spearman(A,B); top-1 regret w.r.t. C for A and B.

### Frozen decision bar (PROTOCOL.md, committed before any step-4 run)
For `ft_block`:
1. Gap exists: D or G shows LeWM success lower by >= 15 points in the less-familiar
   condition, paired cluster-bootstrap 95% CI excluding 0, on the common feasible set,
   while the reference planner's difference under evaluation seed E is < 5 points.
2. Localization (labels are descriptive, not proofs of mechanism):
   - linear-readout deficit: peg readout error >= 2x T readout error AND >= 1.5x the
     pixel-ridge peg error (weak linear accessibility, not proof of information loss);
   - observed-state cost mismatch: Spearman(B,C) in the hard condition < its control by
     >= 0.2, with peg readout < 1.25x pixel-ridge error;
   - prediction-associated degradation: Spearman(B,C) >= 0.5 in BOTH conditions (adequate
     absolute agreement), Spearman(A,B) in the hard condition lower than control by >= 0.2,
     and Spearman(A,C) lower by >= 0.2.
3. Coverage: `ft_mixed` counts as a sufficient remedy only if (a) success in the hard
   condition improves by >= 10 points over `ft_block` (paired CI excluding 0), (b) the
   control condition regresses by < 5 points, and (c) the gap shrinks by >= 50%.
   Otherwise report only "reduced disparity" (or none).

## Scope decisions from review

Astra suggested deferring the submodule replacement and package restructure. Kept,
because the project owner explicitly requires an updated LeWM and a clean, modular,
scalable repository; the restructure is limited to what the probe needs.

## Delegation

Codex `gpt-6.1-sol` implements workstreams sequentially with tests; Claude reviews
diffs and runs long jobs in the background.
- W1: restructure (submodule, pyproject/uv, paths/device/runs, .gitignore), smoke-eval CLI, pilot benchmarks (a),(b).
- W2: PushTPeg env (snapshot, render_state, proprio), policies, collection CLI, tests; pilot (d).
- W3: finetune loop + parity tests; pilot (c).
- W4: conditions, reference planner (wheel CEM + callbacks), rollout eval, stats, tests.
- W5: readout, A/B/C, results summarizer.
