# LeWM Adaptive Compute V1 Preregistered Plan

Status: frozen before encoding or examining any newly selected episode.

Pre-execution amendment, 2026-07-13 America/Phoenix: the initial draft used
60 contiguous episodes (40/10/10). A subsequently completed read-only resource
audit verified strict Cube checkpoint loading and finite encode/predict calls on
the local Apple M5, and measured a comparable 200-episode extraction at about
20 seconds. Before any new Cube episode was encoded or any outcome examined,
the sample was increased to 600 episodes (420/90/90) to provide substantially
stronger episode-clustered uncertainty. Architecture, hypotheses, metrics,
seeds, gates, and decision thresholds are unchanged. Selection is now the
deterministic procedure below rather than a contiguous range.

Pre-execution protocol clarification, 2026-07-13 America/Phoenix: Stage B
cannot be fit after using the final test split for the Stage A go/no-go without
violating the untouched-test requirement. Therefore the same Stage A response
and exact-budget oracle-headroom rules are first applied on calibration. The
fixed ridge gate is fit and its budget frozen only when that calibration screen
passes. The final Stage A decision and, when screened in, Stage B evaluation
are then computed together in one test pass. If calibration screens the gate
out but test later shows oracle headroom, the conservative final category is
`selective_headroom_gate_failed`, with the reason explicitly recorded as
`gate_not_fit_calibration_screen`. No test outcome changes gate fitting or
budget selection.

Pre-full-run implementation clarification, 2026-07-13 America/Phoenix: the
disjoint six-episode engineering smoke run showed that PyTorch's default random
initialization makes an untrained residual projection perturb the already-good
frozen LeWM output. Before extracting any full-run episode, the final linear
layer of `R_phi` is therefore fixed to zero weight and zero bias at
initialization, matching residual-safe/AdaLN-zero practice. All depths start as
the depth-0 prediction, gradients still train the shared block, and no full-run
calibration or test outcome informed this choice.

## Question and scope

This experiment is a direct frozen-LeWM latent-refinement pilot on Cube. It
tests whether repeated applications of one shared, action-conditioned residual
block improve the released LeWM prediction, whether same-transition gains are
heterogeneous enough for selective allocation, and whether a causal gate can
exploit that heterogeneity. The Fetch state-space refiner and prior residual,
kNN, branching, and contact-error analyses are context only and are not inputs
to the decision.

The released encoder, action encoder, predictor, projector, and prediction
projector remain frozen. Depth 0 is the released prediction after `pred_proj`:

```text
z0 = LeWM.predict(z_history, action_encoder(action_history))[:, -1]
z(k+1) = z(k) + R_phi(z_history, action_history, z(k), iteration=k)
```

`R_phi` is one shared MLP at every iteration. Its inputs are the three encoded
history latents, the three raw 25-dimensional blocked actions, the current
prediction, and a learned iteration embedding. It has two 256-unit GELU hidden
layers and a 192-dimensional residual output. Only `R_phi` is trained.

## Frozen inputs and provenance

- Cube source:
  `/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5`.
  Verified layout: 10,000 episodes, 201 rows per episode, 2,010,000 rows.
- Released Cube checkpoint config and weights:
  `runs/lewm_transfer/cube/cache/model/config.json` and `weights.pt`.
- Checkpoint SHA-256:
  `2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89`.
- Expected model contract: history 3, frameskip 5, raw blocked-action dimension
  25, latent dimension 192, image size 224.
- Prior diagnostic episodes 0--29 are excluded. Prior arrays contain no full
  history/action inputs and will not be used for training or evaluation.

The checkpoint's original episode-level training manifest is unavailable.
Consequently, episodes 30--89 are fresh to this diagnostic/refiner experiment,
but are not asserted to be unseen during original LeWM pretraining. The result
is a confirmatory test of the new refiner conditional on a released checkpoint,
not a clean test of base-model dataset generalization.

## Frozen split and indexing

- Build the eligible set of episode ordinals `30..9999`, excluding all prior
  Cube diagnostic episodes `0..29` by construction.
- With NumPy `default_rng(260713)`, draw 600 ordinals without replacement and
  preserve the RNG's returned order.
- Refiner and gate training: first 420 selected episodes.
- Calibration, early stopping, whitening, primary-seed selection, and gate
  compute-budget selection: next 90 selected episodes.
- Test: final 90 selected episodes, evaluated only after all choices above are
  frozen.

For target strided model step `t`, each example uses `z[t-3:t]`, action blocks
`a[t-3:t]`, and target `z[t]`. With 41 strided embeddings and history 3, the
expected counts are 38 transitions per episode: 15,960 train, 3,420 calibration,
and 3,420 test. The implementation must assert pairwise-disjoint episode sets,
unique row keys, expected temporal ordering, and no test-derived fitting.

## Hypotheses and exits

Exits are depths `{0, 1, 2, 4}`. Depth 8 is excluded from V1 because the pilot
can answer its preregistered question with four calls and because stability at
larger recurrent depth is unknown.

- H1 compute response: at least one fixed depth in `{1,2,4}` has positive
  same-sample gain over depth 0 on test.
- H2 selective opportunity: at a budget of exactly one or two block calls per
  test transition, a target-informed, exact-budget oracle improves on the
  corresponding uniform fixed depth.
- H3 gateability: a gate fit on train-only gain labels and prediction-time
  features can allocate the calibration-selected exact budget on test better
  than uniform and can also beat depth 0.

For sample `i` and depth `k`, the primary benefit is:

```text
G_i(k) = MSE(target_i, z_i(0)) - MSE(target_i, z_i(k))
```

The primary loss is raw per-dimension latent MSE. Calibration-fit whitened MSE
is a required sensitivity analysis. Whitening uses only calibration target
latents, an eigendecomposition of their covariance, and an eigenvalue floor of
`1e-6 * largest_eigenvalue`; the fitted transform is then frozen. Robust
relative gain is `G_i(k) / max(loss_i(0), q10_calibration_loss0, 1e-8)`.

## Training and model selection

- Seeds: `260713`, `260714`, `260715`.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`.
- Batch size: 256; maximum 250 epochs; gradient norm clip 1.0.
- Objective: equal-weight mean raw latent MSE at depths 1, 2, and 4, plus
  `1e-4` times the mean squared residual update across the four calls.
- Early stopping: calibration equal-weight depth-1/2/4 objective, patience 30,
  minimum improvement `1e-7`; restore the best refiner state.
- Primary seed: the seed with the smallest frozen calibration objective; ties
  resolve in the listed seed order. Test results do not select a seed.

Depth 0 must be bitwise equal to the stored direct released-LeWM prediction and
must make zero refiner calls. The frozen base remains in evaluation mode and no
base parameter may require or receive a gradient.

## Stage A comparisons and go/no-go rules

For the primary seed, report depth 0 and uniform fixed depths 1, 2, and 4. At
budgets `N` and `2N` calls, an exact multiple-choice dynamic program selects an
oracle depth from `{0,1,2,4}` per sample using target loss, subject to total
calls equaling the budget. The oracle is explicitly diagnostic and cannot be a
gate input. Also report:

- a random exact-budget allocation generated without target or prediction
  features;
- the oracle depth multiset permuted across samples, preserving its exact depth
  histogram and calls.

All allocations execute/gather permitted prefix depths, and depth `k` costs
exactly `k` shared-block calls. Oracle versus uniform is a paired loss
comparison at identical total calls.

Go/no-go rules use episode-clustered 95% percentile intervals with 1,000
bootstrap replicates and seed `260713`:

1. `no_compute_response` if no fixed depth has raw mean gain whose CI lower
   bound exceeds zero.
2. Otherwise, `compute_helpful_but_not_heterogeneous` if neither the one-call
   nor two-call oracle advantage over uniform has CI lower bound above zero and
   mean advantage at least 0.5% of depth-0 mean MSE.
3. Otherwise continue to Stage B with `selective_compute_opportunity`.

Whitened and robust-relative results can qualify or weaken interpretation but
cannot overturn a failed primary raw-MSE gate. Any nonfinite output, split
violation, depth-0 mismatch, base mutation, or budget mismatch voids the
affected claim.

## Stage B causal gate

Stage B runs only after Stage A passes. A fixed ridge regressor (`alpha=10`) is
fit on training examples to predict total gain at depths 1, 2, and 4. Inputs
are compact summaries constructed solely from latent history, raw action
history, and depth-0 prediction: per-token norms, history changes, action
norms/changes, prediction norm, and predictor-to-last-history difference.
Targets may be used to form train-only gain labels, but target, future state,
contact, kinematics, phase, regime, and any target-derived field are rejected
as gate features.

For each candidate mean budget in `{1,2}` calls, predicted gains drive the same
exact-budget optimizer on calibration. Calibration selects the budget with the
largest adaptive-minus-uniform raw-MSE gain while requiring adaptive to improve
on depth 0; if neither qualifies, it selects one call and records calibration
failure. Ties select fewer calls. The ridge weights, feature normalization, and
budget are then frozen and evaluated once on test.

Stage B reports adaptive versus depth 0, adaptive versus uniform at exactly the
same calls, adaptive versus a random exact-budget allocation, and adaptive
versus a permutation of its own depth multiset. It also reports depth
distribution, gate regret to the exact-budget oracle, calibration, and results
by predicted-benefit quantile.

The learned gate passes only if both paired test comparisons have raw mean
benefit with episode-clustered CI lower bound above zero:

- depth-0 loss minus adaptive loss; and
- matched-budget uniform loss minus adaptive loss.

If Stage A has oracle headroom but either comparison fails, the final decision
is `selective_headroom_gate_failed`. If both pass, the decision is
`adaptive_beats_original_and_uniform`.

## Compute, timing, interpretation, and artifacts

Report actual per-sample depths, total block calls, an analytic dense-linear
FLOP estimate, trainable parameter count, and measured batched refiner latency
on the available Apple M5 device when feasible. Latency is secondary because
exact block calls are the controlled compute quantity.

Contact/interaction, transport-free/static motion, normalized phase, and event
labels are attached only after predictions for post-hoc interpretation and
figures. Their thresholds are fit on calibration where needed. They never enter
the refiner or gate and do not define the primary struggle metric.

Required artifacts under this new directory are cached latent inputs with
provenance, split manifest, refiner checkpoints, machine-readable metrics,
`decision.json`, figures for depth curves, gain distribution, budget curves,
and post-hoc allocation by regime, plus `REPORT.md`. No prior result directory
will be modified. No commit, push, download beyond missing Python runtime
packages, or external posting is authorized.

## Stopping and validity rules

- If strict checkpoint loading or fresh episode extraction fails, complete unit
  and synthetic smoke tests, record `blocked_runtime`, and give the exact
  missing resource/command.
- If the full 600-episode run exceeds practical local time, fall back without
  examining full-run test outputs to a preregistered smoke subset of episodes
  30--35 (train 30--33, calibration 34, test 35), mark it engineering-only,
  and do not make confirmatory statistical claims.
- Do not tune architecture, seeds, loss weights, gate alpha, splits, or decision
  thresholds after viewing test outcomes. Any necessary deviation must be
  timestamped and labeled exploratory in `REPORT.md` and `decision.json`.
