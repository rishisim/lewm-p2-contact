# LeWM Adaptive Compute Discovery: First-Proof Search

Status: discovery only. This plan is written before reading any V3 calibration
target. V3 test targets are forbidden in every discovery phase. The sole final
judge is the untouched V3 calibration episode set, consumed once only after the
internal tournament and its settings are frozen.

## Scientific question

Can a causal, transition-local policy allocate extra latent refinement calls so
that it beats the strongest transition-independent randomized depth mixture at
matched expected calls and contributes a nondominated error/compute point,
without changing the known-good V1 first exit?

Primary error is raw latent MSE. Whitening fitted on discovery-fit targets is a
robustness metric. Contact and other physical labels are forbidden as model or
critic inputs and may be attached only after the final policy is frozen.

## Five distinct falsifiable ideas

### 1. Repaired V3 control: frozen anchor, separated optimization

- Mechanism: keep the entire V1 call-0 computation frozen; detach causal
  features; fit later-depth parameters with solver MSE only; fit critics only
  after the solver freezes, with fresh train-only normalization and epoch 0 as
  a selectable checkpoint.
- Failure addressed: directly removes V3 gradient leakage, stale scaling,
  mixed optimizer, missing epoch-0, and loss-scale domination while changing as
  little of the recurrent design as possible.
- Cheapest decisive experiment: two seeds on discovery-fit, compare the exact
  epoch-0 d1 tensor and fixed later exits on internal validation.
- Likely failure: the tiny later-depth degrees of freedom cannot correct the
  slightly drifting recurrent V1 update.
- Kill criterion: any d1 bit differs, or no later exit passes preregistered
  internal no-regression (`mean(d1-dk) >= -1e-6`) and no critic operating point
  beats the matched randomized mixture.

### 2. Stagewise anchored residual cascade (primary)

- Mechanism: freeze `z1 = z0 + R1(x,z0)` bitwise. Train a separate
  zero-output-initialized adapter for z2 from detached z1. Freeze and accept it
  only if internal validation improves its preceding exit, then repeat for z3.
  Earlier exits are never in an optimizer and alpha is constrained to [0,1].
- Failure addressed: removes destructive interference and makes every new
  stage start at an exact no-op with an eligible epoch-0 checkpoint.
- Cheapest decisive experiment: three seeds, one 128-unit adapter per stage,
  early stopping on grouped internal-validation episodes.
- Likely failure: later error is not predictably reducible from causal inputs,
  so adapters learn average corrections that do not transfer.
- Kill criterion: a stage's clustered improvement CI does not exclude zero;
  reject that stage and all dependent deeper stages. A retained noninferior
  stage is allowed only if its verifier produces a matched-compute advantage.

### 3. Contractive shared corrector with shortcut consistency

- Mechanism: one zero-initialized shared later-stage corrector is called after
  frozen z1. Its residual is bounded by a learned sigmoid step and trained with
  target MSE plus update-contraction and one-step/two-step shortcut consistency.
- Failure addressed: V1's naive repeated calls drift; an explicitly stable
  dynamical correction may make repeated computation useful without separate
  unrestricted blocks.
- Cheapest decisive experiment: two seeds with the same train/validation split
  and parameter scale as the cascade.
- Likely failure: contraction favors a harmless no-op instead of useful error
  reduction, or shared calls still create a biased fixed point.
- Kill criterion: no later fixed exit improves d1 or supplies a verifier
  advantage under the no-regression bound; reject on increasing update norms.

### 4. Verifier-only selective reuse of frozen V1 calls

- Mechanism: do not train a solver. Reuse frozen V1 d1/d2/d4 candidates and fit
  independent local gain critics to decide whether another frozen call is
  worthwhile.
- Failure addressed: isolates allocation quality from solver optimization and
  provides a clean causal control for V2's weak ranking result.
- Cheapest decisive experiment: cross-fitted per-depth critics and budgets
  1.25/1.5/2.0 on internal validation.
- Likely failure: repeated V1 candidates have too little positive-gain mass;
  critic error consumes all oracle headroom.
- Kill criterion: critic rank correlation is nonpositive or no adaptive point
  beats the exact matched randomized depth mixture.

### 5. Disagreement/uncertainty-triggered residual ensemble

- Mechanism: train multiple anchored stage-2 adapters on episode bootstraps and
  continue only when a lower confidence bound on predicted marginal gain is
  positive; solver disagreement is a causal uncertainty feature.
- Failure addressed: conservative routing can avoid rare catastrophic negative
  gains and explicitly handles rare positive gain.
- Cheapest decisive experiment: reuse stagewise seeds as an ensemble and add
  disagreement to the critic.
- Likely failure: ensemble disagreement measures epistemic variance rather than
  reducible error and raises overhead.
- Kill criterion: LCB routing is dominated by the mean critic at every useful
  budget or its gate FLOPs erase the claimed compute advantage.

## Independent referee checklist

Every candidate is rejected if any item fails:

1. Causality: inference inputs are row-local history, past/current action
   blocks, current exit, last update, call index, and solver disagreement only.
2. Leakage: no target, future latent/frame/action, physical label, episode-wide
   statistic, validation statistic, or cross-transition rank enters inference.
3. Stability: d0 and d1 are bitwise anchors; gate loss cannot reach a solver;
   epoch 0 is eligible; accepted later stages freeze before the next stage.
4. Compute: processed sample-block calls, parameters, analytic dense FLOPs, and
   wall-clock latency are recorded. Baselines match realized expected calls.
5. Relevance: beating uniform d2 is insufficient. The required comparator is
   the strongest transition-independent mixture at matched calls, and fixed d1
   plus every retained deeper exit are always shown.

Pilots 1, 2, 3, and the cheap verifier-only control (4) enter the executable
internal tournament. Idea 5 is evaluated by LCB critic scoring using the pilot
2 seed ensemble; it does not receive a separately tuned solver.

Pre-calibration adversarial amendment: the first internal implementation pass
showed that a three-member full-feature critic ensemble can cost more analytic
FLOPs than the later solver adapters themselves. Before any V3 calibration
access, add a preregistered compact-critic control using only the 11 row-local
norm/convergence summaries and a 32/16 MLP at each depth. It uses the identical
cross-fitting, seeds, LCB rules, budgets, and kill criteria. Both full and
compact critics remain reported; this amendment tests whether a block-call
frontier survives honest gate-overhead accounting and is not calibrated on V3
calibration data.

## Isolation and phases

- `fit`: extract only the 420 V3 training episodes directly from the source
  HDF5. The existing V3 `cube_inputs.npz` is never opened because its target
  member also contains forbidden test targets. Split the 420 episodes into
  discovery-fit/internal-validation by episode using the frozen config seed.
- Freeze candidate family, selected seed policy, critic family, thresholds,
  budgets, metrics, and bootstrap procedure in `frozen_tournament.json`, then
  hash it.
- `judge`: extract only the 90 V3 calibration episodes directly from source.
  A one-shot marker and expected cache contents prevent reuse or iteration.
  No V3 test episode is extracted or loaded.
- Discovery failure ends the study. Only a passing gate authorizes creation of
  `runs/lewm_adaptive_compute_v4/` or any fresh confirmatory episode.

## Frozen internal criteria

Internal and final discovery judgments use episode-clustered percentile 95%
CIs. Later-stage acceptance requires positive fixed-exit improvement with CI
lower bound above zero; the only exception is mean noninferiority within
`1e-6` raw MSE when the verifier is necessary and passes every adaptive test.
Critics must have positive out-of-fold rank correlation and ordered realized
gain quantiles. At budgets near 1.25, 1.5, 2.0, and 2.5 calls, at least one
causal adaptive point must beat the strongest transition-independent randomized
mixture at exactly matched expected calls, be nondominated, and not hide a
failure against fixed d1 or a stronger deeper exit. Oracle results are
diagnostic only.
