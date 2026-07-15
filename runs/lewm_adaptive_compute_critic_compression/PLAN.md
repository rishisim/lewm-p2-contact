# LeWM Critic-Compression Discovery Plan

Status: discovery protocol fixed before any V3 calibration target access. The
existing stagewise solver is immutable. This study changes only detached gate
models and their causal inference path.

## Question and permitted conclusion

Can the full 1,046-coordinate causal critic signal be compressed into one
deployable student that improves allocation at matched real compute under a
66,240-FLOP per evaluated-decision gate cap? A discovery pass recommends a
separate preregistered V4 confirmation; it is not fresh-data confirmation.

## Data and nesting

Only the mechanically isolated 420-episode V3 train cache is discovery data.
Two repeats of three grouped outer folds estimate performance. Every family
and hyperparameter is selected using three grouped inner folds of each outer
training set. No transition from an episode crosses a fold. Teacher scores for
student-training rows are cross-fitted inside the corresponding outer training
set. Outer evaluation rows never train their student, teacher, normalizer,
feature selector, compute price, mixture, or whitening transform.

## Candidate ladder

All students share parameters across the three decisions and receive a causal
one-hot depth code. The ladder is: (A) full-coordinate regularized linear;
(B) cross-fitted-teacher distilled linear with realized-gain anchoring and
pairwise ranking; (C) fold-local stable sparse raw coordinates; (D) shared
full-coordinate GELU bottlenecks of width 8 or 16; (E) detached linear/tiny
bottleneck heads on the already-available current prediction and last update.
The prior 11 summaries remain a negative/control family. The rich 128/64
critic is training-time teacher and diagnostic only.

## Routing and metrics

Prices are selected from inner/OOF discovery predictions near 1.25, 1.5, 2.0,
and 2.5 mean calls and then applied unchanged to the outer fold. Evaluation is
causal sequential stopping. Primary loss is raw latent MSE; whitening is fit on
the outer training targets only. Comparators at the student's exact realized
calls are the strongest transition-independent randomized depth mixture, its
analytic expected loss, fixed exits, exact histogram randomization, score
permutation, and the full teacher as a diagnostic.

Episode-clustered paired bootstrap intervals use 2,000 discovery replicates.
The discovery gate requires at least one budget that significantly beats the
matched randomized and analytic mixtures, fixed d1, and histogram null; has
positive whitened direction; is nondominated in calls and fully counted FLOPs;
passes all mechanical audits; and is not a one-repeat or one-fold accident.

## Freeze and one-shot judge

If no family passes, write `critic_compression_internal_failed` and do not
create or read a calibration cache. If one passes, select exactly one family,
configuration, and operating point using discovery OOF results; refit one
student on all discovery episodes using cross-fitted teacher labels; serialize
it; freeze and hash the complete tournament, source, checkpoint, thresholds,
baselines, metrics, bootstrap, and exclusion policy. Only then create a
one-shot receipt before extracting the 90 V3 calibration episodes directly
from the source HDF5. Calibration cannot choose another student or operating
point. Its frozen pass rule is the same raw comparisons, positive whitened
direction, call/FLOP nondominance, and all audits. V3 test targets and the old
combined target NPZ are forbidden.

## Compute accounting

Report separately the common base predictor, fixed V1 call, later adapters,
raw causal feature transformations, normalization, student layers, and control
flow. Count multiply, add/subtract, divide, square-root, activation estimates,
and comparisons. Report parameters, estimated parameter/feature bytes, and
same-device end-to-end dense and sparse latency. FLOPs, latency, and energy are
distinct and will not be substituted after results are observed.
