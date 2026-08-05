# Consumed-evidence audit and mechanistic diagnosis

Status: completed read-only before any fresh fit, selection, smoke, or
confirmation outcome. No V3 test target, combined V3 cache array, released
HDF5 corpus, raw v005 episode array, or v005 execution array was opened for
this diagnosis.

## Exact claim boundary

V5 v004 is process-valid terminal evidence for one native Cube PlanOracle DGP:
the frozen causal contact-free `stage_dual_r0.01_q0.85` policy lowers both raw
and PlanOracle-native-whitened episode-averaged latent MSE relative to the
strongest transition-independent analytic allocation at exactly matched
counted compute. Its simultaneous lower bounds are `+5.423955e-6` raw and
`+5.135570e-4` native-whitened over 1,600 fresh episodes.

v005 is separate process-valid terminal external-validity evidence over three
one-factor shifts. Four of six DGP-by-endpoint claims are supported. Both
endpoints are supported at PlanOracle action noise 0.2. Only fixed-whitened is
supported under MarkovOracle and PlanOracle random-action probability 0.1.
These separate families do not establish a simultaneous robust four-DGP
claim. Neither package establishes control improvement, contact-aware routing,
universal robustness, wall-clock acceleration, or priority.

## MarkovOracle: observed mechanism

The raw effect is a point loss, `-2.513778663e-6`, with simultaneous lower
bound `-5.677096742e-6`; fixed-whitened remains positive at
`+1.773618066e-4` with lower bound `+8.606212246e-5`.

The oracle-policy intervention induces large shifts in permitted causal
features. At stage 1, RMS standardized mean shifts are 0.82555 for scalar
summaries, 0.57482 for normalized actions, and 0.37431 for the last update.
Named maxima include `normalized_action_t2_d9 = -1.29548` and
`action_change_t1_t2_norm = +1.28747`. The frozen stage-1 score mean moves from
-0.27968 natively to -0.66502 (standardized shift -0.63827); the stage-3 score
shift is -0.83341 standard deviations.

The marginal stage-1 raw solver gain changes sign, from native
`+4.389287e-5` to `-7.500721e-5`. Every stage-1 score bin has negative raw
gain, while fixed-whitened calibration improves toward the high-score bins.
Recorded score/combined-next-gain Spearman correlations are 0.073684,
0.049842, and -0.039782 over 114,000, 10,941, and 4,193 reached rows. Routing
therefore becomes shallower: mean physical calls 1.138772 versus native
1.216678, with depth histogram `[103059, 6748, 3507, 686]`.

The route retains some within-histogram value (`raw_vs_histogram =
+1.208078e-6`) but loses to fixed depth 1 (`-6.182921e-6`) and the exact-compute
analytic allocator (`-2.513779e-6`). Its 129,134 reached gate evaluations cost
1,031,134,990 FLOPs, equivalent to 3,891.663 additional refiner calls. The
analytic comparator can therefore spend mean depth 1.172909 while the adaptive
path physically makes 1.138772 calls. The evidence supports the mechanistic
inference that a DGP-induced feature-to-raw-gain sign/calibration failure,
compounded by exact gate price, prevents raw benefit; it does not identify a
contact or privileged-state mechanism.

## Random-action probability 0.1: observed mechanism

The raw point estimate is an effective tie, `+9.599566e-8`, with simultaneous
lower bound `-3.128984e-6`; fixed-whitened is `+4.263060037e-4` with lower
bound `+3.605363394e-4`.

Gross depth is nearly native: mean calls 1.204798, depth histogram
`[96637, 12788, 3166, 1409]`, and total-variation distance 0.009321 from V5.
The shift is instead concentrated in action-change summaries (roughly
+0.62 to +0.71 standardized units). Raw marginal-gain retention versus V5 is
only 48.66%, 19.31%, and 29.82% across stages. Score/gain ranks remain positive
but weak (0.173844, 0.093925, 0.079117); later high-score raw bins can be
negative even though fixed-whitened bin calibration remains useful.

The adaptive route beats fixed depth 1 by `+5.237612e-6` but loses to
within-episode call-histogram randomization by `-1.810309e-6`, localizing the
defect to raw transition allocation rather than the gross call histogram. Its
135,938 gate evaluations cost 1,085,464,930 FLOPs, or 4,096.712 refiner-call
equivalents. The strongest raw analytic comparator allocates that recovered
budget only between depths 1 and 2. Thus modest causal feature shift, weak
later-stage raw calibration, and overhead erase the raw adaptive advantage.

## Negative control and design implication

Action noise 0.2 supports both endpoints despite comparable gate cost. It
retains about 91--95% of native raw marginal gain and has stronger stagewise
ranks. Therefore neither gate overhead nor an arbitrary action-distribution
shift alone explains failure. The new bounded family must target a single
contact-free policy whose feature-to-endpoint-gain ordering and calibration are
fit across all four DGPs, with stage-1 raw sign safety, later-stage raw yield,
and exact gate price evaluated in an outcome-isolated worst-DGP selection rule.
All numbers above remain diagnostic only and cannot enter new fitting,
normalization, whitening, candidate selection, smoke, or confirmation.
