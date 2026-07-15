# LeWM adaptive computation V4 confirmatory study

## Outcome

The preregistered one-shot decision is **v4_confirmatory_failed**. The frozen b1.25 policy selected 1.382544 solver calls per transition and achieved raw latent MSE 0.076810371. Its primary benefit versus the strongest transition-independent analytic mixture at exactly the same fully counted total FLOPs was -2.8805324e-05 [-4.1832467e-05, -1.5800503e-05]. The conservative hash-seeded integer mixture was rounded upward by 27924 aggregate FLOPs and gave benefit -2.1478225e-05 [-3.7122408e-05, -6.4037037e-06].

The discovery-whitened benefit versus the analytic exact-total-FLOP comparator was -0.00027689918 [-0.00069798496, 0.00015401074]. Under the frozen rule, a positive interval crossing zero is raw-only confirmation; a positive lower bound is a full pass; a negative point estimate is a failure.

Runtime status is **latency_regressed**. This is reported separately and does not alter the statistical/FLOP verdict. Reliable energy measurement was unavailable.

## Primary checks

- Fixed exits d1–d4 raw MSE: d1=0.076771360, d2=0.076855177, d3=0.076847030, d4=0.076843416.
- Exact-total analytic comparison: -2.8805324e-05 [-4.1832467e-05, -1.5800503e-05].
- Exact-total seeded integer comparison: -2.1478225e-05 [-3.7122408e-05, -6.4037037e-06].
- Fixed d1 comparison: -3.9010905e-05 [-5.5005846e-05, -2.331028e-05].
- Exact histogram randomization: -8.8670792e-06 [-2.3283419e-05, 5.8313371e-06].
- Equal-call analytic decomposition (secondary): -2.9822732e-05 [-4.3283004e-05, -1.6500442e-05].
- Block-call nondominated: False.
- Fully counted FLOP nondominated: False.
- Mechanical validity suite: True.
- Oracle headroom: 0.000386993541 raw MSE.

## Physical robustness

Physical labels were attached only after the primary verdict. The frozen motion/phase/action-controlled result is **contact_aware_claim_not_supported**. The fine-bin matched contact-minus-noncontact call contrast was -0.211657 with 95% CI [-0.319129, -0.10448]. The matched prediction-benefit contrast was 0.000170233 with CI [-0.000128552, 0.000471415]. A null contrast limits the contact-specific claim but does not invalidate overall compute-responsive allocation.

## Scope and caveats

This study confirms or rejects adaptive computation only for the frozen released visual latent physical world model, frozen stagewise solver, frozen full-feature linear gate, and one preregistered operating point. It does not establish a first adaptive world model; LoopWM remains relevant adaptive-depth world-model work in text environments. The released LeWM checkpoint has no original pretraining episode manifest, so pretraining membership of newly generated episodes cannot be mechanically disproven. The documented new seeds, exact raw hashes, and duplicate audit establish generation-time freshness against every accessible prior episode outside the permanently untouched V3 test set.

No solver, student, normalization, whitening transform, price, feature, depth encoding, or stopping rule was trained, selected, or recalibrated from V4 outcomes. Smoke episodes were physically separate and excluded forever. All 300 confirmation episodes were evaluated once after the exclusive access receipt.
