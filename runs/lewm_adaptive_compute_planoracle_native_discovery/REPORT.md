# PlanOracle-native adaptive-computation discovery report

## Verdict

`planoracle_prospective_discovery_passed`. The frozen causal, contact-free minimax dual-linear gate passed every preregistered prospective criterion at exact total-FLOP parity. This authorizes only a future, entirely fresh V5 confirmation task; V5 was not launched.

## Facts

- Prospective cohort: 180 fresh PlanOracle episodes, 6,840 transitions; mean model calls 1.435819; fully counted adaptive FLOPs 487,826,333,110.
- Raw benefit vs analytic exact-FLOP mixture: 6.2729142e-06, 95% CI [1.4442872e-06, 1.1091834e-05].
- Raw benefit vs conservative seeded mixture: 6.3539369e-06, CI [9.7394282e-07, 1.1843477e-05].
- Raw benefit vs fixed depth 1: 2.3383601e-05, CI [1.6762053e-05, 3.0170636e-05].
- Raw benefit vs call-histogram randomization: 1.2603036e-05, CI [6.7166464e-06, 1.860946e-05].
- Native-whitened benefit vs analytic mixture: 0.00073304667, CI [0.00039554874, 0.0010771121].
- Native-whitened benefit vs histogram randomization: 0.0012925336, CI [0.00089097695, 0.0017112096].
- Stagewise score/gain Spearman correlations: stage 1=0.2071, stage 2=0.1015, stage 3=0.0815; every sign is positive.
- Conservative comparator used 487,826,467,920 FLOPs, weakly more than adaptive by 134,810. All hash, disjointness, causal/contact exclusion, gradient isolation, exact-call, sparse/dense, histogram, parity, nondominance, and stagewise-sign audits passed.

## Assumptions and interpretation

The declared PlanOracle implementation and deterministic reset amendment define the DGP; this study does not claim reconstruction of or distribution matching to the released corpus. The result supports prospective discovery that causal allocation improves both coordinate systems under this declared DGP. LoopWM remains relevant prior adaptive-computation work; no “first adaptive world model” claim is made.

## Limitations

This is discovery, not confirmation. Whitening was estimated from 120 fit episodes and is DGP-specific. The gate search was deliberately compact. NumPy/Accelerate emitted spurious overflow warnings on finite matrix products; independent recomputation reproduced exact finite scores, calls, and headline effects. Post-decision latency covers the sealed gate path only, not a fresh end-to-end base/refiner benchmark, and is excluded from the verdict. Contact was not used and no contact-aware conclusion follows.

## Decision mapping

All four raw CI lower bounds, both native-whitened CI lower bounds, nondominance, integrity, and positive stagewise signs passed; therefore the immutable mapping yields `planoracle_prospective_discovery_passed`.
