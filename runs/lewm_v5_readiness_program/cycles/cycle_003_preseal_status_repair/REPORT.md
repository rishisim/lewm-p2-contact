# Cycle 003 scale-valid-simultaneous report

Terminal outcome: `cycle_prospective_discovery_failed`

## Facts

- native_whitened_vs_analytic: estimate 0.0004772376264, 95% CI [0.0001730100738, 0.0007880153511]
- native_whitened_vs_within_episode_histogram: estimate 0.0005631225358, 95% CI [0.0004087937037, 0.0007236357451]
- raw_vs_analytic: estimate 3.61511985e-06, 95% CI [-2.261894155e-07, 7.547963359e-06]
- raw_vs_fixed_d1: estimate 2.549346838e-05, 95% CI [2.047451597e-05, 3.080169625e-05]
- raw_vs_seeded: estimate 1.885642986e-06, 95% CI [-2.598494217e-06, 6.332599772e-06]
- raw_vs_within_episode_histogram: estimate 6.170854838e-06, 95% CI [2.482103009e-06, 9.961221742e-06]

Simultaneous co-primary bounds:

- native_whitened_vs_analytic: simultaneous lower 0.0001730100738
- raw_vs_analytic: simultaneous lower -2.261894155e-07

Adaptive raw MSE: 0.003509875812. Adaptive PlanOracle-native-whitened MSE: 0.09873129669.

Refiner model calls: 16151; mean calls: 1.416754386; gate evaluations: 15499; exact total FLOPs: 813044234063; gate feature FLOPs: 58911699; dual-affine score FLOPs: 65033804; separately counted non-FLOP comparison/min operations: 77495.

Synchronized actual sparse latency (separate from FLOPs):

- batch 1: median 0.003145875s, p05 0.0028126331s, p95 0.0053920081s
- batch 32: median 0.006418083s, p05 0.0060286625s, p95 0.0119288662s
- batch 256: median 0.015683792s, p05 0.0153114749s, p95 0.0161076587s
- batch 1024: median 0.055128083s, p05 0.0532700036s, p95 0.0855953706s

## Independent audit

Independent reproduction passed: True. It separately reproduced effects, all intervals, calls, exact compute, stagewise ranks, and terminal mapping.

## Interpretation

The terminal outcome follows the presealed mapping. A discovery pass permits construction of a separately sealed V5 package; it is not itself V5 confirmation.

## Process validity

Contact and privileged fields were excluded by the model/gate input allowlist. The actual result came from sparse execution; dense execution was a separately accounted comparator/numerical shadow. All frozen modules remained unchanged with no gradients.

## Limitations

This is prospective discovery on one frozen simulator/model distribution. It does not remove confirmation uncertainty and does not claim the first adaptive world model; LoopWM is relevant related work.

Zero V5 outcome episodes were generated.
