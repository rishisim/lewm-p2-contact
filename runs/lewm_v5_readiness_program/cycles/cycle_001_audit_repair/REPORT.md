# Cycle 001 audit-repair report

Terminal outcome: `cycle_prospective_discovery_failed`

## Facts

- native_whitened_vs_analytic: estimate 0.0008302119003, 95% CI [0.0005516380708, 0.001120626442]
- native_whitened_vs_within_episode_histogram: estimate 0.0007047791011, 95% CI [0.0005354589815, 0.0008788218086]
- raw_vs_analytic: estimate 6.665827218e-06, 95% CI [2.806210366e-06, 1.053387814e-05]
- raw_vs_fixed_d1: estimate 2.288467206e-05, 95% CI [1.824060653e-05, 2.77740648e-05]
- raw_vs_seeded: estimate 7.18990884e-06, 95% CI [2.555644515e-06, 1.180871835e-05]
- raw_vs_within_episode_histogram: estimate 5.537162585e-06, 95% CI [1.799454708e-06, 9.425377002e-06]

Simultaneous co-primary bounds:

- native_whitened_vs_analytic: simultaneous lower 0.0005959595191
- raw_vs_analytic: simultaneous lower -0.000227586554

Adaptive raw MSE: 0.00384380231. Adaptive PlanOracle-native-whitened MSE: 0.0980414192.

Refiner model calls: 16087; mean calls: 1.411140351; gate evaluations: 15481; exact total FLOPs: 813027132677; gate feature FLOPs: 58843281; dual-affine score FLOPs: 64958276; separately counted non-FLOP comparison/min operations: 77405.

Synchronized actual sparse latency (separate from FLOPs):

- batch 1: median 0.002970792s, p05 0.0027273375s, p95 0.0054744206s
- batch 32: median 0.004487375s, p05 0.0043032627s, p95 0.0073662829s
- batch 256: median 0.015725042s, p05 0.0156699412s, p95 0.015885225s
- batch 1024: median 0.0535215s, p05 0.0532907714s, p95 0.0554038207s

## Independent audit

Independent reproduction passed: True. It separately reproduced effects, all intervals, calls, exact compute, stagewise ranks, and terminal mapping.

## Interpretation

The terminal outcome follows the presealed mapping. A discovery pass permits construction of a separately sealed V5 package; it is not itself V5 confirmation.

## Process validity

Contact and privileged fields were excluded by the model/gate input allowlist. The actual result came from sparse execution; dense execution was a separately accounted comparator/numerical shadow. All frozen modules remained unchanged with no gradients.

## Limitations

This is prospective discovery on one frozen simulator/model distribution. It does not remove confirmation uncertainty and does not claim the first adaptive world model; LoopWM is relevant related work.

Zero V5 outcome episodes were generated.
