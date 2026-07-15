# Cycle 007 MPS threshold-dtype repair report

Terminal outcome: `cycle_prospective_discovery_passed`

## Facts

- native_whitened_vs_analytic: estimate 0.0006087347716, 95% CI [0.0004154267275, 0.0008114536583]
- native_whitened_vs_within_episode_histogram: estimate 0.0006356099501, 95% CI [0.0005006791327, 0.0007753027183]
- raw_vs_analytic: estimate 7.118699226e-06, 95% CI [4.058252006e-06, 1.039072355e-05]
- raw_vs_fixed_d1: estimate 1.968147439e-05, 95% CI [1.569548081e-05, 2.405351336e-05]
- raw_vs_seeded: estimate 7.446025593e-06, 95% CI [3.510244615e-06, 1.165235544e-05]
- raw_vs_within_episode_histogram: estimate 6.137979301e-06, 95% CI [2.840216522e-06, 9.551988544e-06]

Simultaneous co-primary bounds:

- native_whitened_vs_analytic: simultaneous lower 0.0004154267275
- raw_vs_analytic: simultaneous lower 4.058252006e-06

Adaptive raw MSE: 0.003445511274. Adaptive PlanOracle-native-whitened MSE: 0.09790420746.

Refiner model calls: 13898; mean calls: 1.219122807; gate evaluations: 13751; exact total FLOPs: 812433135415; gate feature FLOPs: 52267551; dual-affine score FLOPs: 57534184; separately counted non-FLOP comparison/min operations: 68755.

Synchronized actual sparse latency (separate from FLOPs):

- batch 1: median 0.002769333s, p05 0.0027235374s, p95 0.0028068206s
- batch 32: median 0.005351708s, p05 0.0051435918s, p95 0.0077542626s
- batch 256: median 0.014439125s, p05 0.0143463669s, p95 0.0146398123s
- batch 1024: median 0.0492865s, p05 0.0492174786s, p95 0.0494421083s

## Independent audit

Independent reproduction passed: True. It separately reproduced effects, all intervals, calls, exact compute, stagewise ranks, and terminal mapping.

## Interpretation

The terminal outcome follows the presealed mapping. Stagewise rank remained a statistical criterion; it was not treated as a process-integrity defect. A discovery pass permits construction of a separately sealed V5 package; it is not itself V5 confirmation.

## Process validity

Contact and privileged fields were excluded by the model/gate input allowlist. The actual result came from sparse execution; dense execution was a separately accounted comparator/numerical shadow. All frozen modules remained unchanged with no gradients.

## Limitations

This is prospective discovery on one frozen simulator/model distribution after an isolated 240-episode fit and 120-episode selection design. It does not remove confirmation uncertainty and does not claim the first adaptive world model; LoopWM is relevant related work.

Zero V5 outcome episodes were generated.
