# Cube Event Localization Diagnostic

## Stage A Setup And Pipeline Sanity

- Model repo: `quentinll/lewm-cube`
- Config: `le-wm/.cache/cube_event_diagnostic/model/config.json`
- Weights: `le-wm/.cache/cube_event_diagnostic/model/weights.pt`
- Source HDF5: `/tmp/lewm_cube_stage0/cube_single_expert.h5`
- Pixel subset shape/dtype: `(6030, 224, 224, 3)` / `uint8`
- Frame count: `6030`; sum(ep_len): `6030`
- Lance subset: `le-wm/diagnostics/cube_event_localization/cube_first30_pixel_subset.lance`
- History size: `3`
- Frameskip/action block: `5` raw steps, action input dim `25`
- Pixel normalization: uint8/255 then ImageNet mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- MPS settings: `PYTORCH_ENABLE_MPS_FALLBACK=1`, precision float32, pin_memory false
- MSE mean/std/min/max: `0.141483` / `0.278465` / `0.00181259` / `1.8917`
- %NaN/Inf: `0`; %zero: `0`
- Identity MSE mean: `0.309381`; model/identity: `0.457308`

## Stage B Contact Labels And Aggregate

- Full-data contact stats: `{'total_onsets': 70, 'median_onsets_per_traj': 2.0, 'contact_fraction': 0.4235489220563847, 'median_bout_length': 42.0, 'bouts_per_episode': 2.0}`
- Pre-check contact stats: `{'trajectories': 30, 'total_onsets': 70, 'median_onsets_per_traj': 2.0, 'median_bout_length': 42.0, 'contact_fraction': 0.4235489220563847, 'bouts_per_episode': 2.0}`
- Geometry agreement at 0.04m: `81.542%`
- Contact/non-contact MSE ratio: `4.76519`

## Stage C-E Event Table

| event | raw_odds_ratio | fisher_p_greater | aligned_lift_t0_over_tminus5 | localized_peak_pass | position_adjusted_odds_ratio | position_adjusted_lr_p | trajectory_wilcoxon_p | tag | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sensor grasp onset | 0 | 1 | 0.27 | False | 1.471e-07 | 6.224e-06 | 1 | USABLE | no |
| Sensor release | 0 | 1 | 0.07338 | False | 6.255e-07 | 0.0004988 | 1 | USABLE | no |
| Effector velocity/accel change | 4.493 | 4.596e-09 | 12.97 | True | 5.772 | 9.675e-11 | 2.317e-05 | USABLE | explains |
| Block velocity/accel discontinuity | 15.28 | 2.652e-30 | 16.6 | True | 12.68 | 4.493e-26 | 6.444e-06 | CIRCULAR | explains |
| Orientation delta | 18.94 | 3.236e-35 | 8.608 | True | 14.24 | 6.14e-28 | 2.272e-06 | CIRCULAR-ish | explains |
| Normalized trajectory position | 0 | 1 | 0.2169 | False | 2.714e-07 | 0.00422 | 1 | CONTROL | no |

## Multi-Phase De-Confounding

- First-grasp/bout-1 aligned lift: `nan`
- First-grasp/bout-1 note: first-grasp t-5 baseline unavailable after the 3-frame history gate
- Second-bout release/place aligned lift: `0.142462`
- Decision logic: spike at both phases means event signal, not trajectory position; spike at only one phase flags likely phase confounding.

## Position Control Note

- Position-only odds ratio: `0.0746954`
- Position-only LR p: `1.4191e-11`
- Position-only flag: FLAG: normalized position alone strongly predicts high-MSE rows

## Verification

- Lance row count == source step count: True (6030 == 6030)
- grasp-onset alignment n@t0 == grasp onset count: True (70 == 70)
- release alignment n@t0 == release count: True (63 == 63)
- angle wrapping pi/-pi smoke value: 0.02; pass=True
- pre-check vs full-data contact divergence: none
- py_compile self-check: passed

## Artifact Checks

- cube_event_records.csv: exists=True, nonempty=True, size=215528
- sensor_grasp_onset_aligned_curve.csv: exists=True, nonempty=True, size=2761
- sensor_grasp_onset_aligned_curve.png: exists=True, nonempty=True, size=96883
- sensor_release_aligned_curve.csv: exists=True, nonempty=True, size=2768
- sensor_release_aligned_curve.png: exists=True, nonempty=True, size=100898
- effector_kinematic_change_aligned_curve.csv: exists=True, nonempty=True, size=2751
- effector_kinematic_change_aligned_curve.png: exists=True, nonempty=True, size=116460
- block_kinematic_discontinuity_aligned_curve.csv: exists=True, nonempty=True, size=2758
- block_kinematic_discontinuity_aligned_curve.png: exists=True, nonempty=True, size=92871
- orientation_delta_aligned_curve.csv: exists=True, nonempty=True, size=2754
- orientation_delta_aligned_curve.png: exists=True, nonempty=True, size=92395
- normalized_trajectory_position_aligned_curve.csv: exists=True, nonempty=True, size=1668
- normalized_trajectory_position_aligned_curve.png: exists=True, nonempty=True, size=87886
- phase_grasp_bout1_aligned_curve.csv: exists=True, nonempty=True, size=1623
- phase_release_bout2_aligned_curve.csv: exists=True, nonempty=True, size=2763
- phase_grasp_bout1_aligned_curve.png: exists=True, nonempty=True, size=77076
- phase_release_bout2_aligned_curve.png: exists=True, nonempty=True, size=103581
- cube_event_aligned_average.png: exists=True, nonempty=True, size=339571
- cube_event_raster.png: exists=True, nonempty=True, size=560834

## Five-Line Plain-Language Summary

1. Cube pixel LeWM one-step MSE was finite and non-collapsed: mean 0.1415, model/identity 0.4573.
2. Sensor contact labels matched the pre-check: 70 onsets, contact fraction 0.424.
3. Contact-step aggregate MSE ratio was 4.765x versus non-contact steps.
4. Grasp-onset aligned lift was 0.270; release aligned lift was 0.073.
5. Event verdicts are reported with both position-adjusted and trajectory-level tests in the table above.
